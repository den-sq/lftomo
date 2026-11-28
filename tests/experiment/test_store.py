from datetime import datetime, timedelta, timezone
import json
import os

import pytest

from lftomo.config import Recon
from lftomo.experiment import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	TrialAttemptRecord,
)
from lftomo.experiment.runner import (
	ExperimentInputs,
	IndependentExperimentRunner,
	atomic_write_model,
)
from lftomo.experiment.store import (
	ClaimRecord,
	PersistedExperimentRunner,
	PersistedExperimentStore,
	ResumePolicy,
)
from tests.experiment.process_helpers import AlphaExecutor, TimeoutOneExecutor

from lftomo.util.shapes import ProjOrder
from lftomo.util.window import TomoWindow


HASHES = [f"{index:064x}" for index in range(1, 10)]


def config(values=(0.01, 0.02), *, timeout=None):
	return ExperimentConfig.model_validate({
		"name": "persistent-test",
		"candidate_space": {
			"pipeline": [{
				"id": "phase", "kind": "phase", "algorithm": "paganin",
				"sweep": {"kind": "list", "params": {"alpha": list(values)}},
			}],
		},
		"sampler": {"name": "exhaustive", "seed": 1},
		"resources": {"max_concurrency": 2},
		"metrics": [{
			"id": "quality", "metric": "quality", "protocol_version": "1",
			"direction": "minimize",
		}],
		"objectives": [{"metric_id": "quality"}],
		"windows": [{"id": "screen", "y_start": 0, "y_stop": 4}],
		"fidelities": [{
			"id": "screen", "windows": ["screen"],
			**({"timeout_seconds": timeout} if timeout is not None else {}),
		}],
		"aggregation": {"metric_ids": ["quality"]},
	})


def inputs(content_hash=HASHES[0]):
	# The persistence executor has one sample per configured detector row.
	return ExperimentInputs(
		source=TomoWindow.from_bounds(ProjOrder(1, 4, 1), dtype="uint16"),
		dataset=DatasetIdentity(dataset_id="fixture", content_hash=content_hash),
		configuration=ConfigurationIdentity(scan_config_hash=HASHES[1]),
		code=CodeIdentity(version="test", git_commit="abcdef123456"),
		environment=EnvironmentIdentity(
			python="3.11", platform="test", backend="numpy",
		),
	)


def runner(
	root, executor, *, experiment_inputs=None, values=(0.01, 0.02), timeout=None,
):
	return IndependentExperimentRunner(
		config(values, timeout=timeout), experiment_inputs or inputs(), root, executor,
	)


@pytest.mark.system
def test_clean_resume_reuses_successes_and_force_rerun_appends_history(tmp_path):
	call_root = tmp_path / "calls"
	execute = AlphaExecutor(call_root=call_root)

	first = PersistedExperimentRunner(runner(tmp_path, execute)).run()
	second = PersistedExperimentRunner(runner(tmp_path, execute)).run()
	assert len(list(call_root.glob("*.json"))) == 2
	assert [item.attempt_id for item in second.attempts] == [
		item.attempt_id for item in first.attempts
	]
	assert all(item.status == "succeeded" for item in second.attempts)

	forced = PersistedExperimentRunner(
		runner(tmp_path, execute), policy=ResumePolicy(force=True),
	).run()
	assert len(list(call_root.glob("*.json"))) == 4
	assert all(item.sequence == 1 for item in forced.attempts)
	inspection = PersistedExperimentStore(tmp_path).inspect()
	assert all([item.sequence for item in history.attempts] == [0, 1]
		for history in inspection.trials)
	manifest = json.loads((tmp_path / "manifest.json").read_text())
	assert len(manifest["attempt_ids"]) == 4


@pytest.mark.system
def test_failed_and_interrupted_attempts_require_explicit_retry(tmp_path):
	execute = AlphaExecutor(fail_on=0.01)

	failed = PersistedExperimentRunner(
		runner(tmp_path, execute, values=(0.01,)),
	).run().attempts[0]
	assert failed.status == "failed"
	execute.fail_on = None
	skipped = PersistedExperimentRunner(
		runner(tmp_path, execute, values=(0.01,)),
	).run().attempts[0]
	assert skipped.status == "skipped"
	retried = PersistedExperimentRunner(
		runner(tmp_path, execute, values=(0.01,)),
		policy=ResumePolicy(retry_failed=True),
	).run().attempts[0]
	assert retried.status == "succeeded"
	assert retried.sequence == 2
	assert retried.previous_attempt_id == skipped.attempt_id

	interrupted_root = tmp_path / "interrupted"
	interrupted = PersistedExperimentRunner(runner(
		interrupted_root,
		AlphaExecutor(interrupt=True),
		values=(0.01,),
	)).run().attempts[0]
	assert interrupted.status == "interrupted"
	resumed = PersistedExperimentRunner(
		runner(
			interrupted_root,
			AlphaExecutor(),
			values=(0.01,),
		),
		policy=ResumePolicy(retry_interrupted=True),
	).run().attempts[0]
	assert resumed.status == "succeeded"


@pytest.mark.system
def test_timeout_failure_requires_retry_and_tighter_policy_rechecks_success(tmp_path):
	timed_out = PersistedExperimentRunner(runner(
		tmp_path, TimeoutOneExecutor(), values=(0.01,), timeout=2.0,
	)).run().attempts[0]
	assert timed_out.status == "failed"
	assert timed_out.failure.type == "TrialTimeoutError"

	skipped = PersistedExperimentRunner(runner(
		tmp_path, AlphaExecutor(), values=(0.01,), timeout=2.0,
	)).run().attempts[0]
	assert skipped.status == "skipped"
	stale_output = (
		tmp_path / "trials" / timed_out.trial_id.value
		/ "windows" / "stale" / "output" / "partial.tif"
	)
	stale_output.parent.mkdir(parents=True)
	stale_output.write_bytes(b"partial")
	retried = PersistedExperimentRunner(
		runner(
			tmp_path, TimeoutOneExecutor(hang_on=999),
			values=(0.01,), timeout=2.0,
		),
		policy=ResumePolicy(retry_failed=True),
	).run().attempts[0]
	assert retried.status == "succeeded"
	assert retried.trial_id == timed_out.trial_id
	assert retried.sequence == 2
	success_output = (
		tmp_path / "trials" / retried.trial_id.value
		/ "windows" / "completed" / "output" / "completed.txt"
	)
	assert success_output.is_file()
	assert not stale_output.exists()
	assert any(
		path.name.startswith("windows.retry-")
		for path in (
			tmp_path / "quarantine" / "trials" / retried.trial_id.value
		).iterdir()
	)

	prepared = runner(
		tmp_path, AlphaExecutor(), values=(0.01,), timeout=1e-9,
	).prepare()[1][0]
	decision = PersistedExperimentStore(tmp_path).decide(
		prepared.identity, ResumePolicy(), timeout_seconds=prepared.timeout_seconds,
	)
	assert decision.action == "execute"
	assert decision.prior_attempt.attempt_id == retried.attempt_id
	assert "exceeded the current timeout policy" in decision.reason


@pytest.mark.system
def test_identity_mismatch_has_fail_skip_and_rerun_policies(tmp_path):
	execute = AlphaExecutor()
	PersistedExperimentRunner(runner(
		tmp_path, execute, experiment_inputs=inputs(HASHES[0]), values=(0.01,),
	)).run()
	changed = runner(
		tmp_path, execute, experiment_inputs=inputs(HASHES[2]), values=(0.01,),
	)
	with pytest.raises(ValueError, match="dataset.content_hash"):
		PersistedExperimentRunner(changed).run()
	incompatible = PersistedExperimentRunner(
		changed, policy=ResumePolicy(mismatch="skip"),
	).run().attempts[0]
	assert incompatible.status == "incompatible"
	assert incompatible.failure.details["diagnostics"][0]["field"] == "dataset.content_hash"
	rerun = PersistedExperimentRunner(
		changed, policy=ResumePolicy(mismatch="rerun"),
	).run().attempts[0]
	assert rerun.status == "succeeded"
	assert rerun.sequence == 1


def test_claims_exclude_concurrent_processes_and_stale_recovery_is_explicit(tmp_path):
	base = runner(
		tmp_path, AlphaExecutor(),
		values=(0.01,),
	)
	store = PersistedExperimentStore(tmp_path)
	store.initialize(base.config)
	trial = base.prepare()[1][0]
	store.register_trial(trial.identity)

	with store.claim(trial.identity, ResumePolicy()):
		with pytest.raises(RuntimeError, match="claimed"):
			with store.claim(trial.identity, ResumePolicy()):
				pass

	claim_path = trial.output_root / "claim.json"
	started = datetime.now(timezone.utc) - timedelta(hours=2)
	running = TrialAttemptRecord(
		attempt_id=f"attempt-{trial.identity.trial_id.value[:12]}-0000",
		sequence=0,
		trial_id=trial.identity.trial_id,
		identity=trial.identity,
		status="running",
		created_at=started,
		started_at=started,
	)
	atomic_write_model(
		trial.output_root / "attempts" / f"{running.attempt_id}.json", running,
	)
	partial = trial.output_root / "partial"
	partial.mkdir()
	(partial / "unfinished.tif").write_bytes(b"partial")
	atomic_write_model(claim_path, ClaimRecord(
		owner="abandoned", pid=999999, host="other-host",
		claimed_at=datetime.now(timezone.utc) - timedelta(hours=2),
	))
	with pytest.raises(RuntimeError, match="recover_stale"):
		with store.claim(trial.identity, ResumePolicy(stale_after_seconds=1)):
			pass
	with store.claim(trial.identity, ResumePolicy(
		recover_stale=True, stale_after_seconds=1,
	)):
		assert claim_path.exists()
	assert not claim_path.exists()
	assert any((tmp_path / "quarantine").rglob("*stale-claim*"))
	recovered = store.load_attempts(trial.identity)[0]
	assert recovered.status == "interrupted"
	assert recovered.failure.type == "AbandonedClaim"
	assert not partial.exists()


def test_partial_and_invalid_attempt_files_are_quarantined(tmp_path):
	base = runner(
		tmp_path, AlphaExecutor(),
		values=(0.01,),
	)
	store = PersistedExperimentStore(tmp_path)
	store.initialize(base.config)
	trial = base.prepare()[1][0]
	store.register_trial(trial.identity)
	attempts = trial.output_root / "attempts"
	attempts.mkdir(parents=True)
	invalid = attempts / "broken.json"
	invalid.write_text("{not-json", encoding="utf-8")
	partial = trial.output_root / "partial"
	partial.mkdir()
	(partial / "volume.tif").write_bytes(b"partial")

	inspection = store.inspect()
	assert inspection.quarantined
	assert not invalid.exists()
	quarantined = store.quarantine_partial_outputs(trial.identity)
	assert quarantined and not partial.exists()
	assert not store.load_attempts(trial.identity)


def test_store_rejects_unknown_schema_and_identity_is_directly_reconstructable(tmp_path):
	base = runner(
		tmp_path, AlphaExecutor(),
		values=(0.01,),
	)
	store = PersistedExperimentStore(tmp_path)
	store.initialize(base.config)
	trial = base.prepare()[1][0]
	path = store.register_trial(trial.identity) / "identity.json"
	reconstructed = type(trial.identity).model_validate_json(path.read_text())
	assert reconstructed.model_dump(mode="json") == trial.identity.model_dump(mode="json")

	metadata = json.loads((tmp_path / "store.json").read_text())
	metadata["schema_version"] = 999
	(tmp_path / "store.json").write_text(json.dumps(metadata), encoding="utf-8")
	with pytest.raises(ValueError, match="unsupported experiment store schema"):
		store.inspect()
