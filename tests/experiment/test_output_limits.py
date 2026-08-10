from pathlib import Path
import threading

import numpy as np
import pytest

from lftomo.config import Recon
from lftomo.evaluation import PreviewContract
from lftomo.experiment import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	OutputLedgerIntegrityError,
	OutputLimitExceeded,
	PendingOutput,
	ProgressEventWriter,
	export_blinded_previews,
	read_progress_events,
)
from lftomo.experiment.outputs import OutputBudgetLedger
from lftomo.experiment.runner import (
	ExperimentInputs,
	ExperimentRunPolicy,
	IndependentExperimentRunner,
	TrialExecution,
)
from lftomo.experiment.store import PersistedExperimentRunner, ResumePolicy
import lftomo.experiment.outputs as outputs_module


HASHES = [f"{index:064x}" for index in range(1, 4)]


def _config(
	*,
	values=(0.01,),
	experiment_limit=16,
	fidelity_limit=16,
	policy_limit=16,
	concurrency=1,
):
	return ExperimentConfig.model_validate({
		"name": "output-ledger-test",
		"candidate_space": Recon.model_validate({
			"pipeline": [{
				"id": "phase", "kind": "phase", "algorithm": "paganin",
				"sweep": {"kind": "list", "params": {"alpha": list(values)}},
			}],
		}).model_dump(mode="json"),
		"sampler": {"name": "exhaustive", "seed": 7},
		"resources": {
			"max_concurrency": concurrency,
			"max_output_bytes": experiment_limit,
		},
		"metrics": [{
			"id": "quality", "metric": "quality", "protocol_version": "1",
			"direction": "minimize",
		}],
		"objectives": [{"metric_id": "quality"}],
		"windows": [{"id": "screen", "y_start": 0, "y_stop": 2}],
		"fidelities": [{
			"id": "screen", "windows": ["screen"],
			"max_output_bytes": fidelity_limit,
		}],
		"aggregation": {"metric_ids": ["quality"]},
		"output": {"screening": "configured_writers", "max_bytes": policy_limit},
	})


def _inputs():
	return ExperimentInputs(
		dataset=DatasetIdentity(dataset_id="fixture", content_hash=HASHES[0]),
		configuration=ConfigurationIdentity(scan_config_hash=HASHES[1]),
		code=CodeIdentity(version="test", git_commit="abcdef123456"),
		environment=EnvironmentIdentity(
			python="3.11", platform="test", backend="numpy",
		),
	)


def _executor(size: int, *, slices=1, barrier=None):
	def execute(identity, output_root):
		temporary = output_root / "partial"
		temporary.mkdir(parents=True, exist_ok=True)
		pending = []
		for index in range(slices):
			path = temporary / f"slice-{index}.tif"
			path.write_bytes(bytes([index + 1]) * size)
			pending.append(PendingOutput(
				role="screening_output",
				temporary_path=path,
				relative_path=(
					Path("published") / "screen" / identity.candidate.value
					/ f"slice-{index}.tif"
				),
			))
		if barrier is not None:
			barrier.wait(timeout=5)
		return TrialExecution(metrics={"quality": 1}, pending_outputs=tuple(pending))
	return execute


def _runner(root, config, execute, *, policy=None, writer=None):
	return IndependentExperimentRunner(
		config, _inputs(), root, execute, policy=policy, event_writer=writer,
	)


def test_unknown_estimate_enforces_actual_bytes_at_and_over_limit(tmp_path):
	exact_root = tmp_path / "exact"
	config = _config(experiment_limit=5, fidelity_limit=5, policy_limit=5)
	writer = ProgressEventWriter(exact_root, config.experiment_id, durable=True)
	exact = _runner(exact_root, config, _executor(5), writer=writer).run()

	assert exact.preflight.estimated_output_bytes is None
	assert exact.attempts[0].status == "succeeded"
	assert exact.output_accounting.attempt_committed_bytes == 0
	assert exact.output_accounting.experiment_committed_bytes == 5
	assert exact.attempts[0].output_accounting.attempt_committed_bytes == 5
	assert exact.attempts[0].result.outputs[0].path.read_bytes() == b"\x01" * 5
	assert "output.committed" in {
		event.kind for event in read_progress_events(exact_root / "progress.jsonl")
	}

	over_root = tmp_path / "over"
	over_writer = ProgressEventWriter(over_root, config.experiment_id, durable=True)
	over = _runner(over_root, config, _executor(6), writer=over_writer).run()
	attempt = over.attempts[0]
	assert attempt.status == "failed"
	assert attempt.failure.type == "OutputLimitExceeded"
	assert attempt.failure.details == {
		"scope": "experiment",
		"limit_bytes": 5,
		"committed_bytes": 0,
		"requested_bytes": 6,
	}
	assert not any((over_root / "published").rglob("*.tif"))
	assert "output.limit_exceeded" in {
		event.kind for event in read_progress_events(over_root / "progress.jsonl")
	}


def test_cumulative_slices_and_policy_scope_are_enforced_before_publication(tmp_path):
	config = _config(experiment_limit=20, fidelity_limit=20, policy_limit=5)
	summary = _runner(tmp_path, config, _executor(3, slices=2)).run()
	attempt = summary.attempts[0]

	assert attempt.status == "failed"
	assert attempt.failure.type == "OutputLimitExceeded"
	assert attempt.failure.details["scope"] == "policy"
	assert attempt.failure.details["requested_bytes"] == 6
	assert not any((tmp_path / "published").rglob("*.tif"))


@pytest.mark.parametrize(("limits", "scope"), [
	((5, 20, 20), "experiment"),
	((20, 5, 20), "fidelity"),
	((20, 20, 5), "policy"),
])
def test_each_output_limit_scope_is_independently_enforced(tmp_path, limits, scope):
	config = _config(
		experiment_limit=limits[0], fidelity_limit=limits[1], policy_limit=limits[2],
	)
	attempt = _runner(tmp_path, config, _executor(6)).run().attempts[0]

	assert attempt.status == "failed"
	assert attempt.failure.details["scope"] == scope


def test_publication_failure_rolls_back_every_file(monkeypatch, tmp_path):
	config = _config(experiment_limit=20, fidelity_limit=20, policy_limit=20)
	original_replace = outputs_module.os.replace
	publication_count = 0

	def fail_second_publication(source, destination):
		nonlocal publication_count
		source = Path(source)
		if source.parent.name == "new":
			publication_count += 1
			if publication_count == 2:
				raise OSError("injected publication failure")
		return original_replace(source, destination)

	monkeypatch.setattr(outputs_module.os, "replace", fail_second_publication)
	attempt = _runner(tmp_path, config, _executor(3, slices=2)).run().attempts[0]

	assert attempt.status == "failed"
	assert attempt.failure.type == "OSError"
	assert not any((tmp_path / "published").rglob("*.tif"))
	assert not any((tmp_path / ".output-transactions").glob("*/transaction.json"))


def test_concurrent_finalists_serialize_against_one_remaining_budget(tmp_path):
	barrier = threading.Barrier(2)
	config = _config(
		values=(0.01, 0.02), experiment_limit=7, fidelity_limit=7,
		policy_limit=7, concurrency=2,
	)
	summary = _runner(tmp_path, config, _executor(4, barrier=barrier)).run()

	assert sorted(attempt.status for attempt in summary.attempts) == ["failed", "succeeded"]
	assert summary.output_accounting.experiment_committed_bytes == 4
	assert len(tuple((tmp_path / "published").rglob("*.tif"))) == 1
	failed = next(attempt for attempt in summary.attempts if attempt.status == "failed")
	assert failed.failure.type == "OutputLimitExceeded"
	assert failed.failure.details["committed_bytes"] == 4


def test_force_rerun_replaces_output_without_double_counting(tmp_path):
	config = _config(experiment_limit=5, fidelity_limit=5, policy_limit=5)
	first = PersistedExperimentRunner(
		_runner(tmp_path, config, _executor(4)),
	).run()
	second = PersistedExperimentRunner(
		_runner(tmp_path, config, _executor(4)),
		policy=ResumePolicy(force=True),
	).run()

	assert first.output_accounting.experiment_committed_bytes == 4
	assert second.output_accounting.experiment_committed_bytes == 4
	assert second.attempts[0].sequence == 1
	assert len(tuple((tmp_path / "published").rglob("*.tif"))) == 1


def test_rejected_replacement_preserves_the_prior_successful_output(tmp_path):
	config = _config(experiment_limit=5, fidelity_limit=5, policy_limit=5)
	first = PersistedExperimentRunner(
		_runner(tmp_path, config, _executor(4)),
	).run()
	path = first.attempts[0].result.outputs[0].path
	second = PersistedExperimentRunner(
		_runner(tmp_path, config, _executor(6)),
		policy=ResumePolicy(force=True),
	).run()

	assert second.attempts[0].status == "failed"
	assert second.attempts[0].failure.details["requested_bytes"] == 2
	assert path.read_bytes() == b"\x01" * 4
	assert second.output_accounting.experiment_committed_bytes == 4


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_resume_detects_missing_or_changed_committed_output(tmp_path, damage):
	config = _config()
	first = PersistedExperimentRunner(
		_runner(tmp_path, config, _executor(4)),
	).run()
	path = first.attempts[0].result.outputs[0].path
	if damage == "missing":
		path.unlink()
	else:
		path.write_bytes(b"changed")

	with pytest.raises(OutputLedgerIntegrityError, match=damage):
		PersistedExperimentRunner(_runner(tmp_path, config, _executor(4))).run()


def test_resume_quarantines_unrecorded_managed_output_and_rebuilds_ledger(tmp_path):
	config = _config()
	first = PersistedExperimentRunner(
		_runner(tmp_path, config, _executor(4)),
	).run()
	orphan = tmp_path / "published" / "orphan.tif"
	orphan.write_bytes(b"orphan")

	resumed = PersistedExperimentRunner(
		_runner(tmp_path, config, _executor(4)),
	).run()

	assert resumed.attempts[0].attempt_id == first.attempts[0].attempt_id
	assert not orphan.exists()
	assert any((tmp_path / "quarantine" / "output-ledger").rglob("orphan.tif"))
	assert resumed.output_accounting.experiment_committed_bytes == 4


def test_explicit_override_is_enforced_consistently_and_recorded(tmp_path):
	config = _config(experiment_limit=1, fidelity_limit=1, policy_limit=1)
	summary = _runner(
		tmp_path,
		config,
		_executor(4),
		policy=ExperimentRunPolicy(override_limits=True),
	).run()

	assert summary.attempts[0].status == "succeeded"
	assert summary.output_accounting.experiment_committed_bytes == 4
	assert summary.output_accounting.limits.override_limits is True
	manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
	assert '"override_limits": true' in manifest


def test_blinded_preview_uses_the_shared_ledger_and_leaves_no_rejected_file(tmp_path):
	config = _config(experiment_limit=1, fidelity_limit=1, policy_limit=1)
	ledger = OutputBudgetLedger(tmp_path, config, "screen")
	contract = PreviewContract(
		scale_min=0, scale_max=1, blinded=True, format="png",
	)

	with pytest.raises(OutputLimitExceeded) as raised:
		export_blinded_previews(
			{"a" * 64: np.ones((4, 4), dtype=np.float32)},
			tmp_path,
			contract,
			blinding_key="private-test-key",
			ledger=ledger,
			attempt_id="attempt-preview-0000",
			trial_id="b" * 64,
			staging_root=tmp_path / "preview-staging",
		)

	assert raised.value.scope == "experiment"
	assert not any((tmp_path / "previews").rglob("*.png"))
	assert not (tmp_path / "private" / "preview-mapping" / "mapping.json").exists()
	assert not (tmp_path / "preview-staging").exists()
