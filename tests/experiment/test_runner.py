from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import threading
import time

from click.testing import CliRunner
import psutil
import pytest
import yaml

from lftomo.config import Recon
from lftomo.experiment import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	PathResourceError,
	ProgressEventWriter,
	read_progress_events,
)
from lftomo.experiment.runner import (
	ExperimentInputs,
	ExperimentRunPolicy,
	IndependentExperimentRunner,
)
import lftomo.experiment.runner as runner_module
from lftomo.experiment.selection import select_candidates
from lftomo.experiment.store import PersistedExperimentRunner
from lftomo.experiment.cli import main
from tests.experiment.process_helpers import (
	AlphaExecutor,
	CrashOneExecutor,
	DisabledCacheExecutor,
	HangingManagedExecutor,
	HangingProcessTreeExecutor,
	TimeoutOneExecutor,
)

from lftomo.util.shapes import ProjOrder
from lftomo.util.window import TomoWindow


HASHES = [f"{index:064x}" for index in range(1, 8)]


def candidate_space(values=(0.01, 0.02)):
	return Recon.model_validate({
		"pipeline": [{
			"id": "phase",
			"kind": "phase",
			"algorithm": "paganin",
			"sweep": {"kind": "list", "params": {"alpha": list(values)}},
		}],
	})


def experiment_config(
	*, candidate=None, sampler=None, concurrency=2, output_limit=None, timeout=None,
):
	return ExperimentConfig.model_validate({
		"name": "runner-test",
		"candidate_space": (candidate or candidate_space()).model_dump(mode="json"),
		"sampler": sampler or {"name": "exhaustive", "seed": 7},
		"resources": {
			"max_concurrency": concurrency,
			**({"max_output_bytes": output_limit} if output_limit is not None else {}),
		},
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
		"output": {"screening": "metrics_only"},
	})


def inputs():
	# The fake executor needs only the four configured detector rows.
	return ExperimentInputs(
		source=TomoWindow.from_bounds(ProjOrder(1, 4, 1), dtype="uint16"),
		dataset=DatasetIdentity(dataset_id="fixture", content_hash=HASHES[0]),
		configuration=ConfigurationIdentity(scan_config_hash=HASHES[1]),
		code=CodeIdentity(version="test", git_commit="abcdef123456"),
		environment=EnvironmentIdentity(
			python="3.11", platform="test", backend="numpy",
		),
	)


def alpha(identity):
	return identity.resolved_recon.pipeline[0].params["alpha"]


def test_exhaustive_selection_is_deterministic_and_deduplicates_resolved_candidates():
	config = experiment_config(candidate=candidate_space((0.01, 0.01, 0.02)))

	first = select_candidates(config)
	second = select_candidates(config)

	assert [item.candidate_key for item in first] == [item.candidate_key for item in second]
	assert [item.recon.pipeline[0].params["alpha"] for item in first] == [0.01, 0.02]
	assert first[0].duplicate_coordinates == 1
	assert first[0].sampler.coordinate == {"enumeration_index": 0}


def test_seeded_sobol_selection_is_reproducible_recorded_and_budgeted():
	continuous = Recon.model_validate({
		"pipeline": [{
			"id": "phase", "kind": "phase", "algorithm": "paganin",
			"sweep": {"kind": "domain", "params": {
				"alpha": {"kind": "log", "start": 0.001, "stop": 0.1},
			}},
		}],
	})
	config = experiment_config(
		candidate=continuous,
		sampler={"name": "sobol", "seed": 41, "budget": 8},
	)

	first = select_candidates(config)
	second = select_candidates(config)

	assert len(first) == 8
	assert [item.sampler.coordinate for item in first] == [
		item.sampler.coordinate for item in second
	]
	assert [item.candidate_key for item in first] == [item.candidate_key for item in second]
	assert all(0.001 <= item.recon.pipeline[0].params["alpha"] <= 0.1 for item in first)
	assert all("q0" in item.sampler.coordinate for item in first)


def test_exhaustive_budget_is_a_guard_not_silent_truncation():
	config = experiment_config(sampler={"name": "exhaustive", "seed": 0, "budget": 1})

	with pytest.raises(ValueError, match="exceeds sampler budget"):
		select_candidates(config)


@pytest.mark.system
def test_independent_concurrent_execution_has_deterministic_identity_and_leaderboard(tmp_path):
	config = experiment_config()
	runner = IndependentExperimentRunner(
		config, inputs(), tmp_path,
		AlphaExecutor(
			mutate_identity=True,
			call_root=tmp_path / "rendezvous",
			rendezvous_participants=2,
		),
	)
	summary = runner.run()

	artifacts = [attempt.observations["artifacts"] for attempt in summary.attempts]
	assert {item["pid"] for item in artifacts}.isdisjoint({os.getpid()})
	assert len({item["pid"] for item in artifacts}) == 2
	assert max(item["started_monotonic"] for item in artifacts) < min(
		item["finished_monotonic"] for item in artifacts
	)
	assert [entry.metrics["quality"] for entry in summary.leaderboard] == [0.01, 0.02]
	assert [entry.pareto.rank for entry in summary.leaderboard] == [0, 1]
	assert all(entry.output_root.name == entry.trial_id for entry in summary.leaderboard)
	assert all(attempt.identity.resolved_recon.pipeline[0].params["alpha"] != 999
		for attempt in summary.attempts)
	assert (tmp_path / "experiment-results.json").is_file()
	assert (tmp_path / "experiment-results.yaml").is_file()
	assert (tmp_path / "leaderboard.md").is_file()
	payload = json.loads((tmp_path / "experiment-results.json").read_text())
	assert [item["trial_id"] for item in payload["leaderboard"]] == [
		item.trial_id for item in summary.leaderboard
	]


@pytest.mark.system
def test_failure_isolation_and_interruption_never_publish_success(tmp_path):
	summary = IndependentExperimentRunner(
		experiment_config(), inputs(), tmp_path, AlphaExecutor(fail_on=0.01),
	).run()
	by_status = {attempt.status: attempt for attempt in summary.attempts}

	assert set(by_status) == {"failed", "succeeded"}
	assert by_status["failed"].failure.message == "candidate exploded"
	assert by_status["failed"].result is None
	failed_record = json.loads(next(
		path for path in tmp_path.glob("trials/*/attempts/*.json")
		if json.loads(path.read_text())["status"] == "failed"
	).read_text())
	assert failed_record["result"] is None

	interrupted_root = tmp_path / "interrupted"
	interrupted = IndependentExperimentRunner(
		experiment_config(candidate=candidate_space((0.01,))),
		inputs(), interrupted_root, AlphaExecutor(interrupt=True),
	).run().attempts[0]
	assert interrupted.status == "interrupted"
	assert interrupted.result is None


@pytest.mark.system
def test_attempt_finish_is_clamped_when_wall_clock_moves_backwards(monkeypatch, tmp_path):
	base = datetime(2026, 8, 5, tzinfo=timezone.utc)
	timestamps = iter((
		base,
		base + timedelta(microseconds=2),
		base + timedelta(microseconds=1),
		base + timedelta(microseconds=3),
	))
	monkeypatch.setattr(runner_module, "_utcnow", lambda: next(timestamps))

	summary = IndependentExperimentRunner(
		experiment_config(candidate=candidate_space((0.01,)), concurrency=1),
		inputs(), tmp_path, AlphaExecutor(),
	).run()
	attempt = summary.attempts[0]

	assert attempt.status == "succeeded"
	assert attempt.finished_at == attempt.started_at
	assert attempt.result.finalized_at == attempt.started_at


def test_preflight_reports_validation_failures_and_enforces_resource_guards(tmp_path):
	invalid = candidate_space((0.01,))
	config = experiment_config(candidate=invalid, output_limit=100)
	runner = IndependentExperimentRunner(
		config, inputs(), tmp_path, AlphaExecutor(quality=1),
		validator=lambda _recon: (_ for _ in ()).throw(ValueError("invalid complete pipeline")),
		policy=ExperimentRunPolicy(estimated_output_bytes_per_trial=50),
	)
	preflight, prepared = runner.prepare()
	assert not prepared
	assert preflight.validation_failures[0].candidate_key
	with pytest.raises(ValueError, match="candidate preflight failed"):
		runner.run()

	guarded = IndependentExperimentRunner(
		experiment_config(output_limit=50), inputs(), tmp_path / "guarded",
		AlphaExecutor(quality=1),
		policy=ExperimentRunPolicy(estimated_output_bytes_per_trial=30),
	)
	with pytest.raises(PathResourceError, match="override the configured guard") as raised:
		guarded.prepare()
	assert raised.value.report.diagnostics[0].code == "guard_exceeded"
	overridden = IndependentExperimentRunner(
		experiment_config(output_limit=50), inputs(), tmp_path / "override",
		AlphaExecutor(quality=1),
		policy=ExperimentRunPolicy(
			estimated_output_bytes_per_trial=30, override_limits=True,
		),
	)
	assert overridden.prepare()[0].estimated_output_bytes == 60


def test_disabled_prefix_plan_adds_no_phantom_cache_resource_request(tmp_path):
	runner = IndependentExperimentRunner(
		experiment_config(candidate=candidate_space((0.01,))),
		inputs(),
		tmp_path / "results",
		DisabledCacheExecutor(tmp_path / "unused-cache"),
	)
	preflight, prepared = runner.prepare()

	assert prepared
	assert preflight.prefix_cache["enabled"] is False
	assert runner.path_resources.cache_root is None
	assert runner.path_resources.estimated_cache_bytes is None
	assert not (tmp_path / "unused-cache").exists()


def test_gpu_and_memory_capabilities_bound_concurrency(tmp_path):
	document = experiment_config(concurrency=4).model_dump(mode="json")
	document["resources"].update({"requires_gpu": True, "memory_gb": 1})
	config = ExperimentConfig.model_validate(document)
	base = dict(
		config=config, inputs=inputs(), root=tmp_path,
		executor=AlphaExecutor(quality=1),
	)
	with pytest.raises(ValueError, match="gpu_slots=0"):
		IndependentExperimentRunner(**base).prepare()
	runner = IndependentExperimentRunner(
		**base,
		policy=ExperimentRunPolicy(
			gpu_slots=2, estimated_memory_bytes_per_trial=700 * 1024**2,
		),
	)
	assert runner.prepare()[0].max_concurrency == 1


@pytest.mark.system
def test_native_process_crash_is_isolated_to_its_candidate(tmp_path):
	summary = IndependentExperimentRunner(
		experiment_config(), inputs(), tmp_path, CrashOneExecutor(),
	).run()
	by_status = {attempt.status: attempt for attempt in summary.attempts}

	assert set(by_status) == {"failed", "succeeded"}
	assert by_status["failed"].failure.type == "TrialProcessExitError"
	assert "code 17" in by_status["failed"].failure.message
	assert by_status["succeeded"].result.metrics == {"quality": 0.02}


def test_deadline_boundary_is_strict_at_configured_limit():
	assert not runner_module._deadline_expired(10.0, 10.999999, 1.0)
	assert runner_module._deadline_expired(10.0, 11.0, 1.0)
	assert runner_module._deadline_expired(10.0, 11.000001, 1.0)
	assert not runner_module._deadline_expired(10.0, 100.0, None)


def test_timeout_is_recorded_execution_policy_not_trial_identity(tmp_path):
	short_config = experiment_config(
		candidate=candidate_space((0.01,)), concurrency=1, timeout=30,
	)
	long_config = experiment_config(
		candidate=candidate_space((0.01,)), concurrency=1, timeout=300,
	)
	short = IndependentExperimentRunner(
		short_config, inputs(), tmp_path / "short", AlphaExecutor(),
	).prepare()[1][0]
	long = IndependentExperimentRunner(
		long_config, inputs(), tmp_path / "long", AlphaExecutor(),
	).prepare()[1][0]

	assert short.timeout_seconds == 30
	assert long.timeout_seconds == 300
	assert short.identity.trial_id == long.identity.trial_id
	assert "timeout_seconds" not in type(short.identity.fidelity).model_fields
	assert short_config.experiment_id != long_config.experiment_id


@pytest.mark.system
def test_timeout_is_attempt_local_structured_and_quarantines_partial_output(tmp_path):
	# Include spawned-interpreter startup in the contract while leaving enough
	# room for the fixture to publish its last-known execution context first.
	config = experiment_config(timeout=2.0)
	writer = ProgressEventWriter(tmp_path, config.experiment_id, durable=True)
	summary = PersistedExperimentRunner(IndependentExperimentRunner(
		config, inputs(), tmp_path, TimeoutOneExecutor(), event_writer=writer,
	)).run()
	by_alpha = {alpha(attempt.identity): attempt for attempt in summary.attempts}
	timed_out = by_alpha[0.01]
	succeeded = by_alpha[0.02]

	assert timed_out.status == "failed"
	assert timed_out.failure.type == "TrialTimeoutError"
	assert timed_out.execution_elapsed_seconds >= 2.0
	details = timed_out.failure.details
	assert details["timeout_seconds"] == 2.0
	assert details["execution_elapsed_seconds"] == timed_out.execution_elapsed_seconds
	assert details["overshoot_seconds"] >= 0
	assert details["last_window_id"] == "timeout-window"
	assert details["last_stage_id"] == "timeout-stage"
	assert details["last_stage_algorithm"] == "blocking-output"
	assert details["termination"]["confirmed_process_exit"] is True
	assert details["termination"]["remaining_descendant_pids"] == []
	timed_out_root = tmp_path / "trials" / timed_out.trial_id.value
	assert not (timed_out_root / "windows").exists()
	quarantined = next(
		(tmp_path / "quarantine" / "trials" / timed_out.trial_id.value).glob(
			"windows.timeout-*"
		)
	)
	assert (quarantined / "timeout-window" / "output" / "partial-output.txt").is_file()

	assert succeeded.status == "succeeded"
	assert succeeded.result.metrics == {"quality": 0.02}
	assert (
		tmp_path / "trials" / succeeded.trial_id.value
		/ "windows" / "completed" / "output" / "completed.txt"
	).is_file()

	process_file = timed_out_root / "timeout-process-tree.json"
	pids = json.loads(process_file.read_text(encoding="utf-8"))
	deadline = time.monotonic() + 5
	while any(psutil.pid_exists(pid) for pid in pids.values()) and time.monotonic() < deadline:
		time.sleep(0.05)
	assert not any(psutil.pid_exists(pid) for pid in pids.values())

	events = read_progress_events(tmp_path / "progress.jsonl")
	timeout_event = next(
		event for event in events
		if event.kind == "trial.timeout" and event.attempt_id == timed_out.attempt_id
	)
	finalized = next(
		event for event in events
		if event.kind == "trial.finalized" and event.attempt_id == timed_out.attempt_id
	)
	assert timeout_event.sequence < finalized.sequence
	assert timeout_event.details == details
	assert timeout_event.failure_type == "TrialTimeoutError"


@pytest.mark.system
@pytest.mark.parametrize("detached_descendant", [False, True])
def test_cancellation_terminates_noncooperative_trial_process_tree(
		tmp_path, detached_descendant,
):
	token = runner_module.CancellationToken()
	runner = IndependentExperimentRunner(
		experiment_config(candidate=candidate_space((0.01,)), concurrency=1),
		inputs(), tmp_path,
		HangingProcessTreeExecutor(detached_descendant=detached_descendant),
		cancellation=token,
	)
	result = []
	thread = threading.Thread(target=lambda: result.append(runner.run()))
	thread.start()
	process_file = None
	deadline = time.monotonic() + 10
	while process_file is None and time.monotonic() < deadline:
		process_file = next(tmp_path.glob("trials/*/process-tree.json"), None)
		time.sleep(0.01)
	assert process_file is not None
	pids = json.loads(process_file.read_text(encoding="utf-8"))
	assert token.request("operator stop")
	thread.join(timeout=10)
	assert not thread.is_alive()
	assert result[0].attempts[0].status == "interrupted"
	deadline = time.monotonic() + 5
	while any(psutil.pid_exists(pid) for pid in pids.values()) and time.monotonic() < deadline:
		time.sleep(0.05)
	assert not any(psutil.pid_exists(pid) for pid in pids.values())


def _segment_exists(name: str) -> bool:
	shared_memory_path = Path("/dev/shm") / name.lstrip("/")
	if not shared_memory_path.parent.is_dir():
		pytest.skip("safe shared-memory inspection requires /dev/shm")
	return shared_memory_path.exists()


@pytest.mark.system
def test_cancellation_allows_managed_trial_to_unlink_shared_segments(tmp_path):
	token = runner_module.CancellationToken()
	runner = IndependentExperimentRunner(
		experiment_config(candidate=candidate_space((0.01,)), concurrency=1),
		inputs(), tmp_path, HangingManagedExecutor(), cancellation=token,
	)
	result = []
	thread = threading.Thread(target=lambda: result.append(runner.run()))
	thread.start()
	process_file = None
	deadline = time.monotonic() + 10
	while process_file is None and time.monotonic() < deadline:
		process_file = next(tmp_path.glob("trials/*/managed-process-tree.json"), None)
		time.sleep(0.01)
	assert process_file is not None
	segment_name = json.loads(process_file.read_text(encoding="utf-8"))["segment_name"]
	assert _segment_exists(segment_name)
	assert token.request("operator stop")
	thread.join(timeout=10)
	assert not thread.is_alive()
	assert result[0].attempts[0].status == "interrupted"
	deadline = time.monotonic() + 10
	while _segment_exists(segment_name) and time.monotonic() < deadline:
		time.sleep(0.05)
	assert not _segment_exists(segment_name)


def test_experiment_cli_exposes_validate_dry_run_and_run(tmp_path):
	document = experiment_config().model_dump(mode="json")
	document["metrics"] = [{
		"id": "rings", "metric": "artifact.ring_score",
		"protocol_version": "1", "direction": "minimize",
	}]
	document["objectives"] = [{"metric_id": "rings"}]
	document["aggregation"]["metric_ids"] = ["rings"]
	config_path = tmp_path / "experiment.yaml"
	config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

	cli = CliRunner()
	help_result = cli.invoke(main, ["--help"])
	assert help_result.exit_code == 0
	assert all(command in help_result.output for command in (
		"validate", "dry-run", "run", "inspect", "resume", "rerun",
		"tier2-plan", "tier2-audit",
	))
	validate_result = cli.invoke(main, ["validate", str(config_path), "--json-output"])
	assert validate_result.exit_code == 0
	payload = json.loads(validate_result.output)
	assert payload["valid"] is True
	assert payload["sampler"]["name"] == "exhaustive"
