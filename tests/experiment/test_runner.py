from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time

from click.testing import CliRunner
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
)
from lftomo.experiment.runner import (
	ExperimentInputs,
	ExperimentRunPolicy,
	IndependentExperimentRunner,
	TrialExecution,
)
import lftomo.experiment.runner as runner_module
from lftomo.experiment.selection import select_candidates
from lftomo.experiment.cli import main


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


def experiment_config(*, candidate=None, sampler=None, concurrency=2, output_limit=None):
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
		"fidelities": [{"id": "screen", "windows": ["screen"]}],
		"aggregation": {"metric_ids": ["quality"]},
		"output": {"screening": "metrics_only"},
	})


def inputs():
	return ExperimentInputs(
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


def test_independent_concurrent_execution_has_deterministic_identity_and_leaderboard(tmp_path):
	config = experiment_config()
	lock = threading.Lock()
	active = 0
	maximum_active = 0
	identity_objects = []

	def execute(identity, output_root):
		nonlocal active, maximum_active
		identity_objects.append(id(identity.resolved_recon))
		value = alpha(identity)
		identity.resolved_recon.pipeline[0].params["alpha"] = 999
		with lock:
			active += 1
			maximum_active = max(maximum_active, active)
		time.sleep(0.03 if value == 0.01 else 0.01)
		with lock:
			active -= 1
		return TrialExecution(metrics={"quality": value})

	runner = IndependentExperimentRunner(config, inputs(), tmp_path, execute)
	summary = runner.run()

	assert maximum_active == 2
	assert len(set(identity_objects)) == 2
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


def test_failure_isolation_and_interruption_never_publish_success(tmp_path):
	def execute(identity, _output_root):
		if alpha(identity) == 0.01:
			raise RuntimeError("candidate exploded")
		return TrialExecution(metrics={"quality": 0.02})

	summary = IndependentExperimentRunner(
		experiment_config(), inputs(), tmp_path, execute,
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
		inputs(), interrupted_root,
		lambda *_args: (_ for _ in ()).throw(InterruptedError("stopped")),
	).run().attempts[0]
	assert interrupted.status == "interrupted"
	assert interrupted.result is None


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
		inputs(), tmp_path,
		lambda *_args: TrialExecution(metrics={"quality": 0.01}),
	).run()
	attempt = summary.attempts[0]

	assert attempt.status == "succeeded"
	assert attempt.finished_at == attempt.started_at
	assert attempt.result.finalized_at == attempt.started_at


def test_preflight_reports_validation_failures_and_enforces_resource_guards(tmp_path):
	invalid = candidate_space((0.01,))
	config = experiment_config(candidate=invalid, output_limit=100)
	runner = IndependentExperimentRunner(
		config, inputs(), tmp_path,
		lambda *_args: TrialExecution(metrics={"quality": 1}),
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
		lambda *_args: TrialExecution(metrics={"quality": 1}),
		policy=ExperimentRunPolicy(estimated_output_bytes_per_trial=30),
	)
	with pytest.raises(PathResourceError, match="override the configured guard") as raised:
		guarded.prepare()
	assert raised.value.report.diagnostics[0].code == "guard_exceeded"
	overridden = IndependentExperimentRunner(
		experiment_config(output_limit=50), inputs(), tmp_path / "override",
		lambda *_args: TrialExecution(metrics={"quality": 1}),
		policy=ExperimentRunPolicy(
			estimated_output_bytes_per_trial=30, override_limits=True,
		),
	)
	assert overridden.prepare()[0].estimated_output_bytes == 60


def test_gpu_and_memory_capabilities_bound_concurrency(tmp_path):
	document = experiment_config(concurrency=4).model_dump(mode="json")
	document["resources"].update({"requires_gpu": True, "memory_gb": 1})
	config = ExperimentConfig.model_validate(document)
	base = dict(
		config=config, inputs=inputs(), root=tmp_path,
		executor=lambda *_args: TrialExecution(metrics={"quality": 1}),
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
