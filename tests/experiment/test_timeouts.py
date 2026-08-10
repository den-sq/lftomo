from pathlib import Path
import shutil
import threading
import time

import pytest

from lftomo.config import Recon
from lftomo.experiment import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	PendingOutput,
	ProgressInfrastructureError,
	ProgressEventWriter,
	TimeoutIsolationUnsupported,
	read_progress_events,
)
from lftomo.experiment.events import current_execution_context
from lftomo.experiment.outputs import OutputBudgetLedger
from lftomo.experiment.runner import (
	ExperimentInputs,
	ExperimentRunPolicy,
	IndependentExperimentRunner,
	TrialExecution,
)
from lftomo.experiment.store import PersistedExperimentRunner, ResumePolicy


HASHES = [f"{index:064x}" for index in range(1, 4)]


def _config(*, values=(0.01,), timeout=2.0, concurrency=1):
	return ExperimentConfig.model_validate({
		"name": "timeout-test",
		"candidate_space": Recon.model_validate({
			"pipeline": [{
				"id": "phase", "kind": "phase", "algorithm": "paganin",
				"sweep": {"kind": "list", "params": {"alpha": list(values)}},
			}],
		}).model_dump(mode="json"),
		"sampler": {"name": "exhaustive", "seed": 7},
		"resources": {"max_concurrency": concurrency},
		"metrics": [{
			"id": "quality", "metric": "quality", "protocol_version": "1",
			"direction": "minimize",
		}],
		"objectives": [{"metric_id": "quality"}],
		"windows": [{"id": "screen", "y_start": 0, "y_stop": 2}],
		"fidelities": [{
			"id": "screen", "windows": ["screen"], "timeout_seconds": timeout,
		}],
		"aggregation": {"metric_ids": ["quality"]},
		"output": {"screening": "metrics_only"},
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


class IsolatedTestExecutor:
	supports_isolated_execution = True

	def __init__(self, root: Path, mode: str, config: ExperimentConfig | None = None):
		self.root = root
		self.mode = mode
		self.config = config

	def __call__(self, identity, output_root):
		alpha = identity.resolved_recon.pipeline[0].params["alpha"]
		mode = "noncooperative" if self.mode == "mixed" and alpha == 0.01 else self.mode
		if self.mode == "mixed" and alpha != 0.01:
			mode = "success"
		if mode == "success":
			return TrialExecution(metrics={"quality": alpha})
		if mode == "failure":
			raise ValueError("remote candidate failure")
		if mode == "progress_failure":
			raise ProgressInfrastructureError("remote progress projection failed")
		if mode in ("committed_output", "commit_then_hang"):
			assert self.config is not None
			context = current_execution_context()
			assert context is not None
			temporary = output_root / "partial" / "final.tif"
			temporary.parent.mkdir(parents=True, exist_ok=True)
			temporary.write_bytes(b"complete-output")
			ledger = OutputBudgetLedger(self.root, self.config, "screen")
			records, accounting = ledger.publish(
				(PendingOutput(
					role="screening_output",
					temporary_path=temporary,
					relative_path=(
						Path("published") / "screen" / identity.trial_id.value / "final.tif"
					),
				),),
				attempt_id=context.progress.attempt_id,
				trial_id=identity.trial_id.value,
			)
			context.output_event(
				"committed", "test output committed",
				output_bytes=accounting.attempt_committed_bytes,
				details=accounting.model_dump(mode="json"),
			)
			if mode == "committed_output":
				return TrialExecution(
					metrics={"quality": alpha}, outputs=records,
					outputs_committed=True, output_accounting=accounting,
				)
			while True:
				time.sleep(0.01)
		if mode == "cooperative":
			context = current_execution_context()
			assert context is not None
			context.window_started("screen", position=1, total=1)
			context.stage_started("slow-stage", "test.slow", position=1, total=1)
			while True:
				time.sleep(0.01)
				context.heartbeat()
		if mode == "noncooperative":
			partial = output_root / "partial"
			partial.mkdir(parents=True, exist_ok=True)
			context = current_execution_context()
			assert context is not None
			publication_copy = (
				self.root / ".output-transactions"
				/ f"{context.progress.attempt_id}-simulated" / "new" / "00000000"
			)
			publication_copy.parent.mkdir(parents=True, exist_ok=True)
			marker = self.root / f"worker-{identity.candidate.value[:12]}.bin"
			while True:
				with marker.open("ab") as stream:
					stream.write(b"x")
					stream.flush()
				with (partial / "unfinished.tif").open("ab") as stream:
					stream.write(b"partial")
				with publication_copy.open("ab") as stream:
					stream.write(b"publication-copy")
				time.sleep(0.01)
		raise AssertionError(f"unknown test mode {mode}")

	def cleanup_attempt_outputs(self, identity, _attempt_id):
		partial = self.root / "trials" / identity.trial_id.value / "partial"
		shutil.rmtree(partial, ignore_errors=True)


def _runner(root, config, executor, *, failure="continue", writer=None):
	return IndependentExperimentRunner(
		config,
		_inputs(),
		root,
		executor,
		policy=ExperimentRunPolicy(failure=failure),
		event_writer=writer,
	)


def test_timeout_bearing_fidelity_rejects_nonisolatable_executor(tmp_path):
	runner = _runner(
		tmp_path, _config(), lambda *_args: TrialExecution(metrics={"quality": 1}),
	)

	with pytest.raises(TimeoutIsolationUnsupported, match="killable isolation"):
		runner.prepare()

	document = _config().model_dump(mode="json")
	document["output"]["screening"] = "configured_writers"
	output_config = ExperimentConfig.model_validate(document)
	with pytest.raises(TimeoutIsolationUnsupported, match="commit them inside"):
		_runner(
			tmp_path / "output",
			output_config,
			IsolatedTestExecutor(tmp_path / "output", "success"),
		).prepare()


def test_timeout_remains_part_of_the_declared_trial_identity(tmp_path):
	first_config = _config(timeout=1)
	second_config = _config(timeout=2)
	first = _runner(
		tmp_path / "first", first_config,
		IsolatedTestExecutor(tmp_path / "first", "success"),
	).prepare()[1][0]
	second = _runner(
		tmp_path / "second", second_config,
		IsolatedTestExecutor(tmp_path / "second", "success"),
	).prepare()[1][0]

	assert first.identity.fidelity.timeout_seconds == 1
	assert second.identity.fidelity.timeout_seconds == 2
	assert first.identity.trial_id != second.identity.trial_id


def test_attempt_completes_inside_deadline_and_remote_failure_type_is_preserved(tmp_path):
	success = _runner(
		tmp_path / "success", _config(timeout=3),
		IsolatedTestExecutor(tmp_path / "success", "success"),
	).run().attempts[0]
	failure = _runner(
		tmp_path / "failure", _config(timeout=3),
		IsolatedTestExecutor(tmp_path / "failure", "failure"),
	).run().attempts[0]

	assert success.status == "succeeded"
	assert success.elapsed_seconds < 3
	assert failure.status == "failed"
	assert failure.failure.type == "ValueError"
	assert failure.failure.message == "remote candidate failure"
	assert "remote candidate failure" in failure.failure.traceback


def test_isolated_progress_infrastructure_failure_still_aborts_coordinator(tmp_path):
	runner = _runner(
		tmp_path, _config(timeout=3),
		IsolatedTestExecutor(tmp_path, "progress_failure"),
	)

	with pytest.raises(ProgressInfrastructureError, match="remote progress projection failed"):
		runner.run()
	attempt_path = next(tmp_path.glob("trials/*/attempts/*.json"))
	assert '"status": "interrupted"' in attempt_path.read_text(encoding="utf-8")
	assert '"type": "ProgressInfrastructureError"' in attempt_path.read_text(encoding="utf-8")


def test_cooperative_timeout_persists_stage_context_and_event_order(tmp_path):
	config = _config(timeout=1.5)
	writer = ProgressEventWriter(tmp_path, config.experiment_id, durable=True)
	attempt = _runner(
		tmp_path, config, IsolatedTestExecutor(tmp_path, "cooperative"), writer=writer,
	).run().attempts[0]

	assert attempt.status == "failed"
	assert attempt.failure.type == "AttemptTimeoutExceeded"
	assert attempt.failure.stage_id == "slow-stage"
	assert attempt.failure.stage_algorithm == "test.slow"
	assert attempt.failure.window_id == "screen"
	assert attempt.failure.details["timeout_seconds"] == 1.5
	events = read_progress_events(tmp_path / "progress.jsonl")
	kinds = [event.kind for event in events]
	assert kinds.index("trial.timeout") < kinds.index("trial.finalized")
	timeout = next(event for event in events if event.kind == "trial.timeout")
	assert timeout.stage_occurrence_id == "slow-stage"
	assert timeout.details["timeout_seconds"] == 1.5


def test_noncooperative_timeout_is_killed_before_terminal_state_and_partial_cleanup(tmp_path):
	config = _config(timeout=1.5)
	attempt = _runner(
		tmp_path, config, IsolatedTestExecutor(tmp_path, "noncooperative"),
	).run().attempts[0]
	marker = next(tmp_path.glob("worker-*.bin"))
	size = marker.stat().st_size
	time.sleep(0.15)

	assert attempt.status == "failed"
	assert attempt.failure.type == "AttemptTimeoutExceeded"
	assert marker.stat().st_size == size
	assert not (tmp_path / "trials" / attempt.trial_id.value / "partial").exists()
	assert not any((tmp_path / ".output-transactions").iterdir())
	assert not (tmp_path / "published").exists()
	assert attempt.result is None


def test_output_commit_is_finalized_on_success_and_rolled_back_if_attempt_times_out(tmp_path):
	success_config = _config(timeout=3)
	success = _runner(
		tmp_path / "success",
		success_config,
		IsolatedTestExecutor(
			tmp_path / "success", "committed_output", success_config,
		),
	).run().attempts[0]
	assert success.status == "succeeded"
	assert success.result.outputs[0].path.read_bytes() == b"complete-output"
	assert not any((tmp_path / "success" / ".output-transactions").iterdir())

	timeout_config = _config(timeout=1.5)
	timed_out = _runner(
		tmp_path / "timeout",
		timeout_config,
		IsolatedTestExecutor(
			tmp_path / "timeout", "commit_then_hang", timeout_config,
		),
	).run().attempts[0]
	assert timed_out.status == "failed"
	assert timed_out.failure.type == "AttemptTimeoutExceeded"
	assert not any((tmp_path / "timeout" / "published").rglob("*.tif"))
	assert not any((tmp_path / "timeout" / ".output-transactions").iterdir())

	limit_document = _config(timeout=3).model_dump(mode="json")
	limit_document["resources"]["max_output_bytes"] = 1
	limit_document["fidelities"][0]["max_output_bytes"] = 1
	limit_document["output"]["max_bytes"] = 1
	limit_config = ExperimentConfig.model_validate(limit_document)
	limit_root = tmp_path / "limit"
	writer = ProgressEventWriter(limit_root, limit_config.experiment_id, durable=True)
	over_limit = _runner(
		limit_root,
		limit_config,
		IsolatedTestExecutor(limit_root, "committed_output", limit_config),
		writer=writer,
	).run().attempts[0]
	assert over_limit.status == "failed"
	assert over_limit.failure.type == "OutputLimitExceeded"
	assert over_limit.failure.details["scope"] == "experiment"
	assert "output.limit_exceeded" in {
		event.kind for event in read_progress_events(limit_root / "progress.jsonl")
	}


def test_concurrent_timeout_does_not_cancel_unrelated_success(tmp_path):
	config = _config(values=(0.01, 0.02), timeout=1.5, concurrency=2)
	summary = _runner(
		tmp_path, config, IsolatedTestExecutor(tmp_path, "mixed"),
	).run()

	assert sorted(attempt.status for attempt in summary.attempts) == ["failed", "succeeded"]
	failed = next(attempt for attempt in summary.attempts if attempt.status == "failed")
	succeeded = next(attempt for attempt in summary.attempts if attempt.status == "succeeded")
	assert failed.failure.type == "AttemptTimeoutExceeded"
	assert succeeded.result.metrics == {"quality": 0.02}
	assert summary.interruption is None


def test_external_cancellation_terminates_isolated_attempt_without_waiting_for_deadline(tmp_path):
	config = _config(timeout=10)
	runner = _runner(
		tmp_path, config, IsolatedTestExecutor(tmp_path, "noncooperative"),
	)
	result = {}
	thread = threading.Thread(target=lambda: result.setdefault("summary", runner.run()))
	thread.start()
	deadline = time.monotonic() + 5
	while not tuple(tmp_path.glob("worker-*.bin")) and time.monotonic() < deadline:
		time.sleep(0.01)
	assert tuple(tmp_path.glob("worker-*.bin"))
	runner.cancellation.request("operator stopped isolated test")
	thread.join(timeout=3)

	assert not thread.is_alive()
	attempt = result["summary"].attempts[0]
	assert attempt.status == "interrupted"
	assert attempt.failure.type == "ExperimentCancelled"
	assert result["summary"].interruption.reason == "operator stopped isolated test"


@pytest.mark.parametrize(("failure_policy", "attempt_count"), [
	("continue", 2),
	("fail_fast", 1),
])
def test_timeout_honors_continue_and_fail_fast_policy(
	tmp_path, failure_policy, attempt_count,
):
	config = _config(values=(0.01, 0.02), timeout=1.5, concurrency=1)
	summary = _runner(
		tmp_path,
		config,
		IsolatedTestExecutor(tmp_path, "mixed"),
		failure=failure_policy,
	).run()

	assert len(summary.attempts) == attempt_count
	assert any(attempt.status == "failed" for attempt in summary.attempts)
	if failure_policy == "continue":
		assert any(attempt.status == "succeeded" for attempt in summary.attempts)


def test_timed_out_attempt_requires_explicit_retry_and_can_then_succeed(tmp_path):
	config = _config(timeout=1.5)
	first = PersistedExperimentRunner(_runner(
		tmp_path, config, IsolatedTestExecutor(tmp_path, "noncooperative"),
	)).run().attempts[0]
	skipped = PersistedExperimentRunner(_runner(
		tmp_path, config, IsolatedTestExecutor(tmp_path, "success"),
	)).run().attempts[0]
	retried = PersistedExperimentRunner(
		_runner(tmp_path, config, IsolatedTestExecutor(tmp_path, "success")),
		policy=ResumePolicy(retry_failed=True),
	).run().attempts[0]

	assert first.status == "failed"
	assert first.failure.type == "AttemptTimeoutExceeded"
	assert skipped.status == "skipped"
	assert retried.status == "succeeded"
	assert retried.sequence == 2
