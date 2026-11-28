import json
import threading
import time

from click.testing import CliRunner
import pytest

from lftomo.config import Acquisition, Detector, Scan, TIFFStackProjections
from lftomo.experiment import (
	CancellationToken,
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	ExperimentLifecycle,
	ExperimentStatus,
	ProgressEvent,
	ProgressEventWriter,
	ReplayBundle,
	ResumePolicy,
	read_progress_events,
)
from lftomo.experiment.cli import main
from lftomo.experiment.events import current_execution_context
from lftomo.experiment.runner import (
	ExperimentInputs,
	IndependentExperimentRunner,
	TrialExecution,
)

from lftomo.util.window import TomoWindow


HASHES = [f"{index:064x}" for index in range(1, 8)]


def config(values=(0.01, 0.02), *, concurrency=2):
	return ExperimentConfig.model_validate({
		"name": "lifecycle-test",
		"candidate_space": {"pipeline": [{
			"id": "phase", "kind": "phase", "algorithm": "paganin",
			"sweep": {"kind": "list", "params": {"alpha": list(values)}},
		}]},
		"sampler": {"name": "exhaustive", "seed": 7},
		"resources": {"max_concurrency": concurrency},
		"metrics": [{
			"id": "quality", "metric": "artifact.ring_score", "protocol_version": "1",
			"direction": "minimize",
		}],
		"objectives": [{"metric_id": "quality"}],
		"windows": [{"id": "screen", "y_start": 0, "y_stop": 1}],
		"fidelities": [{"id": "screen", "windows": ["screen"]}],
		"aggregation": {"metric_ids": ["quality"]},
	})


def inputs():
	return ExperimentInputs(
		source=TomoWindow.from_bounds(scan().projection_shape(), dtype="uint16"),
		dataset=DatasetIdentity(dataset_id="fixture", content_hash=HASHES[0]),
		configuration=ConfigurationIdentity(scan_config_hash=HASHES[1]),
		code=CodeIdentity(version="test", git_commit="abcdef123456"),
		environment=EnvironmentIdentity(
			python="3.11", platform="test", backend="numpy",
			dependencies={"numpy": "1.26.4"},
		),
	)


def scan(secret=None):
	metadata = {"dataset_content_hash": HASHES[0]}
	if secret is not None:
		metadata["api_token"] = secret
	return Scan(
		energy_keV=20,
		pixel_size_um=1,
		n_projections=2,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1, rows=1, columns=2),
		acquisition=Acquisition(n_projections=2, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="unused"),
		metadata=metadata,
	)


class ReplayableExecutor:
	def __init__(self, scan_model=None, *, failure=None, delay=0):
		self.scan_model = scan_model or scan()
		self.failure = failure
		self.delay = delay

	def validate(self, _recon):
		return None

	def validate_identity(self, _identity):
		return None

	def replay_document(self):
		return {
			"kind": "pipeline",
			"scan": self.scan_model.model_dump(mode="json"),
			"prefix_cache_policy": {"enabled": False},
		}

	def __call__(self, identity, _root):
		context = current_execution_context()
		assert context is not None
		context.window_started("screen", position=1, total=1)
		context.stage_started("phase", "paganin", position=1, total=1)
		try:
			if self.delay:
				deadline = time.monotonic() + self.delay
				while time.monotonic() < deadline:
					time.sleep(0.005)
					context.checkpoint()
			if self.failure is not None:
				raise ValueError(self.failure)
		except BaseException as error:
			import traceback

			context.stage_failed("phase", "paganin", error, traceback.format_exc())
			raise
		context.stage_completed("phase", "paganin")
		context.window_completed("screen")
		alpha = identity.resolved_recon.pipeline[0].params["alpha"]
		return TrialExecution(metrics={"quality": float(alpha)})


def lifecycle(tmp_path, executor, *, values=(0.01, 0.02), callback=None, token=None):
	runner = IndependentExperimentRunner(
		config(values), inputs(), tmp_path / "results", executor,
		validator=executor.validate,
		identity_validator=executor.validate_identity,
	)
	return ExperimentLifecycle(
		runner, callback=callback, cancellation=token,
	)


def sampled_variant_config():
	return ExperimentConfig.model_validate({
		"name": "sampled-variant-progress",
		"candidate_space": {
			"pipeline": [
				{
					"id": "phase-a", "kind": "phase", "algorithm": "paganin",
					"params": {"alpha": 0.01},
				},
				{
					"id": "phase-b", "kind": "phase", "algorithm": "paganin",
					"params": {"alpha": 0.02},
				},
			],
			"variants": [
				{"name": "single", "stages": ["phase-a"]},
				{"name": "chain", "stages": ["phase-a", "phase-b"]},
			],
		},
		"sampler": {"name": "sobol", "seed": 41, "budget": 8},
		"resources": {"max_concurrency": 2},
		"metrics": [{
			"id": "quality", "metric": "artifact.ring_score", "protocol_version": "1",
			"direction": "minimize",
		}],
		"objectives": [{"metric_id": "quality"}],
		"windows": [{"id": "screen", "y_start": 0, "y_stop": 1}],
		"fidelities": [{"id": "screen", "windows": ["screen"]}],
		"aggregation": {"metric_ids": ["quality"]},
	})


class VariantProgressExecutor(ReplayableExecutor):
	def __call__(self, identity, _root):
		import traceback

		context = current_execution_context()
		assert context is not None
		context.window_started("screen", position=1, total=1)
		stages = identity.resolved_recon.pipeline
		for position, stage in enumerate(stages, start=1):
			context.stage_started(
				stage.id, stage.algorithm, position=position, total=len(stages),
			)
			context.heartbeat()
			if identity.sampler.explicit_case == "chain" and position == len(stages):
				error = ValueError("injected chain failure")
				context.stage_failed(stage.id, stage.algorithm, error, traceback.format_exc())
				raise error
			context.stage_completed(stage.id, stage.algorithm)
		context.window_completed("screen")
		return TrialExecution(metrics={"quality": float(len(stages))})


class MixedCancellationExecutor(ReplayableExecutor):
	def __init__(self, call_root):
		super().__init__()
		self.call_root = call_root

	def __call__(self, identity, _root):
		alpha = float(identity.resolved_recon.pipeline[0].params["alpha"])
		self.call_root.mkdir(parents=True, exist_ok=True)
		(self.call_root / f"{identity.trial_id.value}.json").write_text(
			json.dumps({"alpha": alpha}), encoding="utf-8",
		)
		context = current_execution_context()
		context.window_started("screen")
		context.stage_started("phase", "paganin")
		if alpha != 0.01:
			deadline = time.monotonic() + 0.5
			while time.monotonic() < deadline:
				time.sleep(0.005)
				context.checkpoint()
		context.stage_completed("phase", "paganin")
		context.window_completed("screen")
		return TrialExecution(metrics={"quality": alpha})


class MustNotExecute:
	def __call__(self, *_args):
		raise AssertionError("must reuse success")


@pytest.mark.system
def test_progress_reports_aggregate_counts_human_context_and_resume_dispositions(tmp_path):
	executor = VariantProgressExecutor()
	runner = IndependentExperimentRunner(
		sampled_variant_config(), inputs(), tmp_path / "results", executor,
		validator=executor.validate, identity_validator=executor.validate_identity,
	)
	api = ExperimentLifecycle(runner)
	summary = api.run()
	assert sorted(attempt.status for attempt in summary.attempts) == ["failed", "succeeded"]
	events = read_progress_events(api.root / "progress.jsonl")

	started = next(event for event in events if event.kind == "fidelity.started")
	assert started.total_candidates == 2
	assert started.completed_candidates == 0
	assert started.remaining_candidates == 2
	progress = [event for event in events if event.kind == "fidelity.progress"]
	assert [event.completed_candidates for event in progress] == [1, 2]
	assert all(
		event.completed_candidates + event.active_candidates + event.remaining_candidates
		== event.total_candidates
		for event in progress
	)
	completed = next(event for event in events if event.kind == "fidelity.completed")
	assert completed.succeeded_candidates == 1
	assert completed.failed_candidates == 1
	assert completed.active_candidates == 0
	assert completed.remaining_candidates == 0
	experiment_completed = next(
		event for event in events if event.kind == "experiment.completed"
	)
	assert experiment_completed.lifecycle_state == "failed"

	trials = [event for event in events if event.kind == "trial.started"]
	assert {event.structural_variant for event in trials} == {"single", "chain"}
	assert len({event.sampler_position for event in trials}) == 2
	assert all(1 <= event.sampler_position <= 8 for event in trials)
	assert {event.sampler_total for event in trials} == {8}
	stage_starts = [event for event in events if event.kind == "stage.started"]
	assert {(event.stage_position, event.stage_total) for event in stage_starts} == {
		(1, 1), (1, 2), (2, 2),
	}
	assert all(event.window_position == 1 and event.window_total == 1 for event in stage_starts)
	heartbeats = [event for event in events if event.kind == "stage.heartbeat"]
	assert heartbeats and all(event.stage_elapsed_seconds is not None for event in heartbeats)
	failure = next(
		event for event in events
		if event.kind == "trial.finalized" and event.lifecycle_state == "failed"
	)
	assert failure.failure_type == "ValueError"
	assert failure.failure_message == "injected chain failure"

	prior_sequence = events[-1].sequence
	resumed = ExperimentLifecycle(IndependentExperimentRunner(
		sampled_variant_config(), inputs(), api.root, executor,
		validator=executor.validate, identity_validator=executor.validate_identity,
	))
	resumed.resume()
	resume_events = [
		event for event in read_progress_events(api.root / "progress.jsonl")
		if event.sequence > prior_sequence
	]
	resume_completed = next(
		event for event in resume_events if event.kind == "fidelity.completed"
	)
	assert resume_completed.reused_candidates == 1
	assert resume_completed.skipped_candidates == 1
	assert resume_completed.completed_candidates == 2


def test_progress_schema_one_remains_backward_compatible_with_absent_display_fields():
	event = ProgressEvent.model_validate({
		"schema_version": 1,
		"sequence": 1,
		"timestamp": "2026-08-10T00:00:00Z",
		"experiment_id": "legacy",
		"kind": "trial.started",
		"message": "legacy schema-one event",
	})
	assert event.schema_version == 1
	assert event.context_update == "replace"
	assert event.segment_id is None
	assert event.segment_members == ()
	assert event.sampler_position is None
	assert event.remaining_candidates is None


@pytest.mark.system
def test_public_lifecycle_persists_ordered_events_and_store_only_status(tmp_path):
	api = lifecycle(tmp_path, ReplayableExecutor())
	summary = api.run()
	events = read_progress_events(api.root / "progress.jsonl")

	assert len(summary.attempts) == 2
	assert [event.sequence for event in events] == list(range(1, len(events) + 1))
	assert {"preflight.started", "preflight.completed", "trial.claim", "trial.started",
		"window.started", "stage.started", "stage.completed", "window.completed",
		"trial.finalized", "experiment.completed"} <= {event.kind for event in events}
	preflight_event = next(event for event in events if event.kind == "preflight.completed")
	parallelism = preflight_event.details["parallelism"]
	assert parallelism["candidate_process_limit"] == 2
	assert parallelism["pipeline_process_workers_scope"] == "per_candidate"
	assert parallelism["coordinator_divides_pipeline_workers"] is False

	# JSONL is derived: losing it, or ending it with a torn record, cannot alter status.
	(api.root / "progress.jsonl").write_text('{"truncated":', encoding="utf-8")
	status = api.status()
	assert isinstance(status, ExperimentStatus)
	assert status.state == "succeeded"
	assert status.counts == {"succeeded": 2}


def test_concurrent_event_producers_are_serialized_without_corruption(tmp_path):
	writer = ProgressEventWriter(tmp_path, "experiment", durable=True)

	def produce(worker):
		for index in range(25):
			writer.emit("test.update", f"{worker}:{index}")

	threads = [threading.Thread(target=produce, args=(worker,)) for worker in range(4)]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()
	events = read_progress_events(tmp_path / "progress.jsonl")
	assert len(events) == 100
	assert [event.sequence for event in events] == list(range(1, 101))

	with (tmp_path / "progress.jsonl").open("a", encoding="utf-8") as stream:
		stream.write('{"torn":')
	recovered = ProgressEventWriter(tmp_path, "experiment", durable=True)
	recovered.emit("test.recovered", "writer recovered a torn final record")
	events = read_progress_events(tmp_path / "progress.jsonl")
	assert len(events) == 101
	assert events[-1].sequence == 101


@pytest.mark.system
def test_cancellation_stops_new_claims_and_finalizes_inflight_attempts(tmp_path):
	started = threading.Event()
	token = CancellationToken()

	def observe(event):
		if event.kind == "stage.started":
			started.set()

	api = lifecycle(
		tmp_path, ReplayableExecutor(delay=0.5),
		values=(0.01, 0.02, 0.03, 0.04), callback=observe, token=token,
	)
	result = []
	thread = threading.Thread(target=lambda: result.append(api.run()))
	thread.start()
	assert started.wait(timeout=20)
	running = api.status()
	assert running.state == "running"
	assert any(
		item.window_id == "screen" and item.stage_occurrence_id == "phase"
		for item in running.attempts if item.state == "started"
	)
	assert api.request_cancellation("operator stop") is True
	thread.join(timeout=5)
	assert not thread.is_alive()

	summary = result[0]
	assert summary.interruption is not None
	assert summary.interruption.reason == "operator stop"
	assert summary.interruption.unstarted_candidates >= 1
	assert all(item.status == "interrupted" for item in summary.attempts)
	status = api.status()
	assert "started" not in status.counts
	assert status.cancellation.reason == "operator stop"


@pytest.mark.system
def test_cancellation_preserves_compatible_success_for_later_resume(tmp_path):
	call_root = tmp_path / "calls"
	executor = MixedCancellationExecutor(call_root)
	api = None

	def observe(event):
		if event.kind == "trial.finalized" and event.lifecycle_state == "succeeded":
			api.request_cancellation("stop after first success")

	api = lifecycle(
		tmp_path, executor, values=(0.01, 0.02, 0.03), callback=observe,
	)
	first = api.run()
	success = next(item for item in first.attempts if item.status == "succeeded")
	assert first.interruption is not None

	second_runner = IndependentExperimentRunner(
		config((0.01, 0.02, 0.03)), inputs(), api.root, executor,
		validator=executor.validate, identity_validator=executor.validate_identity,
	)
	resumed = ExperimentLifecycle(second_runner).resume(ResumePolicy())
	matching = next(item for item in resumed.attempts if item.trial_id == success.trial_id)
	assert matching.attempt_id == success.attempt_id
	calls = [
		json.loads(path.read_text(encoding="utf-8"))["alpha"]
		for path in call_root.glob("*.json")
	]
	assert calls.count(0.01) == 1
	assert ExperimentLifecycle.status_root(api.root).cancellation is None


@pytest.mark.system
def test_failure_traceback_context_is_redacted_in_store_events_and_bundle(
	tmp_path, monkeypatch,
):
	secret = "do-not-persist-this-secret"
	monkeypatch.setenv("LFTOMO_PREVIEW_BLINDING_KEY", secret)
	api = lifecycle(
		tmp_path,
		ReplayableExecutor(failure=f"failure containing {secret}"),
	)
	summary = api.run()
	attempt = summary.attempts[0]
	assert attempt.status == "failed"
	assert attempt.failure.stage_id == "phase"
	assert attempt.failure.window_id == "screen"
	assert "Traceback" in attempt.failure.traceback

	all_text = "\n".join(
		path.read_text(encoding="utf-8", errors="replace")
		for path in api.root.rglob("*") if path.is_file()
	)
	assert secret not in all_text
	assert "[REDACTED]" in all_text
	bundle = ReplayBundle.model_validate_json(
		(api.root / "invocation" / "replay-bundle.json").read_text(encoding="utf-8")
	)
	assert bundle.key_fingerprints["preview_blinding"]


def test_secret_bearing_scientific_config_is_rejected_before_store_creation(tmp_path):
	document = config((0.01,)).model_dump(mode="json")
	document["metadata"]["api_token"] = "must-remain-external"
	unsafe_config = ExperimentConfig.model_validate(document)
	executor = ReplayableExecutor()
	runner = IndependentExperimentRunner(
		unsafe_config, inputs(), tmp_path / "results", executor,
		validator=executor.validate, identity_validator=executor.validate_identity,
	)
	with pytest.raises(ValueError, match="supply it externally"):
		ExperimentLifecycle(runner).run()
	assert not runner.root.exists()


@pytest.mark.system
def test_replay_bundle_rebuilds_controller_without_original_config_files(tmp_path):
	api = lifecycle(tmp_path, ReplayableExecutor())
	api.run()

	def factory(_config, _scan_path, _document):
		return MustNotExecute()

	replayed = ExperimentLifecycle.from_replay(api.root, executor_factory=factory)

	assert replayed.runner.config == api.runner.config
	assert replayed.runner.inputs.model_dump(mode="json") == api.runner.inputs.model_dump(mode="json")
	assert replayed.root == api.root
	assert (api.root / "invocation" / "normalized-experiment.yaml").is_file()
	assert (api.root / "invocation" / "normalized-scan.yaml").is_file()
	resumed = replayed.resume()
	assert all(item.status == "succeeded" for item in resumed.attempts)


@pytest.mark.system
def test_click_run_consumes_same_lifecycle_controller(monkeypatch, tmp_path):
	executor = ReplayableExecutor()
	runner = IndependentExperimentRunner(
		config((0.01,)), inputs(), tmp_path / "results", executor,
		validator=executor.validate, identity_validator=executor.validate_identity,
	)
	experiment_path = tmp_path / "experiment.yaml"
	scan_path = tmp_path / "scan.yaml"
	experiment_path.write_text("placeholder", encoding="utf-8")
	scan_path.write_text("placeholder", encoding="utf-8")
	monkeypatch.setattr("lftomo.experiment.cli._runner", lambda *_args, **_kwargs: runner)

	result = CliRunner().invoke(main, [
		"run", str(experiment_path), str(scan_path), "--root", str(runner.root),
	])
	assert result.exit_code == 0, result.output
	payload = json.loads(result.output)
	assert payload["experiment_id"] == runner.config.experiment_id
	assert ExperimentLifecycle.status_root(runner.root).state == "succeeded"
