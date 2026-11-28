"""Full-environment regression for prefix-cache/lifecycle adapter integration."""

import json
import time

import h5py
import numpy as np
import pytest
import tifffile as tf

from lftomo.config import (
	Acquisition,
	Detector,
	HDF5DxchangeProjections,
	Scan,
)
from lftomo.experiment import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	ExperimentLifecycle,
	IndependentExperimentRunner,
	PrefixCachePolicy,
	ProgressInfrastructureError,
	ResumePolicy,
	read_progress_events,
)
from lftomo.experiment.cli import PipelineTrialExecutor
from lftomo.experiment.runner import ExperimentInputs
from lftomo.experiment.events import current_execution_context
from lftomo.util.window import TomoWindow


class HistoricalCollisionExecutor:
	def __init__(self, working_executor):
		self.working_executor = working_executor

	def plan_prefix_cache(self, trials):
		return self.working_executor.plan_prefix_cache(trials)

	def replay_document(self):
		return self.working_executor.replay_document()

	def __call__(self, _identity, _root):
		raise TypeError(
			"ExecutionContext.cache_event() got multiple values for argument 'status'"
		)


class CacheProgressFailureExecutor(HistoricalCollisionExecutor):
	def __call__(self, _identity, _root):
		raise ProgressInfrastructureError(
			"failed to publish prefix cache progress: invalid cache progress projection"
		)


class ConcurrentCacheProgressFailureExecutor(HistoricalCollisionExecutor):
	def __init__(self, working_executor, rendezvous):
		super().__init__(working_executor)
		self.rendezvous = rendezvous

	def __call__(self, identity, _root):
		value = float(identity.resolved_recon.pipeline[2].params["epsilon"])
		self.rendezvous.mkdir(parents=True, exist_ok=True)
		(self.rendezvous / f"{identity.trial_id.value}.json").write_text(
			json.dumps({"epsilon": value}), encoding="utf-8",
		)
		deadline = time.monotonic() + 5
		while len(list(self.rendezvous.glob("*.json"))) < 2:
			if time.monotonic() >= deadline:
				raise TimeoutError("concurrent trial rendezvous timed out")
			time.sleep(0.01)
		if value == 0.0001:
			raise ProgressInfrastructureError(
				"failed to publish prefix cache progress: "
				"invalid concurrent cache progress projection"
			)
		context = current_execution_context()
		while True:
			time.sleep(0.01)
			context.checkpoint()


def _lifecycle(
	tmp_path, monkeypatch, *, result_name="results",
	max_concurrency=1, window_count=1,
):
	"""Build a tiny real adapter/lifecycle run with a shared cacheable prefix."""

	import tomopy

	data = np.stack([
		20 + theta + np.broadcast_to(np.linspace(0, 5, 16), (4, 16))
		for theta in range(8)
	]).astype(np.float32)
	h5_path = tmp_path / "scan.h5"
	with h5py.File(h5_path, "w") as handle:
		handle.create_dataset("exchange/data", data=data)
		handle.create_dataset(
			"exchange/data_white", data=np.full((2, 4, 16), 100, np.float32),
		)
		handle.create_dataset(
			"exchange/data_dark", data=np.full((2, 4, 16), 2, np.float32),
		)
	flat_paths = []
	for name, value in zip(
		("pre-gain", "post-gain", "pre-dark", "post-dark"),
		(100, 102, 2, 3),
		strict=True,
	):
		path = tmp_path / f"{name}.tif"
		tf.imwrite(path, np.full((4, 16), value, np.float32))
		flat_paths.append(str(path))
	scan_model = Scan(
		energy_keV=20,
		pixel_size_um=1,
		ssd_cm=10,
		n_projections=8,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1, rows=4, columns=16),
		acquisition=Acquisition(n_projections=8, rotation_end_deg=180),
		projections=HDF5DxchangeProjections(
			format="hdf5_dxchange", path=h5_path,
			data_path="exchange/data", white_path="exchange/data_white",
			dark_path="exchange/data_dark",
		),
		metadata={
			"dataset_id": "prefix-lifecycle-fixture",
			"dataset_content_hash": "c" * 64,
			"flat_fields": flat_paths,
		},
	)
	scan_path = tmp_path / "scan.yaml"
	scan_path.write_text(scan_model.to_yaml(), encoding="utf-8")
	config = ExperimentConfig.model_validate({
		"name": "prefix-cache-lifecycle",
		"candidate_space": {"pipeline": [
			{
				"id": "read", "kind": "read", "algorithm": "hdf5_dxchange",
				"params": {"order": "projection", "calibration_policy": "required"},
			},
			{"id": "normalize", "kind": "normalize", "algorithm": "weighted"},
			{
				"id": "negative-log", "kind": "phase", "algorithm": "negative_log",
				"sweep": {"kind": "list", "params": {"epsilon": [0.0001, 0.001]}},
			},
			{
				"id": "center", "kind": "center", "algorithm": "manual",
				"params": {"value_px": 8},
			},
			{
				"id": "reconstruct", "kind": "recon", "algorithm": "gridrec",
				"params": {"circle_mask": False},
			},
			{
				"id": "evaluate", "kind": "evaluate", "algorithm": "metrics",
				"params": {
					"protocol_version": "1",
					"metrics": [{"kind": "ring", "inner_radius": 2}],
				},
			},
		]},
		"sampler": {"name": "exhaustive", "seed": 11},
		"resources": {"max_concurrency": max_concurrency},
		"metrics": [{
			"id": "rings", "metric": "artifact.ring_score",
			"protocol_version": "1", "direction": "minimize",
		}],
		"objectives": [{"metric_id": "rings"}],
		"windows": [
			{"id": f"tune-{index + 1}", "y_start": index * 2, "y_stop": index * 2 + 2}
			for index in range(window_count)
		],
		"fidelities": [{
			"id": "screen",
			"windows": [f"tune-{index + 1}" for index in range(window_count)],
		}],
		"aggregation": {"metric_ids": ["rings"]},
		"output": {"screening": "metrics_only"},
	})

	def reconstruct(values, *_args, **_kwargs):
		mean = np.asarray(values).mean(axis=0)
		return np.repeat(mean[:, None, :], mean.shape[1], axis=1)

	monkeypatch.setattr(tomopy, "recon", reconstruct)
	policy = PrefixCachePolicy(
		enabled=True,
		scratch=tmp_path / "shared" / "prefix-cache",
		max_disk_bytes=10_000_000,
		max_memory_bytes=10_000_000,
		max_entries=8,
		verify_fraction=0,
	)
	executor = PipelineTrialExecutor(config, scan_path, policy)
	runner = IndependentExperimentRunner(
		config,
		ExperimentInputs(
			source=TomoWindow.from_bounds(scan_model.projection_shape(), dtype=data.dtype),
			dataset=DatasetIdentity(
				dataset_id="prefix-lifecycle-fixture", content_hash="c" * 64,
			),
			configuration=ConfigurationIdentity(scan_config_hash="d" * 64),
			code=CodeIdentity(version="test", git_commit="abcdef123456"),
			environment=EnvironmentIdentity(
				python="3.11", platform="test", backend="numpy",
				dependencies={"numpy": np.__version__},
			),
		),
		tmp_path / result_name,
		executor,
		validator=executor.validate,
		identity_validator=executor.validate_identity,
	)
	return ExperimentLifecycle(runner), executor


@pytest.mark.system
def test_real_pipeline_lifecycle_persists_cache_miss_hit_and_retries_old_failures(
	tmp_path, monkeypatch,
):
	import lftomo.runner as reconstruction_runner

	loaded_paths = []
	load = reconstruction_runner.load_scan

	def record_load(path):
		loaded = load(path)
		loaded_paths.append(loaded[1])
		return loaded

	monkeypatch.setattr(reconstruction_runner, "load_scan", record_load)
	api, executor = _lifecycle(tmp_path, monkeypatch)
	scan_path = tmp_path / "scan.yaml"
	assert loaded_paths == [scan_path]
	scan_document = executor.replay_document()["scan"]
	scan_path.unlink()
	summary = api.run()
	assert all(attempt.status == "succeeded" for attempt in summary.attempts)
	assert loaded_paths == [scan_path]
	assert executor.replay_document()["scan"] == scan_document
	all_events = read_progress_events(api.root / "progress.jsonl")
	cache_events = [
		event for event in all_events
		if event.kind in ("cache.miss", "cache.hit")
	]
	assert [event.kind for event in cache_events] == ["cache.miss", "cache.hit"]
	assert all("status" not in event.details for event in cache_events)
	assert [event.details["cache_status"] for event in cache_events] == ["miss", "hit"]
	assert all(event.cache_bytes and event.cache_bytes > 0 for event in cache_events)
	hit = cache_events[1]
	assert hit.cached_stage_count == 2
	assert [member.occurrence_id for member in hit.cached_stage_members] == [
		"read", "normalize",
	]
	assert all(member.cache_state == "cached" for member in hit.cached_stage_members)
	next_segment = next(
		event for event in all_events
		if event.sequence > hit.sequence and event.kind == "segment.started"
		and event.trial_id == hit.trial_id
	)
	assert next_segment.cached_stage_count == 2
	assert next_segment.segment_members[0].position == 3
	assert next_segment.segment_members[0].total == 6
	build_events = [
		event for event in all_events
		if event.kind.startswith("segment.")
		and any(member.cache_state == "cache_build" for member in event.segment_members)
	]
	assert build_events
	assert build_events[0].segment_members[0].position == 1
	assert build_events[0].segment_members[0].total == 6

	retry_root = tmp_path / "retry-case"
	retry_root.mkdir()
	broken, working_executor = _lifecycle(retry_root, monkeypatch)
	retry_scan_path = retry_root / "scan.yaml"
	assert loaded_paths == [scan_path, retry_scan_path]
	replay_document = working_executor.replay_document()
	retry_scan_path.unlink()

	broken.runner.executor = HistoricalCollisionExecutor(working_executor)
	failed = broken.run()
	assert all(attempt.status == "failed" for attempt in failed.attempts)
	assert all(
		"multiple values for argument 'status'" in attempt.failure.message
		for attempt in failed.attempts
	)

	resumed = ExperimentLifecycle.from_replay(broken.root)
	normalized_path = broken.root / "invocation" / "normalized-scan.yaml"
	assert loaded_paths == [scan_path, retry_scan_path, normalized_path]
	assert resumed.runner.executor.replay_document() == replay_document
	normalized_path.unlink()
	retried = resumed.resume(ResumePolicy(retry_failed=True))
	assert all(attempt.status == "succeeded" for attempt in retried.attempts)
	assert loaded_paths == [scan_path, retry_scan_path, normalized_path]
	assert resumed.runner.executor.replay_document() == replay_document
	assert all(
		[attempt.status for attempt in history.attempts] == ["failed", "succeeded"]
		for history in resumed.inspect().trials
	)


@pytest.mark.system
def test_cache_progress_programming_error_aborts_experiment_not_candidate(
	tmp_path, monkeypatch,
):
	api, working_executor = _lifecycle(tmp_path, monkeypatch)
	api.runner.executor = CacheProgressFailureExecutor(working_executor)
	with pytest.raises(
		ProgressInfrastructureError, match="failed to publish prefix cache progress",
	):
		api.run()
	status = api.status()
	assert status.counts.get("failed", 0) == 0
	assert status.counts.get("interrupted", 0) == 1
	assert status.attempts[0].failure.type == "ProgressInfrastructureError"
	assert read_progress_events(api.root / "progress.jsonl")[-1].kind == "experiment.failed"


@pytest.mark.system
def test_cache_progress_infrastructure_abort_stops_other_worker_without_false_failure(
	tmp_path, monkeypatch,
):
	api, _executor = _lifecycle(
		tmp_path, monkeypatch, max_concurrency=2, window_count=2,
	)
	api.runner.executor = ConcurrentCacheProgressFailureExecutor(
		_executor, tmp_path / "rendezvous",
	)
	with pytest.raises(
		ProgressInfrastructureError,
		match="failed to publish prefix cache progress",
	):
		api.run()

	status = api.status()
	assert len(list((tmp_path / "rendezvous").glob("*.json"))) == 2
	assert status.counts.get("failed", 0) == 0
	assert status.counts.get("interrupted", 0) == 2
	assert {attempt.failure.type for attempt in status.attempts} == {
		"ExperimentCancelled", "ProgressInfrastructureError",
	}
	assert all(attempt.state == "interrupted" for attempt in status.attempts)
	assert read_progress_events(api.root / "progress.jsonl")[-1].kind == "experiment.failed"
