from concurrent.futures import ThreadPoolExecutor
from dataclasses import make_dataclass
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np
import pytest

from lftomo.config import (
	Acquisition,
	Detector,
	Recon,
	Scan,
	StageConfig,
	TIFFStackProjections,
)
from lftomo.experiment import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	IndependentExperimentRunner,
	OutputRecord,
	PrefixCacheDecision,
	PrefixCachePolicy,
	PrefixCachingPipelineExecutor,
	PrefixPlanningCandidate,
	plan_common_prefixes,
)
from lftomo.experiment.runner import ExperimentInputs, TrialExecution
from lftomo.experiment.prefix_cache import _fingerprint, _hash_document
from lftomo.pipeline import BufferSpec, Pipeline, PipelineRunContext, PipelineState, Stage, StageContract
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.storage import FileBackedNPArray, ManagedBuffer, buffer_view
from lftomo.util.shapes import ProjOrder, X
from lftomo.util.window import CTFov, TomoDataSet


class SourceStage(Stage):
	name = "test.source"
	contract = StageContract(
		produces=ProjOrder,
		produces_domain=DataDomain.TRANSMISSION,
		produces_artifacts=("nested",),
		cacheable_prefix=True,
	)

	def __init__(self, values, calls=None):
		self.values = np.asarray(values, dtype=np.float32)
		self.calls = calls

	def run(self, state):
		if self.calls is not None:
			with self.calls["lock"]:
				self.calls["count"] += 1
		values = self.values.copy()
		state = state.with_array(
			values, buffer_spec=BufferSpec(ProjOrder(*values.shape), values.dtype),
		)
		state.artifacts["nested"] = {
			"history": ["source"],
			"array": np.arange(3, dtype=np.float32),
		}
		state.scan.metadata["nested"] = {"history": ["source"]}
		state.recon.metadata["nested"] = {"history": ["source"]}
		return state


class MutatingTailStage(Stage):
	name = "test.tail"
	contract = StageContract(
		accepts=(ProjOrder,),
		produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,),
		produces_domain=PRESERVE,
		mutates_input=True,
		cacheable_prefix=True,
	)

	def __init__(self, factor, *, failure=None):
		self.factor = factor
		self.failure = failure

	def run(self, state):
		np.asarray(state.working_array)[:] *= self.factor
		state.artifacts["nested"]["history"].append(self.factor)
		state.artifacts["nested"]["array"][:] += self.factor
		state.scan.metadata["nested"]["history"].append(self.factor)
		state.recon.metadata["nested"]["history"].append(self.factor)
		if self.failure == "failed":
			raise RuntimeError("downstream failure")
		if self.failure == "interrupted":
			raise InterruptedError("downstream cancellation")
		return state


class UnsafeStage(Stage):
	name = "test.unsafe"
	contract = StageContract(mutates_input=False)

	def run(self, state):
		return state


class ReadOnlyTailStage(Stage):
	name = "test.read_only_tail"
	contract = StageContract(
		accepts=(ProjOrder,),
		produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,),
		produces_domain=PRESERVE,
		mutates_input=False,
		cacheable_prefix=True,
	)

	def run(self, state):
		assert not np.asarray(state.working_array).flags.writeable
		state.artifacts["nested"]["history"].append("read-only")
		return state


class InputScaleStage(Stage):
	name = "test.input_scale"
	contract = StageContract(
		produces=ProjOrder,
		mutates_input=True,
		cacheable_prefix=True,
	)

	def __init__(self, factor):
		self.factor = factor

	def run(self, state):
		np.asarray(state.working_array)[:] *= self.factor
		return state


class ManagedArtifactSourceStage(Stage):
	name = "test.managed_artifact_source"
	contract = StageContract(
		produces=ProjOrder,
		produces_domain=DataDomain.TRANSMISSION,
		produces_artifacts=("large_mask", "center", "qc"),
		managed_artifacts=("large_mask",),
		cacheable_prefix=True,
	)

	def __init__(self, values):
		self.values = np.asarray(values, dtype=np.float32)

	def run(self, state):
		values = self.values.copy()
		state = state.with_array(
			values, buffer_spec=BufferSpec(ProjOrder(*values.shape), values.dtype),
		)
		mask = self.values > self.values.mean()
		if state.storage is not None:
			managed = state.storage.allocate(
				ProjOrder(*mask.shape), mask.dtype, role="large_mask",
			)
			managed.array()[...] = mask
			mask = managed
		state.artifacts.update({
			"large_mask": mask,
			"center": 7.5,
			"qc": {"valid": True},
		})
		return state


class ArtifactTailStage(Stage):
	name = "test.artifact_tail"

	def __init__(self, *, mutates):
		self.mutates = mutates
		self.contract = StageContract(
			accepts=(ProjOrder,), produces=PRESERVE,
			accepts_domain=(DataDomain.TRANSMISSION,), produces_domain=PRESERVE,
			mutates_input=mutates, cacheable_prefix=True,
			consumes_artifacts=("large_mask", "center", "qc"),
		)

	def run(self, state):
		mask = state.artifacts["large_mask"]
		assert isinstance(mask, ManagedBuffer)
		assert buffer_view(mask).flags.writeable is self.mutates
		assert state.artifacts["center"] == 7.5
		assert state.artifacts["qc"] == {"valid": True}
		if self.mutates:
			np.asarray(state.working_array)[:] += 1
		return state


class BlockingPrefixStage(Stage):
	name = "test.blocking_prefix"
	contract = StageContract(
		accepts=(ProjOrder,), produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,), produces_domain=PRESERVE,
		cacheable_prefix=True,
	)

	def __init__(self, entered, release):
		self.entered = entered
		self.release = release

	def run(self, state):
		self.entered.set()
		assert self.release.wait(timeout=10)
		return state


def scan():
	return Scan(
		energy_keV=24,
		pixel_size_um=1.3,
		ssd_cm=20,
		n_projections=8,
		degrees_per_fov=180,
		detector=Detector(rows=3, columns=16, pixel_size_um=1.3),
		acquisition=Acquisition(n_projections=8, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


def state(configs=()):
	return PipelineState(
		working_array=np.empty((0,), dtype=np.float32),
		scan=scan(),
		recon=Recon(pipeline=list(configs)),
	)


def config(stage_id, algorithm, **params):
	return StageConfig(id=stage_id, kind="test", algorithm=algorithm, params=params)


def enabled_policy(tmp_path, **updates):
	document = {
		"enabled": True,
		"scratch": tmp_path / "prefix-cache",
		"max_disk_bytes": 10_000_000,
		"max_memory_bytes": 10_000_000,
		"max_entries": 8,
		**updates,
	}
	return PrefixCachePolicy(
		**document,
	)


def planned_pair(tmp_path, values=None, *, policy=None):
	values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1 if values is None else values
	shared = config("source", "array")
	left_tail = config("tail", "multiply", factor=2)
	right_tail = config("tail", "multiply", factor=3)
	policy = policy or enabled_policy(tmp_path)
	candidates = (
		PrefixPlanningCandidate(
			candidate_id="left",
			configs=(shared, left_tail),
			stages=(SourceStage(values), MutatingTailStage(2)),
			share_identity={"dataset": {"content_hash": "a" * 64}},
			estimated_snapshot_bytes=values.nbytes,
		),
		PrefixPlanningCandidate(
			candidate_id="right",
			configs=(shared, right_tail),
			stages=(SourceStage(values), MutatingTailStage(3)),
			share_identity={"dataset": {"content_hash": "a" * 64}},
			estimated_snapshot_bytes=values.nbytes,
		),
	)
	plan = plan_common_prefixes(candidates, policy)
	return policy, candidates, plan


def _segment_exists(name):
	from multiprocessing.shared_memory import SharedMemory

	shared_memory_path = Path("/dev/shm") / name.lstrip("/")
	if shared_memory_path.parent.is_dir():
		return shared_memory_path.exists()
	if "track" not in inspect.signature(SharedMemory).parameters:
		raise RuntimeError(
			"safe shared-memory inspection requires /dev/shm or Python 3.13+"
		)
	try:
		segment = SharedMemory(name, track=False)
	except FileNotFoundError:
		return False
	segment.close()
	return True


def _crashing_cached_trial(connection, scratch):
	if hasattr(os, "setsid"):
		os.setsid()
	values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1
	policy, candidates, plan = planned_pair(scratch, values=values)
	executor = PrefixCachingPipelineExecutor(policy)
	with PipelineRunContext(max_workers=1) as context:
		initial = state(candidates[0].configs)
		initial.storage = context.storage
		initial.executor = context.executor
		working = context.storage.allocate(
			ProjOrder(*values.shape), values.dtype, role="trial-input",
		)
		np.asarray(working)[...] = values
		initial = initial.with_array(working)
		executor.execute(
			plan.decisions[0], initial,
			(SourceStage(values), MutatingTailStage(2)), candidates[0].configs,
			runtime_identity={"dataset": "killed-trial"},
		)
		names = tuple(dict.fromkeys(
			event["name"] for event in context.storage.allocation_events
			if event["action"] == "allocate"
		))
		connection.send(names)
		connection.close()
		while True:
			time.sleep(1)


def test_prefix_planning_is_deterministic_visible_and_fail_closed(tmp_path):
	policy, candidates, first = planned_pair(tmp_path)
	second = plan_common_prefixes(tuple(reversed(candidates)), policy)

	assert first == second
	assert [item.candidate_id for item in first.decisions] == ["left", "right"]
	assert all(item.action == "cache" for item in first.decisions)
	assert all(item.stage_count == 1 for item in first.decisions)
	assert all(item.stage_ids == ("source",) for item in first.decisions)
	assert first.estimated_unique_snapshot_bytes == candidates[0].estimated_snapshot_bytes

	unsafe = PrefixPlanningCandidate(
		candidate_id="unsafe",
		configs=(config("unsafe", "external"),),
		stages=(UnsafeStage(),),
		share_identity={"dataset": "same"},
	)
	other = PrefixPlanningCandidate(
		candidate_id="unsafe-2",
		configs=unsafe.configs,
		stages=(UnsafeStage(),),
		share_identity=unsafe.share_identity,
	)
	decision = plan_common_prefixes((unsafe, other), policy).decisions[0]
	assert decision.action == "independent"
	assert decision.stage_count == 0
	assert "not explicitly prefix-cacheable" in decision.reason

	disabled = plan_common_prefixes(
		candidates, PrefixCachePolicy(),
	)
	assert PrefixCachePolicy().verify_fraction == pytest.approx(0.05)
	assert all(item.action == "independent" for item in disabled.decisions)
	assert "disabled" in disabled.decisions[0].reason


def test_fingerprints_retain_type_identity_and_accept_canonical_geometry():
	left = make_dataclass("Left", [("value", int)])(3)
	right = make_dataclass("Right", [("value", int)])(3)
	assert _hash_document(left) != _hash_document(right)
	dataset = TomoDataSet(
		CTFov(2, 180, 1), ProjOrder(2, 3, 4), X(1),
		angles=np.array([0, 1], dtype=np.float32),
	)
	assert _fingerprint(dataset)["value"] == _fingerprint(dataset.to_document())


def test_cached_forks_are_concurrent_deeply_isolated_and_failure_safe(tmp_path):
	policy, candidates, plan = planned_pair(tmp_path)
	executor = PrefixCachingPipelineExecutor(policy)
	calls = {"count": 0, "lock": threading.Lock()}
	decisions = {item.candidate_id: item for item in plan.decisions}
	shared = candidates[0].configs[0]
	values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1

	def run(candidate_id, factor, failure=None, **callbacks):
		configs = (shared, config("tail", "multiply", factor=factor))
		return executor.execute(
			decisions[candidate_id],
			state(configs),
			(SourceStage(values, calls), MutatingTailStage(factor, failure=failure)),
			configs,
			runtime_identity={"dataset": {"content_hash": "a" * 64}},
			**callbacks,
		)

	with ThreadPoolExecutor(max_workers=4) as pool:
		results = list(pool.map(
			lambda factor: run("left" if factor == 2 else "right", factor),
			(2, 3, 2, 3),
		))
	assert calls["count"] == 1
	assert sorted(event.status for _result, event in results) == ["hit", "hit", "hit", "miss"]
	assert sum(event.written_bytes > 0 for _result, event in results) == 1
	assert all(event.read_bytes > 0 for _result, event in results)
	for result, event in results:
		factor = result.artifacts["nested"]["history"][-1]
		np.testing.assert_array_equal(result.working_array, values * factor)
		np.testing.assert_array_equal(
			result.artifacts["nested"]["array"], np.arange(3) + factor,
		)
		assert result.scan.metadata["nested"]["history"] == ["source", factor]
		assert result.recon.metadata["nested"]["history"] == ["source", factor]
		assert event.fork_strategy == "explicit_copy"

	for failure, exception, terminal in (
		("failed", RuntimeError, "failed"),
		("interrupted", InterruptedError, "interrupted"),
	):
		span_events = []
		cache_events = []
		with pytest.raises(exception):
			run(
				"left", 2, failure,
				execution_observer=span_events.append,
				cache_observer=cache_events.append,
			)
		assert cache_events[0].status == "hit"
		assert span_events[-1].phase.value == terminal
		assert [member.index for member in span_events[-1].span.members] == [1]
	clean, event = run("left", 2)
	assert event.status == "hit"
	np.testing.assert_array_equal(clean.working_array, values * 2)
	assert calls["count"] == 1

	working_path = next((policy.scratch / "entries").glob("*/*/working.npy"))
	np.testing.assert_array_equal(np.load(working_path), values)


def test_nonmutating_suffix_uses_read_only_map_and_unknown_identity_bypasses(tmp_path):
	policy, candidates, plan = planned_pair(tmp_path)
	executor = PrefixCachingPipelineExecutor(policy)
	decision = plan.decisions[0]
	configs = (candidates[0].configs[0], config("tail", "inspect"))
	values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1

	result, event = executor.execute(
		decision,
		state(configs),
		(SourceStage(values), ReadOnlyTailStage()),
		configs,
		runtime_identity={"dataset": "read-only"},
	)
	assert event.fork_strategy == "read_only"
	assert not np.asarray(result.working_array).flags.writeable
	assert result.artifacts["nested"]["history"] == ["source", "read-only"]

	unsafe_state = state(configs)
	unsafe_state.artifacts["uncontracted"] = object()
	timeline = []
	result, event = executor.execute(
		decision,
		unsafe_state,
		(SourceStage(values), MutatingTailStage(2)),
		configs,
		runtime_identity={"dataset": "unsafe-artifact"},
		execution_observer=lambda span: timeline.append((
			"span", span.phase.value,
			tuple(member.index for member in span.span.members),
		)),
		cache_observer=lambda item: timeline.append(("cache", item.status, ())),
	)
	assert event.status == "bypass"
	assert "lacks a cache identity contract" in event.reason
	assert timeline[0] == ("cache", "bypass", ())
	assert [item[2] for item in timeline if item[0] == "span" and item[1] == "started"] == [
		(0,), (1,),
	]
	np.testing.assert_array_equal(result.working_array, values * 2)


def test_cache_fork_is_manager_issued_and_identity_ignores_buffer_name(tmp_path):
	policy, candidates, plan = planned_pair(tmp_path)
	executor = PrefixCachingPipelineExecutor(policy)
	decision = plan.decisions[0]
	values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1
	configs = candidates[0].configs

	with PipelineRunContext(max_workers=1) as context:
		def execute(role):
			initial = state(configs)
			initial.storage = context.storage
			initial.executor = context.executor
			working = context.storage.allocate(
				ProjOrder(*values.shape), values.dtype, role=role,
			)
			np.asarray(working)[...] = values
			initial = initial.with_array(working)
			return executor.execute(
				decision, initial,
				(SourceStage(values), MutatingTailStage(2)), configs,
				runtime_identity={"dataset": "manager-fork"},
			)

		first, first_event = execute("trial-input:a")
		second, second_event = execute("trial-input:b")
		assert isinstance(first.working_array, ManagedBuffer)
		assert isinstance(second.working_array, ManagedBuffer)
		assert first.working_array.name != second.working_array.name
		assert [first_event.status, second_event.status] == ["miss", "hit"]
		assert any(
			event["action"] == "allocate" and event["role"] == "experiment.cache-fork"
			for event in context.storage.allocation_events
		)
		np.testing.assert_array_equal(np.asarray(first.working_array), values * 2)
		np.testing.assert_array_equal(np.asarray(second.working_array), values * 2)


def test_declared_large_artifact_uses_mutability_aware_managed_handles(tmp_path):
	values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1
	shared = config("source", "managed-artifacts")
	configs = (
		(shared, config("tail", "mutating")),
		(shared, config("tail", "read-only")),
	)
	stages = (
		(ManagedArtifactSourceStage(values), ArtifactTailStage(mutates=True)),
		(ManagedArtifactSourceStage(values), ArtifactTailStage(mutates=False)),
	)
	policy = enabled_policy(tmp_path)
	candidates = tuple(PrefixPlanningCandidate(
		candidate_id=name, configs=candidate_configs, stages=candidate_stages,
		share_identity={"dataset": "managed-artifact"},
		estimated_snapshot_bytes=values.nbytes * 2,
	) for name, candidate_configs, candidate_stages in zip(
		("mutating", "read-only"), configs, stages, strict=True,
	))
	plan = plan_common_prefixes(candidates, policy)
	executor = PrefixCachingPipelineExecutor(policy)

	with PipelineRunContext(max_workers=1) as context:
		results = []
		for candidate, decision in zip(candidates, plan.decisions, strict=True):
			initial = state(candidate.configs)
			initial.storage = context.storage
			initial.executor = context.executor
			result, event = executor.execute(
				decision, initial, candidate.stages, candidate.configs,
				runtime_identity=candidate.share_identity,
			)
			results.append((result, event))

		writable, writable_event = results[0]
		read_only, read_only_event = results[1]
		assert isinstance(writable.artifacts["large_mask"], ManagedBuffer)
		assert not isinstance(writable.artifacts["large_mask"], FileBackedNPArray)
		assert isinstance(read_only.artifacts["large_mask"], FileBackedNPArray)
		assert writable.artifacts["center"] == read_only.artifacts["center"] == 7.5
		assert writable.artifacts["qc"] == read_only.artifacts["qc"] == {"valid": True}
		assert writable_event.fork_strategy == "explicit_copy"
		assert read_only_event.fork_strategy == "read_only"

	manifest = json.loads(next((policy.scratch / "entries").glob("*/*/manifest.json")).read_text())
	assert set(manifest["artifact_files"]) == {"large_mask"}
	assert (next((policy.scratch / "entries").glob("*/*")) /
		manifest["artifact_files"]["large_mask"]).is_file()


def test_cache_pin_blocks_alias_while_prefix_operation_retains_source(tmp_path):
	values = np.arange(128, dtype=np.uint16).reshape(2, 8, 8)
	entered = threading.Event()
	release = threading.Event()
	shared = config("prefix", "blocking")
	left = config("tail", "left")
	right = config("tail", "right")
	policy = enabled_policy(tmp_path)
	candidates = tuple(PrefixPlanningCandidate(
		candidate_id=name, configs=(shared, tail),
		stages=(BlockingPrefixStage(entered, release), InputScaleStage(2)),
		share_identity={"dataset": "pin"}, estimated_snapshot_bytes=values.nbytes,
	) for name, tail in (("left", left), ("right", right)))
	decision = plan_common_prefixes(candidates, policy).decisions[0]
	executor = PrefixCachingPipelineExecutor(policy)

	with PipelineRunContext(max_workers=1) as context:
		raw = context.storage.allocate(ProjOrder(*values.shape), values.dtype, role="raw_staging")
		raw.array()[...] = values
		initial = state(candidates[0].configs)
		initial.storage = context.storage
		initial.executor = context.executor
		initial = initial.with_array(raw)
		initial.domain = DataDomain.TRANSMISSION
		with ThreadPoolExecutor(max_workers=1) as pool:
			future = pool.submit(
				executor.execute, decision, initial, candidates[0].stages,
				candidates[0].configs, runtime_identity=candidates[0].share_identity,
			)
			assert entered.wait(timeout=10)
			context.storage.retire(raw, role="raw_staging")
			refused = context.storage.allocate(
				X(32), np.float32, role="reconstruction", reuse_from=("raw_staging",),
			)
			assert refused.name != raw.name
			release.set()
			future.result(timeout=20)

		reused = context.storage.allocate(
			X(32), np.float32, role="reconstruction:retry", reuse_from=("raw_staging",),
		)
		assert reused.name == raw.name
		pin_events = [
			event for event in context.storage.allocation_events
			if event["name"] == raw.name and event["action"] in {"pin", "unpin"}
		]
		assert [event["action"] for event in pin_events] == ["pin", "unpin"]


def test_killed_cached_trial_leaves_no_manager_segments(tmp_path):
	if not hasattr(os, "killpg"):
		pytest.skip("process-group crash cleanup requires POSIX")
	spawn = multiprocessing.get_context("spawn")
	parent, child = spawn.Pipe(duplex=False)
	process = spawn.Process(
		target=_crashing_cached_trial, args=(child, tmp_path / "killed-cache"),
	)
	process.start()
	child.close()
	try:
		assert parent.poll(20), "cached trial did not publish its managed segments"
		names = parent.recv()
		assert len(names) >= 3
		assert all(_segment_exists(name) for name in names)
		os.killpg(process.pid, signal.SIGKILL)
		process.join(timeout=10)
		assert not process.is_alive()
		deadline = time.monotonic() + 10
		while any(_segment_exists(name) for name in names) and time.monotonic() < deadline:
			time.sleep(0.1)
		assert not any(_segment_exists(name) for name in names)
	finally:
		parent.close()
		if process.is_alive():
			os.killpg(process.pid, signal.SIGKILL)
			process.join(timeout=5)


def test_initial_working_array_content_is_part_of_cache_identity(tmp_path):
	policy = enabled_policy(tmp_path)
	shared = config("scale", "input", factor=2)
	candidates = tuple(PrefixPlanningCandidate(
		candidate_id=f"input-{suffix}",
		configs=(shared, config("tail", suffix)),
		stages=(InputScaleStage(2), UnsafeStage()),
		share_identity={"dataset": "same"},
	) for suffix in ("left", "right"))
	decision = plan_common_prefixes(candidates, policy).decisions[0]
	executor = PrefixCachingPipelineExecutor(policy)

	def execute(values):
		initial = state(candidates[0].configs)
		working = np.asarray(values, dtype=np.float32).reshape(1, 1, -1)
		initial = initial.with_array(
			working, buffer_spec=BufferSpec(ProjOrder(*working.shape), working.dtype),
		)
		return executor.execute(
			decision,
			initial,
			candidates[0].stages,
			candidates[0].configs,
			runtime_identity={"dataset": "same"},
		)

	first, event = execute([1, 2, 3])
	assert event.status == "miss"
	np.testing.assert_array_equal(np.asarray(first.working_array).ravel(), [2, 4, 6])
	_same, event = execute([1, 2, 3])
	assert event.status == "hit"
	changed, event = execute([4, 5, 6])
	assert event.status == "miss"
	assert "initial_working_array.sha256" in event.invalidation_fields
	np.testing.assert_array_equal(np.asarray(changed.working_array).ravel(), [8, 10, 12])


def test_invalidation_corruption_recovery_eviction_and_sampled_verification(tmp_path):
	policy, candidates, plan = planned_pair(
		tmp_path,
		policy=enabled_policy(tmp_path, max_entries=1, verify_fraction=1),
	)
	executor = PrefixCachingPipelineExecutor(policy)
	decision = plan.decisions[0]
	configs = candidates[0].configs
	values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1
	calls = {"count": 0, "lock": threading.Lock()}

	def execute(content_hash):
		timeline = []
		def observe(label):
			return lambda span: timeline.append((
				label, span.phase.value,
				tuple(member.index for member in span.span.members),
			))
		result = executor.execute(
			decision,
			state(configs),
			(SourceStage(values, calls), MutatingTailStage(2)),
			configs,
			runtime_identity={"dataset": {"content_hash": content_hash}},
			execution_observer=observe("suffix"),
			cache_build_execution_observer=observe("build"),
			cache_verify_execution_observer=observe("verify"),
			cache_observer=lambda event: timeline.append(("cache", event.status, ())),
		)
		return result[0], result[1], timeline

	_first, event, timeline = execute("a" * 64)
	assert event.status == "miss"
	assert timeline == [
		("build", "started", (0,)), ("build", "completed", (0,)),
		("cache", "miss", ()),
		("suffix", "started", (1,)), ("suffix", "completed", (1,)),
	]
	_second, event, timeline = execute("a" * 64)
	assert event.status == "hit" and event.verified
	assert timeline == [
		("verify", "started", (0,)), ("verify", "completed", (0,)),
		("cache", "hit", ()),
		("suffix", "started", (1,)), ("suffix", "completed", (1,)),
	]
	assert calls["count"] == 2  # one build plus one sampled independent verification

	_partial = policy.scratch / "builds" / "abandoned"
	_partial.mkdir(parents=True)
	(_partial / "partial").write_text("incomplete", encoding="utf-8")
	(policy.scratch / ".cache.lock").write_text(
		'{"pid": 999999, "claimed_at": "2020-01-01T00:00:00Z"}', encoding="utf-8",
	)
	working_path = next((policy.scratch / "entries").glob("*/*/working.npy"))
	with working_path.open("r+b") as stream:
		stream.seek(-1, 2)
		stream.write(b"x")
	_rebuilt, event, timeline = execute("a" * 64)
	assert event.status == "rebuilt"
	assert timeline[0:2] == [
		("build", "started", (0,)), ("build", "completed", (0,)),
	]
	assert ("cache", "rebuilt", ()) in timeline
	assert event.recovered_partials >= 2
	assert any((policy.scratch / "quarantine" / "corrupt").iterdir())

	_changed, event, _timeline = execute("b" * 64)
	assert event.status == "miss"
	assert "runtime.dataset.content_hash" in event.invalidation_fields
	assert event.evicted_keys
	assert len(list((policy.scratch / "entries").glob("*/*/manifest.json"))) == 1


def _independent(stages, configs):
	return Pipeline(list(stages)).run(state(configs))


def test_real_stripe_and_phase_mutation_chains_match_independent_execution(tmp_path):
	from lftomo.pipeline.stages.phase import NegativeLogStage, PaganinFilterStage
	from lftomo.pipeline.stages.stripe import VoLargeStripeStage, VoSortStripeStage

	rng = np.random.default_rng(7)
	stripe_values = rng.random((8, 3, 16), dtype=np.float32) + 0.25
	shared = config("source", "array")
	stripe = StageConfig(id="stripe-sort", kind="stripe", algorithm="vo_sort", params={"size": 5})
	large_a = StageConfig(id="stripe-large", kind="stripe", algorithm="vo_large", params={"size": 5})
	large_b = StageConfig(id="stripe-large", kind="stripe", algorithm="vo_large", params={"size": 7})
	policy = enabled_policy(tmp_path / "stripe")
	candidates = (
		PrefixPlanningCandidate(
			candidate_id="stripe-a", configs=(shared, stripe, large_a),
			stages=(SourceStage(stripe_values), VoSortStripeStage(size=5), VoLargeStripeStage(size=5)),
			share_identity={"dataset": "stripe"}, estimated_snapshot_bytes=stripe_values.nbytes,
		),
		PrefixPlanningCandidate(
			candidate_id="stripe-b", configs=(shared, stripe, large_b),
			stages=(SourceStage(stripe_values), VoSortStripeStage(size=5), VoLargeStripeStage(size=7)),
			share_identity={"dataset": "stripe"}, estimated_snapshot_bytes=stripe_values.nbytes,
		),
	)
	plan = plan_common_prefixes(candidates, policy)
	assert all(item.stage_ids == ("source", "stripe-sort") for item in plan.decisions)
	executor = PrefixCachingPipelineExecutor(policy)
	for candidate, decision in zip(candidates, plan.decisions):
		cached, _event = executor.execute(
			decision, state(candidate.configs), candidate.stages, candidate.configs,
			runtime_identity=candidate.share_identity,
		)
		fresh_stages = (
			SourceStage(stripe_values), VoSortStripeStage(size=5),
			VoLargeStripeStage(size=candidate.configs[-1].params["size"]),
		)
		independent = _independent(fresh_stages, candidate.configs)
		np.testing.assert_allclose(cached.working_array, independent.working_array)

	phase_values = rng.random((8, 3, 16), dtype=np.float32) * 0.5 + 0.5
	filter_config = StageConfig(
		id="phase-filter", kind="phase", algorithm="paganin_filter", params={"alpha": 0.05},
	)
	logs = (
		StageConfig(id="negative-log", kind="phase", algorithm="negative_log", params={"epsilon": 1e-6}),
		StageConfig(id="negative-log", kind="phase", algorithm="negative_log", params={"epsilon": 1e-7}),
	)
	phase_policy = enabled_policy(tmp_path / "phase")
	phase_candidates = tuple(PrefixPlanningCandidate(
		candidate_id=f"phase-{index}",
		configs=(shared, filter_config, log_config),
		stages=(SourceStage(phase_values), PaganinFilterStage(alpha=0.05), NegativeLogStage(
			epsilon=log_config.params["epsilon"],
		)),
		share_identity={"dataset": "phase"},
		estimated_snapshot_bytes=phase_values.nbytes,
	) for index, log_config in enumerate(logs))
	phase_plan = plan_common_prefixes(phase_candidates, phase_policy)
	assert all(item.stage_ids == ("source", "phase-filter") for item in phase_plan.decisions)
	phase_executor = PrefixCachingPipelineExecutor(phase_policy)
	for candidate, decision in zip(phase_candidates, phase_plan.decisions):
		cached, _event = phase_executor.execute(
			decision, state(candidate.configs), candidate.stages, candidate.configs,
			runtime_identity=candidate.share_identity,
		)
		fresh = _independent((
			SourceStage(phase_values), PaganinFilterStage(alpha=0.05),
			NegativeLogStage(epsilon=candidate.configs[-1].params["epsilon"]),
		), candidate.configs)
		np.testing.assert_allclose(cached.working_array, fresh.working_array, rtol=1e-6)


def test_bounds_bypass_before_cache_creation(tmp_path):
	values = np.ones((2, 3, 4), dtype=np.float32)
	policy = PrefixCachePolicy(
		enabled=True,
		scratch=tmp_path / "tiny",
		max_disk_bytes=8,
		max_memory_bytes=8,
	)
	_policy, candidates, _plan = planned_pair(tmp_path, values=values)
	plan = plan_common_prefixes(candidates, policy)
	assert all(item.action == "bypass" for item in plan.decisions)
	assert "disk bound" in plan.decisions[0].reason
	assert not policy.scratch.exists()


class FullRunnerCachedExecutor:
	def __init__(self, source, policy):
		self.source = source
		self.policy = policy
		self.cache = PrefixCachingPipelineExecutor(policy)
		self.decisions = {}

	def __getstate__(self):
		state = self.__dict__.copy()
		state.pop("cache", None)
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		self.cache = PrefixCachingPipelineExecutor(self.policy)

	def plan_prefix_cache(self, trials):
		from lftomo.pipeline.stages import create_stage

		planned = []
		for trial in trials:
			stages = tuple(create_stage(item) for item in trial.identity.resolved_recon.pipeline)
			planned.append(PrefixPlanningCandidate(
				candidate_id=trial.identity.trial_id.value,
				configs=tuple(trial.identity.resolved_recon.pipeline),
				stages=stages,
				share_identity={
					"dataset": trial.identity.dataset,
					"configuration": trial.identity.configuration,
					"code": trial.identity.code,
					"environment": trial.identity.environment,
					"fidelity": trial.identity.fidelity,
				},
				estimated_snapshot_bytes=self.source.nbytes,
			))
		plan = plan_common_prefixes(planned, self.policy)
		self.decisions = {item.candidate_id: item for item in plan.decisions}
		return plan

	def __call__(self, identity, output_root):
		from lftomo.pipeline.stages import create_stage

		configs = tuple(identity.resolved_recon.pipeline)
		stages = tuple(create_stage(item) for item in configs)
		working = self.source.copy()
		initial = PipelineState(
			working_array=working, scan=scan(), recon=identity.resolved_recon,
			buffer_spec=BufferSpec(ProjOrder(*working.shape), working.dtype),
			domain=DataDomain.TRANSMISSION,
		)
		result, event = self.cache.execute(
			self.decisions[identity.trial_id.value], initial, stages, configs,
			runtime_identity={
				"dataset": identity.dataset,
				"configuration": identity.configuration,
				"code": identity.code,
				"environment": identity.environment,
				"fidelity": identity.fidelity,
			},
		)
		path = output_root / "screen.npy"
		path.parent.mkdir(parents=True, exist_ok=True)
		np.save(path, np.asarray(result.working_array))
		return TrialExecution(
			metrics={"mean": float(np.asarray(result.working_array).mean())},
			artifacts={"prefix_cache": event.model_dump(mode="json")},
			outputs=(OutputRecord(
				role="screening_output", path=path, bytes=path.stat().st_size,
			),),
		)


class FullRunnerIndependentExecutor:
	def __init__(self, source):
		self.source = source

	def plan_prefix_cache(self, _trials):
		return {"enabled": False, "decisions": []}

	def __call__(self, identity, output_root):
		from lftomo.pipeline.stages import create_stage

		configs = tuple(identity.resolved_recon.pipeline)
		stages = tuple(create_stage(item) for item in configs)
		working = self.source.copy()
		result = Pipeline(list(stages)).run(PipelineState(
			working_array=working, scan=scan(), recon=identity.resolved_recon,
			buffer_spec=BufferSpec(ProjOrder(*working.shape), working.dtype),
			domain=DataDomain.TRANSMISSION,
		))
		path = output_root / "screen.npy"
		path.parent.mkdir(parents=True, exist_ok=True)
		np.save(path, np.asarray(result.working_array))
		return TrialExecution(
			metrics={"mean": float(np.asarray(result.working_array).mean())},
			outputs=(OutputRecord(
				role="screening_output", path=path, bytes=path.stat().st_size,
			),),
		)


def test_full_runner_metrics_outputs_leaderboard_and_identity_are_equivalent(tmp_path):
	rng = np.random.default_rng(11)
	source = rng.random((8, 3, 16), dtype=np.float32) * 0.4 + 0.6
	config_model = ExperimentConfig.model_validate({
		"name": "prefix-cache-equivalence",
		"candidate_space": {"pipeline": [
			{
				"id": "stripe-sort", "kind": "stripe", "algorithm": "vo_sort",
				"params": {"size": 5},
			},
			{
				"id": "stripe-large", "kind": "stripe", "algorithm": "vo_large",
				"params": {"size": 5, "snr": 2.5, "drop_ratio": 0.1},
			},
			{
				"id": "phase-filter", "kind": "phase", "algorithm": "paganin_filter",
				"sweep": {"kind": "list", "params": {"alpha": [0.02, 0.05, 0.1]}},
			},
			{
				"id": "negative-log", "kind": "phase", "algorithm": "negative_log",
				"params": {"epsilon": 1e-6},
			},
		]},
		"sampler": {"name": "exhaustive", "seed": 7},
		"resources": {"max_concurrency": 3},
		"metrics": [{
			"id": "mean", "metric": "mean", "protocol_version": "1",
			"direction": "minimize",
		}],
		"objectives": [{"metric_id": "mean"}],
		"windows": [{"id": "screen", "y_start": 0, "y_stop": 3}],
		"fidelities": [{"id": "screen", "windows": ["screen"]}],
		"aggregation": {"metric_ids": ["mean"]},
	})
	experiment_inputs = ExperimentInputs(
		dataset=DatasetIdentity(dataset_id="fixture", content_hash="1" * 64),
		configuration=ConfigurationIdentity(scan_config_hash="2" * 64),
		code=CodeIdentity(version="test", git_commit="abcdef123456"),
		environment=EnvironmentIdentity(
			python="3.11", platform="test", backend="numpy",
		),
	)

	cached_executor = FullRunnerCachedExecutor(
		source, enabled_policy(tmp_path / "shared"),
	)
	cached = IndependentExperimentRunner(
		config_model, experiment_inputs, tmp_path / "cached", cached_executor,
	).run()
	independent = IndependentExperimentRunner(
		config_model, experiment_inputs, tmp_path / "independent",
		FullRunnerIndependentExecutor(source),
	).run()

	assert cached.preflight.prefix_cache["enabled"] is True
	assert {
		attempt.observations["artifacts"]["prefix_cache"]["status"]
		for attempt in cached.attempts
	} == {"miss", "hit"}
	assert [item.trial_id for item in cached.leaderboard] == [
		item.trial_id for item in independent.leaderboard
	]
	for cached_entry, independent_entry in zip(cached.leaderboard, independent.leaderboard):
		assert cached_entry.candidate_key == independent_entry.candidate_key
		assert cached_entry.status == independent_entry.status
		assert cached_entry.metrics == independent_entry.metrics
		assert cached_entry.pareto == independent_entry.pareto
		assert cached_entry.parameters == independent_entry.parameters
	for cached_attempt, independent_attempt in zip(cached.attempts, independent.attempts):
		assert cached_attempt.identity == independent_attempt.identity
		assert cached_attempt.result.metrics == independent_attempt.result.metrics
		assert cached_attempt.result.pareto == independent_attempt.result.pareto
		assert cached_attempt.result.resolved_recon == independent_attempt.result.resolved_recon
		cached_output = cached_attempt.result.outputs[0]
		independent_output = independent_attempt.result.outputs[0]
		assert cached_output.role == independent_output.role
		assert cached_output.bytes == independent_output.bytes
		np.testing.assert_allclose(np.load(cached_output.path), np.load(independent_output.path))


def test_cli_dry_run_exposes_real_prefix_plan_without_creating_scratch(tmp_path):
	from click.testing import CliRunner
	import yaml

	from lftomo.experiment.cli import main

	scan_document = scan().model_dump(mode="json")
	scan_document["metadata"] = {
		"dataset_content_hash": "3" * 64,
		"flat_fields": np.stack([
			np.ones((3, 16), dtype=np.float32),
			np.ones((3, 16), dtype=np.float32),
			np.zeros((3, 16), dtype=np.float32),
			np.zeros((3, 16), dtype=np.float32),
		]).tolist(),
	}
	scan_path = tmp_path / "scan.yaml"
	scan_path.write_text(yaml.safe_dump(scan_document), encoding="utf-8")
	experiment = {
		"name": "prefix-dry-run",
		"candidate_space": {"pipeline": [
			{"id": "read", "kind": "read", "algorithm": "tiff_stack"},
			{"id": "normalize", "kind": "normalize", "algorithm": "weighted"},
			{
				"id": "stripe", "kind": "stripe", "algorithm": "vo_sort",
				"params": {"size": 5},
			},
			{
				"id": "phase", "kind": "phase", "algorithm": "paganin_filter",
				"sweep": {"kind": "list", "params": {"alpha": [0.02, 0.05]}},
			},
			{
				"id": "log", "kind": "phase", "algorithm": "negative_log",
				"params": {"epsilon": 1e-6},
			},
			{
				"id": "recon", "kind": "recon", "algorithm": "gridrec",
				"params": {"center": 7.5},
			},
			{
				"id": "metrics", "kind": "evaluate", "algorithm": "metrics",
				"params": {"metrics": [{"kind": "ring", "inner_radius": 1}]},
			},
		]},
		"sampler": {"name": "exhaustive", "seed": 1},
		"resources": {"max_concurrency": 2},
		"metrics": [{
			"id": "rings", "metric": "artifact.ring_score",
			"protocol_version": "1", "direction": "minimize",
		}],
		"objectives": [{"metric_id": "rings"}],
		"windows": [{"id": "screen", "y_start": 0, "y_stop": 2}],
		"fidelities": [{"id": "screen", "windows": ["screen"]}],
		"aggregation": {"metric_ids": ["rings"]},
		"output": {"screening": "metrics_only"},
	}
	experiment_path = tmp_path / "experiment.yaml"
	experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
	scratch = tmp_path / "cache"
	result = CliRunner().invoke(main, [
		"dry-run", str(experiment_path), str(scan_path),
		"--root", str(tmp_path / "results"),
		"--prefix-cache",
		"--prefix-cache-root", str(scratch),
		"--prefix-cache-max-disk-bytes", "10000000",
		"--prefix-cache-max-memory-bytes", "10000000",
	])
	assert result.exit_code == 0, result.output
	payload = json.loads(result.output)
	plan = payload["prefix_cache"]
	assert plan["enabled"] is True
	assert len(plan["decisions"]) == 2
	assert all(item["action"] == "cache" for item in plan["decisions"])
	assert all(item["stage_ids"] == ["read", "normalize", "stripe"] for item in plan["decisions"])
	assert not scratch.exists()
