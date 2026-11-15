"""Run-scoped dispatch for shared-memory pipeline kernels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from copy import copy
from dataclasses import dataclass
import multiprocessing
import os
import signal
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from lftomo.pipeline.core import (
	BoundedWorkRequest,
	BufferSpec,
	PipelineState,
	Stage,
	allocate_output,
	commit_output,
	collect_bounded_work,
)
from lftomo.pipeline.dispatch import DispatchKind
from lftomo.storage import ManagedBuffer, buffer_view
from lftomo.util.process import watch_parent_exit
from lftomo.util.shapes import DataRange, IntIter, ProjOrder, ReconOrder, SinoOrder, Theta
from lftomo.util.window import TomoWindow


class WorkerTimeoutError(TimeoutError):
	"""A process task exceeded its Stage deadline and the pool was killed."""


@dataclass(frozen=True)
class _WorkerState:
	scan: Any
	recon: Any
	artifacts: dict[str, Any]
	backend: Any
	dataset: Any
	source_window: TomoWindow | None
	domain: Any


_WORKER_CPU_THREAD_LIMIT: int | None = None


def _worker_initializer(cpu_thread_limit: int) -> None:
	"""Set the run-owned thread ceiling, process group and owner-death watch."""

	global _WORKER_CPU_THREAD_LIMIT
	_WORKER_CPU_THREAD_LIMIT = cpu_thread_limit

	if hasattr(os, "setsid"):
		try:
			os.setsid()
		except OSError:
			pass
	signal.signal(signal.SIGINT, signal.SIG_IGN)
	# The owner cannot submit the executor's normal shutdown sentinels after
	# a fatal signal. Exit even if this worker is idle or executing a task.
	watch_parent_exit(lambda: os._exit(0))


def _limit_worker_cpu_threads(stage: Stage) -> None:
	"""Cap a declared native pool only inside a run-owned process worker."""

	if (
		_WORKER_CPU_THREAD_LIMIT is None
		or stage.contract.resources.cpu_thread_pool != "torch"
	):
		return
	try:
		import torch
	except ImportError:
		# The Stage's numerical boundary owns its actionable dependency error.
		return
	if torch.get_num_threads() > _WORKER_CPU_THREAD_LIMIT:
		torch.set_num_threads(_WORKER_CPU_THREAD_LIMIT)


def _worker_state(
	buffer: ManagedBuffer, payload: _WorkerState, work_range: DataRange,
) -> PipelineState:
	return PipelineState(
		working_array=buffer,
		scan=payload.scan,
		recon=payload.recon,
		artifacts=dict(payload.artifacts),
		backend=payload.backend,
		storage=None,
		dataset=payload.dataset,
		source_window=payload.source_window,
		domain=payload.domain,
		work_range=work_range,
	)


def _work_ranges(state: PipelineState, axis: str) -> Iterator[DataRange]:
	"""Assign one packed-buffer sample per task without deriving geometry."""

	assigned = state.assigned_range(axis)
	axis_type = type(assigned.step)
	for index in assigned.get_iter(IntIter):
		yield axis_type.drange(index, index + 1, 1)


def _work_states(
	state: PipelineState, axis: str, *, work_keys: tuple[str, ...] = (),
) -> Iterator[tuple[np.ndarray, PipelineState]]:
	"""Derive numerical views while retaining the held window and full buffer."""

	data = np.asarray(state.working_array)
	for work_range in _work_ranges(state, axis):
		row_state = copy(state)
		row_state.work_range = work_range
		scoped_envelope = axis == "Theta" and "phase_envelope" in state.artifacts
		scoped_keys = (*work_keys, *(("phase_envelope",) if scoped_envelope else ()))
		row_state.artifacts = dict(state.artifacts) if scoped_keys else state.artifacts
		_scope_bounded_work(row_state, work_keys)
		if scoped_envelope:
			_scope_phase_envelope(row_state)
		index = next(work_range.get_iter(IntIter))
		yield data[state.layout.point_index(**{axis: index})], row_state
		if scoped_keys:
			state.artifacts.update({
				key: value for key, value in row_state.artifacts.items() if key not in scoped_keys
			})


def _scope_phase_envelope(state: PipelineState) -> None:
	"""Derive the assigned projection's view of the prepared envelope."""

	envelope = state.artifacts.get("phase_envelope")
	if isinstance(envelope, ManagedBuffer):
		state.artifacts["phase_envelope"] = envelope[state.work_index(envelope.shape)]
	elif isinstance(envelope, np.ndarray):
		state.artifacts["phase_envelope"] = envelope[state.work_index()]


def run_sinogram_stage(stage: Stage, state: PipelineState) -> PipelineState:
	"""Apply a Stage's ``run_sinogram`` kernel to each detector-row unit.

	The kernel receives a writable, potentially strided ``(Theta, X)`` view
	and a state holding the full buffer, source window, and assigned row range.
	It may mutate the view and return
	``None``, or return replacement values to copy into that same view.
	Artifacts remain shared with the input state. The original state object is
	returned after its array has been updated.

	This is the serial implementation for a Stage's ``run`` method. It accepts
	NumPy arrays and managed buffers without allocating a new working buffer
	or creating an executor; pipeline bounded dispatch owns parallel execution.
	"""

	if state.backend.name != "numpy":
		raise TypeError(f"{stage.name} currently requires the NumPy backend")
	if np.asarray(state.working_array).ndim != 3:
		raise ValueError(f"{stage.name} expects a three-dimensional tomography array")
	for sinogram, row_state in _work_states(state, "Y"):
		result = stage.run_sinogram(sinogram, row_state)
		if result is not None and result is not sinogram:
			sinogram[...] = result
	return state


def run_projection_stage(stage: Stage, state: PipelineState) -> PipelineState:
	"""Run a projection kernel on views derived from the assigned range."""

	if state.backend.name != "numpy":
		raise TypeError(f"{stage.name} currently requires the NumPy backend")
	if np.asarray(state.working_array).ndim != 3:
		raise ValueError(f"{stage.name} expects a three-dimensional tomography array")
	for image, projection_state in _work_states(state, "Theta"):
		result = stage.run_projection(image, projection_state)
		if result is not None and result is not image:
			image[...] = result
	return state


def run_material_retrieval_kernel(source, baseline, payload, stage, work_range, denominator):
	"""Retrieve assigned projections into a coordinator-owned baseline plane."""
	for projection in work_range.get_iter(IntIter):
		index = type(source.shape).point_index(Theta=projection)
		stage.retrieve(
			buffer_view(source)[index], buffer_view(baseline)[index],
			payload.scan, buffer_view(denominator),
		)


def run_material_classification_kernel(current, psi, synthesis, previous, payload,
		stage, work_range, angles_deg, iteration):
	"""Classify assigned rows and synthesize their material line integrals."""
	changes = 0
	for row in work_range.get_iter(IntIter):
		index = type(current.shape).point_index(Y=row)
		changes += stage.classify_row(
			buffer_view(current)[index], buffer_view(psi)[index], buffer_view(synthesis)[index],
			None if previous is None else buffer_view(previous)[ReconOrder.point_index(Y=row)],
			angles_deg, payload.scan, iteration,
		)
	return changes


def run_material_correction_kernel(psi, synthesis, correction, payload, stage, work_range, denominator):
	"""Apply assigned transverse filters using the run-owned executor."""
	for projection in work_range.get_iter(IntIter):
		index = type(psi.shape).point_index(Theta=projection)
		stage.correct(
			buffer_view(psi)[index], buffer_view(synthesis)[index], buffer_view(correction)[index],
			payload.scan, buffer_view(denominator),
		)


def run_multimaterial_stage(stage: Stage, state: PipelineState) -> PipelineState:
	"""Coordinate the composite loop using explicit ranges and managed scratch.

	Read-only configuration, acquisition angles, and reopenable buffers cross IPC.
	All reusable arrays belong to the run storage and are retired even if a
	classification fails. Direct NumPy callers execute the same kernels serially.
	"""
	if state.backend.name != "numpy" or state.layout not in (ProjOrder, SinoOrder):
		raise TypeError("phase.multimaterial requires NumPy tomography data")
	if not np.issubdtype(state.working_array.dtype, np.floating):
		raise TypeError("phase.multimaterial requires floating-point transmission")
	if state.dataset is None or state.source_window is None:
		raise ValueError("phase.multimaterial requires canonical source angles and window geometry")
	angles = np.rad2deg(state.dataset.angles[state.source_window.axis_slice("Theta")])
	if len(angles) != state.working_array.shape.Theta:
		raise ValueError("phase.multimaterial projection count disagrees with source angles")
	# Resolve once in the authoritative parameter model, before worker copies.
	stage.params.line_fit
	from lftomo.pipeline.stages.phase import _resolved_slope_filter

	_resolved_slope_filter(
		state, (state.working_array.shape.Y, state.working_array.shape.X),
		slope=stage.params.line_fit["slope"], pad=stage.params.pad,
	)
	denominator = state.artifacts["phase_filter"]
	executor = state.executor or getattr(state.storage, "pipeline_executor", None)
	parallel = (
		executor is not None and state.storage is not None
		and isinstance(state.working_array, ManagedBuffer)
	)
	payload = _payload(state)

	def dispatch(kernel, arguments):
		return executor.submit_all(kernel, arguments) if parallel else [kernel(*args) for args in arguments]

	requests = _prepare_bounded_work(state, (stage,))
	work = state.artifacts

	try:
		dispatch(run_material_retrieval_kernel, (
			(
				state.working_array, work["_multimaterial_baseline"], payload, stage, work_range, denominator,
			)
			for work_range in _work_ranges(state, "Theta")
		))
		norms, changes = [], []
		for iteration in range(stage.params.n_iter + 1):
			np.subtract(
				buffer_view(work["_multimaterial_baseline"]), buffer_view(work["_multimaterial_correction"]),
				out=buffer_view(work["_multimaterial_current"]), casting="unsafe",
			)
			counts = dispatch(run_material_classification_kernel, (
				(
					work["_multimaterial_current"], work["_multimaterial_psi"],
					work["_multimaterial_synthesis"], work.get("_multimaterial_previous_labels"),
					payload, stage, work_range, angles, iteration,
				)
				for work_range in _work_ranges(state, "Y")
			))
			if iteration == stage.params.n_iter:
				break
			dispatch(run_material_correction_kernel, (
				(
					work["_multimaterial_psi"], work["_multimaterial_synthesis"],
					work["_multimaterial_correction"], payload, stage, work_range, denominator,
				)
				for work_range in _work_ranges(state, "Theta")
			))
			# Reuse the synthesized-phase plane after its last consumer this pass.
			np.square(buffer_view(work["_multimaterial_correction"]), out=buffer_view(work["_multimaterial_psi"]))
			norms.append(float(np.sqrt(buffer_view(work["_multimaterial_psi"]).mean())))
			if iteration:
				changes.append(float(sum(counts) / work["_multimaterial_previous_labels"].shape.prod()))
		np.asarray(state.working_array)[...] = buffer_view(work["_multimaterial_current"])
		state.artifacts["correction_norms"] = norms
		state.artifacts["label_change"] = changes
		return state
	finally:
		_release_bounded_work(state, requests)


def _scope_bounded_work(state: PipelineState, work_keys: tuple[str, ...]) -> None:
	"""Derive scratch views from the invocation's assigned range."""

	for key in work_keys:
		handle = state.artifacts[key]
		state.artifacts[key] = handle[state.work_index(handle.shape)]


def run_projection_kernel(
	handle: ManagedBuffer,
	stage: Stage,
	payload: _WorkerState,
	work_range: DataRange,
) -> dict[str, Any]:
	"""Module-level adapter that mutates only the assigned shared projections."""

	state = _worker_state(handle, payload, work_range)
	before = set(state.artifacts)
	_limit_worker_cpu_threads(stage)
	result = stage.run(state)
	return {
		key: value for key, value in result.artifacts.items()
		if key not in before and not isinstance(value, ManagedBuffer)
	}


def run_sinogram_chain_kernel(
	handle: ManagedBuffer,
	stages: tuple[Stage, ...],
	payload: _WorkerState,
	work_range: DataRange,
	collect_timings: bool = False,
	work_keys: tuple[str, ...] = (),
) -> tuple[float, ...]:
	"""Process assigned rows through a complete bounded Stage chain."""

	state = _worker_state(handle, payload, work_range)
	timings = [0.0] * len(stages) if collect_timings else []
	for working, row_state in _work_states(state, "Y", work_keys=work_keys):
		for index, stage in enumerate(stages):
			_limit_worker_cpu_threads(stage)
			started = time.perf_counter() if collect_timings else None
			result = stage.run_sinogram(working, row_state)
			if started is not None:
				timings[index] += time.perf_counter() - started
			if result is not None and result is not working:
				working[...] = result
	return tuple(timings)


def _prepare_bounded_work(
	state: PipelineState, stages: Sequence[Stage], axis: str | None = None,
) -> tuple[BoundedWorkRequest, ...]:
	"""Allocate and initialize Stage-declared work once for the complete segment.

	Requests required for serial fallback use allocate_output's existing
	allocation boundary even without managed dispatch. A mixed-axis composite has
	no single dispatched axis to validate against every requested layout.
	"""

	requests = collect_bounded_work(stages, BufferSpec.from_buffer(state.working_array))
	if not isinstance(state.working_array, ManagedBuffer) or state.storage is None:
		requests = tuple(request for request in requests if request.required_for_fallback)
	if any(request.artifact_key in state.artifacts for request in requests):
		raise ValueError("bounded work artifacts are private to one segment")
	if axis is not None and any(
		getattr(request.buffer_spec.shape, axis) != getattr(state.working_array.shape, axis)
		for request in requests
	):
		raise ValueError(f"bounded work must preserve the dispatched {axis} extent")
	allocated: list[BoundedWorkRequest] = []
	try:
		for request in requests:
			spec = request.buffer_spec
			buffer, values = allocate_output(
				state, spec.shape, spec.dtype, role=request.role, reuse_from=(request.role,),
			)
			state.artifacts[request.artifact_key] = buffer
			allocated.append(request)
			if request.fill_value is not None:
				values.fill(request.fill_value)
	except BaseException:
		_release_bounded_work(state, tuple(allocated))
		raise
	if requests and state.storage is not None:
		state.artifacts.setdefault("bounded_work_allocations", []).append(
			_bounded_work_report(state, requests, axis),
		)
	return requests


def _bounded_work_report(
	state: PipelineState, requests: tuple[BoundedWorkRequest, ...], axis: str | None,
) -> dict[str, Any]:
	"""Report allocated geometry from the buffers fulfilling Stage requests."""

	buffers = []
	for request in requests:
		buffer = state.artifacts[request.artifact_key]
		buffers.append({
			"artifact_key": request.artifact_key,
			"role": request.role,
			"shape": buffer.shape.as_dict(),
			"dtype": buffer.dtype.str,
			"bytes": buffer.shape.nbytes(buffer.dtype),
		})
	return {
		"scope": "window" if state.source_window is not None else "whole_volume",
		"unit_axis": axis,
		"units": getattr(state.working_array.shape, axis) if axis is not None else None,
		"buffers": buffers,
		"managed_bytes": sum(buffer["bytes"] for buffer in buffers),
		"allocation_count": len(requests),
		"indexing": "buffer_sample_ordinals",
	}


def _release_bounded_work(
	state: PipelineState, requests: tuple[BoundedWorkRequest, ...],
) -> None:
	"""Remove and retire segment work only after all assigned work completes."""

	for request in requests:
		buffer = state.artifacts.pop(request.artifact_key)
		if state.storage is not None and isinstance(buffer, ManagedBuffer):
			state.storage.retire(buffer, role=request.role)


def write_tiff_slice_kernel(
	handle: ManagedBuffer, path: Any, dtype: np.dtype,
) -> None:
	"""Write one slice from a reopenable handle; no ndarray crosses IPC."""

	from lftomo.pipeline.stages.write import _write_array_slice

	# Keep the sliced handle alive for the full write.  Its ``array`` view does
	# not own the reopened shared-memory mapping and becomes invalid when the
	# temporary handle is finalized.
	values = handle.array()
	_write_array_slice(values[0], path, dtype)


def run_layout_fill_kernel(
	source: ManagedBuffer, target: ManagedBuffer,
) -> None:
	"""Copy one projection between manager-owned layout buffers."""

	target.array()[...] = source.array()


def run_affine_fill_kernel(
	source: ManagedBuffer,
	target: ManagedBuffer,
	stage: Stage,
	payload: _WorkerState,
	work_range: DataRange,
) -> None:
	"""Transform one projection into a manager-owned fused output."""

	_limit_worker_cpu_threads(stage)
	from lftomo.pipeline.stages.geometry import transform_affine_projection

	source_values = source.array()
	target_values = target.array()
	row_count = source.shape.Y
	for projection in work_range.get_iter(IntIter):
		image = source_values[type(source.shape).point_index(Theta=projection)]
		transform_affine_projection(
			stage, image, payload.source_window, projection, payload.dataset.physical_shape, row_count,
			target=target_values[type(target.shape).point_index(Theta=projection)],
		)


def run_helical_fill_kernel(
	source: ManagedBuffer,
	target: ManagedBuffer,
	shift_px: int,
	row_start: int,
	row_stop: int,
	crop_valid: bool,
) -> None:
	"""Fill one projection of a manager-owned helical-remap output."""

	source_image = source.array()
	target_image = target.array()
	if crop_valid:
		target_image[:] = source_image[row_start + shift_px:row_stop + shift_px]
		return
	rows = np.arange(row_start, row_stop)
	source_rows = rows + shift_px
	valid = (source_rows >= 0) & (source_rows < source_image.shape[0])
	target_image[valid] = source_image[source_rows[valid]]


def _payload(state: PipelineState) -> _WorkerState:
	scan = state.scan
	recon = state.recon
	if isinstance(state.artifacts.get("flat_fields"), ManagedBuffer):
		metadata = dict(scan.metadata)
		metadata.pop("flat_fields", None)
		scan = scan.model_copy(update={"metadata": metadata})
	if "phase_envelope" in state.artifacts:
		# Replay config is not a second carrier of the already prepared work plane.
		recon = recon.model_copy(update={"pipeline": [
			config.model_copy(update={"params": {
				key: value for key, value in config.params.items() if key != "envelope_m"
			}}) if config.kind == "phase" else config for config in recon.pipeline
		]})
	return _WorkerState(
		scan=scan,
		recon=recon,
		artifacts=dict(state.artifacts),
		backend=state.backend,
		dataset=state.dataset,
		source_window=state.source_window,
		domain=state.domain,
	)


class PipelineExecutor:
	"""Persistent, recoverable process pool owned by one Pipeline run."""

	def __init__(self, max_workers: int, *, timeout_seconds: float | None = None,
			context: str = "spawn"):
		if max_workers < 1:
			raise ValueError("pipeline executor workers must be positive")
		if timeout_seconds is not None and timeout_seconds <= 0:
			raise ValueError("pipeline worker timeout must be positive")
		self.max_workers = max_workers
		self.timeout_seconds = timeout_seconds
		self.context_name = context
		self._pool: ProcessPoolExecutor | None = None
		self._lock = threading.Lock()

	def _ensure_pool(self) -> ProcessPoolExecutor:
		with self._lock:
			if self._pool is None:
				available_cpus = (
					getattr(os, "process_cpu_count", os.cpu_count)()
					or os.cpu_count() or 1
				)
				self._pool = ProcessPoolExecutor(
					max_workers=self.max_workers,
					mp_context=multiprocessing.get_context(self.context_name),
					initializer=_worker_initializer,
					initargs=(max(1, available_cpus // self.max_workers),),
				)
			return self._pool

	def _kill_pool(self) -> None:
		with self._lock:
			pool, self._pool = self._pool, None
		if pool is None:
			return
		# ProcessPoolExecutor has no public immediate worker-termination API on
		# every supported Python version.  Inspecting ``_processes`` is confined
		# to this compatibility boundary and covered by timeout/recovery tests.
		processes = tuple(getattr(pool, "_processes", {}).values())
		for process in processes:
			if process.pid is None:
				continue
			try:
				if hasattr(os, "killpg"):
					os.killpg(process.pid, signal.SIGKILL)
			except ProcessLookupError:
				pass
			# A worker may be unable to create its own process group (notably
			# under some spawn/forkserver supervisors).  Always target the PID
			# as well so timeout recovery never depends on ``setsid`` succeeding.
			if process.is_alive():
				try:
					process.kill()
				except ProcessLookupError:
					pass
		pool.shutdown(wait=False, cancel_futures=True)
		for process in processes:
			process.join(timeout=2)

	def close(self) -> None:
		with self._lock:
			pool, self._pool = self._pool, None
		if pool is not None:
			pool.shutdown(wait=True, cancel_futures=True)

	def submit_all(self, function: Callable, arguments: Iterable[tuple]) -> list[Any]:
		futures = []
		try:
			pool = self._ensure_pool()
			futures = [pool.submit(function, *args) for args in arguments]
			return [future.result(timeout=self.timeout_seconds) for future in futures]
		except TimeoutError as error:
			self._kill_pool()
			raise WorkerTimeoutError(
				f"pipeline worker exceeded {self.timeout_seconds} seconds"
			) from error
		except BrokenProcessPool:
			self._kill_pool()
			raise
		except BaseException:
			# A completed deterministic kernel error does not poison the worker.
			# If sibling tasks remain live, terminate the pool so they cannot mutate
			# shared buffers after the parent has observed failure.
			for future in futures:
				future.cancel()
			if any(not future.done() for future in futures):
				self._kill_pool()
			raise

	def run_projection_stage(self, state: PipelineState, stage: Stage) -> PipelineState:
		"""Dispatch safe projection units while retaining parent-owned handoff."""

		if not isinstance(state.working_array, ManagedBuffer):
			return stage.run(state)
		from lftomo.pipeline.stages import dispatch_for_stage

		spec = dispatch_for_stage(stage)
		if spec.kind is not DispatchKind.PROCESS or not spec.projection_units:
			return stage.run(state)

		original = state
		working_state = state
		if stage.name.startswith("normalize."):
			from lftomo.pipeline.stages.normalize import (
				_managed_flat_fields,
				_normalization_output,
				prepare_normalization_dispatch,
			)

			output, _values = _normalization_output(state)
			if "flat_fields" in state.scan.metadata:
				_managed_flat_fields(state)
			working_state = state.with_array(
				output,
			)
			air_mask_key = getattr(stage.params, "air_mask_artifact", "air_mask")
			air_mask = working_state.artifacts.get(air_mask_key)
			if (
				isinstance(air_mask, np.ndarray)
				and air_mask.ndim == 3
				and working_state.storage is not None
			):
				managed_mask = working_state.storage.allocate(
					working_state.working_array.shape, air_mask.dtype, role="air_mask",
				)
				managed_mask.array()[...] = air_mask
				working_state.artifacts[air_mask_key] = managed_mask
			prepare_normalization_dispatch(stage, working_state)
		else:
			output = state.working_array
		if stage.name in {
			"phase.paganin", "phase.paganin_filter", "phase.paganin_phase_absorption",
		}:
			from lftomo.pipeline.stages.phase import prepare_phase_filter

			prepare_phase_filter(stage, working_state)
		if working_state.layout not in (ProjOrder, SinoOrder):
			return stage.run(state)
		worker_stage = stage
		if "phase_envelope" in working_state.artifacts and hasattr(stage.params, "envelope_m"):
			worker_stage = copy(stage)
			worker_stage.params = stage.params.model_copy(update={"envelope_m": None})
		results = self.submit_all(run_projection_kernel, (
			(
				working_state.working_array, worker_stage, _payload(working_state), work_range,
			)
			for work_range in _work_ranges(working_state, "Theta")
		))
		if stage.name.startswith("normalize."):
			from lftomo.pipeline.stages.normalize import merge_normalization_dispatch

			merge_normalization_dispatch(stage, working_state, results)
			result = commit_output(
				original, output, retire_as="raw_staging",
			)
			result.artifacts.update(working_state.artifacts)
			return result
		for artifacts in results:
			working_state.artifacts.update(artifacts)
		return working_state

	def run_out_of_place_stage(self, state: PipelineState, stage: Stage) -> PipelineState:
		"""Fill a shape/layout-changing output through shared-buffer handles."""

		if not isinstance(state.working_array, ManagedBuffer):
			return stage.run(state)
		from lftomo.pipeline.stages import AffineSkewMovementStage, HelicalIndexRemapStage

		data = np.asarray(state.working_array)
		if stage.name == "layout.convert":
			if state.layout not in (ProjOrder, SinoOrder):
				raise TypeError("layout.convert requires ProjOrder or SinoOrder input")
			if state.layout is stage.contract.produces:
				raise ValueError("layout.convert target is already the current layout")
			shape = state.working_array.shape.to(stage.contract.produces)
			output, _target = allocate_output(
				state, shape, data.dtype, role=f"layout:{stage.contract.produces.__name__}",
			)
			positions = Theta.drange(0, shape.Theta, 1).get_iter(IntIter)
			self.submit_all(run_layout_fill_kernel, (
				(
					state.working_array[state.layout.point_index(Theta=index)],
					output[type(shape).point_index(Theta=index)],
				)
				for index in positions
			))
			return commit_output(
				state, output, retire_as=None,
			)

		if isinstance(stage, AffineSkewMovementStage):
			if stage.target_layout is None:
				layout = state.layout
				if layout not in (ProjOrder, SinoOrder):
					return stage.run(state)
				self.submit_all(run_projection_kernel, (
					(
						state.working_array, stage, _payload(state), work_range,
					)
					for work_range in _work_ranges(state, "Theta")
				))
				return state
			input_layout = state.layout
			if input_layout not in (ProjOrder, SinoOrder):
				raise ValueError("fused affine transition requires a known tomography layout")
			if input_layout is stage.target_layout:
				raise ValueError("fused affine transition must change layout")
			shape = stage.output_shape(state.working_array.shape)
			output, _target = allocate_output(
				state, shape, data.dtype, role="fused_geometry",
			)
			self.submit_all(run_affine_fill_kernel, (
				(
					state.working_array, output, stage, _payload(state), work_range,
				)
				for work_range in _work_ranges(state, "Theta")
			))
			result = commit_output(
				state, output, retire_as=None,
			)
			if state.source_window is not None:
				result.source_window = stage.output_window(state.source_window)
			result.artifacts["fused_layout_transition"] = {
				"from": input_layout.__name__, "to": stage.target_layout.__name__,
				"pad_x": stage.params.pad_x, "skew_degrees": stage.params.skew_degrees,
			}
			return result

		if isinstance(stage, HelicalIndexRemapStage):
			if not stage.params.enabled:
				return state
			source_shape = state.working_array.shape
			if state.source_window is None:
				raise RuntimeError("helical remap requires PipelineState.source_window")
			shape, shifts, start, stop = stage.output_geometry(
				source_shape, stage._pitch(state), state.source_window,
			)
			output, target = allocate_output(
				state, shape, data.dtype, role="helical_remap",
			)
			if not stage.params.crop_valid:
				target.fill(stage.params.fill_value)
			positions = Theta.drange(0, source_shape.Theta, 1).get_iter(IntIter)
			self.submit_all(run_helical_fill_kernel, (
				(
					state.working_array[type(source_shape).point_index(Theta=index)],
					output[type(shape).point_index(Theta=index)],
					int(shifts[index]),
					start, stop, stage.params.crop_valid,
				)
				for index in positions
			))
			result = commit_output(
				state, output, retire_as=None,
			)
			result.artifacts["helical_row_shifts"] = shifts
			return result

		return stage.run(state)

	def run_stage(self, state: PipelineState, stage: Stage) -> PipelineState:
		"""Apply registry dispatch and an explicit nested-core budget."""

		from lftomo.pipeline.stages import dispatch_for_stage

		spec = dispatch_for_stage(stage)
		if spec.kernel == "manager_owned_parallel_fill":
			return self.run_out_of_place_stage(state, stage)
		if spec.kind is not DispatchKind.ENGINE:
			return self.run_projection_stage(state, stage)
		budget = stage.contract.resources.preferred_cpus or self.max_workers
		original_params = stage.params
		original_contract = stage.contract
		if "ncore" in type(original_params).model_fields:
			stage.params = original_params.model_copy(update={
				"ncore": min(self.max_workers, original_params.ncore or budget),
			})
			stage.contract = type(stage).contract_for(stage.params)
		try:
			return stage.run(state)
		finally:
			stage.params = original_params
			stage.contract = original_contract


def run_bounded_sinogram_segment(
	state: PipelineState,
	stages: Sequence[Stage],
	*,
	max_workers: int = 1,
	failure_observer: Callable[[Stage, BaseException], None] | None = None,
	timing_observer: Callable[[tuple[tuple[float, ...], ...]], None] | None = None,
	checkpoint: Callable[[], None] | None = None,
	executor: PipelineExecutor | None = None,
) -> PipelineState:
	"""Dispatch each bounded row once for an entire fused Stage chain."""

	if state.backend.name != "numpy":
		raise TypeError("bounded sinogram segments currently require the NumPy backend")
	if state.layout not in (ProjOrder, SinoOrder):
		raise ValueError("bounded sinogram execution requires tomography layout input")
	if max_workers < 1:
		raise ValueError("bounded sinogram workers must be positive")
	data = np.asarray(state.working_array)
	if data.ndim != 3:
		raise ValueError("bounded sinogram execution expects a three-dimensional array")
	for stage in stages:
		if not callable(getattr(stage, "run_sinogram", None)):
			raise TypeError(f"{stage.name} declares bounded views but has no run_sinogram kernel")
	# The ndarray sinogram path retains its numerical fallback without imposing
	# managed-work dtype/capacity requirements on that path.
	requests = (
		_prepare_bounded_work(state, stages, "Y")
		if isinstance(state.working_array, ManagedBuffer) and state.storage is not None else ()
	)
	work_keys = tuple(request.artifact_key for request in requests)
	member_samples: list[list[float]] = [[] for _stage in stages]
	try:
		if isinstance(state.working_array, ManagedBuffer) and executor is not None:
			if checkpoint is not None:
				checkpoint()
			try:
				rows = executor.submit_all(run_sinogram_chain_kernel, (
					(
						state.working_array, tuple(stages), _payload(state), work_range,
						timing_observer is not None, work_keys,
					)
					for work_range in _work_ranges(state, "Y")
				))
			except BaseException as error:
				if failure_observer is not None:
					failure_observer(stages[0], error)
				raise
			if timing_observer is not None:
				for row in rows:
					if len(row) != len(stages):
						raise RuntimeError("bounded worker returned incomplete member timing data")
					for member_index, elapsed in enumerate(row):
						member_samples[member_index].append(elapsed)
		else:
			# ndarray-only hermetic execution is deliberately serial.
			for working, row_state in _work_states(state, "Y", work_keys=work_keys):
				for member_index, stage in enumerate(stages):
					if checkpoint is not None:
						checkpoint()
					started = time.perf_counter() if timing_observer is not None else None
					result = stage.run_sinogram(working, row_state)
					if started is not None:
						member_samples[member_index].append(time.perf_counter() - started)
					if result is not None and result is not working:
						working[...] = result
	finally:
		_release_bounded_work(state, requests)
	if timing_observer is not None:
		timing_observer(tuple(tuple(samples) for samples in member_samples))
	return state
