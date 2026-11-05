"""Run-scoped dispatch for shared-memory pipeline kernels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import multiprocessing
import os
import signal
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from lftomo.pipeline.core import (
	BufferSpec,
	PipelineState,
	Stage,
	allocate_output,
	commit_output,
)
from lftomo.pipeline.dispatch import DispatchKind
from lftomo.storage import ManagedBuffer
from lftomo.util.process import watch_parent_exit
from lftomo.util.shapes import IntIter, ProjOrder, ReconOrder, SinoOrder, Theta
from lftomo.util.window import DataRef, TomoWindow


class WorkerTimeoutError(TimeoutError):
	"""A process task exceeded its Stage deadline and the pool was killed."""


@dataclass(frozen=True)
class _WorkerState:
	scan: Any
	recon: Any
	artifacts: dict[str, Any]
	backend: Any
	dataset: Any
	source_window: Any
	buffer_spec: BufferSpec
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
	buffer: ManagedBuffer, payload: _WorkerState,
) -> PipelineState:
	buffer_spec = BufferSpec(payload.source_window.shape, payload.source_window.dtype)
	return PipelineState(
		working_array=buffer,
		scan=payload.scan,
		recon=payload.recon,
		artifacts=dict(payload.artifacts),
		backend=payload.backend,
		storage=None,
		dataset=payload.dataset,
		source_window=payload.source_window,
		buffer_spec=buffer_spec,
		domain=payload.domain,
	)


def _buffer_ref(state: PipelineState, shape=None) -> DataRef:
	"""Bind current buffer extents to the source window's global coordinates."""

	shape = state.buffer_spec.shape if shape is None else shape
	position = shape.make_with(0)
	step = shape.make_with(1)
	if state.source_window is not None:
		source = (
			state.source_window if state.layout is ReconOrder
			else state.source_window.with_pad()
		)
		source_position = source.pos.as_dict()
		source_step = state.source_window.step.as_dict()
		position = type(shape)(**{
			name: source_position.get(name, value)
			for name, value in position.as_dict().items()
		})
		step = type(shape)(**{
			name: source_step.get(name, value)
			for name, value in step.as_dict().items()
		})
	if isinstance(state.source_window, TomoWindow):
		return TomoWindow(
			shape, shape.make_with(0), position, step, state.buffer_spec.dtype,
			state.source_window.internal.dtype,
		)
	return DataRef(shape, shape.make_with(0), position, step, state.buffer_spec.dtype)


def _buffer_units(
	state: PipelineState, axis: str, shape=None,
) -> Iterator[tuple[DataRef, tuple[slice, ...]]]:
	return _buffer_ref(state, shape).units(axis)


def _unit_index(
	unit: DataRef, local_index: tuple[slice, ...], shape,
) -> tuple[slice, ...]:
	by_name = dict(zip(unit.shape.names, local_index, strict=True))
	return type(shape).point_index(**{
		name: by_name[name] for name in shape.names if name in by_name
	})


def _unit_handle(
	buffer: ManagedBuffer, unit: DataRef, local_index: tuple[slice, ...],
) -> ManagedBuffer:
	return buffer[_unit_index(unit, local_index, buffer.shape)]


def _unit_states(
	state: PipelineState, axis: str,
) -> Iterator[tuple[np.ndarray, PipelineState]]:
	"""Bind numerical views and states to the same canonical unit iterator."""

	data = np.asarray(state.working_array)
	for unit, local_index in _buffer_units(state, axis):
		row_data = data[_unit_index(unit, local_index, state.buffer_spec.shape)]
		row_state = state.with_array(
			row_data, buffer_spec=BufferSpec(unit.shape, data.dtype),
		)
		row_state.artifacts = state.artifacts
		row_state.source_window = unit
		yield row_data[state.layout.point_index(**{axis: 0})], row_state


def run_sinogram_stage(stage: Stage, state: PipelineState) -> PipelineState:
	"""Apply a Stage's ``run_sinogram`` kernel to each detector-row unit.

	The kernel receives a writable, potentially strided ``(Theta, X)`` view
	and a row-scoped state whose ``BufferSpec`` and ``source_window`` come
	from the canonical unit iterator. It may mutate the view and return
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
	for sinogram, row_state in _unit_states(state, "Y"):
		result = stage.run_sinogram(sinogram, row_state)
		if result is not None and result is not sinogram:
			sinogram[...] = result
	return state


def run_projection_stage(stage: Stage, state: PipelineState) -> PipelineState:
	"""Run a projection kernel on shared views with canonical unit geometry."""

	if state.backend.name != "numpy":
		raise TypeError(f"{stage.name} currently requires the NumPy backend")
	if np.asarray(state.working_array).ndim != 3:
		raise ValueError(f"{stage.name} expects a three-dimensional tomography array")
	for image, projection_state in _unit_states(state, "Theta"):
		result = stage.run_projection(image, projection_state)
		if result is not None and result is not image:
			image[...] = result
	return state


def _scope_vo_work_to_unit(
	state: PipelineState, unit: DataRef, local_index: tuple[slice, ...],
) -> None:
	"""Bind bounded-segment scratch handles to one iterator-produced row slice."""

	from lftomo.pipeline.stages.stripe import (
		VO_FLOAT_WORK_ARTIFACT,
		VO_INDEX_WORK_ARTIFACT,
		VO_MEDIAN_PAD_WORK_ARTIFACT,
	)

	for key in (
		VO_FLOAT_WORK_ARTIFACT, VO_INDEX_WORK_ARTIFACT, VO_MEDIAN_PAD_WORK_ARTIFACT,
	):
		if key in state.artifacts:
			state.artifacts[key] = _unit_handle(
				state.artifacts[key], unit, local_index,
			)


def run_projection_kernel(
	handle: ManagedBuffer,
	stage: Stage,
	payload: _WorkerState,
	local_index: tuple[slice, ...],
) -> dict[str, Any]:
	"""Module-level adapter that mutates one shared projection view."""

	state = _worker_state(handle, payload)
	unit = payload.source_window
	index = local_index[unit.shape.axis_of("Theta")].start
	projection_times = state.artifacts.get("projection_times")
	if isinstance(projection_times, ManagedBuffer):
		projection_times = projection_times.array()
	if projection_times is not None and len(projection_times) == payload.buffer_spec.shape.Theta:
		state.artifacts["projection_times"] = np.asarray(
			projection_times[index:index + 1],
		)
	air_mask_key = getattr(stage.params, "air_mask_artifact", "air_mask")
	air_mask = state.artifacts.get(air_mask_key)
	if isinstance(air_mask, ManagedBuffer) and air_mask.array().ndim == 3:
		state.artifacts[air_mask_key] = _unit_handle(air_mask, unit, local_index)
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
	local_index: tuple[slice, ...],
	collect_timings: bool = False,
) -> tuple[float, ...]:
	"""Process one unit-shaped row through a complete bounded Stage chain."""

	state = _worker_state(handle, payload)
	_scope_vo_work_to_unit(state, payload.source_window, local_index)
	data = np.asarray(state.working_array)
	layout = state.layout
	working = data[layout.point_index(Y=0)]
	timings = []
	for stage in stages:
		_limit_worker_cpu_threads(stage)
		started = time.perf_counter() if collect_timings else None
		result = stage.run_sinogram(working, state)
		if started is not None:
			timings.append(time.perf_counter() - started)
		if result is not None and result is not working:
			working[...] = result
	return tuple(timings)


def _prepare_vo_work(
	state: PipelineState,
	stages: Sequence[Stage],
	row_count: int,
	sinogram_shape: tuple[int, int],
) -> tuple[tuple[ManagedBuffer, str], ...]:
	"""Allocate row-indexed work planes shared by a fused Vo chain."""

	if not any(getattr(stage, "uses_vo_work", False) for stage in stages):
		return ()
	if not isinstance(state.working_array, ManagedBuffer) or state.storage is None:
		return ()
	from lftomo.pipeline.stages.stripe import (
		VO_FLOAT_WORK_ARTIFACT,
		VO_INDEX_WORK_ARTIFACT,
		VO_MEDIAN_PAD_WORK_ARTIFACT,
	)

	if any(key in state.artifacts for key in (
		VO_FLOAT_WORK_ARTIFACT, VO_INDEX_WORK_ARTIFACT, VO_MEDIAN_PAD_WORK_ARTIFACT,
	)):
		raise ValueError("Vo work artifacts are private to one bounded segment")
	theta_count, column_count = sinogram_shape
	if max(theta_count, column_count) > np.iinfo(np.uint32).max:
		raise ValueError("Vo sorting indices exceed uint32 capacity")
	shape = SinoOrder(row_count, theta_count, column_count)
	float_dtype = np.dtype(state.working_array.dtype)
	if not np.issubdtype(float_dtype, np.floating):
		raise TypeError("Vo reusable work requires floating-point pipeline data")
	median_half_width = max(
		(int(getattr(stage, "vo_line_median_half_width", 0)) for stage in stages),
		default=0,
	)
	if float_dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or not float_dtype.isnative:
		median_half_width = 0
	allocated: list[tuple[ManagedBuffer, str]] = []
	try:
		float_work = state.storage.allocate(shape, float_dtype, role="vo_float_work")
		if isinstance(float_work, list):
			raise TypeError("Vo floating-point work allocation returned multiple buffers")
		allocated.append((float_work, "vo_float_work"))
		index_work = state.storage.allocate(shape, np.uint32, role="vo_index_work")
		if isinstance(index_work, list):
			raise TypeError("Vo sorting-index work allocation returned multiple buffers")
		allocated.append((index_work, "vo_index_work"))
		median_pad_work = None
		if median_half_width:
			median_pad_shape = SinoOrder(
				row_count, theta_count, column_count + 2 * median_half_width,
			)
			median_pad_work = state.storage.allocate(
				median_pad_shape, float_dtype, role="vo_median_pad_work",
			)
			if isinstance(median_pad_work, list):
				raise TypeError("Vo median-pad work allocation returned multiple buffers")
			allocated.append((median_pad_work, "vo_median_pad_work"))
			median_pad_work.array().fill(0)
	except BaseException:
		for buffer, role in allocated:
			state.storage.retire(buffer, role=role)
		raise
	state.artifacts[VO_FLOAT_WORK_ARTIFACT] = float_work
	state.artifacts[VO_INDEX_WORK_ARTIFACT] = index_work
	if median_pad_work is not None:
		state.artifacts[VO_MEDIAN_PAD_WORK_ARTIFACT] = median_pad_work
	float_bytes = shape.nbytes(float_dtype)
	index_bytes = shape.nbytes(np.uint32)
	median_pad_bytes = (
		median_pad_shape.nbytes(float_dtype)
		if median_pad_work is not None else 0
	)
	report = {
		"scope": "window" if state.source_window is not None else "whole_volume",
		"rows": row_count,
		"sinogram_shape": [theta_count, column_count],
		"float_dtype": float_dtype.str,
		"index_dtype": np.dtype(np.uint32).str,
		"float_bytes": float_bytes,
		"index_bytes": index_bytes,
		"median_pad_half_width": median_half_width,
		"median_pad_bytes": median_pad_bytes,
		"managed_bytes": float_bytes + index_bytes + median_pad_bytes,
		"allocation_count": 2 + int(median_pad_work is not None),
		"indexing": "units_local_index",
	}
	state.artifacts.setdefault("vo_scratch_allocations", []).append(report)
	return tuple(allocated)


def _release_vo_work(
	state: PipelineState, allocated: tuple[tuple[ManagedBuffer, str], ...],
) -> None:
	"""Retire temporary Vo planes after all dispatched rows have completed."""

	if not allocated:
		return
	from lftomo.pipeline.stages.stripe import (
		VO_FLOAT_WORK_ARTIFACT,
		VO_INDEX_WORK_ARTIFACT,
		VO_MEDIAN_PAD_WORK_ARTIFACT,
	)

	state.artifacts.pop(VO_FLOAT_WORK_ARTIFACT, None)
	state.artifacts.pop(VO_INDEX_WORK_ARTIFACT, None)
	state.artifacts.pop(VO_MEDIAN_PAD_WORK_ARTIFACT, None)
	if state.storage is not None:
		for buffer, role in allocated:
			state.storage.retire(buffer, role=role)


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
) -> None:
	"""Transform one projection into a manager-owned fused output."""

	_limit_worker_cpu_threads(stage)
	from lftomo.pipeline.stages.geometry import transform_affine_projection

	source_values = source.array()
	target_values = target.array()
	row_count = payload.buffer_spec.shape.Y
	image = source_values[type(source.shape).point_index(Theta=0)]
	transform_affine_projection(
		stage, image, payload.source_window, 0, payload.dataset.physical_shape, row_count,
		target=target_values[type(target.shape).point_index(Theta=0)],
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


def _payload(state: PipelineState, source_window: DataRef) -> _WorkerState:
	scan = state.scan
	if isinstance(state.artifacts.get("flat_fields"), ManagedBuffer):
		metadata = dict(scan.metadata)
		metadata.pop("flat_fields", None)
		scan = scan.model_copy(update={"metadata": metadata})
	return _WorkerState(
		scan=scan,
		recon=state.recon,
		artifacts=dict(state.artifacts),
		backend=state.backend,
		dataset=state.dataset,
		source_window=source_window,
		buffer_spec=state.buffer_spec,
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
				output, buffer_spec=BufferSpec(state.buffer_spec.shape, np.dtype(output.dtype)),
			)
			air_mask_key = getattr(stage.params, "air_mask_artifact", "air_mask")
			air_mask = working_state.artifacts.get(air_mask_key)
			if (
				isinstance(air_mask, np.ndarray)
				and air_mask.ndim == 3
				and working_state.storage is not None
			):
				managed_mask = working_state.storage.allocate(
					working_state.buffer_spec.shape, air_mask.dtype, role="air_mask",
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
		results = self.submit_all(run_projection_kernel, (
			(
				_unit_handle(working_state.working_array, unit, local_index), stage,
				_payload(working_state, unit), local_index,
			)
			for unit, local_index in _buffer_units(working_state, "Theta")
		))
		if stage.name.startswith("normalize."):
			from lftomo.pipeline.stages.normalize import merge_normalization_dispatch

			merge_normalization_dispatch(stage, working_state, results)
			result = commit_output(
				original, output, retire_as="raw_staging",
				buffer_spec=BufferSpec(original.buffer_spec.shape, np.dtype(output.dtype)),
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
			shape = state.buffer_spec.shape.to(stage.contract.produces)
			output, _target = allocate_output(
				state, shape, data.dtype, role=f"layout:{stage.contract.produces.__name__}",
			)
			positions = Theta.drange(0, shape.Theta, 1).get_iter(IntIter)
			self.submit_all(run_layout_fill_kernel, (
				(
					state.working_array[type(state.buffer_spec.shape).point_index(Theta=index)],
					output[type(shape).point_index(Theta=index)],
				)
				for index in positions
			))
			return commit_output(
				state, output, retire_as=None, buffer_spec=BufferSpec(shape, data.dtype),
			)

		if isinstance(stage, AffineSkewMovementStage):
			if stage.target_layout is None:
				layout = state.layout
				if layout not in (ProjOrder, SinoOrder):
					return stage.run(state)
				self.submit_all(run_projection_kernel, (
					(
						_unit_handle(state.working_array, unit, local_index), stage,
						_payload(state, unit), local_index,
					)
					for unit, local_index in _buffer_units(state, "Theta")
				))
				return state
			input_layout = state.layout
			if input_layout not in (ProjOrder, SinoOrder):
				raise ValueError("fused affine transition requires a known tomography layout")
			if input_layout is stage.target_layout:
				raise ValueError("fused affine transition must change layout")
			shape = stage.output_shape(state.buffer_spec.shape)
			output, _target = allocate_output(
				state, shape, data.dtype, role="fused_geometry",
			)
			self.submit_all(run_affine_fill_kernel, (
				(
					_unit_handle(state.working_array, unit, local_index),
					_unit_handle(output, unit, local_index), stage, _payload(state, unit),
				)
				for unit, local_index in _buffer_units(state, "Theta")
			))
			result = commit_output(
				state, output, retire_as=None, buffer_spec=BufferSpec(shape, data.dtype),
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
			source_shape = state.buffer_spec.shape
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
				state, output, retire_as=None, buffer_spec=BufferSpec(shape, data.dtype),
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
	shape = state.buffer_spec.shape
	row_count = shape.Y
	sinogram_shape = (shape.Theta, shape.X)
	allocated = _prepare_vo_work(state, stages, row_count, sinogram_shape)
	member_samples: list[list[float]] = [[] for _stage in stages]
	try:
		if isinstance(state.working_array, ManagedBuffer) and executor is not None:
			if checkpoint is not None:
				checkpoint()
			try:
				rows = executor.submit_all(run_sinogram_chain_kernel, (
					(
						_unit_handle(state.working_array, unit, local_index), tuple(stages),
						_payload(state, unit), local_index, timing_observer is not None,
					)
					for unit, local_index in _buffer_units(state, "Y")
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
			for working, row_state in _unit_states(state, "Y"):
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
		_release_vo_work(state, allocated)
	if timing_observer is not None:
		timing_observer(tuple(tuple(samples) for samples in member_samples))
	return state
