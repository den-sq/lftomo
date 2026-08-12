"""Run-scoped dispatch for shared-memory pipeline kernels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import multiprocessing
import os
import signal
import threading
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from lftomo.pipeline.core import (
	PipelineState,
	Stage,
	allocate_output,
	commit_output,
	working_view,
)
from lftomo.pipeline.dispatch import DispatchKind
from lftomo.pipeline.layout import n_rows, projection_view, sinogram_scratch, sinogram_view
from lftomo.storage import ManagedBuffer
from lftomo.util.shapes import ProjOrder, SinoOrder


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
	layout: type[Any] | None
	domain: Any


def _worker_initializer() -> None:
	"""Give each persistent worker a killable process group."""

	if hasattr(os, "setsid"):
		try:
			os.setsid()
		except OSError:
			pass
	signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_state(buffer: ManagedBuffer, payload: _WorkerState) -> PipelineState:
	return PipelineState(
		working_array=buffer,
		scan=payload.scan,
		recon=payload.recon,
		artifacts=dict(payload.artifacts),
		backend=payload.backend,
		storage=None,
		dataset=payload.dataset,
		source_window=payload.source_window,
		layout=payload.layout,
		domain=payload.domain,
	)


def _projection_handle(buffer: ManagedBuffer, layout: type[Any], index: int) -> ManagedBuffer:
	if layout is ProjOrder:
		return buffer[index:index + 1, :, :]
	if layout is SinoOrder:
		return buffer[:, index:index + 1, :]
	raise ValueError("projection dispatch requires ProjOrder or SinoOrder")


def run_projection_kernel(
	buffer: ManagedBuffer,
	layout: type[Any],
	index: int,
	stage: Stage,
	payload: _WorkerState,
	projection_count: int,
) -> dict[str, Any]:
	"""Module-level adapter that mutates one shared projection view."""

	handle = _projection_handle(buffer, layout, index)
	state = _worker_state(handle, payload)
	base = int(payload.artifacts.get("projection_start", 0))
	total = int(payload.artifacts.get("projection_count", projection_count))
	state.artifacts["projection_start"] = base + index
	state.artifacts["projection_count"] = total
	projection_times = state.artifacts.get("projection_times")
	if isinstance(projection_times, ManagedBuffer):
		projection_times = projection_times.array()
	if projection_times is not None and len(projection_times) == projection_count:
		state.artifacts["projection_times"] = np.asarray(
			projection_times[index:index + 1],
		)
	air_mask_key = getattr(stage, "air_mask_artifact", "air_mask")
	air_mask = state.artifacts.get(air_mask_key)
	if isinstance(air_mask, ManagedBuffer) and air_mask.array().ndim == 3:
		state.artifacts[air_mask_key] = _projection_handle(air_mask, layout, index)
	before = set(state.artifacts)
	result = stage.run(state)
	return {
		key: value for key, value in result.artifacts.items()
		if key not in before and not isinstance(value, ManagedBuffer)
	}


def run_sinogram_chain_kernel(
	buffer: ManagedBuffer,
	layout: type[Any],
	index: int,
	stages: tuple[Stage, ...],
	payload: _WorkerState,
) -> None:
	"""Gather/process/scatter one row for a complete bounded Stage chain."""

	data = working_view(_worker_state(buffer, payload))
	if layout is ProjOrder:
		working = sinogram_scratch(data, layout, index)
	else:
		working = sinogram_view(data, layout, index)
	state = _worker_state(buffer, payload)
	from lftomo.pipeline.stages.stripe import SINOGRAM_ROW_ARTIFACT

	state.artifacts[SINOGRAM_ROW_ARTIFACT] = index
	for stage in stages:
		result = stage.run_sinogram(working, state)
		if result is not None and result is not working:
			working = np.asarray(result)
	if layout is ProjOrder:
		sinogram_view(data, layout, index)[:] = working


def _prepare_vo_work(
	state: PipelineState,
	stages: Sequence[Stage],
	row_count: int,
	sinogram_shape: tuple[int, int],
) -> tuple[tuple[ManagedBuffer, str], ...]:
	"""Allocate the two row-indexed planes shared by a fused Vo chain."""

	if not any(getattr(stage, "uses_vo_work", False) for stage in stages):
		return ()
	if not isinstance(state.working_array, ManagedBuffer) or state.storage is None:
		return ()
	from lftomo.pipeline.stages.stripe import (
		VO_FLOAT_WORK_ARTIFACT,
		VO_INDEX_WORK_ARTIFACT,
	)

	if VO_FLOAT_WORK_ARTIFACT in state.artifacts or VO_INDEX_WORK_ARTIFACT in state.artifacts:
		raise ValueError("Vo work artifacts are private to one bounded segment")
	theta_count, column_count = sinogram_shape
	if max(theta_count, column_count) > np.iinfo(np.uint32).max:
		raise ValueError("Vo sorting indices exceed uint32 capacity")
	shape = (row_count, theta_count, column_count)
	float_dtype = np.dtype(state.working_array.dtype)
	if not np.issubdtype(float_dtype, np.floating):
		raise TypeError("Vo reusable work requires floating-point pipeline data")
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
	except BaseException:
		for buffer, role in allocated:
			state.storage.retire(buffer, role=role)
		raise
	state.artifacts[VO_FLOAT_WORK_ARTIFACT] = float_work
	state.artifacts[VO_INDEX_WORK_ARTIFACT] = index_work
	float_bytes = int(np.prod(shape, dtype=np.int64) * float_dtype.itemsize)
	index_bytes = int(np.prod(shape, dtype=np.int64) * np.dtype(np.uint32).itemsize)
	report = {
		"scope": "window" if state.source_window is not None else "whole_volume",
		"rows": row_count,
		"sinogram_shape": [theta_count, column_count],
		"float_dtype": float_dtype.str,
		"index_dtype": np.dtype(np.uint32).str,
		"float_bytes": float_bytes,
		"index_bytes": index_bytes,
		"managed_bytes": float_bytes + index_bytes,
		"allocation_count": 2,
		"indexing": "source_window_row",
	}
	state.artifacts.setdefault("vo_scratch_allocations", []).append(report)
	return tuple(allocated)


def _release_vo_work(
	state: PipelineState, allocated: tuple[tuple[ManagedBuffer, str], ...],
) -> None:
	"""Retire temporary Vo planes after all dispatched rows have completed."""

	from lftomo.pipeline.stages.stripe import SINOGRAM_ROW_ARTIFACT

	state.artifacts.pop(SINOGRAM_ROW_ARTIFACT, None)
	if not allocated:
		return
	from lftomo.pipeline.stages.stripe import (
		VO_FLOAT_WORK_ARTIFACT,
		VO_INDEX_WORK_ARTIFACT,
	)

	state.artifacts.pop(VO_FLOAT_WORK_ARTIFACT, None)
	state.artifacts.pop(VO_INDEX_WORK_ARTIFACT, None)
	if state.storage is not None:
		for buffer, role in allocated:
			state.storage.retire(buffer, role=role)


def write_tiff_slice_kernel(
	buffer: ManagedBuffer, index: int, path: Any, dtype: np.dtype,
) -> None:
	"""Write one slice from a reopenable handle; no ndarray crosses IPC."""

	from lftomo.pipeline.stages.write import _write_array_slice

	# Keep the sliced handle alive for the full write.  Its ``array`` view does
	# not own the reopened shared-memory mapping and becomes invalid when the
	# temporary handle is finalized.
	slice_buffer = buffer[index]
	_write_array_slice(slice_buffer.array(), path, dtype)


def run_layout_fill_kernel(
	source: ManagedBuffer,
	target: ManagedBuffer,
	source_layout: type[Any],
	target_layout: type[Any],
	projection: int,
) -> None:
	"""Copy one projection between manager-owned layout buffers."""

	source_values = source.array()
	target_values = target.array()
	projection_view(target_values, target_layout, projection)[:] = projection_view(
		source_values, source_layout, projection,
	)


def run_affine_fill_kernel(
	source: ManagedBuffer,
	target: ManagedBuffer,
	source_layout: type[Any],
	target_layout: type[Any],
	projection: int,
	projection_count: int,
	row_count: int,
	stage: Stage,
) -> None:
	"""Transform one projection into a manager-owned fused output."""

	from lftomo.pipeline.stages.geometry import transform_affine_projection

	source_values = source.array()
	target_values = target.array()
	image = projection_view(source_values, source_layout, projection)
	projection_view(target_values, target_layout, projection)[:] = (
		transform_affine_projection(
			stage, image, projection, projection_count, row_count,
		)
	)


def run_helical_fill_kernel(
	source: ManagedBuffer,
	target: ManagedBuffer,
	layout: type[Any],
	projection: int,
	shift_px: int,
	row_start: int,
	row_stop: int,
	crop_valid: bool,
) -> None:
	"""Fill one projection of a manager-owned helical-remap output."""

	source_values = source.array()
	target_values = target.array()
	source_image = projection_view(source_values, layout, projection)
	target_image = projection_view(target_values, layout, projection)
	if crop_valid:
		target_image[:] = source_image[row_start + shift_px:row_stop + shift_px]
		return
	rows = np.arange(row_start, row_stop)
	source_rows = rows + shift_px
	valid = (source_rows >= 0) & (source_rows < source_image.shape[0])
	target_image[valid] = source_image[source_rows[valid]]


def _payload(state: PipelineState) -> _WorkerState:
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
		source_window=state.source_window,
		layout=state.layout,
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
				self._pool = ProcessPoolExecutor(
					max_workers=self.max_workers,
					mp_context=multiprocessing.get_context(self.context_name),
					initializer=_worker_initializer,
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
			working_state = state.with_array(output)
			air_mask_key = getattr(stage, "air_mask_artifact", "air_mask")
			air_mask = working_state.artifacts.get(air_mask_key)
			if (
				isinstance(air_mask, np.ndarray)
				and air_mask.ndim == 3
				and working_state.storage is not None
			):
				managed_mask = working_state.storage.allocate(
					air_mask.shape, air_mask.dtype, role="air_mask",
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
		data = working_view(working_state)
		layout = working_state.layout
		if layout not in (ProjOrder, SinoOrder):
			return stage.run(state)
		count = data.shape[0] if layout is ProjOrder else data.shape[1]
		results = self.submit_all(run_projection_kernel, (
			(working_state.working_array, layout, index, stage, _payload(working_state), count)
			for index in range(count)
		))
		if stage.name.startswith("normalize."):
			from lftomo.pipeline.stages.normalize import merge_normalization_dispatch

			merge_normalization_dispatch(stage, working_state, results)
			result = commit_output(original, output, retire_as="raw_staging")
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

		data = working_view(state)
		if stage.name == "layout.convert":
			if state.layout not in (ProjOrder, SinoOrder):
				raise TypeError("layout.convert requires ProjOrder or SinoOrder input")
			if state.layout is stage.target:
				raise ValueError("layout.convert target is already the current layout")
			shape = stage.target(data.shape[1], data.shape[0], data.shape[2])
			output, _target = allocate_output(
				state, shape, data.dtype, role=f"layout:{stage.target.__name__}",
			)
			count = data.shape[0] if state.layout is ProjOrder else data.shape[1]
			self.submit_all(run_layout_fill_kernel, (
				(state.working_array, output, state.layout, stage.target, index)
				for index in range(count)
			))
			return commit_output(state, output, retire_as=None, layout=stage.target)

		if isinstance(stage, AffineSkewMovementStage):
			if stage.target_layout is None:
				layout = state.layout
				if layout not in (ProjOrder, SinoOrder):
					return stage.run(state)
				count = data.shape[0] if layout is ProjOrder else data.shape[1]
				self.submit_all(run_projection_kernel, (
					(state.working_array, layout, index, stage, _payload(state), count)
					for index in range(count)
				))
				return state
			input_layout = state.layout
			if input_layout not in (ProjOrder, SinoOrder):
				raise ValueError("fused affine transition requires a known tomography layout")
			if input_layout is stage.target_layout:
				raise ValueError("fused affine transition must change layout")
			projection_count = data.shape[0] if input_layout is ProjOrder else data.shape[1]
			row_count = data.shape[1] if input_layout is ProjOrder else data.shape[0]
			width = data.shape[2] + 2 * stage.pad_x
			shape = (
				(projection_count, row_count, width)
				if stage.target_layout is ProjOrder else
				(row_count, projection_count, width)
			)
			output, _target = allocate_output(
				state, stage.target_layout(*shape), data.dtype, role="fused_geometry",
			)
			self.submit_all(run_affine_fill_kernel, (
				(
					state.working_array, output, input_layout, stage.target_layout,
					index, projection_count, row_count, stage,
				)
				for index in range(projection_count)
			))
			result = commit_output(
				state, output, retire_as=None, layout=stage.target_layout,
			)
			result.artifacts["fused_layout_transition"] = {
				"from": input_layout.__name__, "to": stage.target_layout.__name__,
				"pad_x": stage.pad_x, "skew_degrees": stage.skew_degrees,
			}
			return result

		if isinstance(stage, HelicalIndexRemapStage):
			if not stage.enabled:
				return state
			from lftomo.pipeline.layout import resolve_layout

			layout = resolve_layout(state, stage.order, stage.name)
			count = data.shape[0] if layout is ProjOrder else data.shape[1]
			y_size = data.shape[1] if layout is ProjOrder else data.shape[0]
			direction = -1 if stage.going_up else 1
			indices = stage.projection_start + np.arange(count)
			shifts = np.rint(direction * stage._pitch(state) * indices).astype(int)
			if stage.crop_valid:
				start = max(0, -int(shifts.min()))
				stop = min(y_size, y_size - int(shifts.max()))
				if start >= stop:
					raise ValueError("helical pitch leaves no detector rows valid across all projections")
				row_count = stop - start
				shape = (
					(count, row_count, data.shape[2]) if layout is ProjOrder else
					(row_count, count, data.shape[2])
				)
			else:
				start, stop = 0, y_size
				shape = data.shape
			output, target = allocate_output(
				state, layout(*shape), data.dtype, role="helical_remap",
			)
			if not stage.crop_valid:
				target.fill(stage.fill_value)
			self.submit_all(run_helical_fill_kernel, (
				(
					state.working_array, output, layout, index, int(shifts[index]),
					start, stop, stage.crop_valid,
				)
				for index in range(count)
			))
			result = commit_output(state, output, retire_as=None)
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
		old_ncore = getattr(stage, "ncore", None)
		old_kwargs_ncore = None
		kwargs = getattr(stage, "kwargs", None)
		if hasattr(stage, "ncore"):
			stage.ncore = min(self.max_workers, old_ncore or budget)
		elif isinstance(kwargs, dict) and "ncore" in kwargs:
			old_kwargs_ncore = kwargs["ncore"]
			kwargs["ncore"] = min(self.max_workers, old_kwargs_ncore or budget)
		try:
			return stage.run(state)
		finally:
			if hasattr(stage, "ncore"):
				stage.ncore = old_ncore
			elif isinstance(kwargs, dict) and "ncore" in kwargs:
				kwargs["ncore"] = old_kwargs_ncore


def run_bounded_sinogram_segment(
	state: PipelineState,
	stages: Sequence[Stage],
	*,
	max_workers: int = 1,
	failure_observer: Callable[[Stage, BaseException], None] | None = None,
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
	data = working_view(state)
	if data.ndim != 3:
		raise ValueError("bounded sinogram execution expects a three-dimensional array")
	for stage in stages:
		if not callable(getattr(stage, "run_sinogram", None)):
			raise TypeError(f"{stage.name} declares bounded views but has no run_sinogram kernel")
	row_count = n_rows(data, state.layout)
	sinogram_shape = (
		(data.shape[0], data.shape[2])
		if state.layout is ProjOrder else (data.shape[1], data.shape[2])
	)
	allocated = _prepare_vo_work(state, stages, row_count, sinogram_shape)
	try:
		if isinstance(state.working_array, ManagedBuffer) and executor is not None:
			if checkpoint is not None:
				checkpoint()
			try:
				executor.submit_all(run_sinogram_chain_kernel, (
					(state.working_array, state.layout, index, tuple(stages), _payload(state))
					for index in range(row_count)
				))
			except BaseException as error:
				if failure_observer is not None:
					failure_observer(stages[0], error)
				raise
			return state

		# ndarray-only hermetic execution is deliberately serial.
		from lftomo.pipeline.stages.stripe import SINOGRAM_ROW_ARTIFACT

		for index in range(row_count):
			state.artifacts[SINOGRAM_ROW_ARTIFACT] = index
			working = sinogram_scratch(data, state.layout, index) if state.layout is ProjOrder else sinogram_view(data, state.layout, index)
			for stage in stages:
				if checkpoint is not None:
					checkpoint()
				result = stage.run_sinogram(working, state)
				if result is not None and result is not working:
					working = np.asarray(result)
			if state.layout is ProjOrder:
				sinogram_view(data, state.layout, index)[:] = working
		state.artifacts.pop(SINOGRAM_ROW_ARTIFACT, None)
	finally:
		_release_vo_work(state, allocated)
	return state
