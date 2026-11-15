"""Serializable records describing validated pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from lftomo.pipeline.core import BufferSpec, Stage, collect_bounded_work
from lftomo.pipeline.dispatch import dispatch_for_stage
from lftomo.pipeline.domains import DataDomain
from lftomo.util.shapes import ImgSlice, ProjOrder, ReconOrder, SinoOrder
from lftomo.util.window import TomoDataSet, TomoWindow


class MaterializationKind(Enum):
	"""Kinds of full-volume materialization visible in a plan."""

	LAYOUT_CONVERSION = auto()
	TOMOPY_BOUNDARY = auto()


class ExecutionKind(Enum):
	"""Execution strategy selected for a contiguous Stage segment."""

	SEQUENTIAL = auto()
	BOUNDED_SINOGRAM = auto()


class BufferAction(Enum):
	"""Lifecycle actions in the run-owned shared allocation schedule."""

	ALLOCATE = auto()
	ALIAS = auto()
	RETIRE = auto()
	PIN = auto()


@dataclass(frozen=True)
class Materialization:
	"""One explicit or library-boundary full-volume copy."""

	kind: MaterializationKind
	reason: str
	hidden: bool = False


@dataclass(frozen=True)
class PlanStep:
	"""Resolved axes and copy behavior for one Stage."""

	index: int
	stage_name: str
	input_layout: type[Any] | None
	output_layout: type[Any] | None
	input_domain: DataDomain | None
	output_domain: DataDomain | None
	materializations: tuple[Materialization, ...] = ()
	warnings: tuple[str, ...] = ()
	input_buffer: BufferSpec | None = None
	output_buffer: BufferSpec | None = None
	allocation_buffer: BufferSpec | None = None
	output_role: str | None = None
	retires_role: str | None = None


@dataclass(frozen=True)
class PlanSegment:
	"""Adjacent Stages sharing one execution strategy."""

	start: int
	stop: int
	kind: ExecutionKind
	stage_names: tuple[str, ...]


@dataclass(frozen=True)
class BufferScheduleEvent:
	"""One logical transition and its selected physical allocation."""

	stage_index: int
	action: BufferAction
	role: str
	bytes: int
	physical_role: str
	proof: str | None = None
	live_roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelinePlan:
	"""Validated ordered Stage plan with concrete execution segments."""

	steps: tuple[PlanStep, ...]
	segments: tuple[PlanSegment, ...] = ()
	warnings: tuple[str, ...] = ()
	allocation_schedule: tuple[BufferScheduleEvent, ...] = ()

	@property
	def full_volume_materializations(self) -> tuple[Materialization, ...]:
		return tuple(item for step in self.steps for item in step.materializations)

	@property
	def shared_memory_peak_bytes(self) -> int:
		"""Physical bytes in the schedule, retained until run-context shutdown."""

		return sum(
			event.bytes for event in self.allocation_schedule
			if event.action is BufferAction.ALLOCATE
		)


def _nbytes(spec: BufferSpec | None) -> int:
	if spec is None:
		return 0
	return spec.shape.nbytes(spec.dtype)


def _read_spec(
	stage: Stage,
	scan: Any,
	dataset: TomoDataSet | None = None,
	source_window: TomoWindow | None = None,
) -> BufferSpec | None:
	if (dataset is None) != (source_window is None):
		raise ValueError("read planning requires dataset and source_window together")
	if source_window is None:
		physical_shape = scan.projection_shape()
	dtype = None if dataset is None else dataset.dtype
	dtype = scan.metadata.get("raw_dtype") if dtype is None else dtype
	if dtype is None:
		source = scan.projections
		path = source.path
		if getattr(source, "format", None) == "hdf5_dxchange" and path.exists():
			import h5py

			with h5py.File(path, "r") as handle:
				if source.data_path in handle:
					dtype = handle[source.data_path].dtype
		elif getattr(source, "format", None) == "tiff_stack" and path.exists():
			from natsort import natsorted
			import tifffile as tf

			files = natsorted(path.glob(source.pattern)) if path.is_dir() else [path]
			if files:
				with tf.TiffFile(files[0]) as handle:
					dtype = handle.series[0].dtype
	dtype = np.dtype("uint16" if dtype is None else dtype)
	if source_window is None:
		from lftomo.pipeline.streaming import _read_source_window

		window = _read_source_window(stage, physical_shape, dtype=dtype)
	else:
		window = source_window.with_pad(["Y"])
	layout = stage.contract.output_layout(ProjOrder)
	shape = window.frange.length.to(layout)
	return BufferSpec(shape, dtype)


def stage_output_spec(
	stage: Stage,
	input_spec: BufferSpec | None,
	scan: Any,
	*,
	dataset: TomoDataSet | None = None,
	source_window: TomoWindow | None = None,
) -> BufferSpec | None:
	"""Resolve the concrete BufferSpec transition for one known Stage."""

	if stage.name.startswith("read."):
		return _read_spec(stage, scan, dataset, source_window)
	if input_spec is None:
		return None
	shape = input_spec.shape
	layout = stage.contract.output_layout(input_spec.layout)
	dtype = input_spec.dtype
	if stage.name.startswith("normalize.") or stage.name == "weighted-normalize":
		# Fragment validation has no run geometry; the canonical run path derives
		# this dtype from the source window rather than storing a parallel setting.
		dtype = np.dtype(
			np.float32 if source_window is None else source_window.internal.dtype
		)
	elif stage.name == "layout.convert":
		shape = shape.to(layout)
	elif stage.name == "geometry.affine_skew_movement" and getattr(stage, "target_layout", None) is not None:
		shape = stage.output_shape(shape)
	elif stage.name == "helical_correct.index_remap" and getattr(stage.params, "enabled", True):
		pitch = stage.params.pitch_px_per_projection
		if pitch is None and scan.helical.pitch_um_per_projection is not None:
			pitch = scan.helical.pitch_um_per_projection / scan.pixel_size_um
		if pitch is not None:
			if source_window is None:
				raise RuntimeError("helical planning requires the canonical source window")
			shape, _shifts, _start, _stop = stage.output_geometry(
				shape, pitch, source_window,
			)
	elif stage.name.startswith("recon."):
		geometry = getattr(stage, "output_geometry", None)
		shape = (
			ReconOrder.of(shape) if geometry is None
			else geometry(shape, source_window)[1]
		)
		dtype = np.dtype(np.float32)
	elif stage.name.startswith("super_resolve."):
		dtype = np.dtype(np.float32)
	return BufferSpec(shape, dtype)


def stage_allocation_spec(
	stage: Stage, input_spec: BufferSpec | None, output_spec: BufferSpec | None,
	*, source_window: TomoWindow | None = None,
) -> BufferSpec | None:
	"""Return the physical allocation spec when the published output is a view."""

	if input_spec is None or output_spec is None or not stage.name.startswith("recon."):
		return output_spec
	geometry = getattr(stage, "output_geometry", None)
	shape = (
		ReconOrder.of(input_spec.shape) if geometry is None
		else geometry(input_spec.shape, source_window)[0]
	)
	return BufferSpec(shape, np.dtype(np.float32))


def _transition_role(stage: Stage) -> tuple[str | None, str | None]:
	if stage.name.startswith("read."):
		return "raw_staging", None
	if stage.name.startswith("normalize.") or stage.name == "weighted-normalize":
		return "normalized", "raw_staging"
	if stage.name == "layout.convert":
		return "layout_converted", "current"
	if stage.name == "geometry.affine_skew_movement" and getattr(stage, "target_layout", None) is not None:
		return "fused_geometry", "current"
	if stage.name == "helical_correct.index_remap" and getattr(stage.params, "enabled", True):
		return "helical_remap", "current"
	if stage.name.startswith("recon."):
		return "reconstruction", "current"
	if stage.name.startswith("super_resolve."):
		return "super_resolution", "current"
	return None, None


def build_allocation_schedule(
	steps: tuple[PlanStep, ...], stages: tuple[Stage, ...] | list[Stage], scan: Any, *,
	enable_aliasing: bool = True, enable_engine_aliasing: bool = True,
	external_pins: tuple[str, ...] = (),
	segments: tuple[PlanSegment, ...] = (),
) -> tuple[BufferScheduleEvent, ...]:
	"""Build the named physical-buffer schedule required for execution."""

	events: list[BufferScheduleEvent] = []
	physical: dict[str, int] = {}
	live: dict[str, str] = {}
	retired: dict[str, list[str]] = {}
	current_role: str | None = None
	bounded_segments = {
		segment.start: segment for segment in segments
		if segment.kind is ExecutionKind.BOUNDED_SINOGRAM
	}
	bounded_steps = {
		index for segment in bounded_segments.values()
		for index in range(segment.start, segment.stop)
	}
	requests = ()
	work_stop = 0

	def publish(index: int, action: BufferAction, role: str, size: int,
			physical_role: str, proof: str | None = None):
		events.append(BufferScheduleEvent(
			stage_index=index, action=action, role=role, bytes=size,
			physical_role=physical_role, proof=proof,
			live_roles=tuple(sorted(live)),
		))

	for step, stage in zip(steps, stages):
		artifact_specs: list[tuple[str, int]] = []
		if "flat_fields" in stage.contract.produces_artifacts and step.output_buffer is not None:
			rows, width = step.output_buffer.shape.Y, step.output_buffer.shape.X
			artifact_specs.append(("flat_fields", 4 * rows * width * np.dtype(np.float32).itemsize))
		if "phase_filter" in stage.contract.produces_artifacts and step.output_buffer is not None:
			shape = step.output_buffer.shape
			resolve_shape = getattr(stage, "phase_filter_shape", None)
			filter_shape = resolve_shape(shape) if resolve_shape is not None else ImgSlice(shape.Y, shape.X)
			artifact_specs.append(("phase_filter", filter_shape.nbytes(np.float64)))
		if "phase_envelope" in stage.contract.managed_artifacts and step.output_buffer is not None:
			artifact_specs.append(("phase_envelope", step.output_buffer.shape.nbytes(np.float32)))
		for artifact_role, artifact_bytes in artifact_specs:
			physical_role = artifact_role
			suffix = 1
			while physical_role in physical:
				suffix += 1
				physical_role = f"{artifact_role}:{suffix}"
			physical[physical_role] = artifact_bytes
			live[artifact_role] = physical_role
			publish(step.index, BufferAction.ALLOCATE, artifact_role,
				artifact_bytes, physical_role, "producer_contract")
		# Bounded members share the collector's maximum capacities. Sequential
		# sinogram dispatch retains its numerical fallback without managed work.
		if (
			step.input_buffer is not None
			and (
				step.index in bounded_segments or (
					step.index not in bounded_steps
					and dispatch_for_stage(stage).kernel != "run_sinogram_chain_kernel"
				)
			)
		):
			segment = bounded_segments.get(step.index)
			work_stop = segment.stop if segment is not None else step.index + 1
			requests = collect_bounded_work(stages[step.index:work_stop], step.input_buffer)
			for request in requests:
				role = request.role
				size = _nbytes(request.buffer_spec)
				selected = next((
					candidate for candidate in retired.get(role, ())
					if candidate not in external_pins and physical[candidate] >= size
				), None) if enable_aliasing else None
				if selected is None:
					selected = role
					suffix = 1
					while selected in physical:
						suffix += 1
						selected = f"{role}:{suffix}"
					physical[selected] = size
					action = BufferAction.ALLOCATE
				else:
					action = BufferAction.ALIAS
				live[role] = selected
				publish(step.index, action, role, size, selected, "bounded_work_requests")
		# Scratch stays live until the complete bounded segment has finished.
		if step.index + 1 == work_stop:
			for request in requests:
				role = request.role
				selected = live.pop(role)
				retired.setdefault(role, []).append(selected)
				publish(step.index, BufferAction.RETIRE, role, physical[selected], selected)
		role = step.output_role
		if role is None or step.output_buffer is None:
			continue
		size = _nbytes(step.allocation_buffer or step.output_buffer)
		selected = None
		proof = None
		if enable_aliasing and role == "reconstruction":
			for candidate in retired.get("raw_staging", ()):
				if candidate not in external_pins and physical[candidate] >= size:
					selected = candidate
					proof = "internal_liveness"
					break
			if (
				selected is None and enable_engine_aliasing
				and current_role == "normalized" and step.input_layout is ProjOrder
				and current_role not in external_pins
				and physical[live[current_role]] >= size
			):
				selected = live[current_role]
				proof = "tomopy_input_materialization"
		if selected is None:
			selected = role
			suffix = 1
			while selected in physical:
				suffix += 1
				selected = f"{role}:{suffix}"
			physical[selected] = size
			action = BufferAction.ALLOCATE
		else:
			action = BufferAction.ALIAS
		live[role] = selected
		publish(step.index, action, role, size, selected, proof)
		if current_role is not None and step.retires_role is not None:
			old_physical = live.pop(current_role)
			retired.setdefault(current_role, []).append(old_physical)
			publish(step.index, BufferAction.RETIRE, current_role,
				physical[old_physical], old_physical)
		current_role = role

	return tuple(events)
