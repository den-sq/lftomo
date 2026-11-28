"""Serializable records describing validated pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from lftomo.pipeline.core import BufferSpec, Stage, collect_bounded_work
from lftomo.pipeline.dispatch import dispatch_for_stage
from lftomo.pipeline.domains import DataDomain
from lftomo.storage import select_buffer_alias
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
	stage_window: TomoWindow | None = None


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
	reuse_from: tuple[str, ...] = ()
	alias_from: str | None = None
	external_pins: tuple[str, ...] = ()


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


def reserve_streaming_buffers(
	plans: tuple[PipelinePlan, ...] | list[PipelinePlan], storage: Any = None,
	*, window_counts: tuple[int, ...] | None = None,
	prepass_stops: tuple[int | None, ...] | None = None,
) -> int:
	"""Reserve shared roles and replay their schedules across directional sweeps.

	The run reserves each role at its largest scheduled size before executing
	any segment. Replay the canonical allocation policies with those capacities
	and the manager's alias selector, retiring the published window between
	executions just as the runner does. Optional exclusive prefix stops describe
	center prepasses. Counts matter when reuse is disabled.
	"""

	if window_counts is None:
		window_counts = (1,) * len(plans)
	if len(window_counts) != len(plans) or any(count < 1 for count in window_counts):
		raise ValueError("window_counts must contain one positive count per plan")
	if prepass_stops is None:
		prepass_stops = (None,) * len(plans)
	if len(prepass_stops) != len(plans):
		raise ValueError("prepass_stops must contain one optional stop per plan")
	capacities: dict[str, int] = {}
	for plan in plans:
		for event in plan.allocation_schedule:
			capacities[event.role] = max(capacities.get(event.role, 0), event.bytes)
	reserve = getattr(storage, "reserve", None)
	if reserve is not None:
		for role, capacity in capacities.items():
			reserve(role, capacity)
	physical: dict[str, int] = {}
	retired: dict[str, list[str]] = {}
	live: dict[str, str] = {}

	def retire(role: str) -> None:
		selected = live.pop(role, None)
		if selected is not None and selected not in live.values():
			candidates = retired.setdefault(role, [])
			if selected not in candidates:
				candidates.append(selected)

	def replay(events: tuple[BufferScheduleEvent, ...]) -> None:
		for event in events:
			if event.action is BufferAction.RETIRE:
				retire(event.role)
				continue
			if event.action not in (BufferAction.ALLOCATE, BufferAction.ALIAS):
				continue
			selected, _proof = select_buffer_alias(
				event.bytes, retired, event.reuse_from,
				capacity=physical.__getitem__,
				pinned=lambda key: key in event.external_pins,
				alias_from=live.get(event.alias_from), proof=event.proof,
			)
			if selected is None:
				selected = event.role
				suffix = 1
				while selected in physical:
					suffix += 1
					selected = f"{event.role}:{suffix}"
				physical[selected] = max(event.bytes, capacities[event.role])
			else:
				for candidates in retired.values():
					while selected in candidates:
						candidates.remove(selected)
			# Producer artifacts replace their previous handle only after the
			# new allocation succeeds, so both must fit during the handoff.
			retire(event.role)
			live[event.role] = selected
		for role in tuple(live):
			retire(role)

	for plan, count, stop in zip(plans, window_counts, prepass_stops, strict=True):
		prepass = tuple(
			event for event in plan.allocation_schedule
			if stop is not None and event.stage_index < stop
		)
		for _ in range(count):
			if prepass:
				replay(prepass)
			replay(plan.allocation_schedule)
	return sum(physical.values())


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
	if dataset is not None and source_window is None:
		raise ValueError("read planning with a dataset requires its source_window")
	if source_window is None:
		physical_shape = scan.projection_shape()
	# A canonical window is sufficient for metadata-only experiment preflight;
	# creating the acquisition dataset remains execution-time source resolution.
	dtype = None if source_window is None else source_window.dtype
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
		window = source_window.with_pad()
	layout = stage.contract.output_layout(ProjOrder)
	shape = window.frange.length.to(layout)
	from lftomo.pipeline.stages.read import read_output_dtype

	return BufferSpec(shape, read_output_dtype(stage, window))


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
		if source_window is not None:
			shape = stage.output_window(source_window, scan).with_pad().frange.length.to(layout)
		elif stage.params.pitch_px_per_projection is not None or scan.helical.pitch_um_per_projection is not None:
			raise RuntimeError("helical planning requires the canonical source window")
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
	initial_buffer: BufferSpec | None = None,
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
			physical_role: str, proof: str | None = None, *,
			reuse_from: tuple[str, ...] = (), alias_from: str | None = None):
		events.append(BufferScheduleEvent(
			stage_index=index, action=action, role=role, bytes=size,
			physical_role=physical_role, proof=proof,
			live_roles=tuple(sorted(live)),
			reuse_from=reuse_from if enable_aliasing else (),
			alias_from=alias_from if enable_aliasing else None,
			external_pins=external_pins,
		))

	def prepare_work(index, work_requests):
		for request in work_requests:
			role = request.role
			size = _nbytes(request.buffer_spec)
			selected, _alias = select_buffer_alias(
				size, retired, (role,) if enable_aliasing else (),
				capacity=physical.__getitem__, pinned=lambda key: key in external_pins,
			)
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
			publish(index, action, role, size, selected, "bounded_work_requests", reuse_from=(role,))

	def retire_work(index, work_requests):
		for request in work_requests:
			role = request.role
			selected = live.pop(role)
			retired.setdefault(role, []).append(selected)
			publish(index, BufferAction.RETIRE, role, physical[selected], selected)

	if initial_buffer is not None:
		physical["raw_staging"] = _nbytes(initial_buffer)
		live["raw_staging"] = "raw_staging"
		current_role = "raw_staging"
		publish(
			-1, BufferAction.ALLOCATE, "raw_staging", _nbytes(initial_buffer), "raw_staging",
			reuse_from=("raw_staging", "normalized", "reconstruction"),
		)

	for step, stage in zip(steps, stages):
		artifact_specs: list[tuple[str, int]] = []
		if "flat_fields" in stage.contract.produces_artifacts and step.output_buffer is not None:
			shape = step.output_buffer.shape if step.stage_window is None else step.stage_window.frange.length
			rows, width = shape.Y, shape.X
			artifact_specs.append(("flat_fields", 4 * rows * width * np.dtype(np.float32).itemsize))
		if "phase_filter" in stage.contract.produces_artifacts and step.output_buffer is not None:
			shape = (step.output_buffer.shape if step.stage_window is None
				else step.stage_window.with_pad().frange.length)
			resolve_shape = getattr(stage, "phase_filter_shape", None)
			filter_shape = resolve_shape(shape) if resolve_shape is not None else ImgSlice(shape.Y, shape.X)
			artifact_specs.append(("phase_filter", filter_shape.nbytes(np.float64)))
		if "phase_envelope" in stage.contract.managed_artifacts and step.output_buffer is not None:
			shape = (step.output_buffer.shape if step.stage_window is None
				else step.stage_window.with_pad().frange.length)
			artifact_specs.append(("phase_envelope", shape.nbytes(np.float32)))
		for artifact_role, artifact_bytes in artifact_specs:
			physical_role = artifact_role
			suffix = 1
			while physical_role in physical:
				suffix += 1
				physical_role = f"{artifact_role}:{suffix}"
			physical[physical_role] = artifact_bytes
			live[artifact_role] = physical_role
			publish(step.index, BufferAction.ALLOCATE, artifact_role,
				artifact_bytes, physical_role, "producer_contract", reuse_from=(artifact_role,))
		# Bounded members share the collector's maximum capacities. Sequential
		# sinogram dispatch retains its numerical fallback without managed work.
		if (
			step.input_buffer is not None
			and not stage.name.startswith("read.")
			and (
				step.index in bounded_segments or (
					step.index not in bounded_steps
					and dispatch_for_stage(stage).kernel != "run_sinogram_chain_kernel"
				)
			)
		):
			segment = bounded_segments.get(step.index)
			work_stop = segment.stop if segment is not None else step.index + 1
			requests = collect_bounded_work(
				stages[step.index:work_stop], step.input_buffer,
				tuple(item.stage_window for item in steps[step.index:work_stop]),
			)
			prepare_work(step.index, requests)
		# Scratch stays live until the complete bounded segment has finished.
		if step.index + 1 == work_stop:
			retire_work(step.index, requests)
		role = step.output_role
		if role is None or step.output_buffer is None:
			continue
		size = _nbytes(step.allocation_buffer or step.output_buffer)
		selected = None
		proof = None
		alias_from = None
		if enable_aliasing and role == "reconstruction":
			if (
				enable_engine_aliasing
				and current_role == "normalized" and step.input_layout is ProjOrder
				and current_role not in external_pins
			):
				alias_from = current_role
			selected, alias = select_buffer_alias(
				size, retired, ("raw_staging",),
				capacity=physical.__getitem__, pinned=lambda key: key in external_pins,
				alias_from=live.get(alias_from), proof="tomopy_input_materialization",
			)
			if selected is not None:
				proof = "internal_liveness" if alias.startswith("retired:") else alias
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
		reuse_from = {
			"raw_staging": ("raw_staging", "reconstruction"),
			"normalized": ("normalized", "reconstruction"),
			"reconstruction": ("raw_staging", "reconstruction"),
		}.get(role, ())
		publish(step.index, action, role, size, selected, proof,
			reuse_from=reuse_from, alias_from=alias_from)
		if current_role is not None and step.retires_role is not None:
			old_physical = live.pop(current_role)
			retired.setdefault(current_role, []).append(old_physical)
			publish(step.index, BufferAction.RETIRE, current_role,
				physical[old_physical], old_physical)
		current_role = role
		if stage.name.startswith("read."):
			# Source Stages prepare their work after the output buffer exists.
			read_work = collect_bounded_work((stage,), step.output_buffer)
			prepare_work(step.index, read_work)
			retire_work(step.index, read_work)

	return tuple(events)
