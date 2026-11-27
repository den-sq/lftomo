"""Stage and Pipeline ABCs for lftomo reconstruction workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
import time
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Sequence
import uuid

import numpy as np
from pydantic import BaseModel, ConfigDict

from lftomo.config import Recon, Scan
from lftomo.pipeline.domains import ContractEffect, DataDomain, PRESERVE
from lftomo.storage import LocalArray, ManagedBuffer, StorageBackend, TypedBuffer, buffer_view
from lftomo.util.shapes import DataRange, DataShape, IntIter, ProjOrder, SinoOrder, Y

if TYPE_CHECKING:
	from lftomo.pipeline.execution import PipelineExecutor
	from lftomo.pipeline.telemetry import ExecutionSpanObserver
	from lftomo.util.window import TomoDataSet, TomoWindow


@dataclass(frozen=True)
class ResourceHints:
	"""Cooperative resource hints declared by a Stage."""

	preferred_cpus: int | None = None
	requires_gpu: bool = False
	# Native CPU pool used inside a process-dispatched Stage. The executor
	# owns its thread ceiling; the Stage declares only which pool it uses.
	cpu_thread_pool: Literal["torch"] | None = None


@dataclass(frozen=True)
class RuntimeRequirement:
	"""Optional runtime capability required by a Stage implementation.

	The declaration lives with the Stage contract so experiment preflight can
	discover dependencies from the authoritative registry instead of maintaining
	a parallel algorithm-key table.
	"""

	capability: str
	package: str
	module: str
	probe_attributes: tuple[str, ...] = ()
	corrective_action: str = ""


def _canonical_pad(value: DataShape) -> ProjOrder:
	"""Normalize semantic tomography-axis padding without owning its geometry."""

	if not set(asdict(value)).issubset({"Theta", "Y", "X"}):
		raise ValueError("padding may use only Theta, Y, and X axes")
	if any(component < 0 for component in asdict(value).values()):
		raise ValueError("axis padding cannot be negative")
	return ProjOrder(Theta=0, Y=0, X=0) + value


@dataclass(frozen=True)
class PadRequirement:
	"""Stage source-context rule resolved to semantic before/after shapes.

	``fixed`` remains the symmetric input. Supplying ``fixed_after`` makes
	``fixed`` the before-side requirement and sets the after side explicitly.
	``full_source_axes`` requests the complete physical source extent along those axes;
	canonical windows clip the resulting padding at the source boundaries.
	"""

	fixed: DataShape = field(default_factory=lambda: ProjOrder(0, 0, 0))
	fixed_after: DataShape | None = None
	phase_fft_y: bool = False
	phase_fft_passes: int = 1
	full_source_axes: tuple[str, ...] = ()

	def __post_init__(self):
		_canonical_pad(self.fixed)
		if self.fixed_after is not None:
			_canonical_pad(self.fixed_after)
		if self.phase_fft_passes < 1:
			raise ValueError("phase FFT passes must be positive")
		if set(self.full_source_axes) - {"Theta", "Y", "X"}:
			raise ValueError("full source axes must use only Theta, Y, and X")

	def resolve(
		self, scan: Any, input_shape: Any, *, physical_shape: DataShape | None = None,
	) -> tuple[ProjOrder, ProjOrder]:
		before = _canonical_pad(self.fixed)
		after = before if self.fixed_after is None else _canonical_pad(self.fixed_after)
		if self.phase_fft_y:
			from lftomo.prep.phase import _phase_pad

			dimension = int(getattr(input_shape, "Y"))
			phase_y = int(_phase_pad(
				dimension,
				float(scan.pixel_size_um) * 1e-4,
				float(scan.ssd_cm),
				float(scan.energy_keV),
			))
			phase = Y(max(0, phase_y) * self.phase_fft_passes)
			before = before + phase
			after = after + phase
		if self.full_source_axes:
			if physical_shape is None:
				raise ValueError("full source context requires the physical source shape")
			full = replace(ProjOrder(0, 0, 0), **{
				axis: getattr(physical_shape, axis) for axis in self.full_source_axes
			})
			before = ProjOrder.param_max(before, full)
			after = ProjOrder.param_max(after, full)
		return before, after


@dataclass(frozen=True)
class StageContract:
	"""Behavioral contract advertised by a Stage before it runs.

	``accepts`` declares the input layout capabilities and ``produces`` declares
	the output layout or ``PRESERVE`` effect consumed by the Pipeline validator.
	Layout classes usually come from ``lftomo.util.shapes``, such as
	``ProjOrder``, ``SinoOrder``, or ``ReconOrder``. ``runtime_requirements``
	declares invariant optional dependencies and bounded post-import API probes
	for preflight.
	"""

	mutates_input: bool = False
	# Prefix caching is fail-closed: a Stage must explicitly promise that its
	# output is deterministic and fully represented by its config, contract,
	# backend, PipelineState inputs, and declared artifacts before reuse.
	cacheable_prefix: bool = False
	window_axes: tuple[str, ...] = ()
	required_window_axis: Literal["Theta", "Y", "X"] | None = None
	preferred_window_axis: Literal["Theta", "Y", "X"] | None = None
	required_metadata: tuple[str, ...] = ()
	consumes_artifacts: tuple[str, ...] = ()
	produces_artifacts: tuple[str, ...] = ()
	# Large arrays that an experiment prefix snapshot must retain as reopenable
	# process-shared buffers. This is producer-declared; small/QC artifacts stay
	# ordinary Python values regardless of size.
	managed_artifacts: tuple[str, ...] = ()
	forks_pipeline: bool = False
	resources: ResourceHints = field(default_factory=ResourceHints)
	accepts: tuple[type[Any], ...] | None = None
	prefers: type[Any] | None = None
	produces: type[Any] | ContractEffect | None = None
	bounded_views_from: tuple[type[Any], ...] = ()
	fused_transition: bool = False
	accepts_domain: tuple[DataDomain, ...] | None = None
	produces_domain: DataDomain | ContractEffect | None = None
	resamples_detector: bool = False
	requires_row_identity: bool = False
	required_pad: PadRequirement = PadRequirement()
	requires_full_axes: tuple[str, ...] = ()
	runtime_requirements: tuple[RuntimeRequirement, ...] = ()
	# Axes whose extent affects this algorithm's numerical result. An empty
	# tuple explicitly declares no sensitivity; None means undeclared so
	# registry tests can catch missing declarations. This describes the
	# algorithm, including when a configured instance is disabled.
	extent_sensitive_axes: tuple[str, ...] | None = None

	def __post_init__(self):
		for name in ("required_window_axis", "preferred_window_axis"):
			axis = getattr(self, name)
			if axis is None:
				continue
			if axis not in ("Theta", "Y", "X"):
				raise ValueError(f"{name} must be Theta, Y, or X")
			if axis not in self.window_axes:
				raise ValueError(f"{name}={axis} must be declared in window_axes")
			if axis in self.requires_full_axes:
				raise ValueError(f"{name}={axis} conflicts with requires_full_axes")
		if self.required_window_axis is not None and self.preferred_window_axis not in (
			None, self.required_window_axis,
		):
			raise ValueError("required_window_axis and preferred_window_axis disagree")
		undeclared = set(self.managed_artifacts) - set(self.produces_artifacts)
		if undeclared:
			raise ValueError(
				"managed_artifacts must also be declared in produces_artifacts: "
				+ ", ".join(sorted(undeclared))
			)

	@property
	def accepted_layouts(self) -> tuple[type[Any], ...] | None:
		"""Return the explicitly declared input layout capabilities."""

		return self.accepts

	@property
	def preferred_layout(self) -> type[Any] | None:
		"""Return the planner preference derived from accepted layouts."""

		if self.prefers is not None:
			return self.prefers
		accepted = self.accepted_layouts
		return None if not accepted else accepted[0]

	def output_layout(self, input_layout: type[Any] | None) -> type[Any] | None:
		"""Resolve the concrete output layout for ``input_layout``."""

		effect = self.produces
		if effect is PRESERVE:
			return input_layout
		return effect

	def output_domain(self, input_domain: DataDomain | None) -> DataDomain | None:
		"""Resolve the concrete output domain for ``input_domain``."""

		effect = PRESERVE if self.produces_domain is None else self.produces_domain
		if effect is PRESERVE:
			return input_domain
		return effect


@dataclass(frozen=True)
class Backend:
	"""Array-backend selector used by Pipeline and validation."""

	name: str
	supports_gpu: bool = False

	def wrap_array(self, array: Any) -> Any:
		"""Return ``array`` in this backend's preferred wrapper type."""

		return array


class NumpyBackend(Backend):
	"""Production Phase-1 backend for raw NumPy and SharedNPArray storage."""

	def __init__(self):
		super().__init__(name="numpy", supports_gpu=False)


@dataclass(frozen=True)
class DaskArrayStub:
	"""Tiny wrapper used when dask.array is unavailable in test environments."""

	array: np.ndarray

	@property
	def shape(self):
		return self.array.shape

	@property
	def dtype(self):
		return self.array.dtype

	def __array__(self, dtype=None, copy=None):
		if copy is False and dtype is not None and np.dtype(dtype) != self.dtype:
			raise ValueError("dtype conversion cannot be honored with copy=False")
		result = np.asarray(self.array, dtype=dtype)
		return result.copy() if copy else result

	def __getitem__(self, view):
		return type(self)(self.array[view])

	def compute(self) -> np.ndarray:
		"""Return the wrapped NumPy array, mirroring dask.array's method name."""

		return self.array


class DaskStubBackend(Backend):
	"""No-op Dask backend proving the ABC does not pin the array type.

	The backend uses ``dask.array.from_array`` when the lightweight dask package
	is installed. It never imports ``dask.distributed``. If dask is absent, it
	falls back to ``DaskArrayStub`` so core contract tests remain hermetic.
	"""

	def __init__(self):
		super().__init__(name="dask-stub", supports_gpu=False)

	def wrap_array(self, array: Any) -> Any:
		try:
			import dask.array as da
		except Exception:
			return DaskArrayStub(np.asarray(array))
		return da.from_array(array)


@dataclass(frozen=True)
class BufferSpec:
	"""Shape and dtype descriptor for planning and serialized boundaries."""

	shape: DataShape
	dtype: np.dtype

	def __post_init__(self):
		if not isinstance(self.shape, DataShape):
			raise TypeError("BufferSpec.shape must be a DataShape")
		object.__setattr__(self, "dtype", np.dtype(self.dtype))

	@property
	def layout(self) -> type[DataShape]:
		return type(self.shape)

	@classmethod
	def from_buffer(cls, buffer: TypedBuffer) -> "BufferSpec":
		"""Describe a live buffer at a planning or serialization boundary."""

		return cls(buffer.shape, buffer.dtype)


@dataclass(frozen=True)
class BoundedWorkRequest:
	"""One segment-owned workspace, shared by its artifact key.

	``buffer_spec`` is the minimum capacity. Only ``grow_axes`` may grow when
	other members request the same key; other extents, dtype and initialization
	must agree. A request describes temporary work, not a published artifact.
	``required_for_fallback`` retains algorithm-required work when execution
	has no managed input/storage pair; optional work uses the numerical fallback.
	"""

	artifact_key: str
	buffer_spec: BufferSpec
	role: str
	fill_value: float | int | None = None
	grow_axes: tuple[str, ...] = ()
	required_for_fallback: bool = False


def collect_bounded_work(
	stages: Sequence[Stage], input_spec: BufferSpec,
) -> tuple[BoundedWorkRequest, ...]:
	"""Resolve shared capacities from Stage-owned requirements without allocating."""

	requests: dict[str, BoundedWorkRequest] = {}
	for stage in stages:
		for request in stage.bounded_work_requests(input_spec):
			previous = requests.get(request.artifact_key)
			if previous is None:
				requests[request.artifact_key] = request
				continue
			before, after = previous.buffer_spec, request.buffer_spec
			if (
				before.layout is not after.layout or before.dtype != after.dtype
				or previous.role != request.role or previous.fill_value != request.fill_value
				or previous.grow_axes != request.grow_axes
				or previous.required_for_fallback != request.required_for_fallback
			):
				raise ValueError(f"incompatible bounded work requests for {request.artifact_key!r}")
			extents = {}
			for axis in before.shape.names:
				first, second = getattr(before.shape, axis), getattr(after.shape, axis)
				if axis not in request.grow_axes and first != second:
					raise ValueError(f"bounded work {request.artifact_key!r} cannot grow along {axis}")
				extents[axis] = max(first, second)
			requests[request.artifact_key] = replace(
				previous, buffer_spec=BufferSpec(before.layout(**extents), before.dtype),
			)
	return tuple(requests.values())


@dataclass
class PipelineState:
	"""Mutable state object passed from Stage to Stage."""

	working_array: TypedBuffer | np.ndarray
	scan: Scan
	recon: Recon
	artifacts: dict[str, Any] = field(default_factory=dict)
	backend: Backend = field(default_factory=NumpyBackend)
	storage: StorageBackend | None = None
	executor: PipelineExecutor | None = None
	dataset: TomoDataSet | None = None
	source_window: TomoWindow | None = None
	domain: DataDomain | None = None
	work_range: DataRange | None = None

	def __post_init__(self):
		if not isinstance(self.working_array, TypedBuffer):
			if np.asarray(self.working_array).size:
				raise TypeError("non-empty working arrays require a TypedBuffer; wrap borrowed arrays in LocalArray")
			return
		shape = self.working_array.shape
		if not isinstance(shape, DataShape):
			raise TypeError("TypedBuffer.shape must be a DataShape")
		if self.work_range is not None:
			if len(self.work_range.step.names) != 1:
				raise ValueError("work_range must assign exactly one buffer axis")
			axis = self.work_range.step.names[0]
			shape.axis_of(axis)
			start = int(getattr(self.work_range.start, axis))
			stop = int(getattr(self.work_range.stop, axis))
			step = int(getattr(self.work_range.step, axis))
			if not 0 <= start <= stop <= getattr(shape, axis) or step < 1:
				raise ValueError("work_range must stay within its buffer axis with a positive step")

	@property
	def layout(self) -> type[DataShape] | None:
		"""Return the stage-contract layout derived from the canonical shape."""

		return type(self.working_array.shape) if isinstance(self.working_array, TypedBuffer) else None

	def assigned_range(self, axis: str) -> DataRange:
		"""Return assigned packed positions, or the full unassigned buffer axis."""

		if not isinstance(self.working_array, TypedBuffer):
			raise ValueError("assigned work requires a bound TypedBuffer")
		self.working_array.shape.axis_of(axis)
		if self.work_range is not None and axis in self.work_range.step.names:
			return self.work_range
		axis_type = next(
			shape_type for shape_type in DataShape.__subclasses__()
			if shape_type.names == (axis,)
		)
		return axis_type.drange(0, getattr(self.working_array.shape, axis), 1)

	def work_indices(self, axis: str) -> Iterator[int]:
		"""Iterate the packed buffer positions assigned on a semantic axis."""

		return self.assigned_range(axis).get_iter(IntIter)

	def work_index(self, shape: type[DataShape] | DataShape | None = None) -> tuple[slice, ...]:
		"""Express the assignment for the working buffer or another layout."""

		if shape is None:
			if not isinstance(self.working_array, TypedBuffer):
				raise ValueError("assigned work requires a bound TypedBuffer")
			shape = self.working_array.shape
		if self.work_range is None:
			return DataShape._shape_type(shape).point_index()
		return self.work_range.slice_index(shape)

	def with_array(
		self,
		working_array: TypedBuffer,
		*,
		domain: DataDomain | None = None,
	) -> "PipelineState":
		"""Return a shallow-copy state with a newly bound typed buffer."""

		if not isinstance(working_array, TypedBuffer):
			raise TypeError("with_array requires a TypedBuffer")
		return PipelineState(
			working_array=working_array,
			scan=self.scan,
			recon=self.recon,
			artifacts=dict(self.artifacts),
			backend=self.backend,
			storage=self.storage,
			executor=self.executor,
			dataset=self.dataset,
			source_window=self.source_window,
			domain=self.domain if domain is None else domain,
			work_range=self.work_range,
		)


def allocate_output(
	state: PipelineState,
	shape: DataShape,
	dtype: np.dtype,
	*,
	role: str,
	reuse_from: tuple[str, ...] = (),
	alias_from_current: bool = False,
	proof: str | None = None,
) -> tuple[TypedBuffer, np.ndarray]:
	"""Allocate a prospective output without changing or retiring current state."""

	if state.storage is None:
		array = np.empty(shape.as_tuple(), dtype=dtype)
		return LocalArray(array, shape), array
	alias_from = state.working_array if alias_from_current and hasattr(state.working_array, "name") else None
	buffer = state.storage.allocate(
		shape, np.dtype(dtype), role=role, reuse_from=reuse_from,
		alias_from=alias_from, proof=proof,
	)
	return buffer, buffer_view(buffer)


def commit_output(
	state: PipelineState,
	output: TypedBuffer,
	*,
	retire_as: str | None,
	domain: DataDomain | None = None,
) -> PipelineState:
	"""Publish a filled output, then retire its predecessor after the handoff."""

	result = state.with_array(output, domain=domain)
	if (
		state.storage is not None
		and output is not state.working_array
		and isinstance(state.working_array, ManagedBuffer)
	):
		state.storage.retire(state.working_array, role=retire_as)
	return result


class StageParams(BaseModel):
	"""Validated parameters owned by a constructed Stage.

	Config dictionaries are input boundaries; runtime consumers use this model.
	Models stay immutable so a resolved contract cannot become stale.
	"""

	model_config = ConfigDict(extra="forbid", frozen=True)


class Stage(ABC):
	"""Base class for sequential reconstruction pipeline Stages."""

	name: str = "stage"
	contract: StageContract = StageContract()
	Params = StageParams

	def __init__(self, **params):
		self.params = self.Params.model_validate(params)
		self.contract = type(self).contract_for(self.params)

	@classmethod
	def contract_for(cls, params: StageParams) -> StageContract:
		"""Resolve parameter-dependent facts from the class declaration."""

		return cls.contract

	@abstractmethod
	def run(self, state: PipelineState) -> PipelineState:
		"""Execute the Stage and return the next PipelineState."""

	def output_window(self, source: TomoWindow) -> TomoWindow:
		"""Return the source-coordinate window published after this Stage."""

		return source

	def bounded_work_requests(self, input_spec: BufferSpec) -> tuple[BoundedWorkRequest, ...]:
		"""Declare reusable segment work from the current window's buffer geometry."""

		return ()


class SinogramStage(Stage):
	"""Stage with one kernel per detector row and canonical bounded views."""

	contract = StageContract(
		accepts=(SinoOrder,), prefers=SinoOrder, produces=PRESERVE,
		produces_domain=PRESERVE, bounded_views_from=(ProjOrder,),
		mutates_input=True, window_axes=("Y",),
	)

	def run(self, state: PipelineState) -> PipelineState:
		from lftomo.pipeline.execution import run_sinogram_stage

		return run_sinogram_stage(self, state)

	@abstractmethod
	def run_sinogram(self, sinogram: np.ndarray, state: PipelineState) -> Any:
		"""Mutate a row view or return replacement values for that row."""


class ProjectionStage(Stage):
	"""Stage with one kernel per projection in its assigned buffer range."""

	contract = StageContract(
		accepts=(ProjOrder, SinoOrder), prefers=ProjOrder, produces=PRESERVE,
	)

	def run(self, state: PipelineState) -> PipelineState:
		from lftomo.pipeline.execution import run_projection_stage

		self.prepare_projections(state)
		return self.finish_projections(run_projection_stage(self, state))

	def prepare_projections(self, state: PipelineState) -> None:
		"""Prepare shared inputs once before serial or process projection work."""

	def finish_projections(self, state: PipelineState) -> PipelineState:
		"""Validate or summarize the completed held window in the coordinator."""

		return state

	@abstractmethod
	def run_projection(self, image: np.ndarray, state: PipelineState) -> Any:
		"""Mutate a projection view or return replacement values for it."""


StageObserver = Callable[
	[
		Literal["started", "completed", "failed"],
		int,
		Stage,
		BaseException | None,
		"TomoWindow | None",
	],
	None,
]


def _allocation_cursor(storage: StorageBackend | None) -> int | None:
	events = getattr(storage, "allocation_events", None)
	return None if events is None else len(events)


def _allocation_changes(storage: StorageBackend | None, cursor: int | None):
	from lftomo.pipeline.telemetry import ManagedAllocationChange

	if cursor is None:
		return ()
	events = getattr(storage, "allocation_events", ())
	return tuple(ManagedAllocationChange(
		action=str(event["action"]),
		role=None if event.get("role") is None else str(event["role"]),
		name=str(event["name"]),
		bytes=int(event["bytes"]),
		alias=None if event.get("alias") is None else str(event["alias"]),
	) for event in events[cursor:])


class Pipeline:
	"""Sequential Pipeline over an ordered list of Stages."""

	def __init__(self, stages: list[Stage], backend: Backend | None = None,
			scan: Scan | None = None, validate: bool = True,
			validation_mode: str = "fragment",
			initial_layout: type[Any] | None = None,
			initial_domain: DataDomain | None = None,
			bounded_workers: int = 1, observer: StageObserver | None = None,
			execution_observer: ExecutionSpanObserver | None = None,
			stage_indices: tuple[int, ...] | None = None,
			checkpoint: Callable[[], None] | None = None):
		if bounded_workers < 1:
			raise ValueError("bounded_workers must be positive")
		self.stages = tuple(stages)
		self.stage_indices = (
			tuple(range(len(self.stages))) if stage_indices is None else tuple(stage_indices)
		)
		if len(self.stage_indices) != len(self.stages):
			raise ValueError("stage_indices must identify every authored Stage occurrence")
		if any(index < 0 for index in self.stage_indices):
			raise ValueError("authored Stage indices cannot be negative")
		if any(right <= left for left, right in zip(
			self.stage_indices, self.stage_indices[1:],
		)):
			raise ValueError("authored Stage indices must be strictly increasing")
		self.backend = backend or NumpyBackend()
		self.validate = validate
		self.validation_mode = validation_mode
		self.initial_layout = initial_layout
		self.initial_domain = initial_domain
		self.bounded_workers = bounded_workers
		self.observer = observer
		self.execution_observer = execution_observer
		self.checkpoint = checkpoint
		self.plan = None
		self._validated = False
		if validate and scan is not None:
			from lftomo.pipeline.validation import validate_pipeline

			self.plan = validate_pipeline(
				self.stages, scan, self.backend, mode=self.validation_mode,
				initial_layout=self.initial_layout,
				initial_domain=self.initial_domain,
			)
			self._validated = True

	def run(self, state: PipelineState) -> PipelineState:
		"""Run each Stage sequentially; individual Stages own concurrency."""

		state.backend = self.backend
		if self.validate and not self._validated:
			from lftomo.pipeline.validation import validate_pipeline

			self.plan = validate_pipeline(
				self.stages,
				state.scan,
				self.backend,
				available_artifacts=state.artifacts,
				mode=self.validation_mode,
				initial_layout=self.initial_layout,
				initial_domain=self.initial_domain,
				enable_aliasing=state.recon.buffer_aliasing,
				enable_engine_aliasing=state.recon.tomopy_input_aliasing,
				dataset=state.dataset,
				source_window=state.source_window,
			)
			self._validated = True
		segments = () if self.plan is None else self.plan.segments
		executor = state.executor or getattr(state.storage, "pipeline_executor", None)
		if not segments:
			from lftomo.pipeline.planning import ExecutionKind, PlanSegment

			segments = tuple(PlanSegment(
				index, index + 1, ExecutionKind.SEQUENTIAL, (stage.name,),
			) for index, stage in enumerate(self.stages))
		from lftomo.pipeline.planning import ExecutionKind
		if self.execution_observer is not None:
			from lftomo.pipeline.dispatch import dispatch_for_stage
			from lftomo.pipeline.telemetry import (
				ExecutionSpan,
				ExecutionSpanEvent,
				ExecutionSpanPhase,
				LogicalStageMember,
				MemberTiming,
				WorkerTimingAggregate,
			)
			run_id = uuid.uuid4().hex
		for ordinal, segment in enumerate(segments):
			segment_stages = self.stages[segment.start:segment.stop]
			if self.execution_observer is not None:
				members = tuple(LogicalStageMember(
					index=self.stage_indices[index],
					stage_name=stage.name,
					stage_type=f"{type(stage).__module__}.{type(stage).__qualname__}",
					dispatch_kind=dispatch_for_stage(stage).kind,
				) for index, stage in enumerate(segment_stages, start=segment.start))
				span = ExecutionSpan(
					run_id=run_id,
					span_id=f"{run_id}:{ordinal}",
					ordinal=ordinal,
					total=len(segments),
					kind=segment.kind,
					member_start=members[0].index,
					member_stop=members[-1].index + 1,
					members=members,
				)
				allocation_cursor = _allocation_cursor(state.storage)
				unavailable = []
				if allocation_cursor is None:
					unavailable.append("managed_allocation_ledger")
				unavailable.append("process_private_memory")
				self.execution_observer(ExecutionSpanEvent(
					phase=ExecutionSpanPhase.STARTED,
					span=span,
					source_window=state.source_window,
					unavailable_observations=tuple(unavailable),
				))
			if segment.kind is ExecutionKind.BOUNDED_SINOGRAM:
				from lftomo.pipeline.execution import run_bounded_sinogram_segment

				expected_layout = state.layout
				expected_domain = state.domain
				stage_indices = {
					id(stage): offset
					for offset, stage in enumerate(segment_stages, start=segment.start)
				}
				if self.observer is not None:
					for offset, stage in enumerate(segment_stages, start=segment.start):
						self.observer("started", offset, stage, None, state.source_window)
				member_samples: tuple[tuple[float, ...], ...] = ()

				def capture_timings(samples: tuple[tuple[float, ...], ...]) -> None:
					nonlocal member_samples
					member_samples = samples

				span_started = time.perf_counter()
				try:
					state = run_bounded_sinogram_segment(
						state,
						segment_stages,
						max_workers=self.bounded_workers,
						failure_observer=(
							None if self.observer is None else lambda stage, error: self.observer(
								"failed",
								stage_indices[id(stage)],
								stage,
								error,
								state.source_window,
							)
						),
						timing_observer=(
							capture_timings if self.execution_observer is not None else None
						),
						checkpoint=self.checkpoint,
						executor=executor,
					)
					wall_seconds = time.perf_counter() - span_started
				except BaseException as error:
					if self.execution_observer is not None:
						phase = (
							ExecutionSpanPhase.INTERRUPTED
							if isinstance(error, (KeyboardInterrupt, InterruptedError)) else
							ExecutionSpanPhase.FAILED
						)
						self.execution_observer(ExecutionSpanEvent(
							phase=phase,
							span=span,
							source_window=state.source_window,
							coordinator_wall_seconds=time.perf_counter() - span_started,
							managed_allocation_changes=_allocation_changes(
								state.storage, allocation_cursor,
							),
							unavailable_observations=tuple(unavailable),
							error_type=f"{type(error).__module__}.{type(error).__qualname__}",
							error_message=str(error),
						))
					raise
				if self.observer is not None:
					for offset, stage in enumerate(segment_stages, start=segment.start):
						self.observer("completed", offset, stage, None, state.source_window)
				for stage in segment_stages:
					expected_layout = stage.contract.output_layout(expected_layout)
					expected_domain = stage.contract.output_domain(expected_domain)
				if expected_layout is not None and state.layout is not expected_layout:
					raise ValueError(
						"Stage output buffer disagrees with its declared layout contract"
					)
				state.domain = expected_domain
				if self.execution_observer is not None:
					member_timings = tuple(MemberTiming(
						member=member,
						worker=(
							WorkerTimingAggregate.from_samples(samples)
							if samples else None
						),
						unavailable=(() if samples else ("bounded_worker_timing",)),
					) for member, samples in zip(members, member_samples))
					self.execution_observer(ExecutionSpanEvent(
						phase=ExecutionSpanPhase.COMPLETED,
						span=span,
						source_window=state.source_window,
						coordinator_wall_seconds=wall_seconds,
						member_timings=member_timings,
						managed_allocation_changes=_allocation_changes(
							state.storage, allocation_cursor,
						),
						unavailable_observations=tuple(unavailable),
					))
				continue
			for index, stage in enumerate(segment_stages, start=segment.start):
				input_layout = state.layout
				input_domain = state.domain
				if self.observer is not None:
					self.observer("started", index, stage, None, state.source_window)
				span_started = time.perf_counter()
				try:
					if self.checkpoint is not None:
						self.checkpoint()
					state = (
						executor.run_stage(state, stage)
						if executor is not None else stage.run(state)
					)
					wall_seconds = time.perf_counter() - span_started
				except BaseException as error:
					if self.observer is not None:
						self.observer("failed", index, stage, error, state.source_window)
					if self.execution_observer is not None:
						phase = (
							ExecutionSpanPhase.INTERRUPTED
							if isinstance(error, (KeyboardInterrupt, InterruptedError)) else
							ExecutionSpanPhase.FAILED
						)
						self.execution_observer(ExecutionSpanEvent(
							phase=phase,
							span=span,
							source_window=state.source_window,
							coordinator_wall_seconds=time.perf_counter() - span_started,
							managed_allocation_changes=_allocation_changes(
								state.storage, allocation_cursor,
							),
							unavailable_observations=tuple(unavailable),
							error_type=f"{type(error).__module__}.{type(error).__qualname__}",
							error_message=str(error),
						))
					raise
				if self.observer is not None:
					self.observer("completed", index, stage, None, state.source_window)
				output_layout = stage.contract.output_layout(input_layout)
				output_domain = stage.contract.output_domain(input_domain)
				if output_layout is not None and state.layout is not output_layout:
					raise ValueError(
						"Stage output buffer disagrees with its declared layout contract"
					)
				if output_domain is not None:
					state.domain = output_domain
				if self.execution_observer is not None:
					member = members[0]
					member_unavailable = (
						("external_engine_internal_timing",)
						if member.dispatch_kind.name == "ENGINE" else ()
					)
					self.execution_observer(ExecutionSpanEvent(
						phase=ExecutionSpanPhase.COMPLETED,
						span=span,
						source_window=state.source_window,
						coordinator_wall_seconds=wall_seconds,
						member_timings=(MemberTiming(
							member=member,
							coordinator_wall_seconds=wall_seconds,
							unavailable=member_unavailable,
						),),
						managed_allocation_changes=_allocation_changes(
							state.storage, allocation_cursor,
						),
						unavailable_observations=tuple(unavailable),
					))
		return state
