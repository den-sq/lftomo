"""Stage and Pipeline ABCs for lftomo reconstruction workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
import time
from typing import TYPE_CHECKING, Any, Callable, Literal
import uuid

import numpy as np

from lftomo.config import Recon, Scan
from lftomo.pipeline.domains import ContractEffect, DataDomain, PRESERVE
from lftomo.storage import ManagedBuffer, StorageBackend, buffer_view
from lftomo.util.shapes import DataShape, ProjOrder, Y

if TYPE_CHECKING:
	from lftomo.pipeline.execution import PipelineExecutor
	from lftomo.pipeline.telemetry import ExecutionSpanObserver
	from lftomo.util.window import TomoDataSet, TomoWindow


@dataclass(frozen=True)
class ResourceHints:
	"""Cooperative resource hints declared by a Stage."""

	preferred_cpus: int | None = None
	requires_gpu: bool = False


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


class Locality(Enum):
	"""Smallest independent unit on which a Stage operates."""

	PROJECTION = auto()
	SINOGRAM = auto()
	VOLUME = auto()
	METADATA = auto()


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
	"""

	fixed: DataShape = field(default_factory=lambda: ProjOrder(0, 0, 0))
	fixed_after: DataShape | None = None
	phase_fft_y: bool = False

	def __post_init__(self):
		_canonical_pad(self.fixed)
		if self.fixed_after is not None:
			_canonical_pad(self.fixed_after)

	def resolve(self, scan: Any, input_shape: Any) -> tuple[ProjOrder, ProjOrder]:
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
			phase = Y(max(0, phase_y))
			before = before + phase
			after = after + phase
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
	locality: Locality | None = None
	bounded_views_from: tuple[type[Any], ...] = ()
	fused_transition: bool = False
	accepts_domain: tuple[DataDomain, ...] | None = None
	produces_domain: DataDomain | ContractEffect | None = None
	resamples_detector: bool = False
	requires_row_identity: bool = False
	required_pad: PadRequirement = PadRequirement()
	requires_full_axes: tuple[str, ...] = ()
	runtime_requirements: tuple[RuntimeRequirement, ...] = ()
	# Whether changing the detector-row FFT extent changes this algorithm's
	# numerical result. None means undeclared, so registry tests can catch a
	# phase Stage that forgot to choose True or False. This describes the
	# algorithm, including when a configured instance is disabled.
	fft_extent_sensitive: bool | None = None

	def __post_init__(self):
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
	"""Canonical logical shape and dtype of one working allocation."""

	shape: DataShape
	dtype: np.dtype

	def __post_init__(self):
		if not isinstance(self.shape, DataShape):
			raise TypeError("BufferSpec.shape must be a DataShape")
		object.__setattr__(self, "dtype", np.dtype(self.dtype))

	@property
	def layout(self) -> type[DataShape]:
		return type(self.shape)


def _buffer_spec(buffer: Any) -> BufferSpec | None:
	shape = getattr(buffer, "shape", None)
	if shape is None:
		return None
	if not isinstance(shape, DataShape):
		if isinstance(buffer, ManagedBuffer):
			raise TypeError("ManagedBuffer.shape must be a DataShape")
		return None
	try:
		if not all(int(value) > 0 for value in shape.as_tuple()):
			return None
	except TypeError:
		return None
	dtype = getattr(buffer, "dtype", None)
	if dtype is None and hasattr(buffer, "array"):
		view = buffer.array
		dtype = getattr(view() if callable(view) else view, "dtype", None)
	return None if dtype is None else BufferSpec(shape, np.dtype(dtype))


@dataclass
class PipelineState:
	"""Mutable state object passed from Stage to Stage."""

	working_array: Any
	scan: Scan
	recon: Recon
	artifacts: dict[str, Any] = field(default_factory=dict)
	backend: Backend = field(default_factory=NumpyBackend)
	storage: StorageBackend | None = None
	executor: PipelineExecutor | None = None
	dataset: TomoDataSet | None = None
	source_window: TomoWindow | None = None
	buffer_spec: BufferSpec | None = None
	domain: DataDomain | None = None

	def __post_init__(self):
		if self.buffer_spec is None:
			self.buffer_spec = _buffer_spec(self.working_array)
		if self.buffer_spec is None:
			if np.asarray(self.working_array).size:
				raise TypeError("non-empty plain arrays require an explicit BufferSpec")
			return
		actual = getattr(self.working_array, "shape", ())
		actual_shape = actual.as_tuple() if isinstance(actual, DataShape) else tuple(actual)
		if actual_shape != self.buffer_spec.shape.as_tuple():
			raise ValueError("working array shape disagrees with BufferSpec.shape")
		actual_dtype = getattr(self.working_array, "dtype", None)
		if actual_dtype is None and hasattr(self.working_array, "array"):
			view = self.working_array.array
			actual_dtype = getattr(view() if callable(view) else view, "dtype", None)
		if actual_dtype is None or np.dtype(actual_dtype) != self.buffer_spec.dtype:
			raise ValueError("working array dtype disagrees with BufferSpec.dtype")

	@property
	def layout(self) -> type[DataShape] | None:
		"""Return the stage-contract layout derived from the canonical shape."""

		return None if self.buffer_spec is None else self.buffer_spec.layout

	def with_array(
		self,
		working_array: Any,
		*,
		buffer_spec: BufferSpec | None = None,
		domain: DataDomain | None = None,
	) -> "PipelineState":
		"""Return a shallow-copy state with a newly bound array and BufferSpec."""

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
			buffer_spec=buffer_spec,
			domain=self.domain if domain is None else domain,
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
) -> tuple[Any, np.ndarray]:
	"""Allocate a prospective output without changing or retiring current state."""

	if state.storage is None:
		array = np.empty(shape.as_tuple(), dtype=dtype)
		return array, array
	alias_from = state.working_array if alias_from_current and hasattr(state.working_array, "name") else None
	buffer = state.storage.allocate(
		shape, np.dtype(dtype), role=role, reuse_from=reuse_from,
		alias_from=alias_from, proof=proof,
	)
	return buffer, buffer_view(buffer)


def commit_output(
	state: PipelineState,
	output: Any,
	*,
	retire_as: str | None,
	buffer_spec: BufferSpec,
	domain: DataDomain | None = None,
) -> PipelineState:
	"""Publish a filled output, then retire its predecessor after the handoff."""

	result = state.with_array(output, buffer_spec=buffer_spec, domain=domain)
	if (
		state.storage is not None
		and output is not state.working_array
		and isinstance(state.working_array, ManagedBuffer)
	):
		state.storage.retire(state.working_array, role=retire_as)
	return result


class Stage(ABC):
	"""Base class for sequential reconstruction pipeline Stages."""

	name: str = "stage"
	contract: StageContract = StageContract()

	@abstractmethod
	def run(self, state: PipelineState) -> PipelineState:
		"""Execute the Stage and return the next PipelineState."""

	def output_window(self, source: TomoWindow) -> TomoWindow:
		"""Return the source-coordinate window published after this Stage."""

		return source


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
						"Stage output BufferSpec disagrees with its declared layout contract"
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
						"Stage output BufferSpec disagrees with its declared layout contract"
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
