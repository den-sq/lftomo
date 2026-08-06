"""Stage and Pipeline ABCs for lftomo reconstruction workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Literal, Protocol

import numpy as np

from lftomo.config import Recon, Scan
from lftomo.pipeline.domains import ContractEffect, DataDomain, PRESERVE


@dataclass(frozen=True)
class ResourceHints:
	"""Cooperative resource hints declared by a Stage."""

	memory_multiplier: float = 1.0
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


@dataclass(frozen=True)
class StageContract:
	"""Behavioral contract advertised by a Stage before it runs.

	``expected_input`` and ``produced_output`` are shape classes, usually from
	``lftomo.util.shapes`` such as ``ProjOrder``, ``SinoOrder``, or
	``ReconOrder``. These contract fields are consumed by the Pipeline validator
	defined by issue #18 / PR #42. ``runtime_requirements`` declares invariant
	optional dependencies and bounded post-import API probes for preflight.
	"""

	expected_input: type[Any] | None = None
	produced_output: type[Any] | None = None
	mutates_input: bool = False
	# Prefix caching is fail-closed: a Stage must explicitly promise that its
	# output is deterministic and fully represented by its config, contract,
	# backend, PipelineState inputs, and declared artifacts before reuse.
	cacheable_prefix: bool = False
	window_axes: tuple[str, ...] = ()
	required_metadata: tuple[str, ...] = ()
	consumes_artifacts: tuple[str, ...] = ()
	produces_artifacts: tuple[str, ...] = ()
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
	runtime_requirements: tuple[RuntimeRequirement, ...] = ()

	@property
	def accepted_layouts(self) -> tuple[type[Any], ...] | None:
		"""Return explicit capabilities or the legacy expected shape."""

		if self.accepts is not None:
			return self.accepts
		if self.expected_input is not None:
			return (self.expected_input,)
		return None

	@property
	def preferred_layout(self) -> type[Any] | None:
		"""Return the planner preference derived from accepted layouts."""

		if self.prefers is not None:
			return self.prefers
		accepted = self.accepted_layouts
		return None if not accepted else accepted[0]

	def output_layout(self, input_layout: type[Any] | None) -> type[Any] | None:
		"""Resolve the concrete output layout for ``input_layout``."""

		effect = self.produced_output if self.produces is None else self.produces
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


class WorkingArray(Protocol):
	"""Minimal shape-bearing protocol for pipeline working arrays."""

	@property
	def shape(self) -> tuple[int, ...]:
		"""Array shape in backend-native form."""


@dataclass
class PipelineState:
	"""Mutable state object passed from Stage to Stage."""

	working_array: np.ndarray | DaskArrayStub | WorkingArray
	scan: Scan
	recon: Recon
	artifacts: dict[str, Any] = field(default_factory=dict)
	backend: Backend = field(default_factory=NumpyBackend)
	# Fragment callers may omit axes for legacy compatibility. Complete pipelines
	# always acquire both from their read Stage before data processing begins.
	layout: type[Any] | None = None
	domain: DataDomain | None = None

	def with_array(
		self,
		working_array: Any,
		*,
		layout: type[Any] | None = None,
		domain: DataDomain | None = None,
	) -> "PipelineState":
		"""Return a shallow-copy state with a different working array."""

		return PipelineState(
			working_array=working_array,
			scan=self.scan,
			recon=self.recon,
			artifacts=dict(self.artifacts),
			backend=self.backend,
			layout=self.layout if layout is None else layout,
			domain=self.domain if domain is None else domain,
		)


class Stage(ABC):
	"""Base class for sequential reconstruction pipeline Stages."""

	name: str = "stage"
	contract: StageContract = StageContract()

	@abstractmethod
	def run(self, state: PipelineState) -> PipelineState:
		"""Execute the Stage and return the next PipelineState."""


StageObserver = Callable[
	[Literal["started", "completed", "failed"], int, Stage, BaseException | None], None,
]


class Pipeline:
	"""Sequential Pipeline over an ordered list of Stages."""

	def __init__(self, stages: list[Stage], backend: Backend | None = None,
			scan: Scan | None = None, validate: bool = True,
			validation_mode: str = "fragment",
			initial_layout: type[Any] | None = None,
			initial_domain: DataDomain | None = None,
			bounded_workers: int = 1, observer: StageObserver | None = None,
			checkpoint: Callable[[], None] | None = None):
		if bounded_workers < 1:
			raise ValueError("bounded_workers must be positive")
		self.stages = tuple(stages)
		self.backend = backend or NumpyBackend()
		self.validate = validate
		self.validation_mode = validation_mode
		self.initial_layout = initial_layout
		self.initial_domain = initial_domain
		self.bounded_workers = bounded_workers
		self.observer = observer
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
			)
			self._validated = True
		segments = () if self.plan is None else self.plan.segments
		if not segments:
			from lftomo.pipeline.planning import ExecutionKind, PlanSegment

			segments = tuple(PlanSegment(
				index, index + 1, ExecutionKind.SEQUENTIAL, (stage.name,),
			) for index, stage in enumerate(self.stages))
		from lftomo.pipeline.planning import ExecutionKind
		for segment in segments:
			segment_stages = self.stages[segment.start:segment.stop]
			if segment.kind is ExecutionKind.BOUNDED_SINOGRAM:
				from lftomo.pipeline.execution import run_bounded_sinogram_segment

				stage_indices = {
					id(stage): offset
					for offset, stage in enumerate(segment_stages, start=segment.start)
				}
				if self.observer is not None:
					for offset, stage in enumerate(segment_stages, start=segment.start):
						self.observer("started", offset, stage, None)
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
							)
						),
						checkpoint=self.checkpoint,
					)
				except BaseException:
					raise
				if self.observer is not None:
					for offset, stage in enumerate(segment_stages, start=segment.start):
						self.observer("completed", offset, stage, None)
				for stage in segment_stages:
					state.layout = stage.contract.output_layout(state.layout)
					state.domain = stage.contract.output_domain(state.domain)
				continue
			for index, stage in enumerate(segment_stages, start=segment.start):
				if self.checkpoint is not None:
					self.checkpoint()
				input_layout = state.layout
				input_domain = state.domain
				if self.observer is not None:
					self.observer("started", index, stage, None)
				try:
					state = stage.run(state)
				except BaseException as error:
					if self.observer is not None:
						self.observer("failed", index, stage, error)
					raise
				if self.observer is not None:
					self.observer("completed", index, stage, None)
				output_layout = stage.contract.output_layout(input_layout)
				output_domain = stage.contract.output_domain(input_domain)
				if output_layout is not None:
					state.layout = output_layout
				if output_domain is not None:
					state.domain = output_domain
		return state
