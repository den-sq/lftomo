"""Stage and Pipeline ABCs for lftomo reconstruction workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from lftomo.config import Recon, Scan


@dataclass(frozen=True)
class ResourceHints:
	"""Cooperative resource hints declared by a Stage."""

	memory_multiplier: float = 1.0
	preferred_cpus: int | None = None
	requires_gpu: bool = False


@dataclass(frozen=True)
class StageContract:
	"""Behavioral contract advertised by a Stage before it runs.

	``expected_input`` and ``produced_output`` are shape classes, usually from
	``lftomo.util.shapes`` such as ``ProjOrder``, ``SinoOrder``, or
	``ReconOrder``. These contract fields are consumed by the Pipeline validator
	defined by issue #18 / PR #42.
	"""

	expected_input: type[Any] | None = None
	produced_output: type[Any] | None = None
	mutates_input: bool = False
	window_axes: tuple[str, ...] = ()
	required_metadata: tuple[str, ...] = ()
	consumes_artifacts: tuple[str, ...] = ()
	produces_artifacts: tuple[str, ...] = ()
	resources: ResourceHints = field(default_factory=ResourceHints)


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

	def with_array(self, working_array: Any) -> "PipelineState":
		"""Return a shallow-copy state with a different working array."""

		return PipelineState(
			working_array=working_array,
			scan=self.scan,
			recon=self.recon,
			artifacts=dict(self.artifacts),
			backend=self.backend,
		)


class Stage(ABC):
	"""Base class for sequential reconstruction pipeline Stages."""

	name: str = "stage"
	contract: StageContract = StageContract()

	@abstractmethod
	def run(self, state: PipelineState) -> PipelineState:
		"""Execute the Stage and return the next PipelineState."""


class Pipeline:
	"""Sequential Pipeline over an ordered list of Stages."""

	def __init__(self, stages: list[Stage], backend: Backend | None = None):
		self.stages = tuple(stages)
		self.backend = backend or NumpyBackend()

	def run(self, state: PipelineState) -> PipelineState:
		"""Run each Stage sequentially; individual Stages own concurrency."""

		state.backend = self.backend
		for stage in self.stages:
			state = stage.run(state)
		return state
