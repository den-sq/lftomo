"""Typed metric registry used by the evaluation Stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from lftomo.evaluation.models import MetricDefinition, MetricRequest


@dataclass(frozen=True)
class KernelValue:
	"""Value returned by a metric kernel before observation qualification."""

	metric: str
	value: float | None
	sample_count: int
	invalid_count: int = 0
	reason: str | None = None

	@property
	def valid(self) -> bool:
		return self.value is not None and self.reason is None and np.isfinite(self.value)


MetricKernel = Callable[[np.ndarray, MetricRequest, dict[str, Any]], list[KernelValue]]


@dataclass(frozen=True)
class MetricRegistration:
	"""One request type, its emitted definitions, and its pure slice kernel."""

	kind: str
	request_type: type
	definitions: tuple[MetricDefinition, ...]
	kernel: MetricKernel


class MetricRegistry:
	"""Deterministic registry of typed metric request families."""

	def __init__(self):
		self._registrations: dict[str, MetricRegistration] = {}
		self._definitions: dict[str, MetricDefinition] = {}

	def register(self, registration: MetricRegistration) -> None:
		if registration.kind in self._registrations:
			raise ValueError(f"metric request kind already registered: {registration.kind}")
		for definition in registration.definitions:
			if definition.name in self._definitions:
				raise ValueError(f"metric already registered: {definition.name}")
		self._registrations[registration.kind] = registration
		for definition in registration.definitions:
			self._definitions[definition.name] = definition

	def registration(self, request: MetricRequest) -> MetricRegistration:
		try:
			registration = self._registrations[request.kind]
		except KeyError as error:
			raise ValueError(f"unknown metric request kind: {request.kind!r}") from error
		if not isinstance(request, registration.request_type):
			raise TypeError(
				f"metric request {request.kind!r} must be {registration.request_type.__name__}"
			)
		return registration

	def definition(self, name: str) -> MetricDefinition:
		try:
			return self._definitions[name]
		except KeyError as error:
			raise ValueError(f"unknown metric: {name!r}") from error

	def evaluate(
		self,
		image: np.ndarray,
		request: MetricRequest,
		context: dict[str, Any],
	) -> list[KernelValue]:
		"""Evaluate one typed request and verify its declared outputs."""

		registration = self.registration(request)
		values = registration.kernel(image, request, context)
		expected = {definition.name for definition in registration.definitions}
		actual = {value.metric for value in values}
		if actual != expected:
			raise RuntimeError(
				f"metric kernel {request.kind!r} emitted {sorted(actual)}, "
				f"expected {sorted(expected)}"
			)
		return values

	@property
	def definitions(self) -> tuple[MetricDefinition, ...]:
		return tuple(self._definitions[name] for name in sorted(self._definitions))
