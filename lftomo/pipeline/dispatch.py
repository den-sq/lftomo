"""Dispatch classifications shared by the Stage registry and executor."""

from dataclasses import dataclass
from enum import Enum, auto


class DispatchKind(Enum):
	"""Executor selected for one registered Stage entry."""

	PROCESS = auto()
	THREAD = auto()
	SERIAL = auto()
	ENGINE = auto()
	EXEMPT = auto()


@dataclass(frozen=True)
class DispatchSpec:
	"""Registry-owned dispatch classification and its callable boundary."""

	kind: DispatchKind
	kernel: str | None = None
	reason: str | None = None
	projection_units: bool = False

	def __post_init__(self):
		if self.kind is DispatchKind.EXEMPT and not self.reason:
			raise ValueError("executor exemptions require a documented reason")
