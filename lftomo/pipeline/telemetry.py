"""Public execution-span telemetry emitted by the canonical Pipeline runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import TYPE_CHECKING, Callable, Iterable

from lftomo.pipeline.dispatch import DispatchKind
from lftomo.pipeline.planning import ExecutionKind

if TYPE_CHECKING:
	from lftomo.util.window import TomoWindow


class ExecutionSpanPhase(str, Enum):
	"""Lifecycle transition observed at one actual execution boundary."""

	STARTED = "started"
	COMPLETED = "completed"
	FAILED = "failed"
	INTERRUPTED = "interrupted"


class ObservationAttribution(str, Enum):
	"""Whether a resource value was measured or assigned by inference."""

	OBSERVED = "observed"
	ATTRIBUTED = "attributed"
	UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class LogicalStageMember:
	"""One authored Stage occurrence executed as part of a span."""

	index: int
	stage_name: str
	stage_type: str
	dispatch_kind: DispatchKind


@dataclass(frozen=True)
class WorkerTimingAggregate:
	"""Bounded worker-invocation elapsed times for one logical member.

	Each sample is one algorithm invocation on one bounded work unit. ``sum``
	is worker work, not coordinator wall time, and concurrent samples may overlap.
	"""

	invocation_count: int
	minimum_seconds: float
	median_seconds: float
	maximum_seconds: float
	sum_seconds: float
	basis: str = "per-algorithm bounded-worker invocation elapsed time"

	@classmethod
	def from_samples(cls, samples: Iterable[float]) -> "WorkerTimingAggregate":
		values = tuple(float(value) for value in samples)
		if not values:
			raise ValueError("worker timing aggregates require at least one sample")
		return cls(
			invocation_count=len(values),
			minimum_seconds=min(values),
			median_seconds=float(median(values)),
			maximum_seconds=max(values),
			sum_seconds=sum(values),
		)


@dataclass(frozen=True)
class MemberTiming:
	"""Timing attributable to one logical Stage member."""

	member: LogicalStageMember
	coordinator_wall_seconds: float | None = None
	worker: WorkerTimingAggregate | None = None
	unavailable: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManagedAllocationChange:
	"""One observed change from the owning managed-storage ledger."""

	action: str
	role: str | None
	name: str
	bytes: int
	alias: str | None = None
	attribution: ObservationAttribution = ObservationAttribution.OBSERVED
	scope: str = "execution_span"


@dataclass(frozen=True)
class ResourceObservation:
	"""Backend-neutral resource observation attached to its sampled scope."""

	metric: str
	value: int | float | None
	unit: str
	scope: str
	attribution: ObservationAttribution
	process_ids: tuple[int, ...] = ()
	detail: str | None = None


@dataclass(frozen=True)
class ExecutionSpan:
	"""One actual execution segment and its ordered authored members."""

	run_id: str
	span_id: str
	ordinal: int
	total: int
	kind: ExecutionKind
	member_start: int
	member_stop: int
	members: tuple[LogicalStageMember, ...]


@dataclass(frozen=True)
class ExecutionSpanEvent:
	"""One lifecycle report emitted for an :class:`ExecutionSpan`."""

	phase: ExecutionSpanPhase
	span: ExecutionSpan
	source_window: TomoWindow | None = None
	coordinator_wall_seconds: float | None = None
	member_timings: tuple[MemberTiming, ...] = ()
	managed_allocation_changes: tuple[ManagedAllocationChange, ...] = ()
	resource_observations: tuple[ResourceObservation, ...] = ()
	unavailable_observations: tuple[str, ...] = ()
	error_type: str | None = None
	error_message: str | None = None


ExecutionSpanObserver = Callable[[ExecutionSpanEvent], None]
