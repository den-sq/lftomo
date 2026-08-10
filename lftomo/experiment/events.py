"""Public lifecycle events, cooperative cancellation, and progress projections.

Attempt projections are authoritative and are replaced atomically before their
corresponding event is appended. ``progress.jsonl`` is therefore a derived live
tail: readers recover status from the store even when its final line is absent or
truncated after a crash.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterator, Literal, Mapping

from pydantic import Field, JsonValue

from lftomo.config import ConfigModel


EventSeverity = Literal["debug", "info", "warning", "error"]
LifecycleState = Literal[
	"planned", "started", "succeeded", "failed", "interrupted", "reused", "skipped",
]
ExperimentState = Literal[
	"planned", "running", "succeeded", "failed", "interrupted", "mixed", "unknown",
]
ProgressCallback = Callable[["ProgressEvent"], None]


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


_SECRET_MARKERS = (
	"secret", "password", "passwd", "token", "api_key", "apikey", "private_key",
	"blinding_key", "credential",
)


def _secret_key(key: str) -> bool:
	normalized = key.lower().replace("-", "_").replace(".", "_")
	parts = set(normalized.split("_"))
	return (
		normalized in _SECRET_MARKERS
		or bool(parts & {"secret", "password", "passwd", "token", "credential"})
		or normalized.endswith(("_api_key", "_private_key", "_blinding_key"))
	)


def active_secret_values() -> tuple[str, ...]:
	"""Collect non-trivial values from environment variables named as secrets."""

	return tuple(sorted({
		value for key, value in os.environ.items()
		if _secret_key(key) and len(value) >= 8
	}))


def redact_secrets(value: Any, *, secrets: tuple[str, ...] = ()) -> JsonValue:
	"""Return JSON-safe data with secret-bearing fields and known values removed."""

	known = tuple(item for item in secrets if item)
	if hasattr(value, "model_dump"):
		value = value.model_dump(mode="json")
	if isinstance(value, Mapping):
		clean: dict[str, JsonValue] = {}
		for raw_key, item in value.items():
			key = str(raw_key)
			if _secret_key(key):
				if "blinding" in key.lower() and item not in (None, ""):
					fingerprint = hashlib.sha256(str(item).encode("utf-8")).hexdigest()[:16]
					clean[f"{key}_fingerprint"] = fingerprint
				continue
			clean[key] = redact_secrets(item, secrets=known)
		return clean
	if isinstance(value, (list, tuple)):
		return [redact_secrets(item, secrets=known) for item in value]
	if isinstance(value, Path):
		value = str(value)
	if isinstance(value, str):
		for secret in known:
			value = value.replace(secret, "[REDACTED]")
		return value
	if value is None or isinstance(value, (int, float, bool)):
		return value
	return redact_secrets(repr(value), secrets=known)


def secret_material_paths(
	value: Any,
	*,
	secrets: tuple[str, ...] = (),
	prefix: str = "$",
) -> tuple[str, ...]:
	"""Locate secret-bearing keys or known secret values before durable writes."""

	if hasattr(value, "model_dump"):
		value = value.model_dump(mode="json")
	paths = []
	if isinstance(value, Mapping):
		for raw_key, item in value.items():
			key = str(raw_key)
			path = f"{prefix}.{key}"
			if _secret_key(key):
				paths.append(path)
				continue
			paths.extend(secret_material_paths(
				item, secrets=secrets, prefix=path,
			))
	elif isinstance(value, (list, tuple)):
		for index, item in enumerate(value):
			paths.extend(secret_material_paths(
				item, secrets=secrets, prefix=f"{prefix}[{index}]",
			))
	elif isinstance(value, str) and any(secret and secret in value for secret in secrets):
		paths.append(prefix)
	return tuple(paths)


class ProgressEvent(ConfigModel):
	"""Versioned, totally ordered experiment progress event."""

	schema_version: Literal[1] = 1
	sequence: int = Field(ge=1)
	timestamp: datetime
	experiment_id: str = Field(min_length=1)
	kind: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
	severity: EventSeverity = "info"
	message: str = Field(min_length=1)
	fidelity_id: str | None = None
	completed_candidates: int | None = Field(default=None, ge=0)
	total_candidates: int | None = Field(default=None, ge=0)
	succeeded_candidates: int | None = Field(default=None, ge=0)
	failed_candidates: int | None = Field(default=None, ge=0)
	interrupted_candidates: int | None = Field(default=None, ge=0)
	skipped_candidates: int | None = Field(default=None, ge=0)
	reused_candidates: int | None = Field(default=None, ge=0)
	active_candidates: int | None = Field(default=None, ge=0)
	remaining_candidates: int | None = Field(default=None, ge=0)
	candidate_key: str | None = None
	trial_id: str | None = None
	attempt_id: str | None = None
	structural_variant: str | None = None
	sampler_index: int | None = Field(default=None, ge=0)
	sampler_position: int | None = Field(default=None, ge=1)
	sampler_total: int | None = Field(default=None, ge=1)
	window_id: str | None = None
	window_position: int | None = Field(default=None, ge=1)
	window_total: int | None = Field(default=None, ge=1)
	window_elapsed_seconds: float | None = Field(default=None, ge=0)
	stage_occurrence_id: str | None = None
	stage_algorithm: str | None = None
	stage_position: int | None = Field(default=None, ge=1)
	stage_total: int | None = Field(default=None, ge=1)
	stage_elapsed_seconds: float | None = Field(default=None, ge=0)
	lifecycle_state: LifecycleState | None = None
	elapsed_seconds: float | None = Field(default=None, ge=0)
	resource_bytes: int | None = Field(default=None, ge=0)
	output_bytes: int | None = Field(default=None, ge=0)
	cache_bytes: int | None = Field(default=None, ge=0)
	failure_type: str | None = None
	failure_message: str | None = None
	traceback: str | None = None
	details: dict[str, JsonValue] = Field(default_factory=dict)


class FailureContext(ConfigModel):
	"""Complete persisted failure context for an attempt projection."""

	type: str
	message: str
	traceback: str | None = None
	window_id: str | None = None
	stage_occurrence_id: str | None = None
	stage_algorithm: str | None = None


class AttemptProgress(ConfigModel):
	"""Atomic authoritative projection for one logical attempt."""

	schema_version: Literal[1] = 1
	experiment_id: str
	fidelity_id: str
	candidate_key: str
	trial_id: str
	attempt_id: str
	structural_variant: str | None = None
	sampler_index: int | None = Field(default=None, ge=0)
	sampler_position: int | None = Field(default=None, ge=1)
	sampler_total: int | None = Field(default=None, ge=1)
	state: LifecycleState
	updated_at: datetime
	started_at: datetime | None = None
	finished_at: datetime | None = None
	elapsed_seconds: float | None = Field(default=None, ge=0)
	window_id: str | None = None
	window_position: int | None = Field(default=None, ge=1)
	window_total: int | None = Field(default=None, ge=1)
	window_elapsed_seconds: float | None = Field(default=None, ge=0)
	stage_occurrence_id: str | None = None
	stage_algorithm: str | None = None
	stage_position: int | None = Field(default=None, ge=1)
	stage_total: int | None = Field(default=None, ge=1)
	stage_elapsed_seconds: float | None = Field(default=None, ge=0)
	resource_bytes: int | None = Field(default=None, ge=0)
	output_bytes: int | None = Field(default=None, ge=0)
	cache_bytes: int | None = Field(default=None, ge=0)
	failure: FailureContext | None = None


class CancellationDetails(ConfigModel):
	"""Structured interruption summary returned by lifecycle calls."""

	requested: bool = False
	reason: str | None = None
	requested_at: datetime | None = None
	interrupted_attempt_ids: tuple[str, ...] = ()
	unstarted_candidates: int = Field(default=0, ge=0)


class ExperimentStatus(ConfigModel):
	"""Store-derived current experiment state; event logs are never consulted."""

	schema_version: Literal[1] = 1
	experiment_id: str
	state: ExperimentState
	updated_at: datetime
	attempts: tuple[AttemptProgress, ...]
	counts: dict[str, int]
	cancellation: CancellationDetails | None = None


class ExperimentCancelled(InterruptedError):
	"""Raised at cooperative boundaries after a cancellation request."""


class ProgressInfrastructureError(RuntimeError):
	"""A durable progress projection or event could not be published safely."""


class CancellationToken:
	"""Thread-safe cooperative cancellation request shared with callers."""

	def __init__(self):
		self._event = threading.Event()
		self._lock = threading.Lock()
		self._reason: str | None = None
		self._requested_at: datetime | None = None

	def request(self, reason: str = "cancellation requested") -> bool:
		"""Request cancellation, returning ``True`` only for the first request."""

		with self._lock:
			if self._event.is_set():
				return False
			self._reason = reason
			self._requested_at = _utcnow()
			self._event.set()
			return True

	@property
	def requested(self) -> bool:
		return self._event.is_set()

	@property
	def reason(self) -> str | None:
		return self._reason

	@property
	def requested_at(self) -> datetime | None:
		return self._requested_at

	def raise_if_requested(self) -> None:
		if self.requested:
			raise ExperimentCancelled(self.reason or "cancellation requested")


class ProgressEventWriter:
	"""Single ordered writer for concurrent producers and optional callbacks."""

	def __init__(
		self,
		root: str | Path,
		experiment_id: str,
		*,
		callback: ProgressCallback | None = None,
		secrets: tuple[str, ...] = (),
		durable: bool = False,
	):
		self.root = Path(root)
		self.path = self.root / "progress.jsonl"
		self.experiment_id = experiment_id
		self.callback = callback
		self.secrets = secrets
		self._lock = threading.RLock()
		self._sequence = self._recover_sequence()
		self._durable = durable
		self._buffer: list[ProgressEvent] = []
		if durable:
			self._repair_truncated_tail()

	def _recover_sequence(self) -> int:
		if not self.path.exists():
			return 0
		return max((event.sequence for event in read_progress_events(self.path)), default=0)

	def _append(self, event: ProgressEvent) -> None:
		self.path.parent.mkdir(parents=True, exist_ok=True)
		payload = json.dumps(event.model_dump(mode="json"), sort_keys=True)
		with self.path.open("a", encoding="utf-8") as stream:
			stream.write(payload + "\n")
			stream.flush()
			os.fsync(stream.fileno())

	def _repair_truncated_tail(self) -> None:
		"""Discard only a torn final JSONL record before future appends."""

		if not self.path.exists():
			return
		data = self.path.read_bytes()
		lines = data.splitlines(keepends=True)
		valid_bytes = 0
		for index, line in enumerate(lines):
			if not line.strip():
				valid_bytes += len(line)
				continue
			try:
				ProgressEvent.model_validate_json(line.strip())
			except ValueError:
				if index != len(lines) - 1:
					raise
				break
			valid_bytes += len(line)
		if valid_bytes < len(data):
			with self.path.open("r+b") as stream:
				stream.truncate(valid_bytes)
				stream.flush()
				os.fsync(stream.fileno())
			data = data[:valid_bytes]
		if data and not data.endswith(b"\n"):
			with self.path.open("ab") as stream:
				stream.write(b"\n")
				stream.flush()
				os.fsync(stream.fileno())

	def activate(self) -> None:
		"""Begin durable output and flush buffered preflight events in order."""

		with self._lock:
			if self._durable:
				return
			self._repair_truncated_tail()
			self._durable = True
			for event in self._buffer:
				self._append(event)
			self._buffer.clear()

	def flush(self) -> None:
		"""Flush buffered events when durability has been activated."""

		with self._lock:
			if self._durable:
				for event in self._buffer:
					self._append(event)
				self._buffer.clear()

	def emit(
		self,
		kind: str,
		message: str,
		*,
		severity: EventSeverity = "info",
		**fields: Any,
	) -> ProgressEvent:
		with self._lock:
			self._sequence += 1
			clean_message = str(redact_secrets(message, secrets=self.secrets))
			clean_fields = redact_secrets(fields, secrets=self.secrets)
			event = ProgressEvent(
				sequence=self._sequence,
				timestamp=_utcnow(),
				experiment_id=self.experiment_id,
				kind=kind,
				severity=severity,
				message=clean_message,
				**clean_fields,
			)
			if self._durable:
				self._append(event)
			else:
				self._buffer.append(event)
			if self.callback is not None:
				try:
					self.callback(event)
				except Exception:
					# Presentation callbacks cannot invalidate scientific execution.
					pass
			return event


def read_progress_events(path: str | Path) -> tuple[ProgressEvent, ...]:
	"""Read valid ordered events, tolerating only a truncated final JSONL line."""

	path = Path(path)
	if not path.exists():
		return ()
	lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
	events = []
	for index, line in enumerate(lines):
		if not line.strip():
			continue
		try:
			event = ProgressEvent.model_validate_json(line)
		except ValueError:
			if index == len(lines) - 1:
				break
			raise
		if events and event.sequence <= events[-1].sequence:
			raise ValueError("progress event sequence is not strictly increasing")
		events.append(event)
	return tuple(events)


ProgressSink = Callable[[AttemptProgress], None]


@dataclass
class ExecutionContext:
	"""Per-attempt event/projection context available inside pipeline execution."""

	writer: ProgressEventWriter | None
	progress_sink: ProgressSink
	cancellation: CancellationToken
	progress: AttemptProgress
	clock_started: float = field(default_factory=time.perf_counter)
	window_clock_started: float | None = None
	stage_clocks_started: dict[str, float] = field(default_factory=dict)

	def _persist_progress(self, kind: str) -> None:
		try:
			self.progress_sink(self.progress)
		except ProgressInfrastructureError:
			raise
		except Exception as error:
			raise ProgressInfrastructureError(
				f"failed to persist progress projection for {kind!r}: {error}"
			) from error

	def _publish(self, kind: str, message: str, **fields: Any) -> None:
		self._persist_progress(kind)
		try:
			if self.writer is not None:
				self.writer.emit(
					kind,
					message,
					fidelity_id=self.progress.fidelity_id,
					candidate_key=self.progress.candidate_key,
					trial_id=self.progress.trial_id,
					attempt_id=self.progress.attempt_id,
					structural_variant=self.progress.structural_variant,
					sampler_index=self.progress.sampler_index,
					sampler_position=self.progress.sampler_position,
					sampler_total=self.progress.sampler_total,
					window_id=self.progress.window_id,
					window_position=self.progress.window_position,
					window_total=self.progress.window_total,
					window_elapsed_seconds=self.progress.window_elapsed_seconds,
					stage_occurrence_id=self.progress.stage_occurrence_id,
					stage_algorithm=self.progress.stage_algorithm,
					stage_position=self.progress.stage_position,
					stage_total=self.progress.stage_total,
					stage_elapsed_seconds=self.progress.stage_elapsed_seconds,
					lifecycle_state=self.progress.state,
					elapsed_seconds=self.progress.elapsed_seconds,
					resource_bytes=self.progress.resource_bytes,
					output_bytes=self.progress.output_bytes,
					cache_bytes=self.progress.cache_bytes,
					**fields,
				)
		except ProgressInfrastructureError:
			raise
		except Exception as error:
			raise ProgressInfrastructureError(
				f"failed to publish progress event {kind!r}: {error}"
			) from error

	def update(self, kind: str, message: str, **changes: Any) -> None:
		changes.setdefault("updated_at", _utcnow())
		changes.setdefault("elapsed_seconds", time.perf_counter() - self.clock_started)
		self.progress = self.progress.model_copy(update=changes)
		self._publish(kind, message)

	def checkpoint(self) -> None:
		self.cancellation.raise_if_requested()

	def window_started(
		self, window_id: str, *, position: int | None = None, total: int | None = None,
	) -> None:
		self.checkpoint()
		self.window_clock_started = time.perf_counter()
		self.update(
			"window.started", (
				f"window {position}/{total} {window_id} started"
				if position is not None and total is not None
				else f"window {window_id} started"
			),
			window_id=window_id, stage_occurrence_id=None, stage_algorithm=None,
			window_position=position, window_total=total, window_elapsed_seconds=0,
			stage_position=None, stage_total=None, stage_elapsed_seconds=None,
		)

	def window_completed(self, window_id: str) -> None:
		window_elapsed = (
			None if self.window_clock_started is None
			else time.perf_counter() - self.window_clock_started
		)
		self.update(
			"window.completed", (
				f"window {self.progress.window_position}/{self.progress.window_total} "
				f"{window_id} completed"
				if self.progress.window_position is not None
				and self.progress.window_total is not None
				else f"window {window_id} completed"
			),
			window_id=window_id, stage_occurrence_id=None, stage_algorithm=None,
			window_elapsed_seconds=window_elapsed,
			stage_position=None, stage_total=None, stage_elapsed_seconds=None,
		)
		self.progress = self.progress.model_copy(update={
			"updated_at": _utcnow(), "window_id": None,
			"window_position": None, "window_total": None,
			"window_elapsed_seconds": None,
		})
		self.window_clock_started = None
		self._persist_progress("window.completed")

	def stage_started(
		self,
		occurrence_id: str,
		algorithm: str | None,
		*,
		position: int | None = None,
		total: int | None = None,
	) -> None:
		self.checkpoint()
		self.stage_clocks_started[occurrence_id] = time.perf_counter()
		self.update(
			"stage.started", (
				f"Stage {position}/{total} {occurrence_id}:{algorithm} started"
				if position is not None and total is not None
				else f"Stage {occurrence_id} started"
			),
			stage_occurrence_id=occurrence_id, stage_algorithm=algorithm,
			stage_position=position, stage_total=total, stage_elapsed_seconds=0,
		)

	def stage_completed(
		self,
		occurrence_id: str,
		algorithm: str | None,
		*,
		position: int | None = None,
		total: int | None = None,
	) -> None:
		started = self.stage_clocks_started.pop(occurrence_id, None)
		stage_elapsed = (
			None if started is None else time.perf_counter() - started
		)
		position = self.progress.stage_position if position is None else position
		total = self.progress.stage_total if total is None else total
		self.update(
			"stage.completed", (
				f"Stage {position}/{total} "
				f"{occurrence_id}:{algorithm} completed"
				if position is not None and total is not None
				else f"Stage {occurrence_id} completed"
			),
			stage_occurrence_id=occurrence_id, stage_algorithm=algorithm,
			stage_position=position, stage_total=total,
			stage_elapsed_seconds=stage_elapsed,
		)
		self.progress = self.progress.model_copy(update={
			"updated_at": _utcnow(),
			"stage_occurrence_id": None,
			"stage_algorithm": None,
			"stage_position": None,
			"stage_total": None,
			"stage_elapsed_seconds": None,
		})
		self._persist_progress("stage.completed")

	def heartbeat(self) -> None:
		"""Publish exact current-boundary elapsed time without inventing a percentage."""

		self.checkpoint()
		if self.progress.stage_occurrence_id is not None:
			started = self.stage_clocks_started.get(self.progress.stage_occurrence_id)
			stage_elapsed = (
				None if started is None else time.perf_counter() - started
			)
			self.update(
				"stage.heartbeat",
				f"Stage {self.progress.stage_occurrence_id} is running",
				stage_elapsed_seconds=stage_elapsed,
			)
		elif self.progress.window_id is not None:
			window_elapsed = (
				None if self.window_clock_started is None
				else time.perf_counter() - self.window_clock_started
			)
			self.update(
				"window.heartbeat", f"window {self.progress.window_id} is running",
				window_elapsed_seconds=window_elapsed,
			)

	def stage_failed(
		self,
		occurrence_id: str,
		algorithm: str | None,
		error: BaseException,
		traceback_text: str,
		*,
		position: int | None = None,
		total: int | None = None,
	) -> None:
		secrets = () if self.writer is None else self.writer.secrets
		message = str(redact_secrets(
			str(error) or type(error).__name__, secrets=secrets,
		))
		traceback_text = str(redact_secrets(traceback_text, secrets=secrets))
		failure = FailureContext(
			type=type(error).__name__, message=message,
			traceback=traceback_text, window_id=self.progress.window_id,
			stage_occurrence_id=occurrence_id, stage_algorithm=algorithm,
		)
		started = self.stage_clocks_started.pop(occurrence_id, None)
		position = self.progress.stage_position if position is None else position
		total = self.progress.stage_total if total is None else total
		self.progress = self.progress.model_copy(update={
			"updated_at": _utcnow(),
			"elapsed_seconds": time.perf_counter() - self.clock_started,
			"stage_occurrence_id": occurrence_id,
			"stage_algorithm": algorithm,
			"stage_position": position,
			"stage_total": total,
			"stage_elapsed_seconds": (
				None if started is None else time.perf_counter() - started
			),
			"failure": failure,
		})
		self._publish(
			"stage.failed", f"Stage {occurrence_id} failed: {failure.message}",
			severity="error", failure_type=failure.type,
			failure_message=failure.message, traceback=traceback_text,
		)

	def cache_event(
		self,
		event: str,
		*,
		cache_bytes: int | None = None,
		details: Mapping[str, Any] | None = None,
	) -> None:
		"""Publish one explicitly projected cache event.

		``event`` owns the lifecycle event kind. Cache implementation status and
		measurements belong in ``details`` and cannot collide with this call surface.
		"""

		try:
			changes = {
				"updated_at": _utcnow(),
				"elapsed_seconds": time.perf_counter() - self.clock_started,
			}
			if cache_bytes is not None:
				changes["cache_bytes"] = cache_bytes
			self.progress = self.progress.model_copy(update=changes)
			self._publish(
				f"cache.{event}", f"prefix cache {event}",
				details={} if details is None else dict(details),
			)
		except ProgressInfrastructureError:
			raise
		except Exception as error:
			raise ProgressInfrastructureError(
				f"failed to project prefix cache event {event!r}: {error}"
			) from error

	def output_event(
		self,
		event: Literal["committed", "limit_exceeded"],
		message: str,
		*,
		output_bytes: int,
		details: Mapping[str, Any],
	) -> None:
		"""Publish a budgeted output decision with its durable byte totals."""

		self.progress = self.progress.model_copy(update={
			"updated_at": _utcnow(),
			"elapsed_seconds": time.perf_counter() - self.clock_started,
			"output_bytes": output_bytes,
		})
		self._publish(
			f"output.{event}", message,
			severity="error" if event == "limit_exceeded" else "info",
			details=dict(details),
		)


_CURRENT_EXECUTION: ContextVar[ExecutionContext | None] = ContextVar(
	"lftomo_experiment_execution", default=None,
)


@contextmanager
def bind_execution_context(context: ExecutionContext) -> Iterator[ExecutionContext]:
	token = _CURRENT_EXECUTION.set(context)
	try:
		yield context
	finally:
		_CURRENT_EXECUTION.reset(token)


def current_execution_context() -> ExecutionContext | None:
	return _CURRENT_EXECUTION.get()
