"""Killable per-attempt execution boundary for fidelity wall-clock deadlines."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import signal
import time
import traceback
from typing import Any

from .events import (
	AttemptProgress,
	AttemptTimeoutExceeded,
	CancellationToken,
	ExecutionContext,
	ExperimentCancelled,
	ProgressInfrastructureError,
	bind_execution_context,
)


class TimeoutIsolationUnsupported(ValueError):
	"""An executor cannot provide the hard termination required by timeout_seconds."""


class IsolatedExecutionError(RuntimeError):
	"""An isolated executor failed; retain its original failure projection."""

	def __init__(
		self,
		failure_type: str,
		message: str,
		remote_traceback: str,
		failure_details: dict[str, Any] | None = None,
	):
		self.failure_type = failure_type
		self.remote_traceback = remote_traceback
		self._failure_details = failure_details or {}
		super().__init__(message)

	def failure_details(self) -> dict[str, Any]:
		return dict(self._failure_details)


class _QueueEventWriter:
	def __init__(self, queue, secrets: tuple[str, ...]):
		self.queue = queue
		self.secrets = secrets

	def emit(self, kind: str, message: str, *, severity: str = "info", **fields: Any) -> None:
		self.queue.put(("event", {
			"kind": kind,
			"message": message,
			"severity": severity,
			"fields": fields,
		}))


def _isolated_worker(
	executor,
	identity,
	output_root: Path,
	progress_document: dict[str, Any],
	clock_started: float,
	deadline: float,
	timeout_seconds: float,
	timeout_signal,
	queue,
	emit_events: bool,
	secrets: tuple[str, ...],
) -> None:
	try:
		if hasattr(os, "setsid"):
			os.setsid()
		queue.put(("ready", {"pid": os.getpid()}))
		progress = AttemptProgress.model_validate(progress_document)

		def progress_sink(value: AttemptProgress) -> None:
			queue.put(("progress", value.model_dump(mode="json")))

		context = ExecutionContext(
			writer=_QueueEventWriter(queue, secrets) if emit_events else None,
			progress_sink=progress_sink,
			cancellation=CancellationToken(),
			progress=progress,
			clock_started=clock_started,
			deadline=deadline,
			timeout_seconds=timeout_seconds,
			attempt_timeout_signal=timeout_signal,
		)
		context.checkpoint()
		with bind_execution_context(context):
			result = executor(identity, output_root)
		context.checkpoint()
		queue.put(("result", result))
	except BaseException as error:
		details = {}
		failure_details = getattr(error, "failure_details", None)
		if callable(failure_details):
			details = failure_details()
		queue.put(("failure", {
			"type": type(error).__name__,
			"message": str(error) or type(error).__name__,
			"traceback": traceback.format_exc(),
			"details": details,
		}))


@dataclass
class _Messages:
	result: Any | None = None
	failure: dict[str, Any] | None = None
	ready: bool = False


def _drain(queue, context: ExecutionContext, messages: _Messages) -> None:
	while True:
		try:
			kind, payload = queue.get_nowait()
		except Empty:
			return
		if kind == "progress":
			progress = AttemptProgress.model_validate(payload)
			context.progress_sink(progress)
			context.progress = progress
		elif kind == "event":
			if context.writer is not None:
				context.writer.emit(
					payload["kind"], payload["message"],
					severity=payload["severity"], **payload["fields"],
				)
		elif kind == "result":
			messages.result = payload
		elif kind == "failure":
			messages.failure = payload
		elif kind == "ready":
			messages.ready = True


def _stop_process(process: multiprocessing.Process) -> None:
	pid = process.pid
	group_signalled = False
	if pid is not None and hasattr(os, "killpg"):
		try:
			os.killpg(pid, signal.SIGTERM)
			group_signalled = True
		except ProcessLookupError:
			pass
	if not group_signalled and process.is_alive():
		process.terminate()
	process.join(timeout=0.25)
	if group_signalled and pid is not None:
		try:
			os.killpg(pid, signal.SIGKILL)
		except ProcessLookupError:
			pass
	if process.is_alive():
		if not group_signalled:
			process.kill()
		process.join(timeout=1)
	if process.is_alive():
		raise RuntimeError("isolated timeout worker could not be terminated")


def run_isolated_executor(
	executor,
	identity,
	output_root: Path,
	context: ExecutionContext,
	*,
	timeout_seconds: float,
) -> Any:
	"""Run one executor in a process group that cannot outlive terminal timeout state."""

	spawn = multiprocessing.get_context("spawn")
	queue = spawn.Queue()
	timeout_signal = spawn.Event()
	deadline = context.clock_started + timeout_seconds
	process = spawn.Process(
		target=_isolated_worker,
		args=(
			executor,
			identity,
			output_root,
			context.progress.model_dump(mode="json"),
			context.clock_started,
			deadline,
			timeout_seconds,
			timeout_signal,
			queue,
			context.writer is not None,
			() if context.writer is None else context.writer.secrets,
		),
	)
	messages = _Messages()
	try:
		process.start()
		while True:
			_drain(queue, context, messages)
			if context.cancellation.requested:
				timeout_signal.set()
				process.join(timeout=0.1)
				_drain(queue, context, messages)
				_stop_process(process)
				raise ExperimentCancelled(
					context.cancellation.reason or "cancellation requested"
				)
			if messages.result is not None or messages.failure is not None:
				process.join(timeout=0.05)
				if not process.is_alive():
					break
			if not process.is_alive():
				break
			remaining = deadline - time.perf_counter()
			if remaining <= 0:
				timeout_signal.set()
				process.join(timeout=0.1)
				_drain(queue, context, messages)
				_stop_process(process)
				raise AttemptTimeoutExceeded(
					timeout_seconds=timeout_seconds,
					elapsed_seconds=time.perf_counter() - context.clock_started,
					window_id=context.progress.window_id,
					stage_occurrence_id=context.progress.stage_occurrence_id,
					stage_algorithm=context.progress.stage_algorithm,
				)
			process.join(timeout=min(0.02, remaining))
		_drain(queue, context, messages)
		if messages.failure is not None:
			if messages.failure["type"] == "AttemptTimeoutExceeded":
				details = messages.failure["details"]
				raise AttemptTimeoutExceeded(
					timeout_seconds=float(details["timeout_seconds"]),
					elapsed_seconds=float(details["elapsed_seconds"]),
					window_id=details.get("window_id"),
					stage_occurrence_id=details.get("stage_occurrence_id"),
					stage_algorithm=details.get("stage_algorithm"),
				)
			if messages.failure["type"] == "ProgressInfrastructureError":
				error = ProgressInfrastructureError(messages.failure["message"])
				error.add_note(messages.failure["traceback"])
				raise error
			if messages.failure["type"] in (
				"ExperimentCancelled", "InterruptedError", "KeyboardInterrupt",
			):
				raise InterruptedError(messages.failure["message"])
			raise IsolatedExecutionError(
				messages.failure["type"],
				messages.failure["message"],
				messages.failure["traceback"],
				messages.failure["details"],
			)
		if messages.result is None:
			raise IsolatedExecutionError(
				"IsolatedWorkerExit",
				f"isolated executor exited with code {process.exitcode} without a result",
				"",
			)
		return messages.result
	finally:
		if process.is_alive():
			_stop_process(process)
		queue.close()
		queue.join_thread()
