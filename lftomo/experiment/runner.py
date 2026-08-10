"""Independent-trial experiment execution, records, and deterministic leaderboards."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any, Callable, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, JsonValue
import yaml

from lftomo.config import ConfigModel, Recon
from lftomo.evaluation import (
	CandidateMetrics,
	HardConstraint,
	ParetoObjective,
	pareto_rank,
)

from .models import (
	CandidateKey,
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	ExperimentManifest,
	FailureRecord,
	FidelityIdentity,
	FinalResult,
	MetricProtocolIdentity,
	OutputAccounting,
	OutputRecord,
	PendingOutput,
	ParetoRecord,
	TrialAttemptRecord,
	TrialDescription,
	TrialIdentity,
	WindowIdentity,
	canonical_json,
)
from .selection import SelectedCandidate, select_candidates
from .runtime import (
	RuntimeCompatibilityError,
	RuntimeCompatibilityReport,
	check_runtime_compatibility,
)
from .resources import (
	PathResourceRequest,
	PathResourceReport,
	require_path_resources,
)
from .events import (
	AttemptProgress,
	CancellationDetails,
	CancellationToken,
	ExecutionContext,
	FailureContext,
	ProgressInfrastructureError,
	ProgressEventWriter,
	ProgressSink,
	active_secret_values,
	bind_execution_context,
	redact_secrets,
)
from .outputs import (
	OutputBudgetLedger,
	OutputLimitExceeded,
	output_role_is_counted,
)


FailurePolicy = Literal["continue", "fail_fast"]


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


def _finish_timestamp(started_at: datetime) -> datetime:
	"""Return a wall-clock finish that cannot violate lifecycle ordering."""

	return max(_utcnow(), started_at)


def json_value(value: Any) -> JsonValue:
	"""Convert runtime artifacts to the experiment record's JSON value contract."""

	if hasattr(value, "model_dump"):
		return json_value(value.model_dump(mode="json"))
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {str(key): json_value(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [json_value(item) for item in value]
	if isinstance(value, np.generic):
		return value.item()
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	return repr(value)


def atomic_write_text(path: Path, text: str) -> None:
	"""Publish a text artifact atomically with a durable temporary file."""

	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
	try:
		with temporary.open("x", encoding="utf-8") as stream:
			stream.write(text)
			stream.flush()
			os.fsync(stream.fileno())
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def atomic_write_model(path: Path, value: Any) -> None:
	"""Publish a JSON model atomically so partial writes are never successful records."""

	payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
	atomic_write_text(path, json.dumps(json_value(payload), sort_keys=True, indent=2) + "\n")


class ExperimentInputs(ConfigModel):
	"""Identity-bearing inputs shared by selected candidates in one invocation."""

	dataset: DatasetIdentity
	configuration: ConfigurationIdentity
	code: CodeIdentity
	environment: EnvironmentIdentity
	description: TrialDescription = Field(default_factory=TrialDescription)


class ExperimentRunPolicy(ConfigModel):
	"""Invocation controls that do not change scientific trial identity."""

	failure: FailurePolicy = "continue"
	override_limits: bool = False
	estimated_output_bytes_per_trial: int | None = Field(default=None, ge=0)
	estimated_memory_bytes_per_trial: int | None = Field(default=None, gt=0)
	gpu_slots: int = Field(default=0, ge=0)


class ValidationFailure(ConfigModel):
	candidate_key: str
	message: str


class ExperimentPreflight(ConfigModel):
	"""Dry-run-visible deterministic execution plan and safety estimates."""

	experiment_id: str
	fidelity_id: str
	sampler: str
	seed: int
	requested_coordinates: int
	unique_candidates: int
	deduplicated_coordinates: int
	max_concurrency: int
	estimated_output_bytes: int | None
	trial_output_roots: tuple[Path, ...]
	candidates: tuple[dict[str, JsonValue], ...]
	validation_failures: tuple[ValidationFailure, ...] = ()
	prefix_cache: dict[str, JsonValue] | None = None
	runtime: RuntimeCompatibilityReport
	path_resources: PathResourceReport


class TrialExecution(ConfigModel):
	"""Executor result before #129 attempt/final-result wrapping."""

	metrics: dict[str, JsonValue] = Field(default_factory=dict)
	window_metrics: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
	artifacts: dict[str, JsonValue] = Field(default_factory=dict)
	outputs: tuple[OutputRecord, ...] = ()
	pending_outputs: tuple[PendingOutput, ...] = ()
	outputs_committed: bool = False
	output_accounting: OutputAccounting | None = None
	peak_memory_bytes: int | None = Field(
		default=None,
		ge=0,
		description=(
			"Executor-measured per-trial peak memory. When omitted, the runner records "
			"the process-wide RUSAGE_SELF high-water mark instead."
		),
	)


class LeaderboardEntry(ConfigModel):
	trial_id: str
	candidate_key: str
	status: Literal["succeeded", "failed", "interrupted", "skipped", "incompatible"]
	parameters: dict[str, JsonValue]
	metrics: dict[str, JsonValue]
	pareto: ParetoRecord = Field(default_factory=ParetoRecord)
	elapsed_seconds: float | None = Field(default=None, ge=0)
	peak_memory_bytes: int | None = Field(default=None, ge=0)
	output_root: Path
	failure: FailureRecord | None = None


class ExperimentRunSummary(ConfigModel):
	schema_version: Literal[1] = 1
	experiment_id: str
	preflight: ExperimentPreflight
	attempts: tuple[TrialAttemptRecord, ...]
	leaderboard: tuple[LeaderboardEntry, ...]
	output_accounting: OutputAccounting | None = None
	interruption: CancellationDetails | None = None


@dataclass(frozen=True)
class PreparedTrial:
	selected: SelectedCandidate
	identity: TrialIdentity
	output_root: Path
	attempt_id: str
	sampler_total: int
	sequence: int = 0
	previous_attempt_id: str | None = None


TrialExecutor = Callable[[TrialIdentity, Path], TrialExecution]
CandidateValidator = Callable[[Recon], None]
IdentityValidator = Callable[[TrialIdentity], None]


def _peak_memory_bytes() -> int:
	usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	# Linux reports KiB; macOS reports bytes.
	return int(usage if os.uname().sysname == "Darwin" else usage * 1024)


class IndependentExperimentRunner:
	"""Execute every selected trial with independent state and atomic records."""

	def __init__(
		self,
		config: ExperimentConfig,
		inputs: ExperimentInputs,
		root: str | Path,
		executor: TrialExecutor,
		*,
		validator: CandidateValidator | None = None,
		identity_validator: IdentityValidator | None = None,
		policy: ExperimentRunPolicy | None = None,
		fidelity_id: str | None = None,
		promotion_lineage: dict[str, Any] | None = None,
		runtime_report: RuntimeCompatibilityReport | None = None,
		path_resources: PathResourceRequest | None = None,
		event_writer: ProgressEventWriter | None = None,
		cancellation: CancellationToken | None = None,
		progress_sink: ProgressSink | None = None,
	):
		self.config = config
		self.inputs = inputs
		self.root = Path(root)
		self.executor = executor
		self.validator = validator
		self.identity_validator = identity_validator
		self.policy = policy or ExperimentRunPolicy()
		self.fidelity_id = fidelity_id or config.fidelities[0].id
		self.promotion_lineage = promotion_lineage or {}
		self.runtime_report = runtime_report
		self.path_resources = path_resources
		self.event_writer = event_writer
		self.cancellation = cancellation or CancellationToken()
		self.progress_sink = progress_sink or self._write_progress
		configure_output = getattr(self.executor, "configure_output_staging", None)
		if callable(configure_output):
			configure_output(
				result_root=self.root,
				scratch_root=(
					None if self.path_resources is None else self.path_resources.scratch_root
				),
				override_limits=self.policy.override_limits,
			)
		self.output_ledger = OutputBudgetLedger(
			self.root,
			self.config,
			self.fidelity_id,
			override_limits=self.policy.override_limits,
		)

	def _fidelity(self) -> FidelityIdentity:
		try:
			reference = next(item for item in self.config.fidelities if item.id == self.fidelity_id)
		except StopIteration as error:
			raise ValueError(f"unknown fidelity {self.fidelity_id!r}") from error
		windows = {item.id: item for item in self.config.windows}
		return FidelityIdentity(
			id=reference.id,
			windows=tuple(WindowIdentity(
				id=windows[name].id,
				role=windows[name].role,
				y_start=windows[name].y_start,
				y_stop=windows[name].y_stop,
				theta_start=windows[name].theta_start,
				theta_stop=windows[name].theta_stop,
				halo=windows[name].halo,
			) for name in reference.windows),
			projection_stride=reference.projection_stride,
			projection_limit=reference.projection_limit,
			resource_fraction=reference.resource_fraction,
			memory_gb=reference.memory_gb,
			max_output_bytes=reference.max_output_bytes,
			timeout_seconds=reference.timeout_seconds,
		)

	def _identity(self, selected: SelectedCandidate) -> TrialIdentity:
		protocols = tuple(MetricProtocolIdentity(
			metric_id=reference.id,
			protocol_version=reference.protocol_version,
			config_hash=hashlib.sha256(
				canonical_json(reference).encode("utf-8")
			).hexdigest(),
		) for reference in self.config.metrics)
		return TrialIdentity(
			candidate=CandidateKey(value=selected.candidate_key),
			resolved_recon=selected.recon.model_copy(deep=True),
			dataset=self.inputs.dataset,
			configuration=self.inputs.configuration,
			code=self.inputs.code,
			environment=self.inputs.environment,
			sampler=selected.sampler,
			fidelity=self._fidelity(),
			metric_protocols=protocols,
		)

	def _effective_workers(self) -> int:
		workers = self.config.resources.max_concurrency
		if self.config.resources.requires_gpu:
			if self.policy.gpu_slots < 1:
				raise ValueError(
					"experiment requires GPU trials but gpu_slots=0; declare available GPU slots"
				)
			workers = min(workers, self.policy.gpu_slots)
		memory = self.policy.estimated_memory_bytes_per_trial
		fidelity_memory = self._fidelity().memory_gb
		memory_limits = [
			value for value in (self.config.resources.memory_gb, fidelity_memory)
			if value is not None
		]
		if memory is not None and memory_limits:
			budget = int(min(memory_limits) * 1024**3)
			if memory > budget:
				raise ValueError(
					f"one trial requires an estimated {memory} bytes, exceeding memory budget {budget}"
				)
			workers = min(workers, max(1, budget // memory))
		return max(1, workers)

	def prepare(self) -> tuple[ExperimentPreflight, tuple[PreparedTrial, ...]]:
		"""Select, validate, identify, and guard trials without reading dataset payloads."""

		from lftomo.pipeline.stages import create_stage
		runtime = self.runtime_report or check_runtime_compatibility(
			self.config, backend=self.inputs.environment.backend,
		)
		self.runtime_report = runtime
		if not runtime.compatible:
			raise RuntimeCompatibilityError(runtime)

		selected = select_candidates(self.config)
		requested = (
			len(selected) + sum(item.duplicate_coordinates for item in selected)
		)
		prepared = []
		failures = []
		for item in selected:
			try:
				stages = [create_stage(stage) for stage in item.recon.pipeline]
				if not stages:
					raise ValueError("candidate pipeline must not be empty")
				if self.validator is not None:
					self.validator(item.recon.model_copy(deep=True))
			except Exception as error:
				failures.append(ValidationFailure(
					candidate_key=item.candidate_key,
					message=f"{type(error).__name__}: {error}",
				))
				continue
			identity = self._identity(item)
			if self.identity_validator is not None:
				try:
					self.identity_validator(identity.model_copy(deep=True))
				except Exception as error:
					failures.append(ValidationFailure(
						candidate_key=item.candidate_key,
						message=f"{type(error).__name__}: {error}",
					))
					continue
			trial_root = self.root / "trials" / identity.trial_id.value
			prepared.append(PreparedTrial(
				selected=item,
				identity=identity,
				output_root=trial_root,
				attempt_id=f"attempt-{identity.trial_id.value[:12]}-0000",
				sampler_total=requested,
			))
		estimated = self.policy.estimated_output_bytes_per_trial
		total_output = None if estimated is None else estimated * len(prepared)
		limits = [value for value in (
			self.config.resources.max_output_bytes,
			self.config.output.max_bytes,
			self._fidelity().max_output_bytes,
		) if value is not None]
		workers = self._effective_workers()
		prefix_planner = getattr(self.executor, "plan_prefix_cache", None)
		prefix_cache = None
		if callable(prefix_planner):
			prefix_cache = json_value(prefix_planner(tuple(prepared)))
		path_request = self.path_resources or PathResourceRequest(result_root=self.root)
		if (
			path_request.result_root.expanduser().resolve(strict=False)
			!= self.root.expanduser().resolve(strict=False)
		):
			raise ValueError("path resource result_root must match the experiment runner root")
		path_updates: dict[str, Any] = {}
		if path_request.estimated_result_bytes is None:
			path_updates["estimated_result_bytes"] = total_output
		if (
			path_request.result_guard_bytes is None
			and limits
			and not self.policy.override_limits
		):
			path_updates["result_guard_bytes"] = min(limits)
		if prefix_cache is not None:
			if path_request.cache_root is None and prefix_cache.get("scratch") is not None:
				path_updates["cache_root"] = Path(prefix_cache["scratch"])
			if path_request.estimated_cache_bytes is None:
				path_updates["estimated_cache_bytes"] = prefix_cache.get(
					"estimated_unique_snapshot_bytes"
				)
		path_request = path_request.model_copy(update=path_updates)
		self.path_resources = path_request
		path_report = require_path_resources(path_request)
		preflight = ExperimentPreflight(
			experiment_id=self.config.experiment_id,
			fidelity_id=self.fidelity_id,
			sampler=self.config.sampler.name,
			seed=self.config.sampler.seed,
			requested_coordinates=requested,
			unique_candidates=len(selected),
			deduplicated_coordinates=sum(item.duplicate_coordinates for item in selected),
			max_concurrency=workers,
			estimated_output_bytes=total_output,
			trial_output_roots=tuple(item.output_root for item in prepared),
			candidates=tuple({
				"candidate_key": item.selected.candidate_key,
				"trial_id": item.identity.trial_id.value,
				"parameters": json_value(item.selected.parameters),
				"sampler": json_value(item.selected.sampler),
				"stage_ids": [stage.id for stage in item.selected.recon.pipeline],
				"algorithms": [stage.algorithm for stage in item.selected.recon.pipeline],
			} for item in prepared),
			validation_failures=tuple(failures),
			prefix_cache=prefix_cache,
			runtime=runtime,
			path_resources=path_report,
		)
		return preflight, tuple(prepared)

	def _attempt_path(self, trial: PreparedTrial) -> Path:
		return trial.output_root / "attempts" / f"{trial.attempt_id}.json"

	def _progress_path(self, progress: AttemptProgress) -> Path:
		return (
			self.root / "trials" / progress.trial_id / "progress"
			/ f"{progress.attempt_id}.json"
		)

	def _write_progress(self, progress: AttemptProgress) -> None:
		atomic_write_model(self._progress_path(progress), progress)

	def _initial_progress(self, trial: PreparedTrial, now: datetime) -> AttemptProgress:
		return AttemptProgress(
			experiment_id=self.config.experiment_id,
			fidelity_id=trial.identity.fidelity.id,
			candidate_key=trial.identity.candidate.value,
			trial_id=trial.identity.trial_id.value,
			attempt_id=trial.attempt_id,
			structural_variant=trial.identity.sampler.explicit_case,
			sampler_index=trial.selected.index,
			sampler_position=trial.selected.index + 1,
			sampler_total=trial.sampler_total,
			state="planned",
			updated_at=now,
		)

	def _emit_attempt(
		self,
		kind: str,
		message: str,
		progress: AttemptProgress,
		**fields: Any,
	) -> None:
		self.progress_sink(progress)
		if self.event_writer is not None:
			self.event_writer.emit(
				kind,
				message,
				fidelity_id=progress.fidelity_id,
				candidate_key=progress.candidate_key,
				trial_id=progress.trial_id,
				attempt_id=progress.attempt_id,
				structural_variant=progress.structural_variant,
				sampler_index=progress.sampler_index,
				sampler_position=progress.sampler_position,
				sampler_total=progress.sampler_total,
				window_id=progress.window_id,
				window_position=progress.window_position,
				window_total=progress.window_total,
				window_elapsed_seconds=progress.window_elapsed_seconds,
				stage_occurrence_id=progress.stage_occurrence_id,
				stage_algorithm=progress.stage_algorithm,
				stage_position=progress.stage_position,
				stage_total=progress.stage_total,
				stage_elapsed_seconds=progress.stage_elapsed_seconds,
				lifecycle_state=progress.state,
				elapsed_seconds=progress.elapsed_seconds,
				resource_bytes=progress.resource_bytes,
				output_bytes=progress.output_bytes,
				cache_bytes=progress.cache_bytes,
				**fields,
			)

	def _execute(self, trial: PreparedTrial) -> TrialAttemptRecord:
		created = _utcnow()
		base = {
			"attempt_id": trial.attempt_id,
			"sequence": trial.sequence,
			"previous_attempt_id": trial.previous_attempt_id,
			"trial_id": trial.identity.trial_id,
			"identity": trial.identity,
			"description": self.inputs.description,
			"created_at": created,
		}
		pending = TrialAttemptRecord(**base)
		atomic_write_model(self._attempt_path(trial), pending)
		progress = self._initial_progress(trial, created)
		self._emit_attempt("trial.planned", "trial attempt planned", progress)
		started = _utcnow()
		running = TrialAttemptRecord(**base, status="running", started_at=started)
		atomic_write_model(self._attempt_path(trial), running)
		progress = progress.model_copy(update={
			"state": "started", "updated_at": started, "started_at": started,
		})
		self._emit_attempt("trial.started", "trial attempt started", progress)
		clock = time.perf_counter()
		memory_before = _peak_memory_bytes()
		context = ExecutionContext(
			writer=self.event_writer,
			progress_sink=self.progress_sink,
			cancellation=self.cancellation,
			progress=progress,
			clock_started=clock,
		)
		output_accounting = None
		pending_outputs: tuple[PendingOutput, ...] = ()
		try:
			self.cancellation.raise_if_requested()
			# The deep copy is deliberate: executor mutation can never leak into
			# another logical candidate or the recorded identity.
			execution_identity = trial.identity.model_copy(deep=True)
			with bind_execution_context(context):
				execution = self.executor(execution_identity, trial.output_root)
			legacy_pending = []
			retained_outputs = list(execution.outputs) if execution.outputs_committed else []
			for output in execution.outputs:
				if execution.outputs_committed:
					continue
				if not output_role_is_counted(output.role):
					retained_outputs.append(output)
					continue
				try:
					relative = output.path.resolve().relative_to(trial.output_root.resolve())
				except ValueError as error:
					raise RuntimeError(
						"budgeted executor output must be staged inside its trial root or "
						"returned through TrialExecution.pending_outputs"
					) from error
				legacy_pending.append(PendingOutput(
					role=output.role,
					temporary_path=output.path,
					relative_path=(
						Path("published") / trial.identity.fidelity.id
						/ trial.identity.trial_id.value / "legacy" / relative
					),
				))
			pending_outputs = tuple(execution.pending_outputs) + tuple(legacy_pending)
			if execution.outputs_committed:
				output_accounting = (
					execution.output_accounting
					or self.output_ledger.accounting_for_attempt(trial.attempt_id)
				)
			if pending_outputs:
				published, output_accounting = self.output_ledger.publish(
					pending_outputs,
					attempt_id=trial.attempt_id,
					trial_id=trial.identity.trial_id.value,
				)
				execution = execution.model_copy(update={
					"outputs": tuple(retained_outputs) + published,
					"pending_outputs": (),
				})
				context.output_event(
					"committed",
					f"committed {output_accounting.attempt_committed_bytes} output bytes",
					output_bytes=output_accounting.attempt_committed_bytes,
					details=output_accounting.model_dump(mode="json"),
				)
			elif output_accounting is None:
				output_accounting = self.output_ledger.snapshot()
			finished = _finish_timestamp(started)
			elapsed = time.perf_counter() - clock
			peak = execution.peak_memory_bytes or max(memory_before, _peak_memory_bytes())
			result = FinalResult(
				trial_id=trial.identity.trial_id,
				attempt_id=trial.attempt_id,
				finalized_at=finished,
				resolved_recon=trial.identity.resolved_recon,
				metrics=json_value(execution.metrics),
				outputs=execution.outputs,
				promotion=self.promotion_lineage.get(trial.identity.candidate.value),
			)
			attempt = TrialAttemptRecord(
				**base,
				status="succeeded",
				started_at=started,
				finished_at=finished,
				elapsed_seconds=elapsed,
				peak_memory_bytes=peak,
				output_accounting=output_accounting,
				observations={
					"metrics": json_value(execution.metrics),
					"artifacts": json_value(execution.artifacts),
					"window_metrics": json_value(execution.window_metrics),
				},
				result=result,
			)
		except ProgressInfrastructureError as error:
			self.output_ledger.rollback_attempt(trial.attempt_id)
			for item in pending_outputs:
				try:
					item.temporary_path.unlink(missing_ok=True)
				except OSError:
					pass
			# Progress/store plumbing is experiment infrastructure. It must abort the
			# coordinator instead of masquerading as an independent candidate failure.
			# Persist an interrupted attempt directly when possible so the authoritative
			# store does not retain a stale "started" projection after the abort.
			finished = _finish_timestamp(started)
			traceback_text = traceback.format_exc()
			secrets = (
				active_secret_values() if self.event_writer is None else self.event_writer.secrets
			)
			clean_message = str(redact_secrets(
				str(error) or type(error).__name__, secrets=secrets,
			))
			clean_traceback = str(redact_secrets(traceback_text, secrets=secrets))
			attempt = TrialAttemptRecord(
				**base,
				status="interrupted",
				started_at=started,
				finished_at=finished,
				elapsed_seconds=time.perf_counter() - clock,
				peak_memory_bytes=max(memory_before, _peak_memory_bytes()),
				failure=FailureRecord(
					type=type(error).__name__,
					message=clean_message,
					stage_id=context.progress.stage_occurrence_id,
					stage_algorithm=context.progress.stage_algorithm,
					window_id=context.progress.window_id,
					traceback=clean_traceback,
				),
			)
			atomic_write_model(self._attempt_path(trial), attempt)
			progress = context.progress.model_copy(update={
				"state": "interrupted",
				"updated_at": finished,
				"finished_at": finished,
				"elapsed_seconds": attempt.elapsed_seconds,
				"failure": FailureContext(
					type=type(error).__name__, message=clean_message,
					traceback=clean_traceback,
					window_id=context.progress.window_id,
					stage_occurrence_id=context.progress.stage_occurrence_id,
					stage_algorithm=context.progress.stage_algorithm,
				),
			})
			try:
				self.progress_sink(progress)
			except Exception as projection_error:
				# The infrastructure error may itself be a failed progress sink. The
				# attempt record above remains the best available durable projection.
				error.add_note(
					"authoritative attempt projection was written, but the progress "
					f"projection also failed: {type(projection_error).__name__}: "
					f"{projection_error}"
				)
			self.cancellation.request(f"progress infrastructure failure: {clean_message}")
			raise
		except BaseException as error:
			self.output_ledger.rollback_attempt(trial.attempt_id)
			for item in pending_outputs:
				try:
					item.temporary_path.unlink(missing_ok=True)
				except OSError:
					pass
			finished = _finish_timestamp(started)
			traceback_text = traceback.format_exc()
			secrets = (
				active_secret_values() if self.event_writer is None else self.event_writer.secrets
			)
			clean_message = str(redact_secrets(
				str(error) or type(error).__name__, secrets=secrets,
			))
			clean_traceback = str(redact_secrets(traceback_text, secrets=secrets))
			interrupted = isinstance(error, (KeyboardInterrupt, InterruptedError))
			details = {}
			if isinstance(error, OutputLimitExceeded):
				details = error.failure_details()
				output_accounting = self.output_ledger.snapshot()
				context.output_event(
					"limit_exceeded", clean_message,
					output_bytes=output_accounting.attempt_committed_bytes,
					details={
						**output_accounting.model_dump(mode="json"),
						**details,
					},
				)
			attempt = TrialAttemptRecord(
				**base,
				status="interrupted" if interrupted else "failed",
				started_at=started,
				finished_at=finished,
				elapsed_seconds=time.perf_counter() - clock,
				peak_memory_bytes=max(memory_before, _peak_memory_bytes()),
				output_accounting=output_accounting,
				failure=FailureRecord(
					type=type(error).__name__,
					message=clean_message,
					stage_id=context.progress.stage_occurrence_id,
					stage_algorithm=context.progress.stage_algorithm,
					window_id=context.progress.window_id,
					traceback=clean_traceback,
					details=details,
				),
			)
		atomic_write_model(self._attempt_path(trial), attempt)
		if attempt.status == "succeeded":
			self.output_ledger.finalize_attempt(trial.attempt_id)
			for item in pending_outputs:
				try:
					item.temporary_path.unlink(missing_ok=True)
				except OSError:
					pass
		final_state = attempt.status
		assert final_state in ("succeeded", "failed", "interrupted")
		failure = None
		if attempt.failure is not None:
			failure = FailureContext(
				type=attempt.failure.type,
				message=attempt.failure.message,
				traceback=attempt.failure.traceback,
				window_id=attempt.failure.window_id,
				stage_occurrence_id=attempt.failure.stage_id,
				stage_algorithm=attempt.failure.stage_algorithm,
			)
		output_bytes = None
		if attempt.result is not None:
			known_sizes = [item.bytes for item in attempt.result.outputs if item.bytes is not None]
			output_bytes = sum(known_sizes) if known_sizes else None
		progress = context.progress.model_copy(update={
			"state": final_state,
			"updated_at": attempt.finished_at,
			"finished_at": attempt.finished_at,
			"elapsed_seconds": attempt.elapsed_seconds,
			"resource_bytes": attempt.peak_memory_bytes,
			"output_bytes": output_bytes,
			"failure": failure,
		})
		self._emit_attempt(
			"trial.finalized",
			f"trial attempt {final_state}",
			progress,
			severity="error" if final_state == "failed" else (
				"warning" if final_state == "interrupted" else "info"
			),
			failure_type=None if failure is None else failure.type,
			failure_message=None if failure is None else failure.message,
			traceback=None if failure is None else failure.traceback,
		)
		return attempt

	def _rank(
		self,
		trials: tuple[PreparedTrial, ...],
		attempts: list[TrialAttemptRecord],
	) -> tuple[list[TrialAttemptRecord], tuple[LeaderboardEntry, ...]]:
		metrics = []
		for attempt in attempts:
			if attempt.status != "succeeded" or attempt.result is None:
				continue
			numeric = {
				name: float(value) if isinstance(value, (int, float)) else None
				for name, value in attempt.result.metrics.items()
			}
			metrics.append(CandidateMetrics(
				candidate_key=attempt.identity.candidate.value,
				metrics=numeric,
			))
		directions = {item.id: item.direction for item in self.config.metrics}
		objectives = [ParetoObjective(
			metric=item.metric_id,
			direction=directions[item.metric_id],
		) for item in self.config.objectives if item.role == "objective"]
		constraints = [HardConstraint(
			metric=item.metric_id, relation="<=", threshold=item.threshold,
		) for item in self.config.objectives if item.role == "constraint"]
		outcomes = {
			item.candidate_key: item
			for item in pareto_rank(metrics, objectives, constraints)
		} if metrics and objectives else {}
		updated = []
		for attempt in attempts:
			outcome = outcomes.get(attempt.identity.candidate.value)
			if outcome is None or attempt.result is None:
				updated.append(attempt)
				continue
			pareto = ParetoRecord(
				rank=outcome.rank,
				non_dominated=outcome.non_dominated,
				constraints_satisfied=outcome.constraints_satisfied,
			)
			document = attempt.model_dump(mode="json")
			document["pareto"] = pareto.model_dump(mode="json")
			document["result"]["pareto"] = pareto.model_dump(mode="json")
			updated_attempt = TrialAttemptRecord.model_validate(document)
			updated.append(updated_attempt)
			trial = next(item for item in trials if item.identity.trial_id == attempt.trial_id)
			atomic_write_model(
				trial.output_root / "attempts" / f"{attempt.attempt_id}.json",
				updated_attempt,
			)

		parameters = {item.identity.trial_id.value: item.selected.parameters for item in trials}
		roots = {item.identity.trial_id.value: item.output_root for item in trials}
		leaderboard = tuple(sorted((LeaderboardEntry(
			trial_id=attempt.trial_id.value,
			candidate_key=attempt.identity.candidate.value,
			status=attempt.status,
			parameters=json_value(parameters[attempt.trial_id.value]),
			metrics={} if attempt.result is None else attempt.result.metrics,
			pareto=attempt.pareto,
			elapsed_seconds=attempt.elapsed_seconds,
			peak_memory_bytes=attempt.peak_memory_bytes,
			output_root=roots[attempt.trial_id.value],
			failure=attempt.failure,
		) for attempt in updated), key=lambda item: (
			item.status != "succeeded",
			item.pareto.rank if item.pareto.rank is not None else 10**9,
			item.trial_id,
		)))
		return updated, leaderboard

	def run(self) -> ExperimentRunSummary:
		preflight, trials = self.prepare()
		self.output_ledger.reconcile()
		if preflight.validation_failures:
			raise ValueError(
				"candidate preflight failed: "
				+ "; ".join(item.message for item in preflight.validation_failures)
			)
		attempts: list[TrialAttemptRecord] = []
		workers = preflight.max_concurrency
		unstarted = 0
		with ThreadPoolExecutor(max_workers=workers) as pool:
			remaining = iter(trials)
			future_to_trial: dict[Future, PreparedTrial] = {}
			for _ in range(min(workers, len(trials))):
				trial = next(remaining)
				future_to_trial[pool.submit(self._execute, trial)] = trial
			stop = False
			while future_to_trial:
				done, _ = wait(future_to_trial, return_when=FIRST_COMPLETED)
				for future in done:
					future_to_trial.pop(future)
					attempt = future.result()
					attempts.append(attempt)
					if self.policy.failure == "fail_fast" and attempt.status != "succeeded":
						stop = True
				if self.cancellation.requested:
					stop = True
				while not stop and len(future_to_trial) < workers:
					try:
						trial = next(remaining)
					except StopIteration:
						break
					future_to_trial[pool.submit(self._execute, trial)] = trial
			if stop:
				unstarted = sum(1 for _ in remaining)
		attempts.sort(key=lambda item: item.trial_id.value)
		attempts, leaderboard = self._rank(trials, attempts)
		manifest = ExperimentManifest(
			experiment_id=self.config.experiment_id,
			config=self.config,
			created_at=_utcnow(),
			attempt_ids=tuple(item.attempt_id for item in attempts),
			metadata={
				"runtime_compatibility": json_value(preflight.runtime),
				"path_resource_preflight": json_value(preflight.path_resources),
				"output_accounting": json_value(self.output_ledger.snapshot()),
				"override_limits": self.policy.override_limits,
			},
		)
		atomic_write_model(self.root / "manifest.json", manifest)
		summary = ExperimentRunSummary(
			experiment_id=self.config.experiment_id,
			preflight=preflight,
			attempts=tuple(attempts),
			leaderboard=leaderboard,
			output_accounting=self.output_ledger.snapshot(),
			interruption=CancellationDetails(
				requested=True,
				reason=self.cancellation.reason,
				requested_at=self.cancellation.requested_at,
				interrupted_attempt_ids=tuple(
					item.attempt_id for item in attempts if item.status == "interrupted"
				),
				unstarted_candidates=unstarted,
			) if self.cancellation.requested else None,
		)
		self._write_outputs(summary)
		return summary

	def _write_outputs(self, summary: ExperimentRunSummary) -> None:
		payload = summary.model_dump(mode="json")
		atomic_write_model(self.root / "experiment-results.json", payload)
		atomic_write_text(
			self.root / "experiment-results.yaml",
			yaml.safe_dump(json_value(payload), sort_keys=False),
		)
		lines = [
			"| rank | status | trial | candidate | metrics | output |",
			"| ---: | --- | --- | --- | --- | --- |",
		]
		for entry in summary.leaderboard:
			lines.append(
				f"| {entry.pareto.rank if entry.pareto.rank is not None else '-'} "
				f"| {entry.status} | `{entry.trial_id[:12]}` | "
				f"`{entry.candidate_key[:12]}` | `{canonical_json(entry.metrics)}` "
				f"| `{entry.output_root}` |"
			)
		atomic_write_text(self.root / "leaderboard.md", "\n".join(lines) + "\n")
