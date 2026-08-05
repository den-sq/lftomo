"""Independent-trial experiment execution, records, and deterministic leaderboards."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time
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
	OutputRecord,
	ParetoRecord,
	TrialAttemptRecord,
	TrialDescription,
	TrialIdentity,
	WindowIdentity,
	canonical_json,
)
from .selection import SelectedCandidate, select_candidates


FailurePolicy = Literal["continue", "fail_fast"]


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


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


def _atomic_text(path: Path, text: str) -> None:
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
	_atomic_text(path, json.dumps(json_value(payload), sort_keys=True, indent=2) + "\n")


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


class TrialExecution(ConfigModel):
	"""Executor result before #129 attempt/final-result wrapping."""

	metrics: dict[str, JsonValue] = Field(default_factory=dict)
	artifacts: dict[str, JsonValue] = Field(default_factory=dict)
	outputs: tuple[OutputRecord, ...] = ()
	peak_memory_bytes: int | None = Field(default=None, ge=0)


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


@dataclass(frozen=True)
class PreparedTrial:
	selected: SelectedCandidate
	identity: TrialIdentity
	output_root: Path
	attempt_id: str


TrialExecutor = Callable[[TrialIdentity, Path], TrialExecution]
CandidateValidator = Callable[[Recon], None]


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
		policy: ExperimentRunPolicy | None = None,
		fidelity_id: str | None = None,
	):
		self.config = config
		self.inputs = inputs
		self.root = Path(root)
		self.executor = executor
		self.validator = validator
		self.policy = policy or ExperimentRunPolicy()
		self.fidelity_id = fidelity_id or config.fidelities[0].id

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
				y_start=windows[name].y_start,
				y_stop=windows[name].y_stop,
				theta_start=windows[name].theta_start,
				theta_stop=windows[name].theta_stop,
				halo=windows[name].halo,
			) for name in reference.windows),
			projection_stride=reference.projection_stride,
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
		if memory is not None and self.config.resources.memory_gb is not None:
			budget = int(self.config.resources.memory_gb * 1024**3)
			if memory > budget:
				raise ValueError(
					f"one trial requires an estimated {memory} bytes, exceeding memory budget {budget}"
				)
			workers = min(workers, max(1, budget // memory))
		return max(1, workers)

	def prepare(self) -> tuple[ExperimentPreflight, tuple[PreparedTrial, ...]]:
		"""Select, validate, identify, and guard trials without reading dataset payloads."""

		from lftomo.pipeline.stages import create_stage

		selected = select_candidates(self.config)
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
			trial_root = self.root / "trials" / identity.trial_id.value
			prepared.append(PreparedTrial(
				selected=item,
				identity=identity,
				output_root=trial_root,
				attempt_id=f"attempt-{identity.trial_id.value[:12]}-0000",
			))
		requested = (
			len(selected) + sum(item.duplicate_coordinates for item in selected)
		)
		estimated = self.policy.estimated_output_bytes_per_trial
		total_output = None if estimated is None else estimated * len(prepared)
		limits = [value for value in (
			self.config.resources.max_output_bytes,
			self.config.output.max_bytes,
		) if value is not None]
		if limits and total_output is not None and total_output > min(limits):
			if not self.policy.override_limits:
				raise ValueError(
					f"screening output estimate {total_output} bytes exceeds limit {min(limits)}; "
					"reduce trials/output or rerun with override_limits=True"
				)
		workers = self._effective_workers()
		preflight = ExperimentPreflight(
			experiment_id=self.config.experiment_id,
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
		)
		return preflight, tuple(prepared)

	def _attempt_path(self, trial: PreparedTrial) -> Path:
		return trial.output_root / "attempts" / f"{trial.attempt_id}.json"

	def _execute(self, trial: PreparedTrial) -> TrialAttemptRecord:
		created = _utcnow()
		base = {
			"attempt_id": trial.attempt_id,
			"sequence": 0,
			"trial_id": trial.identity.trial_id,
			"identity": trial.identity,
			"description": self.inputs.description,
			"created_at": created,
		}
		pending = TrialAttemptRecord(**base)
		atomic_write_model(self._attempt_path(trial), pending)
		started = _utcnow()
		running = TrialAttemptRecord(**base, status="running", started_at=started)
		atomic_write_model(self._attempt_path(trial), running)
		clock = time.perf_counter()
		memory_before = _peak_memory_bytes()
		try:
			# The deep copy is deliberate: executor mutation can never leak into
			# another logical candidate or the recorded identity.
			execution_identity = trial.identity.model_copy(deep=True)
			execution = self.executor(execution_identity, trial.output_root)
			finished = _utcnow()
			elapsed = time.perf_counter() - clock
			peak = execution.peak_memory_bytes or max(memory_before, _peak_memory_bytes())
			result = FinalResult(
				trial_id=trial.identity.trial_id,
				attempt_id=trial.attempt_id,
				finalized_at=finished,
				resolved_recon=trial.identity.resolved_recon,
				metrics=json_value(execution.metrics),
				outputs=execution.outputs,
			)
			attempt = TrialAttemptRecord(
				**base,
				status="succeeded",
				started_at=started,
				finished_at=finished,
				elapsed_seconds=elapsed,
				peak_memory_bytes=peak,
				observations={
					"metrics": json_value(execution.metrics),
					"artifacts": json_value(execution.artifacts),
				},
				result=result,
			)
		except BaseException as error:
			finished = _utcnow()
			interrupted = isinstance(error, (KeyboardInterrupt, InterruptedError))
			attempt = TrialAttemptRecord(
				**base,
				status="interrupted" if interrupted else "failed",
				started_at=started,
				finished_at=finished,
				elapsed_seconds=time.perf_counter() - clock,
				peak_memory_bytes=max(memory_before, _peak_memory_bytes()),
				failure=FailureRecord(
					type=type(error).__name__,
					message=str(error) or type(error).__name__,
				),
			)
		atomic_write_model(self._attempt_path(trial), attempt)
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
			atomic_write_model(self._attempt_path(trial), updated_attempt)

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
		if preflight.validation_failures:
			raise ValueError(
				"candidate preflight failed: "
				+ "; ".join(item.message for item in preflight.validation_failures)
			)
		attempts: list[TrialAttemptRecord] = []
		workers = preflight.max_concurrency
		with ThreadPoolExecutor(max_workers=workers) as pool:
			future_to_trial: dict[Future, PreparedTrial] = {
				pool.submit(self._execute, trial): trial for trial in trials
			}
			for future in as_completed(future_to_trial):
				attempt = future.result()
				attempts.append(attempt)
				if self.policy.failure == "fail_fast" and attempt.status != "succeeded":
					for pending in future_to_trial:
						pending.cancel()
					break
		attempts.sort(key=lambda item: item.trial_id.value)
		attempts, leaderboard = self._rank(trials, attempts)
		manifest = ExperimentManifest(
			experiment_id=self.config.experiment_id,
			config=self.config,
			created_at=_utcnow(),
			attempt_ids=tuple(item.attempt_id for item in attempts),
		)
		atomic_write_model(self.root / "manifest.json", manifest)
		summary = ExperimentRunSummary(
			experiment_id=self.config.experiment_id,
			preflight=preflight,
			attempts=tuple(attempts),
			leaderboard=leaderboard,
		)
		self._write_outputs(summary)
		return summary

	def _write_outputs(self, summary: ExperimentRunSummary) -> None:
		payload = summary.model_dump(mode="json")
		atomic_write_model(self.root / "experiment-results.json", payload)
		_atomic_text(
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
		_atomic_text(self.root / "leaderboard.md", "\n".join(lines) + "\n")
