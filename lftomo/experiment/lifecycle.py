"""Stable public API for experiment validation, execution, and recovery."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import traceback
from typing import Any, Callable, Literal

from pydantic import Field, JsonValue
import yaml

from lftomo.config import ConfigModel, Scan
from lftomo.evaluation import validate_experiment_metric_references

from .events import (
	CancellationToken,
	ExperimentStatus,
	ProgressCallback,
	ProgressEventWriter,
	active_secret_values,
	redact_secrets,
	secret_material_paths,
)
from .fidelity import FidelityExperimentRunner, FidelityRunSummary
from .models import ExperimentConfig
from .runner import (
	ExperimentInputs,
	ExperimentPreflight,
	ExperimentRunPolicy,
	ExperimentRunSummary,
	IndependentExperimentRunner,
	atomic_write_model,
	atomic_write_text,
)
from .resources import PathResourceRequest
from .store import PersistedExperimentRunner, PersistedExperimentStore, ResumePolicy, StoreInspection


class ExperimentValidation(ConfigModel):
	"""Structured result of schema and cross-reference validation."""

	schema_version: Literal[1] = 1
	valid: Literal[True] = True
	experiment_id: str
	sampler: dict[str, JsonValue]


class FidelityPreflight(ConfigModel):
	"""Structured preflight plans for a multi-fidelity invocation."""

	schema_version: Literal[1] = 1
	experiment_id: str
	levels: dict[str, ExperimentPreflight]


class ReplayBundle(ConfigModel):
	"""Self-contained non-secret invocation facts for compatible replay."""

	schema_version: Literal[1] = 1
	experiment: ExperimentConfig
	inputs: ExperimentInputs
	run_policy: ExperimentRunPolicy
	resume_policy: ResumePolicy
	path_resources: dict[str, JsonValue]
	executor: dict[str, JsonValue]
	key_fingerprints: dict[str, str] = Field(default_factory=dict)


def load_experiment(path: str | Path) -> ExperimentConfig:
	"""Load and fully validate a versioned experiment YAML document."""

	with Path(path).open("r", encoding="utf-8") as stream:
		config = ExperimentConfig.model_validate(yaml.safe_load(stream))
	validate_experiment_metric_references(config)
	return config


def validate_experiment(config: ExperimentConfig | str | Path) -> ExperimentValidation:
	"""Validate a model or YAML path without reading projection payloads."""

	if not isinstance(config, ExperimentConfig):
		config = load_experiment(config)
	else:
		validate_experiment_metric_references(config)
	return ExperimentValidation(
		experiment_id=config.experiment_id,
		sampler=config.sampler.model_dump(mode="json"),
	)


class ExperimentLifecycle:
	"""Public controller shared by Click and external callers such as hatmehyt."""

	def __init__(
		self,
		runner: IndependentExperimentRunner,
		*,
		callback: ProgressCallback | None = None,
		cancellation: CancellationToken | None = None,
		secrets: tuple[str, ...] | None = None,
	):
		self.runner = runner
		self.root = runner.root
		self.cancellation = cancellation or runner.cancellation
		active_secrets = active_secret_values() if secrets is None else secrets
		self.events = runner.event_writer or ProgressEventWriter(
			self.root,
			runner.config.experiment_id,
			callback=callback,
			secrets=active_secrets,
		)
		if runner.event_writer is not None and callback is not None:
			runner.event_writer.callback = callback
		runner.event_writer = self.events
		runner.cancellation = self.cancellation

	@classmethod
	def from_replay(
		cls,
		root: str | Path,
		*,
		callback: ProgressCallback | None = None,
		cancellation: CancellationToken | None = None,
		path_resources: PathResourceRequest | None = None,
		executor_factory: Callable[
			[ExperimentConfig, Path, dict[str, JsonValue]], Any
		] | None = None,
	) -> "ExperimentLifecycle":
		"""Rebuild an invocation without the original config files or session."""

		root = Path(root)
		bundle = ReplayBundle.model_validate_json(
			(root / "invocation" / "replay-bundle.json").read_text(encoding="utf-8")
		)
		if bundle.executor.get("kind") != "pipeline":
			raise ValueError(f"unsupported replay executor {bundle.executor.get('kind')!r}")
		# Imported lazily to keep the public lifecycle module independent of Click
		# initialization while the complete-pipeline adapter remains reusable.
		scan_path = root / "invocation" / "normalized-scan.yaml"
		executor_document = dict(bundle.executor)
		if path_resources is not None and path_resources.cache_root is not None:
			cache_policy = dict(executor_document.get("prefix_cache_policy", {}))
			if cache_policy.get("enabled"):
				cache_policy["scratch"] = str(path_resources.cache_root)
				executor_document["prefix_cache_policy"] = cache_policy
		if executor_factory is None:
			from .cli import PipelineTrialExecutor

			executor = PipelineTrialExecutor.from_replay(
				bundle.experiment, scan_path, executor_document,
			)
		else:
			executor = executor_factory(bundle.experiment, scan_path, executor_document)

		runner = IndependentExperimentRunner(
			bundle.experiment,
			bundle.inputs,
			root,
			executor,
			validator=getattr(executor, "validate", None),
			identity_validator=getattr(executor, "validate_identity", None),
			policy=bundle.run_policy,
			path_resources=path_resources or PathResourceRequest.model_validate(
				bundle.path_resources
			),
		)
		return cls(runner, callback=callback, cancellation=cancellation)

	def validate(self) -> ExperimentValidation:
		self.events.emit("config.validation.started", "experiment validation started")
		result = validate_experiment(self.runner.config)
		self.events.emit(
			"config.validation.completed", "experiment validation completed",
			details=result.model_dump(mode="json"),
		)
		return result

	def _emit_invocation_checks(self) -> None:
		executor_document = self._executor_document()
		unsafe = secret_material_paths(
			{
				"experiment": self.runner.config,
				"inputs": self.runner.inputs,
				"executor": executor_document,
			},
			secrets=self.events.secrets,
		)
		if unsafe:
			raise ValueError(
				"secret material is not allowed in persisted experiment inputs; "
				f"supply it externally instead: {', '.join(unsafe)}"
			)
		self.events.emit("config.started", "experiment configuration check started")
		validation = validate_experiment(self.runner.config)
		self.events.emit(
			"config.completed", "experiment configuration check completed",
			details=validation.model_dump(mode="json"),
		)
		self.events.emit("environment.started", "runtime environment check started")
		self.events.emit(
			"environment.completed", "runtime environment check completed",
			details=self.runner.inputs.environment.model_dump(mode="json"),
		)

	def preflight(self) -> ExperimentPreflight | FidelityPreflight:
		"""Return a structured dry-run plan without creating the result root."""

		self._emit_invocation_checks()
		self.events.emit("preflight.started", "experiment preflight started")
		if self.runner.config.promotion is None:
			preflight, _ = self.runner.prepare()
			self._emit_preflight_completed(preflight)
			return preflight
		levels = {}
		for fidelity in self.runner.config.fidelities:
			level_runner = IndependentExperimentRunner(
				self.runner.config,
				self.runner.inputs,
				self.runner.root,
				self.runner.executor,
				validator=self.runner.validator,
				identity_validator=self.runner.identity_validator,
				policy=self.runner.policy,
				fidelity_id=fidelity.id,
				path_resources=self.runner.path_resources,
				runtime_report=self.runner.runtime_report,
				event_writer=self.events,
				cancellation=self.cancellation,
			)
			level, _ = level_runner.prepare()
			levels[fidelity.id] = level
			self._emit_preflight_completed(level)
		return FidelityPreflight(
			experiment_id=self.runner.config.experiment_id,
			levels=levels,
		)

	def _emit_preflight_completed(self, preflight: ExperimentPreflight) -> None:
		self.events.emit(
			"preflight.completed", "experiment preflight completed",
			fidelity_id=preflight.fidelity_id,
			total_candidates=preflight.unique_candidates,
			details={
				"runtime": preflight.runtime.model_dump(mode="json"),
				"path_resources": preflight.path_resources.model_dump(mode="json"),
				"parallelism": preflight.parallelism,
			},
		)
		if preflight.prefix_cache is not None:
			self.events.emit(
				"cache.plan", "prefix cache plan completed",
				fidelity_id=preflight.fidelity_id,
				details=preflight.prefix_cache,
			)

	def _executor_document(self) -> dict[str, JsonValue]:
		producer = getattr(self.runner.executor, "replay_document", None)
		if not callable(producer):
			return {"kind": "external", "replayable": False}
		return producer()

	def _persist_replay(self, policy: ResumePolicy) -> None:
		executor_document = self._executor_document()
		if executor_document.get("kind") == "external":
			# The public engine still works with custom executors, but cannot claim
			# their application-specific constructor is replayable.
			return
		unsafe = secret_material_paths(executor_document, secrets=self.events.secrets)
		if unsafe:
			raise ValueError(
				"secret material is not allowed in replayable executor configuration; "
				f"supply it externally instead: {', '.join(unsafe)}"
			)
		key_fingerprints = {}
		blinding_key = os.environ.get("LFTOMO_PREVIEW_BLINDING_KEY")
		if blinding_key:
			key_fingerprints["preview_blinding"] = hashlib.sha256(
				blinding_key.encode("utf-8")
			).hexdigest()[:16]
		path_resources = self.runner.path_resources or PathResourceRequest(
			result_root=self.root,
		)
		document = {
			"experiment": self.runner.config,
			"inputs": self.runner.inputs,
			"run_policy": self.runner.policy,
			"resume_policy": policy,
			"path_resources": path_resources,
			"executor": executor_document,
			"key_fingerprints": key_fingerprints,
		}
		clean = redact_secrets(document, secrets=self.events.secrets)
		bundle = ReplayBundle.model_validate(clean)
		invocation = self.root / "invocation"
		atomic_write_model(invocation / "replay-bundle.json", bundle)
		atomic_write_text(
			invocation / "normalized-experiment.yaml", bundle.experiment.to_yaml(),
		)
		scan_document = bundle.executor.get("scan")
		if isinstance(scan_document, dict):
			scan = Scan.model_validate(scan_document)
			atomic_write_text(invocation / "normalized-scan.yaml", scan.to_yaml())

	def _before_execute(self, policy: ResumePolicy) -> Callable[[ExperimentPreflight], None]:
		def callback(preflight: ExperimentPreflight) -> None:
			self._emit_preflight_completed(preflight)
			self._persist_replay(policy)
		return callback

	def _execute(
		self,
		policy: ResumePolicy,
		*,
		trial_ids: set[str] | None = None,
		invalidate_stale_promotions: bool = False,
	) -> ExperimentRunSummary | FidelityRunSummary:
		self._emit_invocation_checks()
		self.events.emit("preflight.started", "experiment preflight started")
		try:
			if self.runner.config.promotion is not None:
				if trial_ids:
					raise ValueError("trial-id filtering is not supported for multi-fidelity runs")
				summary = FidelityExperimentRunner(
					self.runner.config,
					self.runner.inputs,
					self.runner.root,
					self.runner.executor,
					validator=self.runner.validator,
					identity_validator=self.runner.identity_validator,
					run_policy=self.runner.policy,
					resume_policy=policy,
					invalidate_stale_promotions=invalidate_stale_promotions,
					path_resources=self.runner.path_resources,
					runtime_report=self.runner.runtime_report,
					event_writer=self.events,
					cancellation=self.cancellation,
					before_execute=self._before_execute(policy),
				).run()
			else:
				summary = PersistedExperimentRunner(
					self.runner,
					policy=policy,
					trial_ids=trial_ids,
					event_writer=self.events,
					cancellation=self.cancellation,
					before_execute=self._before_execute(policy),
				).run()
		except BaseException as error:
			self.events.activate()
			self.events.emit(
				"experiment.failed", str(error) or type(error).__name__,
				severity="error", failure_type=type(error).__name__,
				failure_message=str(error) or type(error).__name__,
				traceback=traceback.format_exc(),
				lifecycle_state=(
					"interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError))
					else "failed"
				),
			)
			raise
		interrupted = summary.interruption is not None
		if isinstance(summary, FidelityRunSummary):
			completed_candidates = sum(len(level.attempts) for level in summary.levels)
			total_candidates = sum(
				len(level.attempts) + (
					0 if level.interruption is None
					else level.interruption.unstarted_candidates
				)
				for level in summary.levels
			)
		else:
			completed_candidates = len(summary.attempts)
			total_candidates = len(summary.attempts) + (
				0 if summary.interruption is None
				else summary.interruption.unstarted_candidates
			)
		attempts = (
			[attempt for level in summary.levels for attempt in level.attempts]
			if isinstance(summary, FidelityRunSummary)
			else list(summary.attempts)
		)
		has_terminal_failures = any(
			attempt.status != "succeeded" for attempt in attempts
		)
		message = (
			"experiment interrupted" if interrupted
			else "experiment completed with unsuccessful candidates"
			if has_terminal_failures
			else "experiment completed"
		)
		self.events.emit(
			"experiment.completed",
			message,
			severity="warning" if interrupted or has_terminal_failures else "info",
			lifecycle_state=(
				"interrupted" if interrupted
				else "failed" if has_terminal_failures
				else "succeeded"
			),
			completed_candidates=completed_candidates,
			total_candidates=total_candidates,
		)
		self.events.flush()
		return summary

	def run(self) -> ExperimentRunSummary | FidelityRunSummary:
		return self._execute(ResumePolicy())

	def resume(
		self,
		policy: ResumePolicy | None = None,
		*,
		invalidate_stale_promotions: bool = False,
	) -> ExperimentRunSummary | FidelityRunSummary:
		return self._execute(
			policy or ResumePolicy(),
			invalidate_stale_promotions=invalidate_stale_promotions,
		)

	def rerun(
		self,
		policy: ResumePolicy | None = None,
		*,
		trial_ids: set[str] | None = None,
		invalidate_stale_promotions: bool = False,
	) -> ExperimentRunSummary | FidelityRunSummary:
		base = policy or ResumePolicy()
		forced = base.model_copy(update={"force": True})
		return self._execute(
			forced,
			trial_ids=trial_ids,
			invalidate_stale_promotions=invalidate_stale_promotions,
		)

	def request_cancellation(self, reason: str = "cancellation requested") -> bool:
		requested = self.cancellation.request(reason)
		if requested:
			self.events.emit(
				"cancellation.requested", reason,
				severity="warning", lifecycle_state="interrupted",
			)
		return requested

	def inspect(self) -> StoreInspection:
		return PersistedExperimentStore(self.root).inspect()

	def status(self) -> ExperimentStatus:
		return PersistedExperimentStore(self.root).status()

	@staticmethod
	def inspect_root(root: str | Path) -> StoreInspection:
		return PersistedExperimentStore(root).inspect()

	@staticmethod
	def status_root(root: str | Path) -> ExperimentStatus:
		return PersistedExperimentStore(root).status()
