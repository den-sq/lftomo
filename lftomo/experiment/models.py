"""Versioned experiment configuration, identity, and manifest models."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, model_validator

from lftomo.config import ConfigModel, Recon
from lftomo.pipeline import candidate_key as resolved_candidate_key
from lftomo.pipeline import has_continuous_domains


SchemaVersion = Literal[1]
HashValue = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
StableName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")]


def canonical_json(value: Any) -> str:
	"""Serialize a model or JSON value canonically for hashing and comparison."""

	if hasattr(value, "model_dump"):
		value = value.model_dump(mode="json")
	return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
	return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class SamplerSpec(ConfigModel):
	"""A search policy reference; candidate-space syntax remains in Recon."""

	name: Literal["exhaustive", "sobol", "random"]
	seed: int = 0
	budget: int | None = Field(default=None, gt=0)

	@model_validator(mode="after")
	def _budget_is_explicit_for_sampling(self):
		if self.name != "exhaustive" and self.budget is None:
			raise ValueError(f"{self.name} sampling requires an explicit budget")
		return self


class ResourcePolicy(ConfigModel):
	"""Experiment-level concurrency and resource limits."""

	max_concurrency: int = Field(default=1, gt=0)
	memory_gb: float | None = Field(default=None, gt=0)
	requires_gpu: bool = False
	max_output_bytes: int | None = Field(default=None, gt=0)


class MetricReference(ConfigModel):
	"""Versioned reference to a metric implementation supplied by evaluation."""

	id: StableName
	metric: str = Field(min_length=1)
	protocol_version: str = Field(min_length=1)
	direction: Literal["minimize", "maximize", "constraint"]
	units: str | None = None


class ObjectiveReference(ConfigModel):
	"""How a referenced metric participates in experiment comparison."""

	metric_id: StableName
	role: Literal["objective", "constraint"] = "objective"
	weight: float | None = None
	threshold: float | None = None

	@model_validator(mode="after")
	def _role_fields_match(self):
		if self.role == "constraint" and self.threshold is None:
			raise ValueError("constraint objectives require a threshold")
		if self.role == "objective" and self.threshold is not None:
			raise ValueError("objective thresholds require role='constraint'")
		return self


class WindowReference(ConfigModel):
	"""An explicit detector/projection window referenced by a fidelity level."""

	id: StableName
	y_start: int = Field(ge=0)
	y_stop: int = Field(gt=0)
	theta_start: int | None = Field(default=None, ge=0)
	theta_stop: int | None = Field(default=None, gt=0)
	halo: int = Field(default=0, ge=0)

	@model_validator(mode="after")
	def _stops_follow_starts(self):
		if self.y_stop <= self.y_start:
			raise ValueError("window y_stop must be greater than y_start")
		if (
			self.theta_start is not None
			and self.theta_stop is not None
			and self.theta_stop <= self.theta_start
		):
			raise ValueError("window theta_stop must be greater than theta_start")
		return self


class FidelityReference(ConfigModel):
	"""A named set of windows and sampling limits."""

	id: StableName
	windows: list[StableName] = Field(min_length=1)
	projection_stride: int = Field(default=1, gt=0)
	resource_fraction: float = Field(default=1.0, gt=0, le=1)

	@model_validator(mode="after")
	def _windows_are_unique(self):
		if len(self.windows) != len(set(self.windows)):
			raise ValueError("fidelity windows must be unique")
		return self


class AggregationPolicy(ConfigModel):
	"""Reference-only window aggregation policy for later evaluation."""

	name: Literal["mean", "median", "weighted_mean", "none"] = "mean"
	metric_ids: list[StableName] = Field(min_length=1)
	window_weights: dict[StableName, float] = Field(default_factory=dict)
	invalid: Literal["fail", "exclude", "worst"] = "fail"


class OutputPolicy(ConfigModel):
	"""Screening/final output contract, without implementing promotion."""

	screening: Literal["metrics_only", "metrics_and_previews", "configured_writers"] = "metrics_only"
	final_fidelity_id: StableName | None = None
	write_full_finalists: bool = False
	max_bytes: int | None = Field(default=None, gt=0)

	@model_validator(mode="after")
	def _finalist_fields_match(self):
		if self.write_full_finalists and self.final_fidelity_id is None:
			raise ValueError("full finalist output requires final_fidelity_id")
		return self


class ExperimentConfig(ConfigModel):
	"""Versioned envelope around the canonical Recon candidate space."""

	schema_version: SchemaVersion = 1
	name: StableName
	candidate_space: Recon
	sampler: SamplerSpec
	resources: ResourcePolicy = Field(default_factory=ResourcePolicy)
	metrics: list[MetricReference] = Field(min_length=1)
	objectives: list[ObjectiveReference] = Field(min_length=1)
	windows: list[WindowReference] = Field(min_length=1)
	fidelities: list[FidelityReference] = Field(min_length=1)
	aggregation: AggregationPolicy
	output: OutputPolicy = Field(default_factory=OutputPolicy)
	metadata: dict[str, JsonValue] = Field(default_factory=dict)

	@model_validator(mode="after")
	def _references_are_valid(self):
		metric_ids = [item.id for item in self.metrics]
		if len(metric_ids) != len(set(metric_ids)):
			raise ValueError("metric reference IDs must be unique")
		window_ids = [item.id for item in self.windows]
		if len(window_ids) != len(set(window_ids)):
			raise ValueError("window reference IDs must be unique")
		fidelity_ids = [item.id for item in self.fidelities]
		if len(fidelity_ids) != len(set(fidelity_ids)):
			raise ValueError("fidelity reference IDs must be unique")
		unknown_objectives = sorted({
			item.metric_id for item in self.objectives if item.metric_id not in metric_ids
		})
		unknown_aggregates = sorted(set(self.aggregation.metric_ids) - set(metric_ids))
		unknown_windows = sorted({
			window for fidelity in self.fidelities for window in fidelity.windows
			if window not in window_ids
		})
		if unknown_objectives:
			raise ValueError(f"objectives reference unknown metrics: {unknown_objectives}")
		if unknown_aggregates:
			raise ValueError(f"aggregation references unknown metrics: {unknown_aggregates}")
		if unknown_windows:
			raise ValueError(f"fidelities reference unknown windows: {unknown_windows}")
		metric_directions = {item.id: item.direction for item in self.metrics}
		for objective in self.objectives:
			direction = metric_directions.get(objective.metric_id)
			if direction is not None and (direction == "constraint") != (objective.role == "constraint"):
				raise ValueError(
					f"objective role for {objective.metric_id!r} conflicts with metric direction {direction!r}"
				)
		unknown_weight_windows = sorted(set(self.aggregation.window_weights) - set(window_ids))
		if unknown_weight_windows:
			raise ValueError(
				f"aggregation weights reference unknown windows: {unknown_weight_windows}"
			)
		if self.aggregation.name == "weighted_mean" and not self.aggregation.window_weights:
			raise ValueError("weighted_mean aggregation requires window_weights")
		if (
			self.output.final_fidelity_id is not None
			and self.output.final_fidelity_id not in fidelity_ids
		):
			raise ValueError(
				f"output references unknown fidelity: {self.output.final_fidelity_id!r}"
			)
		if self.sampler.name == "exhaustive" and has_continuous_domains(self.candidate_space):
			raise ValueError("exhaustive sampling cannot consume continuous candidate domains")
		return self

	@property
	def experiment_id(self) -> str:
		"""Return the deterministic identity of this complete experiment config."""

		return _sha256(self)


class CandidateKey(ConfigModel):
	"""Candidate-only identity from the resolved Recon content."""

	algorithm: Literal["sha256"] = "sha256"
	value: HashValue

	@classmethod
	def from_recon(cls, recon: Recon) -> "CandidateKey":
		return cls(value=resolved_candidate_key(recon))


class TrialId(ConfigModel):
	"""Identity of a candidate evaluated against concrete scientific inputs."""

	algorithm: Literal["sha256"] = "sha256"
	value: HashValue


class DatasetIdentity(ConfigModel):
	"""Identity-bearing dataset facts; paths and display labels live elsewhere."""

	dataset_id: str = Field(min_length=1)
	content_hash: HashValue
	acquisition_id: str | None = None


class ConfigurationIdentity(ConfigModel):
	"""Resolved non-candidate configuration and calibration identities."""

	scan_config_hash: HashValue
	calibration_hash: HashValue | None = None


class CodeIdentity(ConfigModel):
	"""Code provenance relevant to deterministic compatibility."""

	package: str = "lftomo"
	version: str = Field(min_length=1)
	git_commit: str = Field(min_length=7)
	dirty: bool = False


class EnvironmentIdentity(ConfigModel):
	"""Runtime facts that may affect numerical compatibility."""

	python: str = Field(min_length=1)
	platform: str = Field(min_length=1)
	backend: str = Field(min_length=1)
	dependencies: dict[str, str] = Field(default_factory=dict)
	hardware: dict[str, JsonValue] = Field(default_factory=dict)


class SamplerIdentity(ConfigModel):
	"""The coordinate or explicit case that selected a resolved candidate."""

	name: str = Field(min_length=1)
	seed: int
	coordinate: dict[str, JsonValue] = Field(default_factory=dict)
	explicit_case: str | None = None


class WindowIdentity(ConfigModel):
	"""Exact global coordinates and halo for one evaluated window."""

	id: StableName
	y_start: int = Field(ge=0)
	y_stop: int = Field(gt=0)
	theta_start: int | None = Field(default=None, ge=0)
	theta_stop: int | None = Field(default=None, gt=0)
	halo: int = Field(default=0, ge=0)

	@model_validator(mode="after")
	def _stops_follow_starts(self):
		if self.y_stop <= self.y_start:
			raise ValueError("window y_stop must be greater than y_start")
		if (
			self.theta_start is not None
			and self.theta_stop is not None
			and self.theta_stop <= self.theta_start
		):
			raise ValueError("window theta_stop must be greater than theta_start")
		return self


class FidelityIdentity(ConfigModel):
	"""Exact fidelity/window set used for this trial."""

	id: StableName
	windows: tuple[WindowIdentity, ...] = Field(min_length=1)
	projection_stride: int = Field(default=1, gt=0)


class MetricProtocolIdentity(ConfigModel):
	"""Versioned metric protocol included in result-comparison identity."""

	metric_id: StableName
	protocol_version: str = Field(min_length=1)
	config_hash: HashValue


class TrialIdentity(ConfigModel):
	"""Complete rerunnable trial facts and deterministic compatibility identity."""

	schema_version: SchemaVersion = 1
	candidate: CandidateKey
	resolved_recon: Recon
	dataset: DatasetIdentity
	configuration: ConfigurationIdentity
	code: CodeIdentity
	environment: EnvironmentIdentity
	sampler: SamplerIdentity
	fidelity: FidelityIdentity
	metric_protocols: tuple[MetricProtocolIdentity, ...] = Field(min_length=1)

	@model_validator(mode="after")
	def _candidate_matches_recon(self):
		if self.resolved_recon.variants or any(
			stage.sweep is not None for stage in self.resolved_recon.pipeline
		):
			raise ValueError("trial identity requires a fully resolved candidate Recon")
		expected = resolved_candidate_key(self.resolved_recon)
		if self.candidate.value != expected:
			raise ValueError(
				"candidate key does not match the canonical resolved_recon content"
			)
		protocol_ids = [item.metric_id for item in self.metric_protocols]
		if len(protocol_ids) != len(set(protocol_ids)):
			raise ValueError("metric protocol IDs must be unique")
		return self

	def identity_payload(self) -> dict[str, Any]:
		"""Return only identity-bearing fields in canonical collection order."""

		payload = self.model_dump(mode="json")
		# The complete Recon remains rerunnable in the record, while its output
		# path and execution guards are descriptive. CandidateKey owns the
		# canonical pipeline-content projection.
		payload["resolved_recon"] = {"candidate_key": self.candidate.value}
		payload["metric_protocols"] = sorted(
			payload["metric_protocols"],
			key=lambda item: (item["metric_id"], item["protocol_version"], item["config_hash"]),
		)
		payload["fidelity"]["windows"] = sorted(
			payload["fidelity"]["windows"], key=lambda item: item["id"],
		)
		return payload

	@property
	def trial_id(self) -> TrialId:
		return TrialId(value=_sha256(self.identity_payload()))


class CompatibilityDiagnostic(ConfigModel):
	"""One field-specific incompatibility between trial identities."""

	field: str
	expected: JsonValue
	actual: JsonValue


class IdentityCompatibility(ConfigModel):
	"""Structured compatibility result for future resume/cache consumers."""

	compatible: bool
	diagnostics: tuple[CompatibilityDiagnostic, ...] = ()


def _identity_differences(expected: Any, actual: Any, prefix: str = "") -> list[CompatibilityDiagnostic]:
	if isinstance(expected, dict) and isinstance(actual, dict):
		diagnostics = []
		for key in sorted(set(expected) | set(actual)):
			field = f"{prefix}.{key}" if prefix else key
			if key not in expected or key not in actual:
				diagnostics.append(CompatibilityDiagnostic(
					field=field,
					expected=expected.get(key),
					actual=actual.get(key),
				))
			else:
				diagnostics.extend(_identity_differences(expected[key], actual[key], field))
		return diagnostics
	if isinstance(expected, list) and isinstance(actual, list):
		diagnostics = []
		if len(expected) != len(actual):
			diagnostics.append(CompatibilityDiagnostic(
				field=f"{prefix}.length",
				expected=len(expected),
				actual=len(actual),
			))
		for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
			diagnostics.extend(_identity_differences(
				expected_item, actual_item, f"{prefix}[{index}]",
			))
		return diagnostics
	if expected != actual:
		return [CompatibilityDiagnostic(field=prefix, expected=expected, actual=actual)]
	return []


def compare_trial_identities(
	expected: TrialIdentity,
	actual: TrialIdentity,
) -> IdentityCompatibility:
	"""Compare identity-bearing fields and report every concrete mismatch."""

	diagnostics = tuple(_identity_differences(
		expected.identity_payload(), actual.identity_payload(),
	))
	return IdentityCompatibility(compatible=not diagnostics, diagnostics=diagnostics)


class TrialDescription(ConfigModel):
	"""Non-identity labels and locations that may change without invalidation."""

	label: str | None = None
	dataset_path: Path | None = None
	notes: str | None = None
	metadata: dict[str, JsonValue] = Field(default_factory=dict)


class FailureRecord(ConfigModel):
	"""Structured failure retained for failed/interrupted attempts."""

	type: str = Field(min_length=1)
	message: str = Field(min_length=1)
	stage_id: str | None = None
	details: dict[str, JsonValue] = Field(default_factory=dict)


class OutputRecord(ConfigModel):
	"""One atomically finalized output artifact."""

	role: str = Field(min_length=1)
	path: Path
	sha256: HashValue | None = None
	bytes: int | None = Field(default=None, ge=0)
	complete: Literal[True] = True


class ParetoRecord(ConfigModel):
	"""Slots populated later by deterministic cross-candidate ranking."""

	rank: int | None = Field(default=None, ge=0)
	non_dominated: bool | None = None
	constraints_satisfied: bool | None = None


class PromotionLineage(ConfigModel):
	"""Auditable parent/child fidelity lineage without implementing promotion."""

	parent_trial_id: TrialId | None = None
	from_fidelity: str | None = None
	decision_id: str | None = None


class FinalResult(ConfigModel):
	"""Atomically finalizable successful result; complete is never implicit."""

	schema_version: SchemaVersion = 1
	trial_id: TrialId
	attempt_id: StableName
	finalized_at: datetime
	resolved_recon: Recon
	metrics: dict[str, JsonValue]
	pareto: ParetoRecord = Field(default_factory=ParetoRecord)
	outputs: tuple[OutputRecord, ...] = ()
	promotion: PromotionLineage | None = None
	complete: Literal[True] = True


AttemptStatus = Literal[
	"pending", "running", "succeeded", "failed", "interrupted", "skipped", "incompatible",
]


class TrialAttemptRecord(ConfigModel):
	"""Append-safe logical attempt with strict lifecycle/result invariants."""

	schema_version: SchemaVersion = 1
	attempt_id: StableName
	sequence: int = Field(ge=0)
	previous_attempt_id: StableName | None = None
	trial_id: TrialId
	identity: TrialIdentity
	description: TrialDescription = Field(default_factory=TrialDescription)
	status: AttemptStatus = "pending"
	created_at: datetime
	started_at: datetime | None = None
	finished_at: datetime | None = None
	elapsed_seconds: float | None = Field(default=None, ge=0)
	peak_memory_bytes: int | None = Field(default=None, ge=0)
	observations: dict[str, JsonValue] = Field(default_factory=dict)
	pareto: ParetoRecord = Field(default_factory=ParetoRecord)
	failure: FailureRecord | None = None
	result: FinalResult | None = None

	@model_validator(mode="after")
	def _lifecycle_is_consistent(self):
		if self.trial_id != self.identity.trial_id:
			raise ValueError("trial_id does not match identity-bearing inputs")
		if self.status == "pending":
			if any(value is not None for value in (
				self.started_at, self.finished_at, self.failure, self.result,
			)):
				raise ValueError("pending attempts cannot contain execution/final state")
		elif self.status == "running":
			if self.started_at is None or self.finished_at is not None or self.result is not None:
				raise ValueError("running attempts require started_at and no final state")
		elif self.status == "succeeded":
			if self.started_at is None or self.finished_at is None or self.result is None:
				raise ValueError("succeeded attempts require timestamps and a finalized result")
			if self.failure is not None:
				raise ValueError("succeeded attempts cannot contain a failure")
			if self.result.trial_id != self.trial_id or self.result.attempt_id != self.attempt_id:
				raise ValueError("final result identity must match its attempt")
			if CandidateKey.from_recon(self.result.resolved_recon) != self.identity.candidate:
				raise ValueError("final result resolved_recon must match the trial candidate")
		elif self.status in ("failed", "interrupted"):
			if self.started_at is None or self.finished_at is None:
				raise ValueError(f"{self.status} attempts require start and finish timestamps")
			if self.failure is None or self.result is not None:
				raise ValueError(f"{self.status} attempts require failure and forbid final result")
		elif self.status in ("skipped", "incompatible"):
			if self.started_at is not None or self.finished_at is None or self.result is not None:
				raise ValueError(
					f"{self.status} attempts require finished_at, no start, and no final result"
				)
			if self.failure is None:
				raise ValueError(f"{self.status} attempts require a structured reason")
		if (
			self.started_at is not None
			and self.finished_at is not None
			and self.finished_at < self.started_at
		):
			raise ValueError("finished_at must not precede started_at")
		return self


class ExperimentManifest(ConfigModel):
	"""Top-level experiment record containing config and append-only attempt IDs."""

	schema_version: SchemaVersion = 1
	experiment_id: HashValue
	config: ExperimentConfig
	created_at: datetime
	attempt_ids: tuple[StableName, ...] = ()
	metadata: dict[str, JsonValue] = Field(default_factory=dict)

	@model_validator(mode="after")
	def _experiment_id_matches_config(self):
		if self.experiment_id != self.config.experiment_id:
			raise ValueError("experiment_id does not match canonical experiment config")
		if len(self.attempt_ids) != len(set(self.attempt_ids)):
			raise ValueError("experiment attempt IDs must be append-unique")
		return self
