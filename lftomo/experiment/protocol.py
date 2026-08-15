"""Tamper-evident Tier-2 experiment planning and result-audit contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from lftomo.config import ConfigModel
from lftomo.evaluation import PreviewContract

from .fidelity import FidelityRunSummary, PreviewBundle
from .models import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	FidelityIdentity,
	MetricProtocolIdentity,
	WindowIdentity,
	canonical_json,
)
from .runner import ExperimentInputs, atomic_write_model, json_value
from .selection import select_candidates


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


def _sha256(value) -> str:
	return hashlib.sha256(canonical_json(json_value(value)).encode("utf-8")).hexdigest()


class Tier2Bounds(ConfigModel):
	"""Hard limits that keep a real-data comparison intentional and bounded."""

	max_candidates: int = Field(default=128, gt=0)
	max_screen_window_rows: int = Field(default=64, gt=0)
	max_screen_projections: int = Field(default=512, gt=0)
	max_output_bytes: int = Field(gt=0)
	max_memory_gb: float = Field(gt=0)


class Tier2ExperimentPlan(ConfigModel):
	"""Content-addressed provenance and review protocol recorded before Tier-2 data."""

	schema_version: Literal[1] = 1
	protocol_id: str = Field(pattern=r"^[0-9a-f]{64}$")
	name: str = Field(min_length=1)
	experiment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
	dataset: DatasetIdentity
	configuration: ConfigurationIdentity
	code: CodeIdentity
	environment: EnvironmentIdentity
	environment_lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
	projection_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
	metric_protocols: tuple[MetricProtocolIdentity, ...]
	fidelities: tuple[FidelityIdentity, ...]
	timeout_seconds_by_fidelity: dict[str, float | None]
	tuning_window_ids: tuple[str, ...]
	held_out_window_ids: tuple[str, ...]
	raw_metric_ids: tuple[str, ...]
	preview: PreviewContract
	blinding_key_id: str = Field(min_length=1)
	final_fidelity_id: str
	write_full_finalists: Literal[True] = True
	require_pareto: Literal[True] = True
	bounds: Tier2Bounds
	created_at: datetime

	def identity_payload(self):
		payload = self.model_dump(mode="json")
		payload.pop("protocol_id")
		payload.pop("created_at")
		return payload


def build_tier2_plan(
	config: ExperimentConfig,
	inputs: ExperimentInputs,
	*,
	environment_lock_hash: str,
	projection_manifest_hash: str,
	preview: PreviewContract,
	blinding_key_id: str,
	bounds: Tier2Bounds,
) -> Tier2ExperimentPlan:
	"""Validate a production protocol and pin every comparison-bearing fact."""

	if config.promotion is None or len(config.fidelities) < 2:
		raise ValueError("Tier-2 protocol requires a multi-fidelity promotion policy")
	if inputs.code.dirty or inputs.code.git_commit.startswith("unknown"):
		raise ValueError("Tier-2 protocol requires a clean, pinned code commit")
	if not preview.blinded:
		raise ValueError("Tier-2 protocol requires blinded previews")
	if config.output.screening != "metrics_and_previews":
		raise ValueError("Tier-2 screening must retain raw metrics and blinded previews")
	if not config.output.write_full_finalists or config.output.final_fidelity_id is None:
		raise ValueError("Tier-2 protocol requires finalist-only full output")
	output_limits = [
		value for value in (config.resources.max_output_bytes, config.output.max_bytes)
		if value is not None
	]
	if not output_limits or min(output_limits) > bounds.max_output_bytes:
		raise ValueError("Tier-2 output guard must be explicit and within protocol bounds")
	memory_limits = [
		value for value in (config.resources.memory_gb,) if value is not None
	]
	if not memory_limits or min(memory_limits) > bounds.max_memory_gb:
		raise ValueError("Tier-2 memory guard must be explicit and within protocol bounds")
	candidate_count = len(select_candidates(config))
	if candidate_count > bounds.max_candidates:
		raise ValueError(
			f"Tier-2 candidate count {candidate_count} exceeds bound {bounds.max_candidates}"
		)
	windows = {item.id: item for item in config.windows}
	screen = config.fidelities[0]
	if screen.projection_limit is None:
		raise ValueError("Tier-2 screening fidelity requires projection_limit")
	if screen.projection_limit > bounds.max_screen_projections:
		raise ValueError("Tier-2 screening projection limit exceeds protocol bounds")
	for name in screen.windows:
		window = windows[name]
		if window.y_stop - window.y_start > bounds.max_screen_window_rows:
			raise ValueError(f"Tier-2 screening window {name!r} exceeds row bound")
	tuning = tuple(sorted(name for name in screen.windows if windows[name].role == "tuning"))
	held_out = tuple(sorted(name for name in screen.windows if windows[name].role == "held_out"))
	if len(tuning) < 2:
		raise ValueError("Tier-2 screening requires at least two tuning windows")
	if not held_out:
		raise ValueError("Tier-2 screening requires at least one held-out window")

	identities = []
	for fidelity in config.fidelities:
		identities.append(FidelityIdentity(
			id=fidelity.id,
			windows=tuple(WindowIdentity(
				id=windows[name].id,
				role=windows[name].role,
				y_start=windows[name].y_start,
				y_stop=windows[name].y_stop,
				theta_start=windows[name].theta_start,
				theta_stop=windows[name].theta_stop,
				halo=windows[name].halo,
			) for name in fidelity.windows),
			projection_stride=fidelity.projection_stride,
			projection_limit=fidelity.projection_limit,
			resource_fraction=fidelity.resource_fraction,
			memory_gb=fidelity.memory_gb,
			max_output_bytes=fidelity.max_output_bytes,
		))
	protocols = tuple(MetricProtocolIdentity(
		metric_id=metric.id,
		protocol_version=metric.protocol_version,
		config_hash=_sha256(metric),
	) for metric in config.metrics)
	payload = {
		"schema_version": 1,
		"name": config.name,
		"experiment_id": config.experiment_id,
		"dataset": inputs.dataset,
		"configuration": inputs.configuration,
		"code": inputs.code,
		"environment": inputs.environment,
		"environment_lock_hash": environment_lock_hash,
		"projection_manifest_hash": projection_manifest_hash,
		"metric_protocols": protocols,
		"fidelities": identities,
		"timeout_seconds_by_fidelity": dict(sorted(
			(fidelity.id, fidelity.timeout_seconds) for fidelity in config.fidelities
		)),
		"tuning_window_ids": tuning,
		"held_out_window_ids": held_out,
		"raw_metric_ids": tuple(sorted(metric.id for metric in config.metrics)),
		"preview": preview,
		"blinding_key_id": blinding_key_id,
		"final_fidelity_id": config.output.final_fidelity_id,
		"write_full_finalists": True,
		"require_pareto": True,
		"bounds": bounds,
	}
	return Tier2ExperimentPlan(
		protocol_id=_sha256(payload),
		created_at=_utcnow(),
		**payload,
	)


def write_tier2_plan(path: str | Path, plan: Tier2ExperimentPlan) -> None:
	"""Atomically publish a pre-data-execution protocol."""

	if plan.protocol_id != _sha256(plan.identity_payload()):
		raise ValueError("Tier-2 protocol_id does not match its content-addressed payload")
	atomic_write_model(Path(path), plan)


class Tier2Audit(ConfigModel):
	protocol_id: str
	passed: bool
	errors: tuple[str, ...] = ()


def audit_tier2_results(
	plan: Tier2ExperimentPlan,
	summary: FidelityRunSummary,
	previews: PreviewBundle,
) -> Tier2Audit:
	"""Check raw metrics, Pareto, blinding, held-out, and finalist artifacts."""

	errors = []
	if summary.experiment_id != plan.experiment_id:
		errors.append("result experiment_id differs from the recorded protocol")
	if previews.contract != plan.preview or not previews.contract.blinded:
		errors.append("preview contract or blinding differs from the recorded protocol")
	required = set(plan.raw_metric_ids)
	for level in summary.levels:
		for attempt in level.attempts:
			if attempt.status != "succeeded" or attempt.result is None:
				continue
			missing = sorted(required - set(attempt.result.metrics))
			if missing:
				errors.append(f"attempt {attempt.attempt_id} lacks raw metrics {missing}")
			if plan.require_pareto and attempt.result.pareto.rank is None:
				errors.append(f"attempt {attempt.attempt_id} lacks Pareto rank")
	for decision in summary.promotions:
		for ranking in decision.rankings:
			missing = sorted(required - set(ranking.held_out_metrics))
			if missing:
				errors.append(
					f"promotion {decision.decision_id} lacks held-out metrics {missing} "
					f"for {ranking.candidate_key}"
				)
	if summary.levels:
		for attempt in summary.levels[-1].attempts:
			if attempt.status == "succeeded" and attempt.result is not None:
				if not any(output.role == "finalist_output" for output in attempt.result.outputs):
					errors.append(f"finalist {attempt.attempt_id} lacks finalized full output")
	return Tier2Audit(
		protocol_id=plan.protocol_id,
		passed=not errors,
		errors=tuple(errors),
	)
