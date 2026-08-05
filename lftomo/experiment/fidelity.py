"""Window aggregation, deterministic promotion, previews, and fidelity orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Literal, Mapping
from uuid import uuid4

import numpy as np
from pydantic import Field, JsonValue

from lftomo.config import ConfigModel
from lftomo.evaluation import (
	CandidateMetrics,
	HardConstraint,
	ParetoObjective,
	PreviewContract,
	pareto_rank,
)

from .models import (
	AggregationPolicy,
	ExperimentConfig,
	PromotionLineage,
	PromotionPolicy,
	TrialId,
	canonical_json,
)
from .runner import (
	CandidateValidator,
	ExperimentInputs,
	ExperimentRunPolicy,
	ExperimentRunSummary,
	IndependentExperimentRunner,
	IdentityValidator,
	TrialExecutor,
	atomic_write_model,
)
from .resources import PathResourceRequest
from .runtime import RuntimeCompatibilityReport
from .store import PersistedExperimentRunner, ResumePolicy


PromotionStatus = Literal[
	"succeeded", "failed", "interrupted", "skipped", "incompatible",
]


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


class WindowMetricAggregates(ConfigModel):
	"""Per-window values plus separately aggregated tuning and held-out values."""

	per_window: dict[str, dict[str, float | None]]
	tuning: dict[str, float | None]
	held_out: dict[str, float | None]


def _finite_values(value: Any) -> tuple[list[float], bool]:
	items = value if isinstance(value, (list, tuple)) else [value]
	valid = []
	invalid = False
	for item in items:
		if isinstance(item, (int, float)) and math.isfinite(float(item)):
			valid.append(float(item))
		else:
			invalid = True
	return valid, invalid


def _reduce(values: list[float], policy: AggregationPolicy, weights=None) -> float:
	array = np.asarray(values, dtype=np.float64)
	if policy.name == "median":
		return float(np.median(array))
	if policy.name == "weighted_mean":
		return float(np.average(array, weights=np.asarray(weights, dtype=np.float64)))
	return float(np.mean(array))


def aggregate_window_metrics(
	window_metrics: Mapping[str, Mapping[str, Any]],
	config: ExperimentConfig,
) -> WindowMetricAggregates:
	"""Aggregate slices within windows, then windows by tuning/held-out role."""

	window_refs = {window.id: window for window in config.windows}
	unknown = sorted(set(window_metrics) - set(window_refs))
	if unknown:
		raise ValueError(f"window metrics reference unknown windows: {unknown}")
	per_window: dict[str, dict[str, float | None]] = {}
	for window_id in sorted(window_metrics):
		per_window[window_id] = {}
		for metric, raw in sorted(window_metrics[window_id].items()):
			values, invalid = _finite_values(raw)
			if invalid and config.aggregation.invalid in ("fail", "worst"):
				value = None
			elif not values:
				value = None
			else:
				within_policy = config.aggregation.model_copy(update={
					"name": "mean" if config.aggregation.name == "weighted_mean"
					else config.aggregation.name,
				})
				value = _reduce(values, within_policy)
			per_window[window_id][metric] = value

	def aggregate_role(role: str) -> dict[str, float | None]:
		window_ids = [
			window_id for window_id in sorted(per_window)
			if window_refs[window_id].role == role
		]
		metrics = sorted({metric for name in window_ids for metric in per_window[name]})
		result = {}
		for metric in metrics:
			pairs = [(name, per_window[name].get(metric)) for name in window_ids]
			invalid = any(value is None for _, value in pairs)
			valid = [(name, value) for name, value in pairs if value is not None]
			if config.aggregation.name == "none":
				if len(valid) == 1 and not invalid:
					result[metric] = valid[0][1]
				else:
					result[metric] = None
			elif invalid and config.aggregation.invalid in ("fail", "worst"):
				result[metric] = None
			elif not valid:
				result[metric] = None
			else:
				weights = [config.aggregation.window_weights.get(name, 1.0) for name, _ in valid]
				result[metric] = _reduce(
					[float(value) for _, value in valid], config.aggregation, weights,
				)
		return result

	return WindowMetricAggregates(
		per_window=per_window,
		tuning=aggregate_role("tuning"),
		held_out=aggregate_role("held_out"),
	)


class PromotionCandidate(ConfigModel):
	candidate_key: str = Field(min_length=1)
	trial_id: str = Field(min_length=1)
	status: PromotionStatus
	tuning_metrics: dict[str, float | None] = Field(default_factory=dict)
	held_out_metrics: dict[str, float | None] = Field(default_factory=dict)


class PromotionRank(ConfigModel):
	position: int = Field(ge=1)
	candidate_key: str
	trial_id: str
	status: PromotionStatus
	eligible: bool
	promoted: bool
	pareto_rank: int | None = Field(default=None, ge=0)
	tie_signature: str | None = None
	reason: str | None = None
	tuning_metrics: dict[str, float | None]
	held_out_metrics: dict[str, float | None]


class PromotionDecision(ConfigModel):
	schema_version: Literal[1] = 1
	decision_id: str = Field(pattern=r"^[0-9a-f]{64}$")
	input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
	protocol_version: str
	from_fidelity: str
	to_fidelity: str
	created_at: datetime
	target_count: int = Field(ge=0)
	cutoff_position: int | None = Field(default=None, ge=1)
	tie_policy: Literal["stable_cut", "include_all"]
	manual_override: bool = False
	rankings: tuple[PromotionRank, ...]
	promoted_candidate_keys: tuple[str, ...]


def _sha256(value: Any) -> str:
	return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def successive_halving(
	candidates: list[PromotionCandidate] | tuple[PromotionCandidate, ...],
	config: ExperimentConfig,
	from_fidelity: str,
	to_fidelity: str,
	*,
	manual_promote: set[str] | None = None,
) -> PromotionDecision:
	"""Produce a stable, auditable promotion decision from tuning metrics only."""

	policy = config.promotion or PromotionPolicy()
	candidates = tuple(sorted(candidates, key=lambda item: item.candidate_key))
	keys = {item.candidate_key for item in candidates}
	if manual_promote is not None:
		unknown = sorted(manual_promote - keys)
		if unknown:
			raise ValueError(f"manual promotion references unknown candidates: {unknown}")
		terminal = sorted(
			item.candidate_key for item in candidates
			if item.candidate_key in manual_promote and item.status != "succeeded"
		)
		if terminal:
			raise ValueError(
				f"manual promotion requires succeeded candidates: {terminal}"
			)
	directions = {item.id: item.direction for item in config.metrics}
	objectives = [item for item in config.objectives if item.role == "objective"]
	constraints = [item for item in config.objectives if item.role == "constraint"]
	valid = []
	worst = []
	excluded: dict[str, str] = {}
	for candidate in candidates:
		if candidate.status != "succeeded":
			if policy.failures == "fail":
				raise ValueError(
					f"candidate {candidate.candidate_key} ended {candidate.status}"
				)
			if policy.failures == "exclude":
				excluded[candidate.candidate_key] = f"terminal status {candidate.status}"
			else:
				worst.append(candidate)
				excluded[candidate.candidate_key] = (
					f"terminal status {candidate.status}; ranked as worst but ineligible "
					"for promotion"
				)
			continue
		required = [item.metric_id for item in (*objectives, *constraints)]
		missing = [
			metric for metric in required
			if candidate.tuning_metrics.get(metric) is None
			or not math.isfinite(float(candidate.tuning_metrics[metric]))
		]
		if missing:
			if policy.invalid == "fail":
				raise ValueError(
					f"candidate {candidate.candidate_key} has invalid tuning metrics {missing}"
				)
			if policy.invalid == "exclude":
				excluded[candidate.candidate_key] = f"invalid tuning metrics: {missing}"
			else:
				worst.append(candidate)
			continue
		violated = [
			item.metric_id for item in constraints
			if float(candidate.tuning_metrics[item.metric_id]) > float(item.threshold)
		]
		if violated:
			excluded[candidate.candidate_key] = f"constraint violation: {violated}"
			continue
		valid.append(candidate)

	pareto_outcomes = {}
	if valid and objectives:
		outcomes = pareto_rank(
			[CandidateMetrics(
				candidate_key=item.candidate_key,
				metrics={name: value for name, value in item.tuning_metrics.items()},
			) for item in valid],
			[ParetoObjective(
				metric=item.metric_id, direction=directions[item.metric_id],
			) for item in objectives],
			[HardConstraint(
				metric=item.metric_id, relation="<=", threshold=float(item.threshold),
			) for item in constraints],
		)
		pareto_outcomes = {item.candidate_key: item for item in outcomes}

	def metric_signature(candidate: PromotionCandidate) -> tuple[float, ...]:
		values = []
		for objective in objectives:
			value = float(candidate.tuning_metrics[objective.metric_id])
			values.append(value if directions[objective.metric_id] == "minimize" else -value)
		return tuple(values)

	valid.sort(key=lambda item: (
		pareto_outcomes[item.candidate_key].rank if item.candidate_key in pareto_outcomes else 0,
		metric_signature(item),
		item.candidate_key,
	))
	worst.sort(key=lambda item: item.candidate_key)
	eligible = valid + [
		item for item in worst if item.candidate_key not in excluded
	]
	target = min(
		len(eligible),
		max(policy.min_survivors, math.ceil(len(eligible) / policy.reduction_factor)),
	) if eligible else 0
	if manual_promote is not None:
		promoted = set(manual_promote)
	elif target == 0:
		promoted = set()
	else:
		promoted = {item.candidate_key for item in eligible[:target]}
		if policy.ties == "include_all" and target < len(eligible):
			boundary = eligible[target - 1]
			boundary_signature = (
				boundary in worst,
				pareto_outcomes[boundary.candidate_key].rank
				if boundary.candidate_key in pareto_outcomes else 10**9,
				metric_signature(boundary) if boundary in valid else (),
			)
			for item in eligible[target:]:
				signature = (
					item in worst,
					pareto_outcomes[item.candidate_key].rank
					if item.candidate_key in pareto_outcomes else 10**9,
					metric_signature(item) if item in valid else (),
				)
				if signature == boundary_signature:
					promoted.add(item.candidate_key)

	ordered = eligible + sorted(
		(item for item in candidates if item.candidate_key in excluded),
		key=lambda item: item.candidate_key,
	)
	rankings = []
	for position, candidate in enumerate(ordered, start=1):
		outcome = pareto_outcomes.get(candidate.candidate_key)
		tie_payload = None
		if candidate in valid:
			tie_payload = _sha256({
				"pareto_rank": outcome.rank if outcome else 0,
				"metrics": metric_signature(candidate),
			})
		elif candidate in worst:
			tie_payload = _sha256({"worst": True})
		rankings.append(PromotionRank(
			position=position,
			candidate_key=candidate.candidate_key,
			trial_id=candidate.trial_id,
			status=candidate.status,
			eligible=candidate not in [item for item in candidates if item.candidate_key in excluded],
			promoted=candidate.candidate_key in promoted,
			pareto_rank=None if outcome is None else outcome.rank,
			tie_signature=tie_payload,
			reason=excluded.get(candidate.candidate_key) or (
				"ranked as worst by policy" if candidate in worst else None
			),
			tuning_metrics=candidate.tuning_metrics,
			held_out_metrics=candidate.held_out_metrics,
		))

	input_payload = [item.model_dump(mode="json") for item in candidates]
	decision_payload = {
		"inputs": input_payload,
		"experiment_id": config.experiment_id,
		"policy": policy.model_dump(mode="json"),
		"from_fidelity": from_fidelity,
		"to_fidelity": to_fidelity,
		"promoted": sorted(promoted),
		"rankings": [item.model_dump(mode="json") for item in rankings],
		"manual": manual_promote is not None,
	}
	return PromotionDecision(
		decision_id=_sha256(decision_payload),
		input_fingerprint=_sha256(input_payload),
		protocol_version=policy.protocol_version,
		from_fidelity=from_fidelity,
		to_fidelity=to_fidelity,
		created_at=_utcnow(),
		target_count=target,
		cutoff_position=None if not promoted else max(
			item.position for item in rankings if item.promoted
		),
		tie_policy=policy.ties,
		manual_override=manual_promote is not None,
		rankings=tuple(rankings),
		promoted_candidate_keys=tuple(sorted(promoted)),
	)


class PromotionDecisionStore:
	"""Persist and validate the active decision at each fidelity boundary."""

	def __init__(self, root: str | Path):
		self.root = Path(root)

	def publish(
		self,
		decision: PromotionDecision,
		*,
		stale: Literal["fail", "invalidate"] = "fail",
	) -> PromotionDecision:
		boundary = (
			self.root / "promotions"
			/ f"{decision.from_fidelity}--{decision.to_fidelity}"
		)
		active = boundary / "active.json"
		if active.exists():
			stored = PromotionDecision.model_validate_json(active.read_text(encoding="utf-8"))
			if stored.decision_id == decision.decision_id:
				return stored
			if stale == "fail":
				raise ValueError(
					"stored promotion boundary is stale: "
					f"{stored.input_fingerprint} != {decision.input_fingerprint}"
				)
			quarantine = (
				self.root / "quarantine" / "promotions" / boundary.name
				/ f"active.{stored.decision_id}.json"
			)
			quarantine.parent.mkdir(parents=True, exist_ok=True)
			os.replace(active, quarantine)
		atomic_write_model(active, decision)
		return decision


def promotion_lineage(
	decision: PromotionDecision,
) -> dict[str, PromotionLineage]:
	"""Build candidate-key lineage attached to promoted next-fidelity results."""

	parents = {item.candidate_key: item.trial_id for item in decision.rankings}
	return {
		candidate_key: PromotionLineage(
			parent_trial_id=TrialId(value=parents[candidate_key]),
			from_fidelity=decision.from_fidelity,
			decision_id=decision.decision_id,
		)
		for candidate_key in decision.promoted_candidate_keys
	}


class PreviewMappingEntry(ConfigModel):
	blind_id: str
	candidate_key: str
	files: tuple[Path, ...]


class PreviewBundle(ConfigModel):
	contract: PreviewContract
	mapping_path: Path
	entries: tuple[PreviewMappingEntry, ...]
	total_bytes: int = Field(ge=0)


def export_blinded_previews(
	images: Mapping[str, np.ndarray | Mapping[str, np.ndarray]],
	root: str | Path,
	contract: PreviewContract,
	*,
	blinding_key: str,
	max_bytes: int | None = None,
) -> PreviewBundle:
	"""Serialize bundle updates so mappings and global byte guards remain atomic."""

	root = Path(root)
	private_root = root / "private" / "preview-mapping"
	private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
	with (private_root / ".lock").open("a+", encoding="utf-8") as lock:
		fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
		try:
			return _export_blinded_previews_unlocked(
				images,
				root,
				contract,
				blinding_key=blinding_key,
				max_bytes=max_bytes,
			)
		finally:
			fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _export_blinded_previews_unlocked(
	images: Mapping[str, np.ndarray | Mapping[str, np.ndarray]],
	root: str | Path,
	contract: PreviewContract,
	*,
	blinding_key: str,
	max_bytes: int | None = None,
) -> PreviewBundle:
	"""Write consistently scaled/cropped previews and a reversible private mapping."""

	import cv2
	import tifffile as tf

	if not blinding_key:
		raise ValueError("blinding_key must not be empty")
	root = Path(root)
	private_root = root / "private" / "preview-mapping"
	private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
	mapping_path = private_root / "mapping.json"
	entry_root = private_root / "entries"
	if mapping_path.exists():
		existing = PreviewBundle.model_validate_json(mapping_path.read_text(encoding="utf-8"))
		if existing.contract != contract:
			raise ValueError("preview contract differs from the existing blinded bundle")
	total = sum(
		path.stat().st_size for path in (root / "previews").glob("*/*")
		if path.is_file() and path.suffix in (".png", ".tif", ".tiff")
	)
	for candidate_key in sorted(images):
		blind_id = hashlib.sha256(
			f"{blinding_key}:{candidate_key}".encode("utf-8")
		).hexdigest()[:16]
		raw = images[candidate_key]
		if isinstance(raw, Mapping):
			slices = sorted(raw.items())
		else:
			array = np.asarray(raw)
			slices = [(f"{index:04d}", image) for index, image in enumerate(
				array if array.ndim == 3 else [array]
			)]
		files = []
		for label, image in slices:
			array = np.asarray(image)
			if array.ndim != 2:
				raise ValueError("preview images must be two-dimensional slices")
			if contract.crop is not None:
				crop = contract.crop
				array = array[crop.y_start:crop.y_stop, crop.x_start:crop.x_stop]
			scaled = np.clip(
				(array.astype(np.float64) - contract.scale_min)
				/ (contract.scale_max - contract.scale_min),
				0,
				1,
			)
			encoded = np.round(scaled * 65535).astype(np.uint16)
			suffix = ".png" if contract.format == "png" else ".tif"
			path = root / "previews" / blind_id / f"{label}{suffix}"
			path.parent.mkdir(parents=True, exist_ok=True)
			temporary = path.with_name(f".{path.stem}-{uuid4().hex}{suffix}")
			if contract.format == "png":
				if not cv2.imwrite(str(temporary), encoded):
					raise OSError(f"failed to write preview {temporary}")
			else:
				tf.imwrite(temporary, encoded)
			encoded_bytes = temporary.stat().st_size
			replaced_bytes = path.stat().st_size if path.exists() else 0
			if max_bytes is not None and total - replaced_bytes + encoded_bytes > max_bytes:
				temporary.unlink()
				raise ValueError(
					f"preview output would exceed configured guard {max_bytes} bytes"
				)
			os.replace(temporary, path)
			total = total - replaced_bytes + encoded_bytes
			files.append(path)
		entry = PreviewMappingEntry(
			blind_id=blind_id, candidate_key=candidate_key, files=tuple(files),
		)
		fragment = entry_root / f"{blind_id}.json"
		if fragment.exists():
			prior = PreviewMappingEntry.model_validate_json(fragment.read_text(encoding="utf-8"))
			entry = entry.model_copy(update={
				"files": tuple(sorted(set(prior.files) | set(entry.files))),
			})
		atomic_write_model(fragment, entry)
	entries = tuple(sorted((
		PreviewMappingEntry.model_validate_json(path.read_text(encoding="utf-8"))
		for path in entry_root.glob("*.json")
	), key=lambda item: item.blind_id))
	bundle = PreviewBundle(
		contract=contract,
		mapping_path=mapping_path,
		entries=entries,
		total_bytes=total,
	)
	atomic_write_model(mapping_path, bundle)
	return bundle


class FidelityRunSummary(ConfigModel):
	schema_version: Literal[1] = 1
	experiment_id: str
	levels: tuple[ExperimentRunSummary, ...]
	promotions: tuple[PromotionDecision, ...]


class FidelityExperimentRunner:
	"""Execute fidelity levels and persist deterministic promotion boundaries."""

	def __init__(
		self,
		config: ExperimentConfig,
		inputs: ExperimentInputs,
		root: str | Path,
		executor: TrialExecutor,
		*,
		validator: CandidateValidator | None = None,
		identity_validator: IdentityValidator | None = None,
		run_policy: ExperimentRunPolicy | None = None,
		resume_policy: ResumePolicy | None = None,
		manual_promotions: Mapping[str, set[str]] | None = None,
		invalidate_stale_promotions: bool = False,
		path_resources: PathResourceRequest | None = None,
		runtime_report: RuntimeCompatibilityReport | None = None,
	):
		if config.promotion is None:
			raise ValueError("fidelity execution requires an explicit promotion policy")
		self.config = config
		self.inputs = inputs
		self.root = Path(root)
		self.executor = executor
		self.validator = validator
		self.identity_validator = identity_validator
		self.run_policy = run_policy
		self.resume_policy = resume_policy or ResumePolicy()
		self.manual_promotions = manual_promotions or {}
		self.invalidate_stale_promotions = invalidate_stale_promotions
		self.path_resources = path_resources
		self.runtime_report = runtime_report

	def run(self) -> FidelityRunSummary:
		levels = []
		decisions = []
		selected_keys: set[str] | None = None
		lineage: dict[str, PromotionLineage] = {}
		decision_store = PromotionDecisionStore(self.root)
		for index, fidelity in enumerate(self.config.fidelities):
			runner = IndependentExperimentRunner(
				self.config,
				self.inputs,
				self.root,
				self.executor,
				validator=self.validator,
				identity_validator=self.identity_validator,
				policy=self.run_policy,
				fidelity_id=fidelity.id,
				promotion_lineage=lineage,
				path_resources=self.path_resources,
				runtime_report=self.runtime_report,
			)
			trial_ids = None
			if selected_keys is not None:
				_, prepared = runner.prepare()
				trial_ids = {
					trial.identity.trial_id.value for trial in prepared
					if trial.identity.candidate.value in selected_keys
				}
			level = PersistedExperimentRunner(
				runner,
				policy=self.resume_policy,
				trial_ids=trial_ids,
			).run()
			levels.append(level)
			if index == len(self.config.fidelities) - 1:
				break
			promotion_candidates = []
			for attempt in level.attempts:
				stored_windows = attempt.observations.get("window_metrics", {})
				if isinstance(stored_windows, dict) and stored_windows:
					aggregates = aggregate_window_metrics(stored_windows, self.config)
					tuning = aggregates.tuning
					held_out = aggregates.held_out
				else:
					tuning = {} if attempt.result is None else {
						name: float(value) if isinstance(value, (int, float)) else None
						for name, value in attempt.result.metrics.items()
					}
					held_out = {}
				promotion_candidates.append(PromotionCandidate(
					candidate_key=attempt.identity.candidate.value,
					trial_id=attempt.trial_id.value,
					status=attempt.status,
					tuning_metrics=tuning,
					held_out_metrics=held_out,
				))
			next_fidelity = self.config.fidelities[index + 1]
			boundary = f"{fidelity.id}--{next_fidelity.id}"
			decision = successive_halving(
				promotion_candidates,
				self.config,
				fidelity.id,
				next_fidelity.id,
				manual_promote=self.manual_promotions.get(boundary),
			)
			decision = decision_store.publish(
				decision,
				stale="invalidate" if self.invalidate_stale_promotions else "fail",
			)
			decisions.append(decision)
			selected_keys = set(decision.promoted_candidate_keys)
			lineage = promotion_lineage(decision)
			if not selected_keys:
				break
		summary = FidelityRunSummary(
			experiment_id=self.config.experiment_id,
			levels=tuple(levels),
			promotions=tuple(decisions),
		)
		atomic_write_model(self.root / "fidelity-results.json", summary)
		return summary
