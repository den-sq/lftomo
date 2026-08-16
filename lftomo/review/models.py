"""Versioned records for blinded experiment review and unblinded exports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from lftomo.config import ConfigModel, Recon
from lftomo.evaluation import PreviewContract
from lftomo.experiment.models import (
	FailureRecord,
	HashValue,
	OutputRecord,
	ParetoRecord,
	PromotionLineage,
	SchemaVersion,
)


BlindId = Annotated[str, Field(pattern=r"^[0-9a-f]{16}$")]
CandidateKeyValue = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SessionId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
PassOneVerdict = Literal["discard", "uncertain", "retain"]
PassTwoVerdict = Literal["reject", "acceptable", "shortlist"]
TriageVerdict = PassOneVerdict | PassTwoVerdict
ReasonTag = Literal[
	"noise",
	"rings",
	"blur_detail_loss",
	"contrast_loss",
	"structural_artifact",
	"display_scaling_problem",
]
PairwiseOutcome = Literal["left", "right", "tie", "cannot_judge"]
NegativeControlKind = Literal[
	"low_ring_high_noise",
	"oversmoothed",
	"prominent_ring",
	"contrast_detail_loss",
	"structural_artifact",
]


class ReviewProtocol(ConfigModel):
	"""Vocabulary contract rendered by review clients."""

	schema_version: SchemaVersion = 1
	protocol_version: str = Field(default="1", min_length=1)
	pass_one_verdicts: tuple[PassOneVerdict, ...] = (
		"discard", "uncertain", "retain",
	)
	pass_two_verdicts: tuple[PassTwoVerdict, ...] = (
		"reject", "acceptable", "shortlist",
	)
	reason_tags: tuple[ReasonTag, ...] = (
		"noise",
		"rings",
		"blur_detail_loss",
		"contrast_loss",
		"structural_artifact",
		"display_scaling_problem",
	)
	pairwise_outcomes: tuple[PairwiseOutcome, ...] = (
		"left", "right", "tie", "cannot_judge",
	)
	tie_convention: Literal["half_credit"] = "half_credit"


class ReviewSession(ConfigModel):
	"""One review session bound to an immutable experiment-output contract."""

	schema_version: SchemaVersion = 1
	session_id: SessionId
	experiment_id: HashValue
	results_root: Path
	results_root_fingerprint: HashValue
	preview_contract: PreviewContract
	reviewer_id: str = Field(min_length=1)
	reviewer_label: str | None = None
	protocol: ReviewProtocol = Field(default_factory=ReviewProtocol)
	created_at: datetime
	updated_at: datetime
	blinded: bool = True

	@model_validator(mode="after")
	def _timestamps_are_ordered(self):
		if self.updated_at < self.created_at:
			raise ValueError("review updated_at must not precede created_at")
		return self


class TriageJudgment(ConfigModel):
	"""One immutable broad-triage or shortlist judgment."""

	schema_version: SchemaVersion = 1
	protocol_version: str = Field(min_length=1)
	pass_number: Literal[1, 2]
	blind_id: BlindId
	verdict: TriageVerdict
	reason_tags: tuple[ReasonTag, ...] = ()
	timestamp: datetime
	sequence: int = Field(ge=0)

	@field_validator("reason_tags")
	@classmethod
	def _reason_tags_are_unique(cls, values: tuple[ReasonTag, ...]):
		if len(values) != len(set(values)):
			raise ValueError("reason tags must be unique")
		return values

	@model_validator(mode="after")
	def _verdict_matches_pass(self):
		allowed = (
			{"discard", "uncertain", "retain"}
			if self.pass_number == 1
			else {"reject", "acceptable", "shortlist"}
		)
		if self.verdict not in allowed:
			raise ValueError(
				f"verdict {self.verdict!r} is not valid for pass {self.pass_number}"
			)
		return self


class PairwiseVote(ConfigModel):
	"""One immutable comparison in the exact order shown to the reviewer."""

	schema_version: SchemaVersion = 1
	protocol_version: str = Field(min_length=1)
	left_blind_id: BlindId
	right_blind_id: BlindId
	outcome: PairwiseOutcome
	timestamp: datetime
	sequence: int = Field(ge=0)

	@model_validator(mode="after")
	def _candidates_are_distinct(self):
		if self.left_blind_id == self.right_blind_id:
			raise ValueError("pairwise votes require two different candidates")
		return self


class PairScheduleConfig(ConfigModel):
	"""Reproducible policy for selecting pairwise review comparisons."""

	schema_version: SchemaVersion = 1
	policy_version: Literal["1"] = "1"
	round_robin_max: int = Field(default=12, gt=1)
	swiss_extra_rounds: int = Field(default=2, ge=0)
	seed: int = 0


class ScheduledPair(ConfigModel):
	"""One scheduled comparison in its required display order."""

	left_blind_id: BlindId
	right_blind_id: BlindId
	round_number: int = Field(gt=0)

	@model_validator(mode="after")
	def _candidates_are_distinct(self):
		if self.left_blind_id == self.right_blind_id:
			raise ValueError("scheduled pairs require two different candidates")
		return self


class PairSchedule(ConfigModel):
	"""Deterministic schedule progress derived from candidates and raw votes."""

	schema_version: SchemaVersion = 1
	policy: Literal["round_robin", "swiss"]
	protocol: ReviewProtocol
	config: PairScheduleConfig
	candidate_set_hash: HashValue
	pairs_total: int = Field(ge=0)
	pairs_consumed: int = Field(ge=0)
	remaining: tuple[ScheduledPair, ...] = ()
	complete: bool

	@model_validator(mode="after")
	def _progress_is_consistent(self):
		if self.pairs_consumed > self.pairs_total:
			raise ValueError("consumed pair count exceeds the schedule total")
		if self.complete != (self.pairs_consumed == self.pairs_total):
			raise ValueError("schedule completion differs from pair progress")
		if self.pairs_consumed + len(self.remaining) > self.pairs_total:
			raise ValueError("remaining pairs exceed the schedule total")
		if self.complete and self.remaining:
			raise ValueError("a complete schedule cannot have remaining pairs")
		if not self.complete and not self.remaining:
			raise ValueError("an incomplete schedule requires a next pair")
		keys = [
			tuple(sorted((pair.left_blind_id, pair.right_blind_id)))
			for pair in self.remaining
		]
		if len(keys) != len(set(keys)):
			raise ValueError("remaining schedule contains duplicate pairs")
		return self

	@property
	def next_pair(self) -> ScheduledPair | None:
		"""Return the next required display pair, if review is incomplete."""

		return self.remaining[0] if self.remaining else None


class UnblindEvent(ConfigModel):
	"""Explicit audit record authorizing an unblinded operation."""

	schema_version: SchemaVersion = 1
	protocol_version: str = Field(min_length=1)
	reviewer_id: str = Field(min_length=1)
	scope: Literal["candidate", "session", "export"]
	blind_ids: tuple[BlindId, ...] = ()
	timestamp: datetime
	sequence: int = Field(ge=0)

	@model_validator(mode="after")
	def _scope_has_targets(self):
		if self.scope in ("candidate", "export") and not self.blind_ids:
			raise ValueError(f"unblind scope {self.scope!r} requires blind_ids")
		if len(self.blind_ids) != len(set(self.blind_ids)):
			raise ValueError("unblind blind_ids must be unique")
		return self


class RankingConfig(ConfigModel):
	"""Numerical policy for regularized Bradley--Terry derivation."""

	schema_version: SchemaVersion = 1
	tie_convention: Literal["half_credit"] = "half_credit"
	pseudo_count_tau: float = Field(default=1.0, gt=0, allow_inf_nan=False)
	tolerance: float = Field(default=1e-10, gt=0, allow_inf_nan=False)
	max_iterations: int = Field(default=10_000, gt=0)


class MissingPair(ConfigModel):
	left_blind_id: BlindId
	right_blind_id: BlindId


class RankingEntry(ConfigModel):
	rank: int = Field(gt=0)
	blind_id: BlindId
	strength: float = Field(gt=0, allow_inf_nan=False)


class RankingRecord(ConfigModel):
	"""Reproducible ranking derived from, but never replacing, raw votes."""

	schema_version: SchemaVersion = 1
	ranking_id: HashValue
	derived_at: datetime
	protocol: ReviewProtocol
	config: RankingConfig
	vote_set_hash: HashValue
	candidate_set_hash: HashValue
	votes_total: int = Field(ge=0)
	votes_fitted: int = Field(ge=0)
	excluded_cannot_judge: int = Field(ge=0)
	entries: tuple[RankingEntry, ...]
	incomplete: bool
	missing_pairs: tuple[MissingPair, ...] = ()
	converged: bool
	iterations: int = Field(ge=0)

	@model_validator(mode="after")
	def _entries_are_a_total_order(self):
		blind_ids = [item.blind_id for item in self.entries]
		if len(blind_ids) != len(set(blind_ids)):
			raise ValueError("ranking entries must have unique blind IDs")
		if [item.rank for item in self.entries] != list(range(1, len(self.entries) + 1)):
			raise ValueError("ranking entries must use consecutive one-based ranks")
		return self


class StageChainEntry(ConfigModel):
	position: int = Field(gt=0)
	stage_id: str | None = None
	kind: str
	algorithm: str | None = None


class CandidateVariation(ConfigModel):
	sampler: str
	seed: int
	parameters: dict[str, JsonValue] = Field(default_factory=dict)
	explicit_case: str | None = None


class ResolvedAttempt(ConfigModel):
	attempt_id: str
	sequence: int = Field(ge=0)
	status: str
	created_at: datetime
	started_at: datetime | None = None
	finished_at: datetime | None = None
	elapsed_seconds: float | None = Field(default=None, ge=0)
	execution_elapsed_seconds: float | None = Field(default=None, ge=0)
	peak_memory_bytes: int | None = Field(default=None, ge=0)
	metrics: dict[str, JsonValue] = Field(default_factory=dict)
	observations: dict[str, JsonValue] = Field(default_factory=dict)
	pareto: ParetoRecord = Field(default_factory=ParetoRecord)
	promotion: PromotionLineage | None = None
	outputs: tuple[OutputRecord, ...] = ()
	failure: FailureRecord | None = None


class ResolvedTrial(ConfigModel):
	trial_id: HashValue
	fidelity_id: str
	attempt_ids: tuple[str, ...]
	latest_attempt: ResolvedAttempt | None = None


class CandidateReport(ConfigModel):
	"""Complete deliberate-unblinding report for one resolved candidate."""

	schema_version: SchemaVersion = 1
	matched_namespace: Literal["blind_id", "candidate_key", "trial_id"]
	matched_id: str
	blind_id: BlindId | None = None
	candidate_key: CandidateKeyValue
	algorithm_chain: tuple[StageChainEntry, ...]
	variations: tuple[CandidateVariation, ...]
	resolved_recon: Recon
	trials: tuple[ResolvedTrial, ...]
	preview_paths: tuple[Path, ...] = ()


class CandidateInventoryEntry(ConfigModel):
	"""Compact joinable candidate summary for plural lookup."""

	blind_id: BlindId | None = None
	candidate_key: CandidateKeyValue
	trial_ids: tuple[HashValue, ...]
	fidelity_ids: tuple[str, ...]
	latest_statuses: tuple[str, ...]
	best_rank: int | None = Field(default=None, ge=0)
	promoted: bool = False
	algorithm_chain: tuple[str, ...]


class ReviewExportSelection(ConfigModel):
	"""Reviewer-selected anchors, controls, and confirmation shortlist."""

	positive_anchors: tuple[BlindId, ...] = ()
	negative_controls: dict[NegativeControlKind, tuple[BlindId, ...]] = Field(
		default_factory=dict,
	)
	confirmation_shortlist: tuple[BlindId, ...] = ()

	@model_validator(mode="after")
	def _selections_are_unique(self):
		groups = {
			"positive_anchors": self.positive_anchors,
			"confirmation_shortlist": self.confirmation_shortlist,
			**self.negative_controls,
		}
		for name, values in groups.items():
			if len(values) != len(set(values)):
				raise ValueError(f"{name} contains duplicate blind IDs")
		return self


class ReviewedCandidate(ConfigModel):
	blind_id: BlindId
	candidate_key: CandidateKeyValue
	trial_ids: tuple[HashValue, ...]
	triage_judgments: tuple[TriageJudgment, ...]
	pairwise_votes: tuple[PairwiseVote, ...]
	ranking_position: int | None = Field(default=None, gt=0)
	resolved_recon: Recon


class ConfirmationCandidate(ConfigModel):
	candidate_key: CandidateKeyValue
	resolved_recon: Recon


class ReviewExport(ConfigModel):
	"""Unblinded machine-readable anchor and rerun contract."""

	schema_version: SchemaVersion = 1
	session_id: SessionId
	experiment_id: HashValue
	exported_at: datetime
	ranking_id: HashValue | None = None
	positive_anchors: tuple[ReviewedCandidate, ...]
	negative_controls: dict[NegativeControlKind, tuple[ReviewedCandidate, ...]]
	confirmation_shortlist: tuple[ConfirmationCandidate, ...]
