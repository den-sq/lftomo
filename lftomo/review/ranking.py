"""Deterministic regularized Bradley--Terry ranking for review votes."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Iterable, Sequence

from lftomo.experiment.models import _sha256, canonical_json

from .models import (
	BlindId,
	MissingPair,
	PairwiseVote,
	RankingConfig,
	RankingEntry,
	RankingRecord,
	ReviewProtocol,
)
from .protocol import current_review_protocol


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


def _vote_set_hash(votes: Sequence[PairwiseVote]) -> str:
	ordered = sorted(canonical_json(vote) for vote in votes)
	return _sha256(ordered)


def derive_bradley_terry(
	votes: Iterable[PairwiseVote],
	candidates: Iterable[BlindId],
	config: RankingConfig | None = None,
	*,
	protocol: ReviewProtocol | None = None,
	derived_at: datetime | None = None,
) -> RankingRecord:
	"""Derive an order-independent finite Bradley--Terry ranking.

	``cannot_judge`` votes remain in the consumed-vote hash but are excluded from
	the fit. Ties contribute one half-win to each candidate. A symmetric virtual
	pair with total weight ``pseudo_count_tau`` regularizes every candidate pair.
	"""

	config = config or RankingConfig()
	protocol = protocol or current_review_protocol()
	if protocol.tie_convention != config.tie_convention:
		raise ValueError("ranking tie convention differs from the review protocol")
	vote_list = tuple(votes)
	raw_candidates = tuple(candidates)
	candidate_ids = tuple(sorted(set(raw_candidates)))
	if len(candidate_ids) != len(raw_candidates):
		raise ValueError("ranking candidates must be unique")
	known = set(candidate_ids)
	for vote in vote_list:
		unknown = {vote.left_blind_id, vote.right_blind_id} - known
		if unknown:
			raise ValueError(f"votes reference unknown candidates: {sorted(unknown)}")
		if vote.protocol_version != protocol.protocol_version:
			raise ValueError(
				f"vote protocol {vote.protocol_version!r} differs from "
				f"ranking protocol {protocol.protocol_version!r}"
			)

	index = {candidate: offset for offset, candidate in enumerate(candidate_ids)}
	count = len(candidate_ids)
	wins = [[0.0] * count for _ in range(count)]
	judgeable_pairs: set[tuple[str, str]] = set()
	excluded = 0
	for vote in vote_list:
		if vote.outcome == "cannot_judge":
			excluded += 1
			continue
		left = index[vote.left_blind_id]
		right = index[vote.right_blind_id]
		judgeable_pairs.add(tuple(sorted((vote.left_blind_id, vote.right_blind_id))))
		if vote.outcome == "left":
			wins[left][right] += 1.0
		elif vote.outcome == "right":
			wins[right][left] += 1.0
		else:
			wins[left][right] += 0.5
			wins[right][left] += 0.5

	all_pairs = [
		(candidate_ids[left], candidate_ids[right])
		for left in range(count)
		for right in range(left + 1, count)
	]
	missing = tuple(
		MissingPair(left_blind_id=left, right_blind_id=right)
		for left, right in all_pairs
		if (left, right) not in judgeable_pairs
	)
	for left in range(count):
		for right in range(left + 1, count):
			prior_half = config.pseudo_count_tau / 2.0
			wins[left][right] += prior_half
			wins[right][left] += prior_half

	strengths = [1.0] * count
	converged = True
	iterations = 0
	if count > 1:
		converged = False
		win_totals = [math.fsum(row) for row in wins]
		for iteration in range(1, config.max_iterations + 1):
			updated = []
			for left in range(count):
				denominator = math.fsum(
					(wins[left][right] + wins[right][left])
					/ (strengths[left] + strengths[right])
					for right in range(count)
					if right != left
				)
				updated.append(win_totals[left] / denominator)
			log_mean = math.fsum(math.log(value) for value in updated) / count
			updated = [value / math.exp(log_mean) for value in updated]
			delta = max(
				abs(math.log(new_value / old_value))
				for new_value, old_value in zip(updated, strengths, strict=True)
			)
			strengths = updated
			iterations = iteration
			if delta <= config.tolerance:
				converged = True
				break

	ordered = sorted(
		zip(candidate_ids, strengths, strict=True),
		key=lambda item: (-item[1], item[0]),
	)
	entries = tuple(
		RankingEntry(rank=rank, blind_id=blind_id, strength=strength)
		for rank, (blind_id, strength) in enumerate(ordered, start=1)
	)
	vote_hash = _vote_set_hash(vote_list)
	candidate_hash = _sha256(candidate_ids)
	payload = {
		"protocol": protocol.model_dump(mode="json"),
		"config": config.model_dump(mode="json"),
		"vote_set_hash": vote_hash,
		"candidate_set_hash": candidate_hash,
		"entries": [item.model_dump(mode="json") for item in entries],
		"votes_total": len(vote_list),
		"votes_fitted": len(vote_list) - excluded,
		"excluded_cannot_judge": excluded,
		"missing_pairs": [item.model_dump(mode="json") for item in missing],
		"converged": converged,
		"iterations": iterations,
	}
	return RankingRecord(
		ranking_id=_sha256(payload),
		derived_at=derived_at or _utcnow(),
		protocol=protocol,
		config=config,
		vote_set_hash=vote_hash,
		candidate_set_hash=candidate_hash,
		votes_total=len(vote_list),
		votes_fitted=len(vote_list) - excluded,
		excluded_cannot_judge=excluded,
		entries=entries,
		incomplete=bool(missing or excluded or not converged),
		missing_pairs=missing,
		converged=converged,
		iterations=iterations,
	)
