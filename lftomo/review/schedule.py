"""Deterministic pairwise-comparison scheduling for blinded review."""

from __future__ import annotations

from typing import Iterable, Sequence

from lftomo.experiment.models import _sha256, canonical_json

from .models import (
	BlindId,
	PairSchedule,
	PairScheduleConfig,
	PairwiseVote,
	ReviewProtocol,
	ScheduledPair,
)
from .protocol import current_review_protocol


PairKey = tuple[str, str]


def _pair_key(left: str, right: str) -> PairKey:
	return tuple(sorted((left, right)))


def _order_key(config: PairScheduleConfig, blind_id: str) -> tuple[str, str]:
	return (
		_sha256({
			"policy_version": config.policy_version,
			"seed": config.seed,
			"blind_id": blind_id,
		}),
		blind_id,
	)


def _base_order(
	candidates: Sequence[str], config: PairScheduleConfig,
) -> tuple[str, ...]:
	return tuple(sorted(candidates, key=lambda item: _order_key(config, item)))


def _first_votes(votes: Sequence[PairwiseVote]) -> dict[PairKey, PairwiseVote]:
	first = {}
	for vote in sorted(votes, key=lambda item: (item.sequence, canonical_json(item))):
		first.setdefault(_pair_key(vote.left_blind_id, vote.right_blind_id), vote)
	return first


def _display_assignment(
	first: str,
	second: str,
	positions: dict[str, int],
	meeting_number: int = 0,
) -> tuple[str, str]:
	count = len(positions)
	low, high = sorted((first, second), key=positions.__getitem__)
	distance = positions[high] - positions[low]
	low_is_left = distance <= count // 2
	left, right = (low, high) if low_is_left else (high, low)
	if meeting_number % 2:
		left, right = right, left
	return left, right


def _avoid_boundary_repeat(
	pairs: tuple[ScheduledPair, ...],
	previous: ScheduledPair | None,
) -> tuple[ScheduledPair, ...]:
	if previous is None or len(pairs) < 2:
		return pairs
	previous_ids = {previous.left_blind_id, previous.right_blind_id}
	for index, pair in enumerate(pairs):
		if previous_ids.isdisjoint((pair.left_blind_id, pair.right_blind_id)):
			return pairs[index:] + pairs[:index]
	return pairs


def _round_robin_rounds(
	base: tuple[str, ...],
) -> tuple[tuple[ScheduledPair, ...], ...]:
	if len(base) < 2:
		return ()
	positions = {candidate: index for index, candidate in enumerate(base)}
	rotation: list[str | None] = list(base)
	if len(rotation) % 2:
		rotation.append(None)
	rounds = []
	previous = None
	for round_index in range(len(rotation) - 1):
		pairs = []
		for table in range(len(rotation) // 2):
			first = rotation[table]
			second = rotation[-1 - table]
			if first is None or second is None:
				continue
			left, right = _display_assignment(first, second, positions)
			pairs.append(ScheduledPair(
				left_blind_id=left,
				right_blind_id=right,
				round_number=round_index + 1,
			))
		round_pairs = _avoid_boundary_repeat(tuple(pairs), previous)
		rounds.append(round_pairs)
		if round_pairs:
			previous = round_pairs[-1]
		rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
	return tuple(rounds)


def _round_robin_schedule(
	candidate_ids: tuple[str, ...],
	first_votes: dict[PairKey, PairwiseVote],
	config: PairScheduleConfig,
	protocol: ReviewProtocol,
) -> PairSchedule:
	base = _base_order(candidate_ids, config)
	pairs = tuple(pair for round_pairs in _round_robin_rounds(base) for pair in round_pairs)
	remaining = tuple(
		pair for pair in pairs
		if _pair_key(pair.left_blind_id, pair.right_blind_id) not in first_votes
	)
	pairs_total = len(pairs)
	pairs_consumed = pairs_total - len(remaining)
	return PairSchedule(
		policy="round_robin",
		protocol=protocol,
		config=config,
		candidate_set_hash=_sha256(candidate_ids),
		pairs_total=pairs_total,
		pairs_consumed=pairs_consumed,
		remaining=remaining,
		complete=pairs_consumed == pairs_total,
	)


def _swiss_round(
	*,
	round_number: int,
	base: tuple[str, ...],
	scores: dict[str, float],
	prior_byes: set[str],
	prior_pairs: set[PairKey],
	pair_occurrences: dict[PairKey, int],
	previous: ScheduledPair | None,
) -> tuple[tuple[ScheduledPair, ...], str | None]:
	positions = {candidate: index for index, candidate in enumerate(base)}
	standings = sorted(base, key=lambda item: (-scores[item], positions[item]))
	bye = None
	if len(base) % 2:
		bye = next(
			(candidate for candidate in reversed(standings) if candidate not in prior_byes),
			None,
		)
		if bye is None:
			raise ValueError("Swiss schedule exhausted unique byes")
	available = [candidate for candidate in standings if candidate != bye]
	left_candidates = available[::2]
	right_candidates = available[1::2]
	standings_position = {
		candidate: index for index, candidate in enumerate(available)
	}
	preferences = {
		left: tuple(sorted(right_candidates, key=lambda right: (
			abs(standings_position[left] - standings_position[right]),
			standings_position[right],
		)))
		for left in left_candidates
	}
	matched_right: dict[str, str] = {}

	def assign(left: str, seen: set[str]) -> bool:
		for right in preferences[left]:
			if right in seen or _pair_key(left, right) in prior_pairs:
				continue
			seen.add(right)
			prior_left = matched_right.get(right)
			if prior_left is None or assign(prior_left, seen):
				matched_right[right] = left
				return True
		return False

	rematch_free = all(assign(left, set()) for left in left_candidates)
	candidate_pairs = None
	if rematch_free:
		matched_left = {left: right for right, left in matched_right.items()}
		candidate_pairs = tuple((left, matched_left[left]) for left in left_candidates)
	if candidate_pairs is None:
		fallback = []
		while available:
			first = available.pop(0)
			partner_index = next((
				index for index, candidate in enumerate(available)
				if _pair_key(first, candidate) not in prior_pairs
			), 0)
			fallback.append((first, available.pop(partner_index)))
		candidate_pairs = tuple(fallback)
	pairs = []
	for first, second in candidate_pairs:
		key = _pair_key(first, second)
		left, right = _display_assignment(
			first, second, positions, pair_occurrences.get(key, 0),
		)
		pairs.append(ScheduledPair(
			left_blind_id=left,
			right_blind_id=right,
			round_number=round_number,
		))
	return _avoid_boundary_repeat(tuple(pairs), previous), bye


def _apply_standing_vote(
	scores: dict[str, float], vote: PairwiseVote,
) -> None:
	if vote.outcome == "left":
		scores[vote.left_blind_id] += 1.0
	elif vote.outcome == "right":
		scores[vote.right_blind_id] += 1.0
	elif vote.outcome == "tie":
		scores[vote.left_blind_id] += 0.5
		scores[vote.right_blind_id] += 0.5


def _swiss_schedule(
	candidate_ids: tuple[str, ...],
	first_votes: dict[PairKey, PairwiseVote],
	config: PairScheduleConfig,
	protocol: ReviewProtocol,
) -> PairSchedule:
	base = _base_order(candidate_ids, config)
	rounds_total = (len(base) - 1).bit_length() + config.swiss_extra_rounds
	if rounds_total > len(base):
		raise ValueError("Swiss schedule rounds cannot exceed the candidate count")
	pairs_per_round = len(base) // 2
	pairs_total = rounds_total * pairs_per_round
	scores = {candidate: 0.0 for candidate in base}
	prior_byes: set[str] = set()
	prior_pairs: set[PairKey] = set()
	scored_pairs: set[PairKey] = set()
	pair_occurrences: dict[PairKey, int] = {}
	pairs_consumed = 0
	previous = None
	for round_number in range(1, rounds_total + 1):
		round_pairs, bye = _swiss_round(
			round_number=round_number,
			base=base,
			scores=scores,
			prior_byes=prior_byes,
			prior_pairs=prior_pairs,
			pair_occurrences=pair_occurrences,
			previous=previous,
		)
		remaining = tuple(
			pair for pair in round_pairs
			if _pair_key(pair.left_blind_id, pair.right_blind_id) not in first_votes
		)
		pairs_consumed += len(round_pairs) - len(remaining)
		for pair in round_pairs:
			key = _pair_key(pair.left_blind_id, pair.right_blind_id)
			prior_pairs.add(key)
			pair_occurrences[key] = pair_occurrences.get(key, 0) + 1
		if remaining:
			return PairSchedule(
				policy="swiss",
				protocol=protocol,
				config=config,
				candidate_set_hash=_sha256(candidate_ids),
				pairs_total=pairs_total,
				pairs_consumed=pairs_consumed,
				remaining=remaining,
				complete=False,
			)
		for pair in round_pairs:
			key = _pair_key(pair.left_blind_id, pair.right_blind_id)
			if key in scored_pairs:
				continue
			_apply_standing_vote(scores, first_votes[key])
			scored_pairs.add(key)
		if bye is not None:
			prior_byes.add(bye)
		if round_pairs:
			previous = round_pairs[-1]
	return PairSchedule(
		policy="swiss",
		protocol=protocol,
		config=config,
		candidate_set_hash=_sha256(candidate_ids),
		pairs_total=pairs_total,
		pairs_consumed=pairs_total,
		remaining=(),
		complete=True,
	)


def derive_pair_schedule(
	votes: Iterable[PairwiseVote],
	candidates: Iterable[BlindId],
	config: PairScheduleConfig | None = None,
	*,
	protocol: ReviewProtocol | None = None,
) -> PairSchedule:
	"""Derive deterministic comparison progress without persistent schedule state."""

	config = config or PairScheduleConfig()
	protocol = protocol or current_review_protocol()
	if protocol.tie_convention != "half_credit":
		raise ValueError("pair scheduling requires the half_credit tie convention")
	vote_list = tuple(votes)
	raw_candidates = tuple(candidates)
	candidate_ids = tuple(sorted(set(raw_candidates)))
	if len(candidate_ids) != len(raw_candidates):
		raise ValueError("schedule candidates must be unique")
	known = set(candidate_ids)
	for vote in vote_list:
		unknown = {vote.left_blind_id, vote.right_blind_id} - known
		if unknown:
			raise ValueError(f"votes reference unknown candidates: {sorted(unknown)}")
		if vote.protocol_version != protocol.protocol_version:
			raise ValueError(
				f"vote protocol {vote.protocol_version!r} differs from "
				f"schedule protocol {protocol.protocol_version!r}"
			)
	first_votes = _first_votes(vote_list)
	if len(candidate_ids) <= config.round_robin_max:
		return _round_robin_schedule(candidate_ids, first_votes, config, protocol)
	return _swiss_schedule(candidate_ids, first_votes, config, protocol)
