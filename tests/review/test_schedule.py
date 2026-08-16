from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import math

import pytest

from lftomo.review import (
	PairScheduleConfig,
	PairwiseVote,
	ReviewProtocol,
	derive_bradley_terry,
	derive_pair_schedule,
)


NOW = datetime(2026, 8, 16, tzinfo=timezone.utc)


def candidates(count):
	return tuple(f"{index:016x}" for index in range(1, count + 1))


def vote(sequence, left, right, outcome="left", protocol_version="1"):
	return PairwiseVote(
		protocol_version=protocol_version,
		left_blind_id=left,
		right_blind_id=right,
		outcome=outcome,
		timestamp=NOW,
		sequence=sequence,
	)


def pair_key(pair):
	return tuple(sorted((pair.left_blind_id, pair.right_blind_id)))


def complete_schedule(candidate_ids, config, prefix=()):
	votes = list(prefix)
	emitted = []
	while True:
		schedule = derive_pair_schedule(votes, candidate_ids, config)
		if schedule.complete:
			return tuple(emitted), tuple(votes), schedule
		pair = schedule.next_pair
		assert pair is not None
		emitted.append(pair)
		outcome = ("left", "right", "tie")[len(votes) % 3]
		sequence = max((item.sequence for item in votes), default=-1) + 1
		votes.append(vote(
			sequence, pair.left_blind_id, pair.right_blind_id, outcome,
		))


@pytest.mark.parametrize("count", [2, 5, 12])
def test_round_robin_is_complete_and_unique(count):
	candidate_ids = candidates(count)
	schedule = derive_pair_schedule((), candidate_ids)
	keys = [pair_key(pair) for pair in schedule.remaining]
	assert schedule.policy == "round_robin"
	assert schedule.pairs_total == count * (count - 1) // 2
	assert len(keys) == len(set(keys)) == schedule.pairs_total
	assert set(keys) == set(itertools.combinations(candidate_ids, 2))
	for _, grouped in itertools.groupby(
		schedule.remaining, key=lambda pair: pair.round_number,
	):
		round_pairs = tuple(grouped)
		shown = [blind_id for pair in round_pairs
			for blind_id in (pair.left_blind_id, pair.right_blind_id)]
		assert len(shown) == len(set(shown))
	if count >= 5:
		assert all(
			{left.left_blind_id, left.right_blind_id}.isdisjoint(
				{right.left_blind_id, right.right_blind_id},
			)
			for left, right in itertools.pairwise(schedule.remaining)
		)


@pytest.mark.parametrize("count", [5, 12])
def test_round_robin_balances_display_sides(count):
	schedule = derive_pair_schedule((), candidates(count))
	sides = defaultdict(Counter)
	for pair in schedule.remaining:
		sides[pair.left_blind_id]["left"] += 1
		sides[pair.right_blind_id]["right"] += 1
	assert all(
		abs(counts["left"] - counts["right"]) <= 1
		for counts in sides.values()
	)


@pytest.mark.parametrize("count", [5, 13])
def test_schedule_is_deterministic_and_candidate_order_independent(count):
	candidate_ids = candidates(count)
	config = PairScheduleConfig(seed=47)
	first = derive_pair_schedule((), candidate_ids, config)
	second = derive_pair_schedule((), candidate_ids, config)
	reordered = derive_pair_schedule((), reversed(candidate_ids), config)
	assert first == second == reordered
	assert first.model_dump_json() == reordered.model_dump_json()
	assert first.protocol == ReviewProtocol()
	assert first.config.policy_version == "1"


def test_swiss_seed_changes_the_first_round():
	candidate_ids = candidates(13)
	first = derive_pair_schedule((), candidate_ids, PairScheduleConfig(seed=1))
	second = derive_pair_schedule((), candidate_ids, PairScheduleConfig(seed=2))
	assert first.policy == second.policy == "swiss"
	assert tuple(map(pair_key, first.remaining)) != tuple(map(pair_key, second.remaining))


@pytest.mark.parametrize("count", [5, 13])
def test_resume_recomputation_matches_uninterrupted_schedule(count):
	candidate_ids = candidates(count)
	config = PairScheduleConfig(seed=23)
	full_pairs, full_votes, final = complete_schedule(candidate_ids, config)
	assert final.complete
	for cut in sorted({0, 1, len(full_votes) // 2, len(full_votes) - 1}):
		continued, resumed_votes, resumed = complete_schedule(
			candidate_ids, config, full_votes[:cut],
		)
		assert continued == full_pairs[cut:]
		assert resumed_votes == full_votes
		assert resumed == final


def test_cannot_judge_consumes_for_schedule_but_remains_missing_for_ranking():
	candidate_ids = candidates(3)
	initial = derive_pair_schedule((), candidate_ids)
	shown = initial.next_pair
	assert shown is not None
	votes = [vote(
		0, shown.left_blind_id, shown.right_blind_id, "cannot_judge",
	)]
	resumed = derive_pair_schedule(votes, candidate_ids)
	assert resumed.pairs_consumed == 1
	assert resumed.next_pair != shown
	_, completed_votes, completed = complete_schedule(candidate_ids, PairScheduleConfig(), votes)
	assert completed.complete
	ranking = derive_bradley_terry(completed_votes, candidate_ids, derived_at=NOW)
	assert ranking.excluded_cannot_judge == 1
	assert [
		(item.left_blind_id, item.right_blind_id) for item in ranking.missing_pairs
	] == [pair_key(shown)]
	assert ranking.incomplete


def test_re_review_does_not_change_progress_or_swiss_standings():
	candidate_ids = candidates(13)
	config = PairScheduleConfig(seed=31)
	first_round = derive_pair_schedule((), candidate_ids, config).remaining
	votes = [
		vote(index, pair.left_blind_id, pair.right_blind_id, "tie")
		for index, pair in enumerate(first_round)
	]
	without_re_review = derive_pair_schedule(votes, candidate_ids, config)
	re_review = vote(
		len(votes),
		first_round[0].left_blind_id,
		first_round[0].right_blind_id,
		"left",
	)
	with_re_review = derive_pair_schedule([*votes, re_review], candidate_ids, config)
	assert with_re_review == without_re_review


def test_advanced_vote_only_counts_when_its_pair_is_scheduled():
	candidate_ids = candidates(13)
	config = PairScheduleConfig(seed=37)
	first = derive_pair_schedule((), candidate_ids, config)
	first_votes = [
		vote(index, pair.left_blind_id, pair.right_blind_id, "tie")
		for index, pair in enumerate(first.remaining)
	]
	second = derive_pair_schedule(first_votes, candidate_ids, config)
	advanced_pair = second.remaining[0]
	advanced_vote = vote(
		len(first_votes),
		advanced_pair.left_blind_id,
		advanced_pair.right_blind_id,
		"left",
	)
	before = derive_pair_schedule([advanced_vote], candidate_ids, config)
	assert before.pairs_consumed == 0
	assert before.remaining == first.remaining
	after = derive_pair_schedule([advanced_vote, *first_votes], candidate_ids, config)
	assert after.pairs_consumed == len(first.remaining) + 1
	assert advanced_pair not in after.remaining


def test_validation_matches_ranking_contract():
	candidate_ids = candidates(3)
	with pytest.raises(ValueError, match="schedule candidates must be unique"):
		derive_pair_schedule((), (*candidate_ids, candidate_ids[0]))
	with pytest.raises(ValueError, match="votes reference unknown candidates"):
		derive_pair_schedule(
			[vote(0, candidate_ids[0], "ffffffffffffffff")], candidate_ids,
		)
	with pytest.raises(ValueError, match="vote protocol"):
		derive_pair_schedule(
			[vote(0, candidate_ids[0], candidate_ids[1], protocol_version="2")],
			candidate_ids,
		)
	unsupported = ReviewProtocol.model_construct(tie_convention="davidson")
	with pytest.raises(ValueError, match="half_credit"):
		derive_pair_schedule((), candidate_ids, protocol=unsupported)


def test_policy_boundary_and_swiss_bounds_byes_and_rematches():
	assert derive_pair_schedule((), candidates(12)).policy == "round_robin"
	candidate_ids = candidates(13)
	config = PairScheduleConfig(seed=0)
	emitted, _, final = complete_schedule(candidate_ids, config)
	rounds_total = math.ceil(math.log2(len(candidate_ids))) + config.swiss_extra_rounds
	assert final.policy == "swiss"
	assert final.pairs_total == rounds_total * (len(candidate_ids) // 2) == 36
	assert final.pairs_consumed == final.pairs_total
	assert final.complete
	by_round = defaultdict(list)
	for pair in emitted:
		by_round[pair.round_number].append(pair)
	assert sorted(by_round) == list(range(1, rounds_total + 1))
	byes = []
	for round_pairs in by_round.values():
		shown = {
			blind_id for pair in round_pairs
			for blind_id in (pair.left_blind_id, pair.right_blind_id)
		}
		assert len(round_pairs) == len(candidate_ids) // 2
		byes.append((set(candidate_ids) - shown).pop())
	assert len(byes) == len(set(byes))
	keys = [pair_key(pair) for pair in emitted]
	assert len(keys) == len(set(keys))


def test_swiss_repairs_adjacent_rematches_and_consumes_advanced_vote():
	candidate_ids = candidates(13)
	config = PairScheduleConfig(seed=43)
	first = derive_pair_schedule((), candidate_ids, config)
	votes = [
		vote(index, pair.left_blind_id, pair.right_blind_id, "tie")
		for index, pair in enumerate(first.remaining)
	]
	second = derive_pair_schedule(votes, candidate_ids, config)
	assert {pair_key(pair) for pair in first.remaining}.isdisjoint(
		{pair_key(pair) for pair in second.remaining}
	)


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_candidates_is_empty_and_complete(count):
	schedule = derive_pair_schedule((), candidates(count))
	assert schedule.policy == "round_robin"
	assert schedule.pairs_total == schedule.pairs_consumed == 0
	assert schedule.remaining == ()
	assert schedule.next_pair is None
	assert schedule.complete
