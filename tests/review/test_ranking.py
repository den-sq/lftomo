from datetime import datetime, timezone
import math
import random

from lftomo.review import PairwiseVote, RankingConfig, derive_bradley_terry


NOW = datetime(2026, 8, 16, tzinfo=timezone.utc)
CANDIDATES = tuple(f"{index:016x}" for index in range(1, 4))


def vote(sequence, left, right, outcome):
	return PairwiseVote(
		protocol_version="1",
		left_blind_id=left,
		right_blind_id=right,
		outcome=outcome,
		timestamp=NOW,
		sequence=sequence,
	)


def test_half_credit_ties_and_cannot_judge_are_explicit():
	votes = [
		vote(0, CANDIDATES[0], CANDIDATES[1], "left"),
		vote(1, CANDIDATES[1], CANDIDATES[2], "tie"),
		vote(2, CANDIDATES[1], CANDIDATES[2], "tie"),
		vote(3, CANDIDATES[1], CANDIDATES[2], "tie"),
		vote(4, CANDIDATES[0], CANDIDATES[2], "cannot_judge"),
	]
	record = derive_bradley_terry(votes, CANDIDATES, derived_at=NOW)
	order = [item.blind_id for item in record.entries]
	# Ignoring ties leaves candidates 2 and 3 at zero wins and would put 2
	# first by fallback ID. Half-credit fits the ties and carries candidate 2's
	# loss to candidate 1 through the comparison graph, so candidate 3 is higher.
	assert order.index(CANDIDATES[2]) < order.index(CANDIDATES[1])
	assert record.config.tie_convention == "half_credit"
	assert record.excluded_cannot_judge == 1
	assert record.votes_fitted == 4
	assert record.incomplete
	assert [(item.left_blind_id, item.right_blind_id) for item in record.missing_pairs] == [
		(CANDIDATES[0], CANDIDATES[2]),
	]


def test_ranking_is_order_independent_and_complete_separation_stays_finite():
	votes = [
		vote(0, CANDIDATES[0], CANDIDATES[1], "left"),
		vote(1, CANDIDATES[0], CANDIDATES[2], "left"),
		vote(2, CANDIDATES[1], CANDIDATES[2], "left"),
	]
	shuffled = list(votes)
	random.Random(17).shuffle(shuffled)
	config = RankingConfig(pseudo_count_tau=0.5, tolerance=1e-12)
	forward = derive_bradley_terry(
		votes, CANDIDATES, config, derived_at=NOW,
	)
	reordered = derive_bradley_terry(
		shuffled, reversed(CANDIDATES), config, derived_at=NOW,
	)
	assert forward == reordered
	repeated = derive_bradley_terry(votes, CANDIDATES, config)
	assert repeated.entries == forward.entries
	assert repeated.ranking_id == forward.ranking_id
	assert forward.entries[0].blind_id == CANDIDATES[0]
	assert all(math.isfinite(item.strength) and item.strength > 0
		for item in forward.entries)
	assert forward.converged
	assert not forward.incomplete
