import copy

import pytest

from lftomo.evaluation import (
	AggregateScoreConfig,
	CandidateMetrics,
	HardConstraint,
	MetricObservation,
	ParetoObjective,
	aggregate_observations,
	aggregate_score,
	pareto_rank,
)
from lftomo.evaluation.models import AggregationConfig


def observation(value, *, window="a", valid=True, reason=None):
	return MetricObservation(
		metric="quality",
		window_id=window,
		slice_index=0,
		value=value if valid else None,
		valid=valid,
		reason=reason,
		direction="minimize",
		units="ratio",
		normalization="test",
		protocol_version="1",
		sample_count=10,
		invalid_count=0 if valid else 10,
	)


def test_aggregation_retains_raw_counts_and_explicit_invalid_handling():
	raw = [
		observation(1, window="a"),
		observation(3, window="b"),
		observation(None, window="b", valid=False, reason="undefined"),
	]
	failed = aggregate_observations(raw, AggregationConfig(invalid="fail"))["quality"]
	excluded = aggregate_observations(raw, AggregationConfig(invalid="exclude"))["quality"]
	worst = aggregate_observations(
		raw,
		AggregationConfig(invalid="worst", worst_values={"quality": 9}),
	)["quality"]
	weighted = aggregate_observations(
		raw[:2],
		AggregationConfig(
			method="weighted_mean",
			invalid="exclude",
			window_weights={"a": 3, "b": 1},
		),
	)["quality"]

	assert not failed.valid and failed.value is None
	assert excluded.value == pytest.approx(2)
	assert excluded.observation_count == 3
	assert excluded.valid_observation_count == 2
	assert excluded.sample_count == 30
	assert excluded.invalid_count == 10
	assert worst.value == pytest.approx((1 + 3 + 9) / 3)
	assert weighted.value == pytest.approx(1.5)
	assert [item.value for item in raw] == [1, 3, None]


def test_pareto_ranking_honors_direction_constraints_ties_and_input_order():
	candidates = [
		CandidateMetrics(candidate_key="d", metrics={"error": 0.4, "cnr": 8, "invalid": 0}),
		CandidateMetrics(candidate_key="b", metrics={"error": 0.2, "cnr": 5, "invalid": 0}),
		CandidateMetrics(candidate_key="a", metrics={"error": 0.2, "cnr": 5, "invalid": 0}),
		CandidateMetrics(candidate_key="c", metrics={"error": 0.1, "cnr": 9, "invalid": 2}),
		CandidateMetrics(candidate_key="e", metrics={"error": None, "cnr": 10, "invalid": 0}),
	]
	objectives = [
		ParetoObjective(metric="error", direction="minimize"),
		ParetoObjective(metric="cnr", direction="maximize"),
	]
	constraints = [HardConstraint(metric="invalid", relation="<=", threshold=0)]
	forward = pareto_rank(candidates, objectives, constraints)
	reverse = pareto_rank(list(reversed(candidates)), objectives, constraints)
	by_key = {item.candidate_key: item for item in forward}

	assert forward == reverse
	assert [item.candidate_key for item in forward] == ["a", "b", "c", "d", "e"]
	assert by_key["a"].non_dominated and by_key["b"].non_dominated
	assert by_key["d"].non_dominated
	assert not by_key["c"].constraints_satisfied
	assert not by_key["e"].valid_objectives
	assert by_key["c"].rank > by_key["a"].rank
	assert by_key["e"].rank > by_key["a"].rank


def test_optional_aggregate_score_does_not_change_raw_metrics_or_pareto():
	candidate = CandidateMetrics(
		candidate_key="a", metrics={"error": 0.2, "cnr": 5, "raw": 99},
	)
	before = copy.deepcopy(candidate.metrics)
	config = AggregateScoreConfig.model_validate({
		"terms": [
			{"metric": "error", "direction": "minimize", "weight": 2, "scale": 0.1},
			{"metric": "cnr", "direction": "maximize", "weight": 1, "scale": 5},
		],
	})

	assert aggregate_score(candidate, config) == pytest.approx(1)
	assert candidate.metrics == before
	assert pareto_rank(
		[candidate],
		[ParetoObjective(metric="error", direction="minimize")],
	)[0].non_dominated
