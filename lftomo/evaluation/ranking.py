"""Pure deterministic Pareto and optional aggregate-score utilities."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from lftomo.config import ConfigModel


class CandidateMetrics(ConfigModel):
	"""Raw aggregate metrics for one candidate."""

	candidate_key: str = Field(min_length=1)
	metrics: dict[str, float | None]


class ParetoObjective(ConfigModel):
	metric: str = Field(min_length=1)
	direction: Literal["minimize", "maximize"]


class HardConstraint(ConfigModel):
	metric: str = Field(min_length=1)
	relation: Literal["<=", ">="]
	threshold: float


class ParetoOutcome(ConfigModel):
	candidate_key: str
	rank: int
	non_dominated: bool
	constraints_satisfied: bool
	valid_objectives: bool


def _finite_metric(candidate: CandidateMetrics, name: str) -> float | None:
	value = candidate.metrics.get(name)
	return float(value) if value is not None and math.isfinite(value) else None


def _constraints_satisfied(
	candidate: CandidateMetrics,
	constraints: tuple[HardConstraint, ...],
) -> bool:
	for constraint in constraints:
		value = _finite_metric(candidate, constraint.metric)
		if value is None:
			return False
		if constraint.relation == "<=" and value > constraint.threshold:
			return False
		if constraint.relation == ">=" and value < constraint.threshold:
			return False
	return True


def _dominates(
	a: CandidateMetrics,
	b: CandidateMetrics,
	objectives: tuple[ParetoObjective, ...],
) -> bool:
	better = False
	for objective in objectives:
		a_value = _finite_metric(a, objective.metric)
		b_value = _finite_metric(b, objective.metric)
		assert a_value is not None and b_value is not None
		if objective.direction == "minimize":
			if a_value > b_value:
				return False
			better = better or a_value < b_value
		else:
			if a_value < b_value:
				return False
			better = better or a_value > b_value
	return better


def pareto_rank(
	candidates: list[CandidateMetrics] | tuple[CandidateMetrics, ...],
	objectives: list[ParetoObjective] | tuple[ParetoObjective, ...],
	constraints: list[HardConstraint] | tuple[HardConstraint, ...] = (),
) -> tuple[ParetoOutcome, ...]:
	"""Return deterministic non-dominated ranks honoring hard constraints."""

	objectives = tuple(objectives)
	constraints = tuple(constraints)
	if not objectives:
		raise ValueError("Pareto ranking requires at least one objective")
	keys = [candidate.candidate_key for candidate in candidates]
	if len(keys) != len(set(keys)):
		raise ValueError("candidate keys must be unique for Pareto ranking")
	valid = {
		candidate.candidate_key: all(
			_finite_metric(candidate, objective.metric) is not None
			for objective in objectives
		)
		for candidate in candidates
	}
	feasible = {
		candidate.candidate_key: _constraints_satisfied(candidate, constraints)
		for candidate in candidates
	}
	remaining = {
		candidate.candidate_key: candidate
		for candidate in candidates
		if valid[candidate.candidate_key] and feasible[candidate.candidate_key]
	}
	ranks: dict[str, int] = {}
	rank = 0
	while remaining:
		front = [
			candidate
			for candidate in remaining.values()
			if not any(
				other.candidate_key != candidate.candidate_key
				and _dominates(other, candidate, objectives)
				for other in remaining.values()
			)
		]
		for candidate in front:
			ranks[candidate.candidate_key] = rank
			remaining.pop(candidate.candidate_key)
		rank += 1
	invalid_rank = rank
	for candidate in candidates:
		ranks.setdefault(candidate.candidate_key, invalid_rank)
	return tuple(
		ParetoOutcome(
			candidate_key=candidate.candidate_key,
			rank=ranks[candidate.candidate_key],
			non_dominated=ranks[candidate.candidate_key] == 0
				and valid[candidate.candidate_key]
				and feasible[candidate.candidate_key],
			constraints_satisfied=feasible[candidate.candidate_key],
			valid_objectives=valid[candidate.candidate_key],
		)
		for candidate in sorted(candidates, key=lambda item: item.candidate_key)
	)


class AggregateScoreTerm(ConfigModel):
	"""Explicit affine normalization for one optional score contribution."""

	metric: str
	direction: Literal["minimize", "maximize"]
	weight: float = Field(gt=0)
	offset: float = 0
	scale: float = Field(gt=0)


class AggregateScoreConfig(ConfigModel):
	"""Configured scalar score; lower is always better."""

	terms: tuple[AggregateScoreTerm, ...] = Field(min_length=1)
	missing: Literal["fail", "exclude"] = "fail"

	@model_validator(mode="after")
	def _metrics_are_unique(self):
		names = [term.metric for term in self.terms]
		if len(names) != len(set(names)):
			raise ValueError("aggregate score metric terms must be unique")
		return self


def aggregate_score(candidate: CandidateMetrics, config: AggregateScoreConfig) -> float | None:
	"""Calculate an explicitly normalized score without mutating raw metrics."""

	total = 0.0
	used_weight = 0.0
	for term in config.terms:
		value = _finite_metric(candidate, term.metric)
		if value is None:
			if config.missing == "fail":
				return None
			continue
		normalized = (value - term.offset) / term.scale
		if term.direction == "maximize":
			normalized = -normalized
		total += term.weight * normalized
		used_weight += term.weight
	return total / used_weight if used_weight else None
