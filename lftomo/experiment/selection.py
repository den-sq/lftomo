"""Deterministic exhaustive and space-filling experiment candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.stats import qmc

from lftomo.config import (
	CategoricalDomain,
	DiscreteDomain,
	LinearDomain,
	LogDomain,
	ParameterDomain,
	Recon,
	StageCase,
	StageConfig,
)
from lftomo.pipeline import candidate_key, domain_values, expand_recon_sweeps

from .models import ExperimentConfig, SamplerIdentity


@dataclass(frozen=True)
class SelectedCandidate:
	"""One unique, fully resolved candidate and its sampler provenance."""

	index: int
	recon: Recon
	parameters: dict[str, Any]
	candidate_key: str
	sampler: SamplerIdentity
	duplicate_coordinates: int = 0


def _structure(recon: Recon, unit: float) -> tuple[str | None, list[StageConfig]]:
	if not recon.variants:
		return None, [stage.model_copy(deep=True) for stage in recon.pipeline]
	index = min(int(unit * len(recon.variants)), len(recon.variants) - 1)
	variant = recon.variants[index]
	by_id = {stage.id: stage for stage in recon.pipeline}
	return variant.name, [by_id[stage_id].model_copy(deep=True) for stage_id in variant.stages]


def _domain_dimension(domain: ParameterDomain) -> int:
	return 1


def _stage_dimensions(stage: StageConfig) -> int:
	if stage.sweep is None:
		return 0
	if stage.sweep.kind == "cases":
		return 1 + max((sum(_domain_dimension(item) for item in case.domains.values())
			for case in stage.sweep.cases), default=0)
	if stage.sweep.kind == "domain":
		return len(stage.sweep.params)
	return 1


def sampling_dimensions(recon: Recon) -> int:
	"""Return the fixed unit-cube dimensionality used by sampled selection."""

	dimensions = 1 if recon.variants else 0
	for stage in recon.pipeline:
		dimensions += _stage_dimensions(stage)
	return dimensions


def _sample_domain(domain: ParameterDomain, unit: float) -> Any:
	if isinstance(domain, (DiscreteDomain, CategoricalDomain)) or domain.count is not None:
		values = domain_values(domain)
		return values[min(int(unit * len(values)), len(values) - 1)]
	if isinstance(domain, LinearDomain):
		return domain.start + unit * (domain.stop - domain.start)
	assert isinstance(domain, LogDomain)
	start = math.log(domain.start, domain.base)
	stop = math.log(domain.stop, domain.base)
	return domain.base ** (start + unit * (stop - start))


def _domain_from_stage(stage: StageConfig, name: str) -> ParameterDomain:
	from lftomo.pipeline.sweep import parameter_domains

	return parameter_domains(stage.sweep)[name]


def _sample_case(
	stage: StageConfig,
	case: StageCase,
	units: np.ndarray,
	prefix: str,
) -> tuple[StageConfig, dict[str, Any]]:
	resolved = stage.model_copy(deep=True)
	resolved.algorithm = case.algorithm
	resolved.params.update(case.params)
	values: dict[str, Any] = {
		f"{prefix}.case": case.name,
		f"{prefix}.algorithm": case.algorithm,
	}
	for offset, name in enumerate(sorted(case.domains)):
		value = _sample_domain(case.domains[name], float(units[offset]))
		resolved.params[name] = value
		values[f"{prefix}.{name}"] = value
	resolved.sweep = None
	return resolved, values


def _sample_stage(
	stage: StageConfig,
	units: np.ndarray,
	index: int,
) -> tuple[StageConfig, dict[str, Any]]:
	prefix = f"stage[{stage.id}]" if stage.id is not None else f"stage[{index}]"
	if stage.sweep is None:
		return stage.model_copy(deep=True), {}
	if stage.sweep.kind == "cases":
		case_index = min(int(float(units[0]) * len(stage.sweep.cases)), len(stage.sweep.cases) - 1)
		return _sample_case(stage, stage.sweep.cases[case_index], units[1:], prefix)
	if stage.sweep.kind == "domain":
		resolved = stage.model_copy(deep=True)
		values = {}
		for offset, name in enumerate(sorted(stage.sweep.params)):
			value = _sample_domain(_domain_from_stage(stage, name), float(units[offset]))
			resolved.params[name] = value
			values[f"{prefix}.{name}"] = value
		resolved.sweep = None
		return resolved, values

	# Legacy finite sweep forms are sampled as one categorical axis over their
	# complete Stage-local expansion. This preserves zip/list case coupling.
	temporary = Recon(pipeline=[stage.model_copy(deep=True)])
	variants = expand_recon_sweeps(temporary)
	choice = min(int(float(units[0]) * len(variants)), len(variants) - 1)
	selected = variants[choice]
	return selected.recon.pipeline[0], selected.parameters


def _resolve_unit_candidate(recon: Recon, units: np.ndarray) -> tuple[Recon, dict[str, Any]]:
	cursor = 0
	structure_unit = float(units[cursor]) if recon.variants else 0.0
	if recon.variants:
		cursor += 1
	variant, stages = _structure(recon, structure_unit)
	resolved_stages = []
	parameters: dict[str, Any] = {}
	original_index = {stage.id: index for index, stage in enumerate(recon.pipeline)}
	for fallback_index, stage in enumerate(stages):
		dimensions = _stage_dimensions(stage)
		stage_units = units[cursor:cursor + dimensions]
		cursor += dimensions
		index = original_index.get(stage.id, fallback_index)
		resolved, values = _sample_stage(stage, stage_units, index)
		resolved_stages.append(resolved)
		parameters.update(values)
	# Validate the final materialized candidate as one coherent model.  Assigning
	# the selected subset to a copy first would make its still-attached structural
	# variants temporarily reference omitted Stage IDs, and validate_assignment
	# correctly rejects that intermediate state.
	document = recon.model_dump(mode="python")
	document.update({
		"pipeline": [stage.model_dump(mode="python") for stage in resolved_stages],
		"variants": [],
	})
	resolved = Recon.model_validate(document)
	if variant is not None:
		parameters = {"pipeline.variant": variant, **parameters}
	return resolved, parameters


def _deduplicate(candidates: list[SelectedCandidate]) -> tuple[SelectedCandidate, ...]:
	by_key: dict[str, SelectedCandidate] = {}
	counts: dict[str, int] = {}
	for candidate in candidates:
		if candidate.candidate_key in by_key:
			counts[candidate.candidate_key] = counts.get(candidate.candidate_key, 0) + 1
			continue
		by_key[candidate.candidate_key] = candidate
		counts[candidate.candidate_key] = 0
	return tuple(
		SelectedCandidate(
			index=index,
			recon=item.recon,
			parameters=item.parameters,
			candidate_key=item.candidate_key,
			sampler=item.sampler,
			duplicate_coordinates=counts[item.candidate_key],
		)
		for index, item in enumerate(by_key.values())
	)


def select_candidates(config: ExperimentConfig) -> tuple[SelectedCandidate, ...]:
	"""Select deterministic, deduplicated candidates from an ExperimentConfig."""

	if config.sampler.name == "exhaustive":
		runs = expand_recon_sweeps(config.candidate_space)
		if config.sampler.budget is not None and len(runs) > config.sampler.budget:
			raise ValueError(
				f"exhaustive candidate count {len(runs)} exceeds sampler budget "
				f"{config.sampler.budget}; raise the budget or narrow the space"
			)
		return _deduplicate([
			SelectedCandidate(
				index=run.index,
				recon=run.recon.model_copy(deep=True),
				parameters=dict(run.parameters),
				candidate_key=run.candidate_key,
				sampler=SamplerIdentity(
					name="exhaustive",
					seed=config.sampler.seed,
					coordinate={"enumeration_index": run.index},
					explicit_case=run.variant,
				),
			)
			for run in runs
		])

	budget = config.sampler.budget
	assert budget is not None
	dimensions = sampling_dimensions(config.candidate_space)
	if dimensions == 0:
		points = np.zeros((1, 0), dtype=np.float64)
	elif config.sampler.name == "sobol":
		exponent = math.ceil(math.log2(budget))
		points = qmc.Sobol(
			d=dimensions, scramble=True, seed=config.sampler.seed,
		).random_base2(exponent)[:budget]
	else:
		points = np.random.default_rng(config.sampler.seed).random((budget, dimensions))
	candidates = []
	for sample_index, point in enumerate(points):
		resolved, parameters = _resolve_unit_candidate(config.candidate_space, point)
		key = candidate_key(resolved)
		coordinate = {f"q{index}": float(value) for index, value in enumerate(point)}
		coordinate.update({f"resolved.{name}": value for name, value in parameters.items()})
		candidates.append(SelectedCandidate(
			index=sample_index,
			recon=resolved,
			parameters=parameters,
			candidate_key=key,
			sampler=SamplerIdentity(
				name=config.sampler.name,
				seed=config.sampler.seed,
				coordinate=coordinate,
			),
		))
	return _deduplicate(candidates)
