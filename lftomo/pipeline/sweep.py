"""Materialize the canonical candidate space into concrete reconstruction runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
from typing import Any, Iterable

import numpy as np
from pydantic import TypeAdapter

from lftomo.config import (
	CategoricalDomain,
	DiscreteDomain,
	LinearDomain,
	LogDomain,
	ParameterDomain,
	Recon,
	StageCase,
	StageConfig,
	SweepConfig,
)


class CandidateSpaceError(ValueError):
	"""Base error for candidate materialization failures."""


class ContinuousDomainError(CandidateSpaceError):
	"""Raised when an exhaustive runner encounters a continuous domain."""


class CandidateLimitError(CandidateSpaceError):
	"""Raised before materialization when a candidate limit is exceeded."""


@dataclass(frozen=True)
class SweepRun:
	"""One complete, directly rerunnable candidate produced from a Recon."""

	index: int
	recon: Recon
	parameters: dict[str, Any]
	candidate_key: str
	variant: str | None = None

	@property
	def output_key(self) -> str:
		"""Stable directory-safe name for this candidate."""

		return f"candidate-{self.candidate_key[:16]}"


@dataclass(frozen=True)
class _StageVariant:
	stage: StageConfig
	values: dict[str, Any]


_DOMAIN_ADAPTER = TypeAdapter(ParameterDomain)


def _sequence(value: Any, name: str) -> list[Any]:
	if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
		raise CandidateSpaceError(f"sweep parameter {name!r} must be a sequence")
	return list(value)


def _linspace(value: Any, name: str) -> list[Any]:
	if not isinstance(value, dict):
		raise CandidateSpaceError(f"linspace parameter {name!r} must be a mapping")
	unknown = set(value) - {"start", "stop", "num", "endpoint"}
	if unknown or not {"start", "stop", "num"} <= set(value):
		raise CandidateSpaceError(
			f"linspace parameter {name!r} requires start, stop, and num; "
			f"unknown keys: {sorted(unknown)}"
		)
	values = np.linspace(
		value["start"],
		value["stop"],
		num=value["num"],
		endpoint=value.get("endpoint", True),
	)
	return [item.item() for item in values]


def parameter_domains(sweep: SweepConfig) -> dict[str, ParameterDomain]:
	"""Return validated typed domains from a domain sweep."""

	if sweep.kind != "domain":
		raise CandidateSpaceError("typed parameter domains require sweep kind 'domain'")
	if not sweep.params:
		raise CandidateSpaceError("sweep.params must not be empty")
	return {
		name: _DOMAIN_ADAPTER.validate_python(spec)
		for name, spec in sweep.params.items()
	}


def domain_values(domain: ParameterDomain, name: str = "parameter") -> list[Any]:
	"""Exhaust one finite domain, rejecting continuous domains explicitly."""

	if isinstance(domain, (DiscreteDomain, CategoricalDomain)):
		return list(domain.values)
	if domain.count is None:
		raise ContinuousDomainError(
			f"domain {name!r} is continuous; select it with an experiment sampler "
			"instead of the ordinary exhaustive runner"
		)
	if isinstance(domain, LinearDomain):
		values = np.linspace(
			domain.start, domain.stop, num=domain.count, endpoint=domain.endpoint,
		)
	else:
		assert isinstance(domain, LogDomain)
		start = np.log(domain.start) / np.log(domain.base)
		stop = np.log(domain.stop) / np.log(domain.base)
		values = np.logspace(
			start, stop, num=domain.count, endpoint=True, base=domain.base,
		)
	return [item.item() for item in values]


def _domain_grid(domains: dict[str, ParameterDomain]) -> list[dict[str, Any]]:
	if not domains:
		return [{}]
	values = {name: domain_values(domain, name) for name, domain in domains.items()}
	return [
		dict(zip(values, items, strict=True))
		for items in product(*values.values())
	]


def expand_sweep(sweep: SweepConfig) -> list[dict[str, Any]]:
	"""Expand one finite parameter sweep into parameter overrides.

	Algorithm cases are materialized by expand_recon_sweeps because an
	algorithm selection cannot be represented as parameter overrides alone.
	"""

	if sweep.kind == "cases":
		raise CandidateSpaceError(
			"algorithm cases must be expanded as part of a Stage/Recon candidate"
		)
	if not sweep.params:
		raise CandidateSpaceError("sweep.params must not be empty")

	if sweep.kind == "domain":
		return _domain_grid(parameter_domains(sweep))

	if sweep.kind == "list":
		if set(sweep.params) == {"cases"}:
			cases = _sequence(sweep.params["cases"], "cases")
			if not all(isinstance(case, dict) for case in cases):
				raise CandidateSpaceError("list sweep cases must be mappings")
			return [dict(case) for case in cases]
		if len(sweep.params) != 1:
			raise CandidateSpaceError("list sweeps accept one parameter or a 'cases' list")
		name, values = next(iter(sweep.params.items()))
		return [{name: value} for value in _sequence(values, name)]

	if sweep.kind == "linspace":
		values = {name: _linspace(spec, name) for name, spec in sweep.params.items()}
	else:
		values = {name: _sequence(spec, name) for name, spec in sweep.params.items()}

	if sweep.kind == "zip":
		lengths = {len(items) for items in values.values()}
		if len(lengths) != 1:
			raise CandidateSpaceError("zip sweep parameter sequences must have equal lengths")
		return [
			dict(zip(values, items, strict=True))
			for items in zip(*values.values(), strict=True)
		]

	# Grid and legacy linspace preserve their existing Cartesian ordering.
	return [
		dict(zip(values, items, strict=True))
		for items in product(*values.values())
	]


def _case_variants(stage: StageConfig, case: StageCase, prefix: str) -> list[_StageVariant]:
	variants = []
	for overrides in _domain_grid(case.domains):
		concrete = stage.model_copy(deep=True)
		concrete.algorithm = case.algorithm
		concrete.params.update(case.params)
		concrete.params.update(overrides)
		concrete.sweep = None
		values: dict[str, Any] = {
			f"{prefix}.case": case.name,
			f"{prefix}.algorithm": case.algorithm,
		}
		values.update({
			f"{prefix}.{name}": value
			for name, value in {**case.params, **overrides}.items()
		})
		variants.append(_StageVariant(concrete, values))
	return variants


def _stage_variants(stage: StageConfig, index: int) -> list[_StageVariant]:
	prefix = f"stage[{stage.id}]" if stage.id is not None else f"stage[{index}]"
	if stage.sweep is None:
		return [_StageVariant(stage.model_copy(deep=True), {})]
	if stage.sweep.kind == "cases":
		return [
			variant
			for case in stage.sweep.cases
			for variant in _case_variants(stage, case, prefix)
		]
	variants = []
	for overrides in expand_sweep(stage.sweep):
		concrete = stage.model_copy(deep=True)
		concrete.params.update(overrides)
		concrete.sweep = None
		qualified = {f"{prefix}.{name}": value for name, value in overrides.items()}
		variants.append(_StageVariant(concrete, qualified))
	return variants


def _structures(recon: Recon) -> list[tuple[str | None, list[tuple[int, StageConfig]]]]:
	indexed = list(enumerate(recon.pipeline))
	if not recon.variants:
		return [(None, indexed)]
	by_id = {stage.id: (index, stage) for index, stage in indexed}
	return [
		(variant.name, [by_id[stage_id] for stage_id in variant.stages])
		for variant in recon.variants
	]


def _stage_variant_count(stage: StageConfig) -> int:
	if stage.sweep is None:
		return 1
	if stage.sweep.kind != "cases":
		return len(expand_sweep(stage.sweep))
	return sum(len(_domain_grid(case.domains)) for case in stage.sweep.cases)


def count_recon_candidates(recon: Recon) -> int:
	"""Count finite candidates without constructing their Cartesian product."""

	total = 0
	for _, structure in _structures(recon):
		count = 1
		for _, stage in structure:
			count *= _stage_variant_count(stage)
		total += count
	return total


def has_continuous_domains(recon: Recon) -> bool:
	"""Return whether any parameter or algorithm case has a continuous domain."""

	for stage in recon.pipeline:
		if stage.sweep is None:
			continue
		if stage.sweep.kind == "domain":
			if any(
				isinstance(domain, (LinearDomain, LogDomain)) and domain.count is None
				for domain in parameter_domains(stage.sweep).values()
			):
				return True
		if stage.sweep.kind == "cases":
			for case in stage.sweep.cases:
				if any(
					isinstance(domain, (LinearDomain, LogDomain)) and domain.count is None
					for domain in case.domains.values()
				):
					return True
	return False


def candidate_key(recon: Recon) -> str:
	"""Hash canonical resolved candidate content independently of enumeration."""

	payload = {
		"pipeline": [
			stage.model_dump(mode="json", exclude={"sweep"})
			for stage in recon.pipeline
		],
		"pipeline_template": recon.pipeline_template,
		"metadata": recon.model_dump(mode="json")["metadata"],
	}
	encoded = json.dumps(
		payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
	).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def expand_recon_sweeps(recon: Recon, *, max_candidates: int | None = None) -> list[SweepRun]:
	"""Return independent, complete, sweep-free candidates in deterministic order."""

	limit = recon.candidate_limits.max_candidates if max_candidates is None else max_candidates
	count = count_recon_candidates(recon)
	if count > limit:
		raise CandidateLimitError(
			f"candidate space contains {count} runs, exceeding max_candidates={limit}; "
			"narrow the space or raise recon.candidate_limits.max_candidates explicitly"
		)

	runs: list[SweepRun] = []
	for variant_name, structure in _structures(recon):
		stage_variants = [
			_stage_variants(stage, original_index)
			for original_index, stage in structure
		]
		for combination in product(*stage_variants):
			concrete = recon.model_copy(deep=True)
			concrete.variants = []
			concrete.pipeline = [item.stage for item in combination]
			parameters = {
				name: value
				for item in combination
				for name, value in item.values.items()
			}
			if variant_name is not None:
				parameters = {"pipeline.variant": variant_name, **parameters}
			key = candidate_key(concrete)
			runs.append(SweepRun(
				index=len(runs),
				recon=concrete,
				parameters=parameters,
				candidate_key=key,
				variant=variant_name,
			))
	return runs
