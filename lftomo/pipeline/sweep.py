"""Expand declarative stage sweeps into concrete reconstruction runs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np

from lftomo.config import Recon, StageConfig, SweepConfig


@dataclass(frozen=True)
class SweepRun:
	"""One concrete pipeline produced by expanding a reconstruction spec."""

	index: int
	recon: Recon
	parameters: dict[str, Any]


def _sequence(value: Any, name: str) -> list[Any]:
	if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
		raise ValueError(f"sweep parameter {name!r} must be a sequence")
	return list(value)


def _linspace(value: Any, name: str) -> list[Any]:
	if not isinstance(value, dict):
		raise ValueError(f"linspace parameter {name!r} must be a mapping")
	unknown = set(value) - {"start", "stop", "num", "endpoint"}
	if unknown or not {"start", "stop", "num"} <= set(value):
		raise ValueError(
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


def expand_sweep(sweep: SweepConfig) -> list[dict[str, Any]]:
	"""Expand one Stage sweep into parameter overrides."""
	if not sweep.params:
		raise ValueError("sweep.params must not be empty")

	if sweep.kind == "list":
		if set(sweep.params) == {"cases"}:
			cases = _sequence(sweep.params["cases"], "cases")
			if not all(isinstance(case, dict) for case in cases):
				raise ValueError("list sweep cases must be mappings")
			return [dict(case) for case in cases]
		if len(sweep.params) != 1:
			raise ValueError("list sweeps accept one parameter or a 'cases' list")
		name, values = next(iter(sweep.params.items()))
		return [{name: value} for value in _sequence(values, name)]

	if sweep.kind == "linspace":
		values = {name: _linspace(spec, name) for name, spec in sweep.params.items()}
	else:
		values = {name: _sequence(spec, name) for name, spec in sweep.params.items()}

	if sweep.kind == "zip":
		lengths = {len(items) for items in values.values()}
		if len(lengths) != 1:
			raise ValueError("zip sweep parameter sequences must have equal lengths")
		return [dict(zip(values, items, strict=True)) for items in zip(*values.values(), strict=True)]

	# grid and linspace both form the Cartesian product when several parameters
	# are present. A single linspace remains the common one-dimensional case.
	return [dict(zip(values, items, strict=True)) for items in product(*values.values())]


def expand_recon_sweeps(recon: Recon) -> list[SweepRun]:
	"""Return independent concrete Recon models for every sweep combination."""
	stage_variants: list[list[tuple[StageConfig, dict[str, Any]]]] = []
	for stage_index, stage in enumerate(recon.pipeline):
		if stage.sweep is None:
			stage_variants.append([(stage.model_copy(deep=True), {})])
			continue
		variants = []
		for overrides in expand_sweep(stage.sweep):
			concrete = stage.model_copy(deep=True)
			concrete.params.update(overrides)
			concrete.sweep = None
			qualified = {f"stage[{stage_index}].{name}": value for name, value in overrides.items()}
			variants.append((concrete, qualified))
		stage_variants.append(variants)

	runs = []
	for index, combination in enumerate(product(*stage_variants)):
		concrete = recon.model_copy(deep=True)
		concrete.pipeline = [stage for stage, _ in combination]
		parameters = {name: value for _, values in combination for name, value in values.items()}
		runs.append(SweepRun(index=index, recon=concrete, parameters=parameters))
	return runs
