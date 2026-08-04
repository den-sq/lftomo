"""Deterministic aggregation of raw per-slice/window metric observations."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from lftomo.evaluation.models import (
	AggregatedMetric,
	AggregationConfig,
	MetricObservation,
)


def aggregate_observations(
	observations: list[MetricObservation] | tuple[MetricObservation, ...],
	config: AggregationConfig,
) -> dict[str, AggregatedMetric]:
	"""Aggregate observations without discarding raw values or invalid counts."""

	groups: dict[str, list[MetricObservation]] = defaultdict(list)
	for observation in observations:
		groups[observation.metric].append(observation)
	result = {}
	for metric in sorted(groups):
		items = groups[metric]
		first = items[0]
		for item in items[1:]:
			if (
				item.direction, item.units, item.normalization, item.protocol_version
			) != (
				first.direction, first.units, first.normalization, first.protocol_version
			):
				raise ValueError(f"inconsistent observation definitions for metric {metric!r}")
		valid_items = [item for item in items if item.valid and item.value is not None]
		invalid_items = [item for item in items if not item.valid or item.value is None]
		values = [float(item.value) for item in valid_items]
		weights = [
			config.window_weights.get(item.window_id, 1.0)
			for item in valid_items
		]
		reason = None
		if invalid_items and config.invalid == "fail":
			value = None
			reason = (
				f"{len(invalid_items)} invalid observation(s): "
				+ "; ".join(sorted({item.reason or "undefined" for item in invalid_items}))
			)
		elif config.invalid == "worst" and invalid_items:
			if metric not in config.worst_values:
				value = None
				reason = f"missing finite worst_values entry for metric {metric!r}"
			else:
				values.extend([config.worst_values[metric]] * len(invalid_items))
				weights.extend([
					config.window_weights.get(item.window_id, 1.0)
					for item in invalid_items
				])
				value = _aggregate(values, weights, config)
		else:
			value = _aggregate(values, weights, config) if values else None
			if value is None:
				reason = "no valid observations"
		result[metric] = AggregatedMetric(
			metric=metric,
			value=value,
			valid=value is not None and reason is None,
			reason=reason,
			direction=first.direction,
			units=first.units,
			normalization=first.normalization,
			protocol_version=first.protocol_version,
			method=config.method,
			invalid_policy=config.invalid,
			observation_count=len(items),
			valid_observation_count=len(valid_items),
			sample_count=sum(item.sample_count for item in items),
			invalid_count=sum(item.invalid_count for item in items),
		)
	return result


def _aggregate(values: list[float], weights: list[float], config: AggregationConfig) -> float:
	array = np.asarray(values, dtype=np.float64)
	if config.method == "median":
		return float(np.median(array))
	if config.method == "weighted_mean":
		weight_array = np.asarray(weights, dtype=np.float64)
		if not np.isfinite(weight_array).all() or np.any(weight_array <= 0):
			raise ValueError("window aggregation weights must be finite and positive")
		return float(np.average(array, weights=weight_array))
	return float(np.mean(array))
