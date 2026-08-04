"""Validation seam between experiment metric references and the registry."""

from __future__ import annotations

from lftomo.evaluation.metrics import METRIC_REGISTRY
from lftomo.evaluation.registry import MetricRegistry
from lftomo.experiment import ExperimentConfig


def validate_experiment_metric_references(
	config: ExperimentConfig,
	registry: MetricRegistry = METRIC_REGISTRY,
) -> None:
	"""Validate experiment references against concrete versioned definitions."""

	errors = []
	for reference in config.metrics:
		try:
			definition = registry.definition(reference.metric)
		except ValueError as error:
			errors.append(f"{reference.id}: {error}")
			continue
		if reference.protocol_version != definition.protocol_version:
			errors.append(
				f"{reference.id}: protocol_version {reference.protocol_version!r} "
				f"!= registry {definition.protocol_version!r}"
			)
		if reference.direction != definition.direction:
			errors.append(
				f"{reference.id}: direction {reference.direction!r} "
				f"!= registry {definition.direction!r}"
			)
		if reference.units is not None and reference.units != definition.units:
			errors.append(
				f"{reference.id}: units {reference.units!r} != registry {definition.units!r}"
			)
	if errors:
		raise ValueError("experiment metric reference validation failed: " + "; ".join(errors))
