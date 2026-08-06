"""Shared runtime-requirement declarations for built-in Pipeline Stages."""

from __future__ import annotations

from lftomo.pipeline.core import RuntimeRequirement


_TOMOPY_CORRECTIVE_ACTION = (
	"install the lftomo tomopy extra with TomoPy 1.15.3 or newer in this interpreter "
	"(for example: python -m pip install 'lftomo[tomopy]')"
)


def tomopy_requirement(*probe_attributes: str) -> RuntimeRequirement:
	"""Return the shared TomoPy requirement with Stage-specific API probes."""

	return RuntimeRequirement(
		capability="tomopy",
		package="tomopy",
		module="tomopy",
		probe_attributes=probe_attributes,
		corrective_action=_TOMOPY_CORRECTIVE_ACTION,
	)


ASTRA_RUNTIME_REQUIREMENT = RuntimeRequirement(
	capability="astra-cuda",
	package="astra-toolbox",
	module="astra",
	corrective_action="install a compatible ASTRA Toolbox build in this interpreter",
)


__all__ = ["ASTRA_RUNTIME_REQUIREMENT", "tomopy_requirement"]
