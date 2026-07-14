"""Pipeline contract validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lftomo.pipeline.core import Backend, NumpyBackend, Stage


@dataclass
class PipelineValidationError(ValueError):
	"""Base class for Pipeline validation failures."""

	stage: str
	field: str
	message: str

	def __str__(self) -> str:
		return f"{self.stage}: {self.field}: {self.message}"


class ShapeCompatibilityError(PipelineValidationError):
	"""Raised when adjacent Stage shape contracts do not compose."""


class MissingMetadataError(PipelineValidationError):
	"""Raised when a Stage requires metadata absent from Scan."""


class MissingArtifactError(PipelineValidationError):
	"""Raised when a Stage consumes an artifact no upstream Stage produces."""


class BackendCompatibilityError(PipelineValidationError):
	"""Raised when a Stage requires a capability the backend does not provide."""


class MutationForkError(PipelineValidationError):
	"""Raised when a fork assumes a shared prefix after upstream mutation."""


def _has_path(obj: Any, dotted_path: str) -> bool:
	current = obj
	for part in dotted_path.split("."):
		if isinstance(current, dict):
			if part not in current:
				return False
			current = current[part]
		else:
			if not hasattr(current, part):
				return False
			current = getattr(current, part)
		if current is None:
			return False
	return True


def validate_pipeline(
	stages: tuple[Stage, ...] | list[Stage],
	scan: Any,
	backend: Backend | None = None,
	available_artifacts: Any = (),
) -> None:
	"""Validate an ordered Stage sequence before execution."""

	backend = backend or NumpyBackend()
	produced_artifacts = set(available_artifacts)
	previous_stage: Stage | None = None
	upstream_mutated = False

	for index, stage in enumerate(stages):
		stage_name = f"{index}:{stage.name}"
		contract = stage.contract

		if previous_stage is not None:
			previous_shape = previous_stage.contract.produced_output
			expected_shape = contract.expected_input
			if previous_shape is not None and expected_shape is not None and previous_shape is not expected_shape:
				raise ShapeCompatibilityError(
					stage=stage_name,
					field="expected_input",
					message=f"expected {expected_shape.__name__}, got {previous_shape.__name__} from {previous_stage.name}",
				)

		for metadata_path in contract.required_metadata:
			if not _has_path(scan, metadata_path):
				raise MissingMetadataError(
					stage=stage_name,
					field=f"required_metadata.{metadata_path}",
					message="missing from Scan config",
				)

		for artifact_name in contract.consumes_artifacts:
			if artifact_name not in produced_artifacts:
				raise MissingArtifactError(
					stage=stage_name,
					field=f"consumes_artifacts.{artifact_name}",
					message="not produced by an upstream Stage",
				)

		if contract.resources.requires_gpu and not backend.supports_gpu:
			raise BackendCompatibilityError(
				stage=stage_name,
				field="resources.requires_gpu",
				message=f"Stage requires GPU but backend {backend.name!r} does not support GPU execution",
			)

		if contract.forks_pipeline and upstream_mutated:
			raise MutationForkError(
				stage=stage_name,
				field="forks_pipeline",
				message="cannot share a prefix after an upstream Stage mutated input state",
			)

		upstream_mutated = upstream_mutated or contract.mutates_input
		produced_artifacts.update(contract.produces_artifacts)
		previous_stage = stage
