"""Pipeline contract validation and concrete plan construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from lftomo.pipeline.core import Backend, Locality, NumpyBackend, Stage
from lftomo.pipeline.domains import DataDomain
from lftomo.pipeline.planning import (
	Materialization,
	MaterializationKind,
	ExecutionKind,
	PipelinePlan,
	PlanSegment,
	PlanStep,
	_transition_role,
	build_allocation_schedule,
	stage_allocation_spec,
	stage_output_spec,
)
from lftomo.util.window import TomoDataSet, TomoWindow
from lftomo.util.shapes import ProjOrder, SinoOrder


class ValidationMode(str, Enum):
	"""Validation scope for a complete YAML pipeline or reusable fragment."""

	COMPLETE = "complete"
	FRAGMENT = "fragment"


@dataclass
class PipelineValidationError(ValueError):
	"""Base class for Pipeline validation failures."""

	stage: str
	field: str
	message: str

	def __str__(self) -> str:
		return f"{self.stage}: {self.field}: {self.message}"


class ShapeCompatibilityError(PipelineValidationError):
	"""Raised when adjacent Stage layout contracts do not compose."""


class DomainCompatibilityError(PipelineValidationError):
	"""Raised when Stage domains do not compose monotonically."""


class CompletePipelineError(PipelineValidationError):
	"""Raised when a complete runner/YAML pipeline violates its skeleton."""


class RowIdentityError(PipelineValidationError):
	"""Raised when a detector-row operation follows detector resampling."""


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


def _layout_name(layout: type[Any] | None) -> str:
	return "unknown" if layout is None else layout.__name__


def _domain_name(domain: DataDomain | None) -> str:
	return "unknown" if domain is None else domain.name.lower()


def _complete_skeleton(stages: tuple[Stage, ...] | list[Stage]) -> None:
	if not stages or not stages[0].name.startswith("read."):
		raise CompletePipelineError(
			stage="0:<pipeline>", field="read",
			message="complete pipelines must begin with exactly one read Stage",
		)
	read_indices = [index for index, stage in enumerate(stages) if stage.name.startswith("read.")]
	if read_indices != [0]:
		raise CompletePipelineError(
			stage=f"{read_indices[-1]}:{stages[read_indices[-1]].name}", field="read",
			message="complete pipelines may only read at index 0",
		)


def validate_pipeline(
	stages: tuple[Stage, ...] | list[Stage],
	scan: Any,
	backend: Backend | None = None,
	available_artifacts: Any = (),
	*,
	mode: ValidationMode | str = ValidationMode.FRAGMENT,
	initial_layout: type[Any] | None = None,
	initial_domain: DataDomain | None = None,
	enable_aliasing: bool = True,
	enable_engine_aliasing: bool = True,
	external_pins: tuple[str, ...] = (),
	dataset: TomoDataSet | None = None,
	source_window: TomoWindow | None = None,
) -> PipelinePlan:
	"""Validate a Stage sequence and return its resolved execution plan."""

	try:
		mode = ValidationMode(mode)
	except ValueError as error:
		raise ValueError("validation mode must be 'complete' or 'fragment'") from error
	if mode is ValidationMode.COMPLETE:
		_complete_skeleton(stages)
		current_layout = None
		current_domain = None
	else:
		current_layout = initial_layout
		current_domain = initial_domain

	backend = backend or NumpyBackend()
	produced_artifacts = set(available_artifacts)
	upstream_mutated = False
	row_identity_available = True
	normalize_transitions = 0
	attenuation_transitions = 0
	steps: list[PlanStep] = []
	warnings: list[str] = []
	current_buffer = None
	current_window = source_window

	for index, stage in enumerate(stages):
		occurrence = getattr(stage, "occurrence_id", None)
		stage_name = f"{occurrence if occurrence is not None else index}:{stage.name}"
		contract = stage.contract
		input_layout = current_layout
		input_domain = current_domain
		input_buffer = current_buffer

		accepted = contract.accepted_layouts
		if (
			current_layout is not None
			and accepted is not None
			and current_layout not in accepted
			and current_layout not in contract.bounded_views_from
		):
			expected = ", ".join(item.__name__ for item in accepted)
			raise ShapeCompatibilityError(
				stage=stage_name, field="accepts",
				message=f"accepts ({expected}), got {_layout_name(current_layout)}; add an explicit layout.convert",
			)

		accepted_domains = contract.accepts_domain
		if current_domain is not None and accepted_domains is not None and current_domain not in accepted_domains:
			expected = ", ".join(item.name.lower() for item in accepted_domains)
			raise DomainCompatibilityError(
				stage=stage_name, field="accepts_domain",
				message=f"accepts ({expected}), got {_domain_name(current_domain)}",
			)

		output_layout = contract.output_layout(current_layout)
		output_domain = contract.output_domain(current_domain)
		if (
			current_layout in (ProjOrder, SinoOrder)
			and output_layout in (ProjOrder, SinoOrder)
			and output_layout is not current_layout
			and stage.name != "layout.convert"
			and not contract.fused_transition
		):
			raise ShapeCompatibilityError(
				stage=stage_name, field="produces",
				message="tomography layout changes require layout.convert or fused_transition=True",
			)
		if current_domain is not None and output_domain is not None and output_domain.value < current_domain.value:
			raise DomainCompatibilityError(
				stage=stage_name, field="produces_domain",
				message=f"domain cannot move backward from {_domain_name(current_domain)} to {_domain_name(output_domain)}",
			)

		if current_domain is DataDomain.RAW and output_domain is DataDomain.TRANSMISSION:
			if not stage.name.startswith("normalize.") and stage.name != "weighted-normalize":
				raise DomainCompatibilityError(
					stage=stage_name, field="produces_domain",
					message="only normalize Stages may convert raw counts to transmission",
				)
			normalize_transitions += 1
		if current_domain is DataDomain.TRANSMISSION and output_domain is DataDomain.ATTENUATION:
			attenuation_transitions += 1

		for metadata_path in contract.required_metadata:
			if not _has_path(scan, metadata_path):
				raise MissingMetadataError(
					stage=stage_name, field=f"required_metadata.{metadata_path}",
					message="missing from Scan config",
				)

		for artifact_name in contract.consumes_artifacts:
			if artifact_name not in produced_artifacts:
				raise MissingArtifactError(
					stage=stage_name, field=f"consumes_artifacts.{artifact_name}",
					message="not produced by an upstream Stage",
				)

		if contract.resources.requires_gpu and not backend.supports_gpu:
			raise BackendCompatibilityError(
				stage=stage_name, field="resources.requires_gpu",
				message=f"Stage requires GPU but backend {backend.name!r} does not support GPU execution",
			)

		if contract.forks_pipeline and upstream_mutated:
			raise MutationForkError(
				stage=stage_name, field="forks_pipeline",
				message="cannot share a prefix after an upstream Stage mutated input state",
			)

		if contract.requires_row_identity and not row_identity_available:
			raise RowIdentityError(
				stage=stage_name, field="requires_row_identity",
				message="detector row identity was destroyed by an upstream resampling Stage",
			)

		step_warnings: list[str] = []
		preferred = contract.preferred_layout
		if input_layout is not None and preferred is not None and input_layout is not preferred:
			message = (
				f"{stage_name} accepts {_layout_name(input_layout)} but prefers {preferred.__name__}"
			)
			step_warnings.append(message)
			warnings.append(message)

		materializations: list[Materialization] = []
		if stage.name == "layout.convert":
			reason = str(getattr(stage, "reason", "")).strip()
			if not reason:
				raise ShapeCompatibilityError(
					stage=stage_name, field="reason",
					message="YAML-authored layout conversions require a non-empty reason",
				)
			if input_layout is not None and output_layout is input_layout:
				raise ShapeCompatibilityError(
					stage=stage_name, field="produces",
					message="layout.convert must change the current layout",
				)
			materializations.append(Materialization(
				MaterializationKind.LAYOUT_CONVERSION, reason=reason,
			))
		elif stage.name.startswith("recon.") and input_layout is ProjOrder:
			materializations.append(Materialization(
				MaterializationKind.TOMOPY_BOUNDARY,
				reason="TomoPy materializes its contiguous SinoOrder working array",
				hidden=True,
			))

		output_buffer = stage_output_spec(
			stage, input_buffer, scan, dataset=dataset, source_window=current_window,
		)
		allocation_buffer = stage_allocation_spec(
			stage, input_buffer, output_buffer, source_window=current_window,
		)
		output_role, retires_role = _transition_role(stage)
		steps.append(PlanStep(
			index=index, stage_name=stage.name,
			input_layout=input_layout, output_layout=output_layout,
			input_domain=input_domain, output_domain=output_domain,
			materializations=tuple(materializations), warnings=tuple(step_warnings),
			input_buffer=input_buffer, output_buffer=output_buffer,
			allocation_buffer=allocation_buffer,
			output_role=output_role, retires_role=retires_role,
		))
		upstream_mutated = upstream_mutated or contract.mutates_input
		row_identity_available = row_identity_available and not contract.resamples_detector
		produced_artifacts.update(contract.produces_artifacts)
		current_layout = output_layout
		current_domain = output_domain
		current_buffer = output_buffer
		if current_window is not None and output_layout in (ProjOrder, SinoOrder):
			current_window = stage.output_window(current_window)

	if mode is ValidationMode.COMPLETE:
		if normalize_transitions != 1:
			raise CompletePipelineError(
				stage="<pipeline>", field="normalize",
				message=f"expected exactly one raw-to-transmission transition, found {normalize_transitions}",
			)
		if any(stage.name.startswith("recon.") for stage in stages) and attenuation_transitions != 1:
			raise CompletePipelineError(
				stage="<pipeline>", field="attenuation",
				message=f"expected exactly one transmission-to-attenuation transition, found {attenuation_transitions}",
			)

	segments: list[PlanSegment] = []
	index = 0
	while index < len(stages):
		stage = stages[index]
		step = steps[index]
		contract = stage.contract
		bounded = (
			step.input_layout is ProjOrder
			and contract.locality is Locality.SINOGRAM
			and ProjOrder in contract.bounded_views_from
			and step.output_layout is step.input_layout
			and step.output_domain is step.input_domain
			and not contract.consumes_artifacts
			and not contract.produces_artifacts
			and not contract.forks_pipeline
			and callable(getattr(stage, "run_sinogram", None))
		)
		if not bounded:
			segments.append(PlanSegment(
				start=index, stop=index + 1, kind=ExecutionKind.SEQUENTIAL,
				stage_names=(stage.name,),
			))
			index += 1
			continue
		stop = index + 1
		while stop < len(stages):
			next_stage = stages[stop]
			next_step = steps[stop]
			next_contract = next_stage.contract
			if not (
				next_step.input_layout is ProjOrder
				and next_contract.locality is Locality.SINOGRAM
				and ProjOrder in next_contract.bounded_views_from
				and next_step.output_layout is next_step.input_layout
				and next_step.output_domain is next_step.input_domain
				and not next_contract.consumes_artifacts
				and not next_contract.produces_artifacts
				and not next_contract.forks_pipeline
				and callable(getattr(next_stage, "run_sinogram", None))
			):
				break
			stop += 1
		segments.append(PlanSegment(
			start=index, stop=stop, kind=ExecutionKind.BOUNDED_SINOGRAM,
			stage_names=tuple(item.name for item in stages[index:stop]),
		))
		index = stop

	resolved_steps = tuple(steps)
	schedule = build_allocation_schedule(
		resolved_steps, stages, scan,
		enable_aliasing=enable_aliasing,
		enable_engine_aliasing=enable_engine_aliasing,
		external_pins=external_pins,
	)
	return PipelinePlan(
		steps=resolved_steps, segments=tuple(segments), warnings=tuple(warnings),
		allocation_schedule=schedule,
	)
