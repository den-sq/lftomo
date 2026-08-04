"""Preserving reconstruction-domain evaluation Pipeline Stage."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import TypeAdapter

from lftomo.config import StageConfig
from lftomo.evaluation.aggregation import aggregate_observations
from lftomo.evaluation.metrics import METRIC_REGISTRY, PROTOCOL_VERSION
from lftomo.evaluation.models import (
	AggregationConfig,
	EvaluationArtifact,
	EvaluationWindow,
	MetricObservation,
	MetricRequest,
	PreviewContract,
	ReferenceMetricRequest,
)
from lftomo.pipeline.core import Locality, PipelineState, ResourceHints, Stage, StageContract
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.util.shapes import ReconOrder


_REQUEST_ADAPTER = TypeAdapter(MetricRequest)


class EvaluationStage(Stage):
	"""Evaluate configured slices/windows without changing reconstruction data."""

	name = "evaluate.metrics"
	contract = StageContract(
		expected_input=ReconOrder,
		produced_output=ReconOrder,
		accepts=(ReconOrder,),
		prefers=ReconOrder,
		produces=PRESERVE,
		accepts_domain=(DataDomain.RECON,),
		produces_domain=PRESERVE,
		locality=Locality.VOLUME,
		mutates_input=False,
		window_axes=("Y",),
		produces_artifacts=("metrics",),
		resources=ResourceHints(memory_multiplier=4.0, preferred_cpus=1),
	)

	def __init__(
		self,
		metrics: list[dict[str, Any] | MetricRequest],
		windows: list[dict[str, Any] | EvaluationWindow] | None = None,
		aggregation: dict[str, Any] | AggregationConfig | None = None,
		preview: dict[str, Any] | PreviewContract | None = None,
		protocol_version: str = PROTOCOL_VERSION,
	):
		if not metrics:
			raise ValueError("evaluate.metrics requires at least one metric request")
		if protocol_version != PROTOCOL_VERSION:
			raise ValueError(
				f"unsupported evaluation protocol_version {protocol_version!r}; "
				f"expected {PROTOCOL_VERSION!r}"
			)
		self.metrics = tuple(
			item if hasattr(item, "kind") else _REQUEST_ADAPTER.validate_python(item)
			for item in metrics
		)
		self.windows = tuple(
			item if isinstance(item, EvaluationWindow) else EvaluationWindow.model_validate(item)
			for item in (windows or ())
		)
		if len({window.id for window in self.windows}) != len(self.windows):
			raise ValueError("evaluation window IDs must be unique")
		self.aggregation = (
			aggregation if isinstance(aggregation, AggregationConfig)
			else AggregationConfig.model_validate(aggregation or {})
		)
		self.preview = (
			preview if isinstance(preview, PreviewContract)
			else PreviewContract.model_validate(preview) if preview is not None
			else None
		)
		self.protocol_version = protocol_version
		for request in self.metrics:
			METRIC_REGISTRY.registration(request)

	@classmethod
	def from_config(cls, config: StageConfig) -> "EvaluationStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("evaluate.metrics currently requires the NumPy backend")
		working = np.asarray(state.working_array)
		if working.ndim != 3:
			raise ValueError(
				"evaluate.metrics expects a three-dimensional (Y, Z, X) reconstruction"
			)
		windows = self.windows or (
			EvaluationWindow(id="full", slice_start=0, slice_stop=working.shape[0]),
		)
		if self.aggregation.method == "weighted_mean":
			window_ids = {window.id for window in windows}
			weight_ids = set(self.aggregation.window_weights)
			if weight_ids != window_ids:
				raise ValueError(
					"weighted_mean window_weights must match configured windows exactly; "
					f"missing={sorted(window_ids - weight_ids)}, "
					f"unknown={sorted(weight_ids - window_ids)}"
				)
		for window in windows:
			if window.slice_stop > working.shape[0]:
				raise ValueError(
					f"evaluation window {window.id!r} [{window.slice_start}, "
					f"{window.slice_stop}) exceeds {working.shape[0]} slices"
				)
		references: dict[str, np.ndarray] = {}
		for request in self.metrics:
			if not isinstance(request, ReferenceMetricRequest):
				continue
			try:
				reference = np.asarray(state.artifacts[request.reference_artifact])
			except KeyError as error:
				raise ValueError(
					f"reference metrics require artifact {request.reference_artifact!r}"
				) from error
			if reference.shape != working.shape:
				raise ValueError(
					f"reference artifact {request.reference_artifact!r} shape "
					f"{reference.shape} does not match reconstruction {working.shape}"
				)
			references[request.reference_artifact] = reference

		observations = []
		definitions = {}
		for window in windows:
			for slice_index in range(window.slice_start, window.slice_stop):
				image = working[slice_index]
				for request in self.metrics:
					METRIC_REGISTRY.registration(request)
					context: dict[str, Any] = {}
					if isinstance(request, ReferenceMetricRequest):
						context["reference"] = references[request.reference_artifact][slice_index]
					for kernel_value in METRIC_REGISTRY.evaluate(image, request, context):
						definition = METRIC_REGISTRY.definition(kernel_value.metric)
						definitions[definition.name] = definition
						valid = kernel_value.valid
						reason = kernel_value.reason
						if not valid and reason is None:
							reason = "metric returned a non-finite value"
						observations.append(MetricObservation(
							metric=kernel_value.metric,
							window_id=window.id,
							slice_index=slice_index,
							value=kernel_value.value if valid else None,
							valid=valid,
							reason=reason,
							direction=definition.direction,
							units=definition.units,
							normalization=definition.normalization,
							protocol_version=definition.protocol_version,
							sample_count=kernel_value.sample_count,
							invalid_count=kernel_value.invalid_count,
						))
		aggregates = aggregate_observations(observations, self.aggregation)
		artifact = EvaluationArtifact(
			protocol_version=self.protocol_version,
			windows=windows,
			requests=self.metrics,
			definitions=definitions,
			observations=tuple(observations),
			aggregates=aggregates,
			preview=self.preview,
			metadata={
				"bounded_axis": "Y",
				"maximum_kernel_input_shape": [working.shape[1], working.shape[2]],
			},
		)
		state.artifacts["metrics"] = artifact.model_dump(mode="json")
		return state
