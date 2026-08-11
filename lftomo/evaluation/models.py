"""Typed evaluation requests, observations, aggregation, and preview contracts."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from lftomo.config import ConfigModel


MetricDirection = Literal["minimize", "maximize", "constraint"]
AggregationMethod = Literal["mean", "median", "weighted_mean"]


class ImageROI(ConfigModel):
	"""Explicit half-open ROI in one reconstructed image slice."""

	y_start: int = Field(ge=0)
	y_stop: int = Field(gt=0)
	x_start: int = Field(ge=0)
	x_stop: int = Field(gt=0)

	@model_validator(mode="after")
	def _stops_follow_starts(self):
		if self.y_stop <= self.y_start or self.x_stop <= self.x_start:
			raise ValueError("ROI stops must be greater than starts")
		return self


class EvaluationWindow(ConfigModel):
	"""Explicit half-open first-axis window in a ReconOrder volume."""

	id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
	slice_start: int = Field(ge=0)
	slice_stop: int = Field(gt=0)
	weight: float = Field(default=1.0, gt=0)

	@model_validator(mode="after")
	def _stop_follows_start(self):
		if self.slice_stop <= self.slice_start:
			raise ValueError("evaluation window stop must be greater than start")
		return self


class ReferenceMetricRequest(ConfigModel):
	"""Request the existing NRMSE/correlation/normalized-bias adapter."""

	kind: Literal["reference"] = "reference"
	reference_artifact: str = "evaluation_reference"


class ValidityMetricRequest(ConfigModel):
	"""Request non-finite and explicit clipping/saturation measurements."""

	kind: Literal["validity"] = "validity"
	clip_min: float | None = None
	clip_max: float | None = None

	@model_validator(mode="after")
	def _clipping_range_is_valid(self):
		if self.clip_min is None and self.clip_max is None:
			raise ValueError("validity metric requires clip_min and/or clip_max")
		if (
			self.clip_min is not None
			and self.clip_max is not None
			and self.clip_max <= self.clip_min
		):
			raise ValueError("clip_max must be greater than clip_min")
		return self


class RingMetricRequest(ConfigModel):
	"""Request a radial high-frequency ring-artifact score."""

	kind: Literal["ring"] = "ring"
	center_y: float | None = None
	center_x: float | None = None
	smoothing_sigma: float = Field(default=2.0, gt=0)
	inner_radius: float = Field(default=2.0, ge=0)
	outer_radius: float | None = Field(default=None, gt=0)


class RoiNoiseCnrMetricRequest(ConfigModel):
	"""Request background-noise and contrast-to-noise measurements."""

	kind: Literal["roi_noise_cnr"] = "roi_noise_cnr"
	background: ImageROI
	signal: ImageROI


MetricRequest = Annotated[
	ReferenceMetricRequest
	| ValidityMetricRequest
	| RingMetricRequest
	| RoiNoiseCnrMetricRequest,
	Field(discriminator="kind"),
]


class AggregationConfig(ConfigModel):
	"""Deterministic aggregation and explicit invalid-observation policy."""

	method: AggregationMethod = "mean"
	invalid: Literal["fail", "exclude", "worst"] = "fail"
	window_weights: dict[str, float] = Field(default_factory=dict)
	worst_values: dict[str, float] = Field(default_factory=dict)

	@model_validator(mode="after")
	def _policy_fields_match(self):
		if self.method == "weighted_mean" and not self.window_weights:
			raise ValueError("weighted_mean requires explicit window_weights")
		if self.invalid == "worst" and not self.worst_values:
			raise ValueError("invalid='worst' requires finite per-metric worst_values")
		if any(not math.isfinite(value) for value in self.worst_values.values()):
			raise ValueError("worst_values must be finite")
		if any(not math.isfinite(value) or value <= 0 for value in self.window_weights.values()):
			raise ValueError("window_weights must be finite and positive")
		return self


class PreviewContract(ConfigModel):
	"""Consistent-scale/blinding metadata for future preview export."""

	scale_min: float
	scale_max: float
	crop: ImageROI | None = None
	blinded: bool = True
	format: Literal["png", "tiff"] = "png"

	@model_validator(mode="after")
	def _scale_is_ordered(self):
		if self.scale_max <= self.scale_min:
			raise ValueError("preview scale_max must be greater than scale_min")
		return self


class MetricDefinition(ConfigModel):
	"""Static registry declaration for one emitted scalar measurement."""

	name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
	direction: MetricDirection
	units: str
	normalization: str
	required_inputs: tuple[str, ...] = ()
	default_aggregation: AggregationMethod = "mean"
	undefined: Literal["record_invalid", "fail_stage"]
	protocol_version: str
	resource_scope: Literal["scalar", "slice", "volume"] = "slice"


class MetricObservation(ConfigModel):
	"""Raw metric value for one slice in one configured window."""

	metric: str
	window_id: str
	slice_index: int = Field(ge=0)
	value: float | None
	valid: bool
	reason: str | None = None
	direction: MetricDirection
	units: str
	normalization: str
	protocol_version: str
	sample_count: int = Field(ge=0)
	invalid_count: int = Field(default=0, ge=0)

	@model_validator(mode="after")
	def _value_matches_validity(self):
		if self.valid and self.value is None:
			raise ValueError("valid observations require a value")
		if self.valid and not math.isfinite(self.value):
			raise ValueError("valid observations require a finite value")
		if not self.valid and not self.reason:
			raise ValueError("invalid observations require a reason")
		return self


class AggregatedMetric(ConfigModel):
	"""Aggregate retaining counts and the exact invalid handling used."""

	metric: str
	value: float | None
	valid: bool
	reason: str | None = None
	direction: MetricDirection
	units: str
	normalization: str
	protocol_version: str
	method: AggregationMethod
	invalid_policy: Literal["fail", "exclude", "worst"]
	observation_count: int = Field(ge=0)
	valid_observation_count: int = Field(ge=0)
	sample_count: int = Field(ge=0)
	invalid_count: int = Field(ge=0)

	@model_validator(mode="after")
	def _value_matches_validity(self):
		if self.valid and (self.value is None or not math.isfinite(self.value)):
			raise ValueError("valid aggregates require a finite value")
		if not self.valid and not self.reason:
			raise ValueError("invalid aggregates require a reason")
		return self


class EvaluationArtifact(ConfigModel):
	"""Versioned, JSON-serializable output of the preserving evaluation Stage."""

	schema_version: Literal[1] = 1
	protocol_version: str
	windows: tuple[EvaluationWindow, ...]
	requests: tuple[MetricRequest, ...]
	definitions: dict[str, MetricDefinition]
	observations: tuple[MetricObservation, ...]
	aggregates: dict[str, AggregatedMetric]
	preview: PreviewContract | None = None
	metadata: dict[str, JsonValue] = Field(default_factory=dict)
