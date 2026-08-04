"""Bounded first metric set for reconstruction-domain evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from lftomo.evaluation.models import (
	ReferenceMetricRequest,
	RingMetricRequest,
	RoiNoiseCnrMetricRequest,
	ValidityMetricRequest,
)
from lftomo.evaluation.registry import KernelValue, MetricRegistration, MetricRegistry
from lftomo.evaluation.models import MetricDefinition


PROTOCOL_VERSION = "1"


def _invalid(names: tuple[str, ...], reason: str, sample_count: int = 0) -> list[KernelValue]:
	return [
		KernelValue(name, None, sample_count=sample_count, invalid_count=sample_count, reason=reason)
		for name in names
	]


def reference_metrics(
	image: np.ndarray,
	request: ReferenceMetricRequest,
	context: dict[str, Any],
) -> list[KernelValue]:
	"""Reproduce the existing Tier-2 NRMSE/correlation/bias calculations."""

	reference = context.get("reference")
	names = (
		"reference.nrmse",
		"reference.correlation",
		"reference.normalized_bias",
	)
	if reference is None:
		raise ValueError(
			f"reference metrics require artifact {request.reference_artifact!r}"
		)
	actual = np.asarray(image, dtype=np.float64)
	expected = np.asarray(reference, dtype=np.float64)
	if actual.shape != expected.shape:
		raise ValueError(
			f"reference metric shape mismatch: {actual.shape} != {expected.shape}"
		)
	count = actual.size
	if count == 0:
		return _invalid(names, "empty image/reference", 0)
	if not np.isfinite(actual).all() or not np.isfinite(expected).all():
		invalid = int(count - np.count_nonzero(np.isfinite(actual) & np.isfinite(expected)))
		return [
			KernelValue(name, None, count, invalid, "non-finite image/reference")
			for name in names
		]
	dynamic_range = float(expected.max() - expected.min())
	if dynamic_range <= 0 or not np.isfinite(dynamic_range):
		return _invalid(names, "reference has no finite dynamic range", count)
	difference = actual - expected
	nrmse = float(np.sqrt(np.mean(difference * difference)) / dynamic_range)
	bias = float(abs(actual.mean() - expected.mean()) / dynamic_range)
	centered_actual = actual - actual.mean()
	centered_expected = expected - expected.mean()
	denominator = float(np.sqrt(
		np.sum(centered_actual * centered_actual)
		* np.sum(centered_expected * centered_expected)
	))
	correlation = (
		float(np.sum(centered_actual * centered_expected) / denominator)
		if denominator else 0.0
	)
	return [
		KernelValue("reference.nrmse", nrmse, count),
		KernelValue("reference.correlation", correlation, count),
		KernelValue("reference.normalized_bias", bias, count),
	]


def validity_metrics(
	image: np.ndarray,
	request: ValidityMetricRequest,
	_context: dict[str, Any],
) -> list[KernelValue]:
	values = np.asarray(image)
	count = values.size
	finite = np.isfinite(values)
	nonfinite_count = int(count - np.count_nonzero(finite))
	clipped = np.zeros(values.shape, dtype=bool)
	if request.clip_min is not None:
		clipped |= finite & (values <= request.clip_min)
	if request.clip_max is not None:
		clipped |= finite & (values >= request.clip_max)
	clipped_count = int(np.count_nonzero(clipped))
	return [
		KernelValue(
			"validity.nonfinite_count", float(nonfinite_count), count,
			invalid_count=nonfinite_count,
		),
		KernelValue(
			"validity.clipped_fraction",
			float(clipped_count / count) if count else None,
			count,
			invalid_count=nonfinite_count,
			reason=None if count else "empty image",
		),
	]


def ring_metric(
	image: np.ndarray,
	request: RingMetricRequest,
	_context: dict[str, Any],
) -> list[KernelValue]:
	values = np.asarray(image, dtype=np.float64)
	count = values.size
	if values.ndim != 2:
		raise ValueError(f"ring metric expects a two-dimensional slice, got {values.shape}")
	if not np.isfinite(values).all():
		invalid = int(count - np.count_nonzero(np.isfinite(values)))
		return [KernelValue("artifact.ring_score", None, count, invalid, "non-finite image")]
	center_y = (values.shape[0] - 1) / 2 if request.center_y is None else request.center_y
	center_x = (values.shape[1] - 1) / 2 if request.center_x is None else request.center_x
	y, x = np.indices(values.shape, dtype=np.float64)
	radius = np.floor(np.hypot(y - center_y, x - center_x)).astype(np.int64)
	outer = (
		min(values.shape) / 2 - 1
		if request.outer_radius is None
		else request.outer_radius
	)
	mask = (radius >= request.inner_radius) & (radius <= outer)
	if np.count_nonzero(mask) < 8:
		return [KernelValue("artifact.ring_score", None, count, 0, "ring annulus is empty or too small")]
	selected_radius = radius[mask]
	selected_values = values[mask]
	sums = np.bincount(selected_radius, weights=selected_values)
	counts = np.bincount(selected_radius)
	valid_bins = counts > 0
	profile = sums[valid_bins] / counts[valid_bins]
	if profile.size < 3:
		return [KernelValue("artifact.ring_score", None, count, 0, "too few radial bins")]
	smooth = gaussian_filter1d(profile, request.smoothing_sigma, mode="nearest")
	dynamic_range = float(values.max() - values.min())
	if dynamic_range <= 0:
		return [KernelValue("artifact.ring_score", None, count, 0, "image has no dynamic range")]
	score = float(np.sqrt(np.mean((profile - smooth) ** 2)) / dynamic_range)
	return [KernelValue("artifact.ring_score", score, int(np.count_nonzero(mask)))]


def _roi(values: np.ndarray, roi, name: str) -> np.ndarray:
	if roi.y_stop > values.shape[0] or roi.x_stop > values.shape[1]:
		raise ValueError(
			f"{name} ROI {(roi.y_start, roi.y_stop, roi.x_start, roi.x_stop)} "
			f"exceeds image shape {values.shape}"
		)
	return values[roi.y_start:roi.y_stop, roi.x_start:roi.x_stop]


def roi_noise_cnr_metrics(
	image: np.ndarray,
	request: RoiNoiseCnrMetricRequest,
	_context: dict[str, Any],
) -> list[KernelValue]:
	values = np.asarray(image, dtype=np.float64)
	if values.ndim != 2:
		raise ValueError(f"ROI metrics expect a two-dimensional slice, got {values.shape}")
	background = _roi(values, request.background, "background")
	signal = _roi(values, request.signal, "signal")
	count = background.size + signal.size
	if not np.isfinite(background).all() or not np.isfinite(signal).all():
		invalid = int(count - np.count_nonzero(np.isfinite(background)) - np.count_nonzero(np.isfinite(signal)))
		return [
			KernelValue("roi.background_noise", None, background.size, invalid, "non-finite ROI"),
			KernelValue("roi.cnr", None, count, invalid, "non-finite ROI"),
		]
	noise = float(np.std(background))
	cnr = (
		float(abs(signal.mean() - background.mean()) / noise)
		if noise > 0 else None
	)
	return [
		KernelValue("roi.background_noise", noise, background.size),
		KernelValue(
			"roi.cnr", cnr, count,
			reason=None if cnr is not None else "background ROI has zero noise",
		),
	]


REFERENCE_DEFINITIONS = (
	MetricDefinition(
		name="reference.nrmse", direction="minimize", units="ratio",
		normalization="reference dynamic range", required_inputs=("reference",),
		default_aggregation="mean", undefined="record_invalid",
		protocol_version=PROTOCOL_VERSION, resource_scope="slice", memory_multiplier=2,
	),
	MetricDefinition(
		name="reference.correlation", direction="maximize", units="unitless",
		normalization="Pearson product-moment", required_inputs=("reference",),
		default_aggregation="mean", undefined="record_invalid",
		protocol_version=PROTOCOL_VERSION, resource_scope="slice", memory_multiplier=2,
	),
	MetricDefinition(
		name="reference.normalized_bias", direction="minimize", units="ratio",
		normalization="absolute mean bias / reference dynamic range",
		required_inputs=("reference",), default_aggregation="mean",
		undefined="record_invalid", protocol_version=PROTOCOL_VERSION,
		resource_scope="slice", memory_multiplier=2,
	),
)

VALIDITY_DEFINITIONS = (
	MetricDefinition(
		name="validity.nonfinite_count", direction="constraint", units="pixels",
		normalization="none", default_aggregation="mean", undefined="fail_stage",
		protocol_version=PROTOCOL_VERSION, resource_scope="slice",
	),
	MetricDefinition(
		name="validity.clipped_fraction", direction="minimize", units="fraction",
		normalization="configured clip bounds", default_aggregation="mean",
		undefined="record_invalid", protocol_version=PROTOCOL_VERSION,
		resource_scope="slice",
	),
)

RING_DEFINITIONS = (
	MetricDefinition(
		name="artifact.ring_score", direction="minimize", units="ratio",
		normalization="radial-profile residual RMS / image dynamic range",
		default_aggregation="mean", undefined="record_invalid",
		protocol_version=PROTOCOL_VERSION, resource_scope="slice", memory_multiplier=1.25,
	),
)

ROI_DEFINITIONS = (
	MetricDefinition(
		name="roi.background_noise", direction="minimize", units="intensity",
		normalization="none", required_inputs=("background_roi",),
		default_aggregation="mean", undefined="record_invalid",
		protocol_version=PROTOCOL_VERSION, resource_scope="slice",
	),
	MetricDefinition(
		name="roi.cnr", direction="maximize", units="ratio",
		normalization="absolute signal-background contrast / background standard deviation",
		required_inputs=("background_roi", "signal_roi"),
		default_aggregation="mean", undefined="record_invalid",
		protocol_version=PROTOCOL_VERSION, resource_scope="slice",
	),
)


def default_metric_registry() -> MetricRegistry:
	registry = MetricRegistry()
	registry.register(MetricRegistration(
		"reference", ReferenceMetricRequest, REFERENCE_DEFINITIONS, reference_metrics,
	))
	registry.register(MetricRegistration(
		"validity", ValidityMetricRequest, VALIDITY_DEFINITIONS, validity_metrics,
	))
	registry.register(MetricRegistration(
		"ring", RingMetricRequest, RING_DEFINITIONS, ring_metric,
	))
	registry.register(MetricRegistration(
		"roi_noise_cnr", RoiNoiseCnrMetricRequest, ROI_DEFINITIONS, roi_noise_cnr_metrics,
	))
	return registry


METRIC_REGISTRY = default_metric_registry()
