"""Two-flat beam tracking, diagnostics, and beam-time schedule fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

from .static_dynamic import _clean


@dataclass(frozen=True)
class BeamSchedule:
	"""Serializable mapping from global scan fraction to calibrated beam time."""

	kind: Literal["linear", "piecewise"] = "linear"
	alpha: float = 0.0
	beta: float = 1.0
	scan_fractions: tuple[float, ...] = ()
	beam_times: tuple[float, ...] = ()

	def __call__(self, value: float | np.ndarray) -> float | np.ndarray:
		values = np.asarray(value, dtype=float)
		if self.kind == "linear":
			result = self.alpha + self.beta * values
		else:
			if len(self.scan_fractions) < 2 or len(self.scan_fractions) != len(self.beam_times):
				raise ValueError("piecewise schedule requires matching scan_fractions and beam_times")
			result = np.interp(values, self.scan_fractions, self.beam_times)
		return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class ScheduleFit:
	schedule: BeamSchedule
	beam_times: tuple[float, ...]
	residual_std: float
	air_pixel_count: int
	air_fraction: float


def geometric_crossfade(pre: Any, post: Any, schedule: Callable | None = None):
	pre, post = _clean(pre), _clean(post)
	log_pre, log_post = np.log(pre), np.log(post)
	schedule = (lambda value: value) if schedule is None else schedule
	return lambda time: np.exp((1 - schedule(time)) * log_pre + schedule(time) * log_post)


def optimal_transport_interp(
	pre: Any, post: Any, transport_sigma: float = 12,
	schedule: Callable | None = None,
):
	"""Safe-default gain evaluator retaining static content in the log mean."""

	if transport_sigma <= 0:
		raise ValueError("transport_sigma must be positive")
	pre, post = _clean(pre), _clean(post)
	log_pre, log_post = np.log(pre), np.log(post)
	average = 0.5 * (log_pre + log_post)
	transport = gaussian_filter(log_post - log_pre, transport_sigma)
	schedule = (lambda value: value) if schedule is None else schedule
	return lambda time: np.exp(average + (schedule(time) - 0.5) * transport)


def estimate_noise_floor(image: Any, patch: int = 64) -> float:
	if patch < 3:
		raise ValueError("noise patch must be at least three pixels")
	image = _clean(image)
	kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
	values = []
	for row in range(0, image.shape[0] - patch + 1, patch):
		for column in range(0, image.shape[1] - patch + 1, patch):
			block = image[row:row + patch, column:column + patch]
			if block.mean() <= 1:
				continue
			response = convolve2d(block, kernel, mode="valid")
			values.append(np.sqrt(np.pi / 2) / (6 * response.size) * np.sum(np.abs(response)))
	return float(np.percentile(values, 10)) if values else float("nan")


def diagnose_beam_change(pre: Any, post: Any, beam_sigma: float = 8) -> dict[str, Any]:
	pre, post = _clean(pre), _clean(post)
	log_ratio = np.log(post) - np.log(pre)
	smooth = gaussian_filter(log_ratio, beam_sigma)
	high = log_ratio - smooth
	noise = estimate_noise_floor(pre)
	bright = float(np.mean(pre[pre > np.percentile(pre, 60)]))
	noise_relative = noise / bright
	centered = high - high.mean()
	variance = float(np.mean(centered * centered))
	lag1 = float(np.mean(centered[:, :-1] * centered[:, 1:]) / variance) if variance else 0.0
	return {
		"intensity_drift_pct": float((np.exp(np.median(log_ratio)) - 1) * 100),
		"noise_floor": noise,
		"noise_relative": noise_relative,
		"smooth_motion_std": float(np.std(smooth)),
		"high_frequency_std": float(np.std(high)),
		"high_frequency_lag1": lag1,
		"motion_to_noise": float(np.std(smooth) / max(noise_relative * np.sqrt(2), 1e-12)),
		"recommendation": "optimal_transport_interp",
	}


def auto_air_mask(projection: Any, reference: Any, ratio_hi: float = 0.95,
			exposure_quantile: float = 0.5) -> np.ndarray:
	projection, reference = _clean(projection), _clean(reference)
	return (
		(projection / reference > ratio_hi)
		& (reference > np.quantile(reference, exposure_quantile))
	)


def _bandpass(values: np.ndarray, low: float = 1.5, high: float = 25) -> np.ndarray:
	return gaussian_filter(values, low) - gaussian_filter(values, high)


def fit_beam_time(
	projection: Any, pre: Any, post: Any, *, mask: np.ndarray | None = None,
	band: tuple[float, float] = (1.5, 25),
) -> float:
	projection, pre, post = _clean(projection), _clean(pre), _clean(post)
	mask = auto_air_mask(projection, pre) if mask is None else np.asarray(mask, dtype=bool)
	if mask.shape != pre.shape or np.count_nonzero(mask) < 16:
		raise ValueError("beam-time fit requires at least 16 valid air pixels")
	direction = _bandpass(np.log(post) - np.log(pre), *band)
	observed = _bandpass(np.log(projection) - np.log(pre), *band)
	valid = mask & np.isfinite(direction) & np.isfinite(observed)
	denominator = float(np.sum(direction[valid] ** 2))
	if denominator <= 1e-18:
		raise ValueError("pre/post beam direction is degenerate in the air region")
	return float(np.sum(observed[valid] * direction[valid]) / denominator)


def calibrate_schedule(
	projections: Any,
	scan_fractions: Any,
	pre: Any,
	post: Any,
	*,
	mask: np.ndarray | None = None,
	kind: Literal["linear", "piecewise"] = "linear",
) -> ScheduleFit:
	projections = np.asarray(projections)
	fractions = np.asarray(scan_fractions, dtype=float)
	if projections.ndim != 3 or len(projections) != len(fractions) or len(fractions) < 2:
		raise ValueError("schedule fitting requires matching projections and at least two fractions")
	if mask is None:
		mask = auto_air_mask(projections[0], pre)
	air_count = int(np.count_nonzero(mask))
	beam_times = np.asarray([
		fit_beam_time(projection, pre, post, mask=mask) for projection in projections
	])
	order = np.argsort(fractions)
	fractions, beam_times = fractions[order], beam_times[order]
	if np.any(np.diff(fractions) <= 0):
		raise ValueError("schedule sample fractions must be strictly increasing")
	if kind == "linear":
		matrix = np.column_stack([np.ones_like(fractions), fractions])
		alpha, beta = np.linalg.lstsq(matrix, beam_times, rcond=None)[0]
		fitted = alpha + beta * fractions
		schedule = BeamSchedule(kind="linear", alpha=float(alpha), beta=float(beta))
	else:
		schedule = BeamSchedule(
			kind="piecewise", scan_fractions=tuple(fractions),
			beam_times=tuple(np.maximum.accumulate(beam_times)),
		)
		fitted = np.asarray(schedule(fractions))
	return ScheduleFit(
		schedule=schedule, beam_times=tuple(float(value) for value in beam_times),
		residual_std=float(np.std(beam_times - fitted)), air_pixel_count=air_count,
		air_fraction=float(air_count / mask.size),
	)
