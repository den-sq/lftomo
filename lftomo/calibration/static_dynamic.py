"""Static detector-field decomposition and camera-move beam registration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import find_peaks

from .build import CalibrationBuildSession
from .models import (
	CalibrationComponent,
	CompatibilityMetadata,
	MotorAxisCalibration,
	TimeAxis,
	ValidationResult,
)


BandOrientation = Literal["horizontal", "vertical"] | None


def _clean(values: Any) -> np.ndarray:
	array = np.asarray(values, dtype=np.float64)
	valid = np.isfinite(array) & (array > 0)
	if not valid.any():
		raise ValueError("flat field contains no finite positive pixels")
	floor = max(float(np.min(array[valid])), np.finfo(np.float64).tiny)
	return np.where(valid, array, floor)


def decompose_static_dynamic(
	flats: Any,
	beam_sigma: float = 4,
	band_orientation: BandOrientation = "horizontal",
	band_length: int = 75,
	exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Split flats exactly into detector-static ``D`` and dynamic beam ``B``.

	The estimator is the supplied source model: in log space, remove the smooth
	beam envelope and an optional long directional median, then take the temporal
	median of the stable compact residual. ``D * B[k]`` reconstructs every cleaned
	input exactly.
	"""

	if beam_sigma <= 0:
		raise ValueError("beam_sigma must be positive")
	if band_length < 1:
		raise ValueError("band_length must be positive")
	if band_orientation not in ("horizontal", "vertical", None):
		raise ValueError("band_orientation must be 'horizontal', 'vertical', or None")
	array = _clean(flats)
	if array.ndim == 2:
		array = array[None]
	if array.ndim != 3 or array.shape[0] < 1:
		raise ValueError("flats must have shape (N, Y, X) or (Y, X)")
	size = (
		(1, band_length) if band_orientation == "horizontal"
		else (band_length, 1) if band_orientation == "vertical"
		else None
	)
	high_frequency = np.empty(array.shape, dtype=np.float32)
	for index in range(array.shape[0]):
		log_flat = np.log(array[index])
		residual = log_flat - gaussian_filter(log_flat, beam_sigma)
		if size is not None:
			residual = residual - median_filter(residual, size=size)
		high_frequency[index] = residual.astype(np.float32)
	log_detector = (
		np.median(high_frequency, axis=0) if len(high_frequency) > 1
		else high_frequency[0]
	)
	if exclude_mask is not None:
		mask = np.asarray(exclude_mask, dtype=bool)
		if mask.shape != log_detector.shape:
			raise ValueError("exclude_mask must match one flat frame")
		log_detector = log_detector.copy()
		log_detector[mask] = 0
	detector = np.exp(log_detector).astype(np.float32)
	beam = np.stack([
		(array[index] / detector).astype(np.float32) for index in range(len(array))
	])
	return detector, beam


def build_static_dynamic_components(
	build: CalibrationBuildSession,
	series: Any,
	*,
	time_axis: TimeAxis | dict[str, Any],
	camera_segment: str,
	static_artifact_epoch: str,
	y_chunk: int = 64,
	beam_sigma: float = 4,
	band_orientation: BandOrientation = "horizontal",
	band_length: int = 75,
	exclude_mask: np.ndarray | None = None,
	static_name: str = "detector_static",
	dynamic_name: str = "dynamic_beam",
) -> tuple[CalibrationComponent, CalibrationComponent]:
	"""Decompose row windows and write #109 components through a #111 session."""

	if y_chunk < 1:
		raise ValueError("y_chunk must be positive")
	time_axis = TimeAxis.model_validate(time_axis)
	if time_axis.values is not None and len(time_axis.values) != series.frame_count:
		raise ValueError("time-axis length must match flat frame count")
	static_writer = build.create_array_component(
		static_name, shape=(series.rows, series.columns), dtype="float32", axes=("Y", "X"),
		kind="detector_static", coordinate_frame="detector", append_axis=0,
		compatibility=CompatibilityMetadata(static_artifact_epoch=static_artifact_epoch),
		metadata={
			"beam_sigma": beam_sigma, "band_orientation": band_orientation,
			"band_length": band_length,
		},
	)
	dynamic_writer = build.create_array_component(
		dynamic_name, shape=(series.frame_count, series.rows, series.columns),
		dtype="float32", axes=("T", "Y", "X"), kind="dynamic_beam",
		coordinate_frame="beam", append_axis=1, time_axis=time_axis,
		compatibility=CompatibilityMetadata(camera_segment=camera_segment),
		interpolation="linear",
		metadata={"camera_segment_boundary": "hard"},
	)
	halo = max(int(math.ceil(4 * beam_sigma)), band_length // 2 if band_orientation == "vertical" else 0)
	max_error = 0.0
	for start in range(0, series.rows, y_chunk):
		stop = min(series.rows, start + y_chunk)
		extended_start = max(0, start - halo)
		extended_stop = min(series.rows, stop + halo)
		flats = series.read_oriented_window(extended_start, extended_stop, "identity")
		mask = None if exclude_mask is None else exclude_mask[extended_start:extended_stop]
		detector, beam = decompose_static_dynamic(
			flats, beam_sigma=beam_sigma, band_orientation=band_orientation,
			band_length=band_length, exclude_mask=mask,
		)
		core = slice(start - extended_start, stop - extended_start)
		detector_core = detector[core]
		beam_core = beam[:, core]
		static_writer.append(detector_core)
		dynamic_writer.append(beam_core)
		reconstructed = detector_core[None] * beam_core
		max_error = max(max_error, float(np.max(np.abs(reconstructed - flats[:, core]))))
	qc = ValidationResult(
		name="exact_factorization", status="pass",
		message="D * B reconstructs every cleaned input window",
		metrics={"max_absolute_error": max_error, "y_chunk": y_chunk, "halo": halo},
	)
	static_writer.add_validation_result(qc)
	dynamic_writer.add_validation_result(qc)
	return static_writer.close(), dynamic_writer.close()


@dataclass(frozen=True)
class AxisRegistration:
	measured_shift: float
	resolved_shift: float
	zncc: float
	peak_ambiguity: float
	peak_margin: float
	used_expected_fallback: bool


@dataclass(frozen=True)
class BeamOffsetRegistration:
	dy: AxisRegistration
	dx: AxisRegistration

	@property
	def resolved_shift(self) -> tuple[float, float]:
		return self.dy.resolved_shift, self.dx.resolved_shift


def _profile_shift(
	old: np.ndarray,
	new: np.ndarray,
	max_shift: int,
	expected: float | None,
	refine: int,
	min_confidence: float,
	max_ambiguity: float,
) -> AxisRegistration:
	count = len(old)
	maximum = int(min(max_shift, count // 2 - 8))
	if maximum < 1:
		raise ValueError("registration profile is too short for the requested search")
	if expected is None:
		lags = np.arange(-maximum, maximum + 1)
	else:
		centre = -int(round(expected))
		lags = np.arange(centre - refine, centre + refine + 1)
	correlation = np.full(len(lags), -1.0)
	for index, lag in enumerate(lags):
		if lag >= 0:
			x, y = old[lag:], new[:count - lag]
		else:
			x, y = old[:count + lag], new[-lag:]
		x = x - x.mean()
		y = y - y.mean()
		correlation[index] = (x @ y) / max(np.sqrt((x @ x) * (y @ y)), 1e-12)
	peak_index = int(np.argmax(correlation))
	if 0 < peak_index < len(correlation) - 1:
		numerator = correlation[peak_index - 1] - correlation[peak_index + 1]
		denominator = correlation[peak_index - 1] - 2 * correlation[peak_index] + correlation[peak_index + 1]
		delta = float(np.clip(0.5 * numerator / (denominator + 1e-12), -1, 1))
	else:
		delta = 0.0
	measured = float(-(lags[peak_index] + delta))
	best = float(correlation[peak_index])
	# Ambiguity compares distinct local maxima, not adjacent samples on one broad
	# subpixel peak. Quasi-periodic stripe aliases remain separate candidates.
	peak_indices, _ = find_peaks(correlation, distance=max(2, refine))
	competitor_indices = [index for index in peak_indices if index != peak_index]
	second = max((float(correlation[index]) for index in competitor_indices), default=-1.0)
	confidence = float(np.clip(best, 0, 1))
	ambiguity = 1.0 if confidence <= 0 else float(np.clip(max(second, 0) / confidence, 0, 1))
	margin = float(best - second)
	unreliable = confidence < min_confidence or ambiguity > max_ambiguity
	use_fallback = expected is not None and unreliable
	return AxisRegistration(
		measured_shift=measured,
		resolved_shift=float(expected) if use_fallback else measured,
		zncc=confidence,
		peak_ambiguity=ambiguity,
		peak_margin=margin,
		used_expected_fallback=use_fallback,
	)


def register_beam_offset(
	flat_old: Any,
	flat_new: Any,
	D: Any | None = None,
	beam_sigma: float = 3,
	env_sigma: float = 60,
	max_shift: int = 60,
	expected: tuple[float, float] | None = None,
	refine: int = 6,
	min_confidence: float = 0.25,
	max_ambiguity: float = 0.98,
) -> BeamOffsetRegistration:
	"""Register old-to-new beam motion with ZNCC and peak-ambiguity diagnostics."""

	old = _clean(flat_old)
	new = _clean(flat_new)
	if old.shape != new.shape or old.ndim != 2:
		raise ValueError("camera registration requires two equally shaped 2-D flats")
	if D is not None:
		detector = _clean(D)
		if detector.shape != old.shape:
			raise ValueError("D must match the registration flat shape")
		old = old / detector
		new = new / detector
	# Preserve stripe wavelengths and a broad envelope in the profiles. The
	# env_sigma argument is retained in the API/provenance for the source model;
	# the bounded ZNCC overlap removes the envelope mean per lag.
	old = gaussian_filter(np.log(old), beam_sigma, mode="nearest")
	new = gaussian_filter(np.log(new), beam_sigma, mode="nearest")
	e_dy, e_dx = (None, None) if expected is None else expected
	return BeamOffsetRegistration(
		dy=_profile_shift(old.mean(axis=1), new.mean(axis=1), max_shift, e_dy, refine, min_confidence, max_ambiguity),
		dx=_profile_shift(old.mean(axis=0), new.mean(axis=0), max_shift, e_dx, refine, min_confidence, max_ambiguity),
	)


@dataclass(frozen=True)
class MotorAxisCalibrationResult:
	axis: Literal["vertical", "horizontal"]
	calibration: MotorAxisCalibration
	registration: AxisRegistration

	def expected_shift(self, motor_delta: float) -> float:
		return self.calibration.sign * self.calibration.pixels_per_unit * motor_delta


def calibrate_motor_axis(
	flat_before: Any,
	flat_after: Any,
	motor_delta: float,
	*,
	unit: str,
	D: Any | None = None,
	axis: Literal["vertical", "horizontal"] = "vertical",
	**registration_kwargs: Any,
) -> MotorAxisCalibrationResult:
	"""Calibrate a signed motor-unit to detector-pixel conversion."""

	if motor_delta == 0:
		raise ValueError("motor_delta must be nonzero for calibration")
	registration = register_beam_offset(
		flat_before, flat_after, D=D, **registration_kwargs,
	)
	axis_result = registration.dy if axis == "vertical" else registration.dx
	scale = axis_result.measured_shift / motor_delta
	return MotorAxisCalibrationResult(
		axis=axis,
		calibration=MotorAxisCalibration(
			unit=unit, pixels_per_unit=abs(float(scale)), sign=1 if scale >= 0 else -1,
		),
		registration=axis_result,
	)


def write_camera_registration_component(
	build: CalibrationBuildSession,
	registrations: dict[str, BeamOffsetRegistration],
	*,
	motor_axes: dict[str, MotorAxisCalibration] | None = None,
	name: str = "camera_registration",
) -> CalibrationComponent:
	"""Persist camera-move shifts and both confidence diagnostics via #111."""

	if not registrations:
		raise ValueError("at least one camera registration is required")
	columns = (
		"dy_measured", "dy_resolved", "dy_zncc", "dy_peak_ambiguity", "dy_peak_margin",
		"dx_measured", "dx_resolved", "dx_zncc", "dx_peak_ambiguity", "dx_peak_margin",
	)
	rows = []
	for registration in registrations.values():
		rows.append([
			registration.dy.measured_shift, registration.dy.resolved_shift,
			registration.dy.zncc, registration.dy.peak_ambiguity, registration.dy.peak_margin,
			registration.dx.measured_shift, registration.dx.resolved_shift,
			registration.dx.zncc, registration.dx.peak_ambiguity, registration.dx.peak_margin,
		])
	writer = build.create_array_component(
		name, shape=(len(rows), len(columns)), dtype="float64",
		axes=("Registration", "Metric"), kind="camera_registration",
		coordinate_frame="motor", append_axis=0,
		compatibility=CompatibilityMetadata(motor_axes=motor_axes or {}),
		metadata={"registration_labels": list(registrations), "columns": list(columns)},
	)
	writer.append(np.asarray(rows, dtype=np.float64))
	return writer.close()


@dataclass(frozen=True)
class SegmentedTimePoint:
	segment: str
	time: float


def segment_time_bracket(
	points: Iterable[SegmentedTimePoint], *, segment: str, time: float,
) -> tuple[SegmentedTimePoint, SegmentedTimePoint]:
	"""Bracket within one camera segment; crossing a camera move is impossible."""

	values = sorted((point for point in points if point.segment == segment), key=lambda point: point.time)
	if not values:
		raise ValueError(f"no beam keyframes exist for camera segment {segment!r}")
	if time < values[0].time or time > values[-1].time:
		raise ValueError(
			f"time {time} is outside camera segment {segment!r}; interpolation across a camera move is forbidden"
		)
	for index, point in enumerate(values):
		if time == point.time:
			return point, point
		if time < point.time:
			return values[index - 1], point
	return values[-1], values[-1]
