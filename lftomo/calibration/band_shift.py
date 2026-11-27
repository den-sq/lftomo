"""Bounded construction of the measured beam shift/stretch schedule."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

from lftomo.util.window import TomoDataSet, TomoWindow
from .build import CalibrationBuildSession
from .models import BeamBandShiftModel, CalibrationComponent, TimeAxis, ValidationResult
from .static_dynamic import _profile_shift


def build_beam_band_shift_schedule(
	build: CalibrationBuildSession,
	source: Any,
	dataset: TomoDataSet,
	*,
	model: BeamBandShiftModel,
	row_windows: Sequence[tuple[int, int]],
	highpass_sigma: float,
	search_shifts: Sequence[float],
	stretch_sigma: float = 25,
	y_chunk: int = 64,
	name: str = "beam_schedule",
) -> CalibrationComponent:
	"""Measure six local offsets and fit ``dy_c + (m - 1) * (y - yc)``.

	White and dark are completed components named by ``model`` in this build.
	The source implements the existing random-frame ``read`` interface. Input
	planes are read by Y chunk; fitting keeps the reference/current row profiles
	and the small schedule. All recorded frames are fitted, with acquisition
	slots taken from ``dataset``.

	Profiles average log dark-subtracted counts across X, with a one-count floor
	matching the normalization gain convention. Gaussian high-pass profiles are
	registered by normalized correlation on the caller's uniform shift grid.
	Only magnification is Gaussian-smoothed along recorded-frame order. Window
	locations, high-pass width and search range are deliberately caller-owned:
	the historical B10 producer settings have not been recovered.
	"""

	physical = dataset.physical_shape
	if (source.frame_count, source.rows, source.columns) != (physical.Theta, physical.Y, physical.X):
		raise ValueError("frame source must match the dataset's full physical shape")
	if source.frame_count < 2 or y_chunk < 1:
		raise ValueError("schedule fitting requires at least two frames and a positive y_chunk")
	if not np.isfinite(highpass_sigma) or highpass_sigma <= 0 or not np.isfinite(stretch_sigma) or stretch_sigma < 0:
		raise ValueError("highpass_sigma must be positive and stretch_sigma nonnegative")
	shifts = np.asarray(search_shifts, dtype=np.float64)
	if (shifts.ndim != 1 or len(shifts) < 3 or not np.isfinite(shifts).all()
			or np.any(np.diff(shifts) <= 0)
			or not np.allclose(np.diff(shifts), shifts[1] - shifts[0], rtol=1e-8, atol=1e-12)):
		raise ValueError("search_shifts must be a finite increasing uniform grid with at least three values")
	if len(row_windows) != 6:
		raise ValueError("band-shift fitting requires six reference row windows")
	windows = [TomoWindow.from_bounds(physical, y=bounds) for bounds in row_windows]
	rows = [np.fromiter(window.int_iter("Y"), dtype=np.int64) for window in windows]
	if any(len(value) < 3 or value[0] + shifts[0] < 0 or value[-1] + shifts[-1] > physical.Y - 1 for value in rows):
		raise ValueError("reference windows and their shift search must remain inside the detector")
	centres = np.array([value.mean() for value in rows])
	design = np.column_stack((np.ones(len(rows)), centres - model.yc))
	if np.linalg.matrix_rank(design) != 2:
		raise ValueError("reference windows must have distinct row centres to fit stretch")

	for component_name, kind in ((model.white_component, "detector_static"), (model.dark_component, "dark")):
		component = build.components.get(component_name)
		if component is None:
			# Preserve the session's actionable incomplete/unknown-component error.
			build.read_component(component_name)
		if component.kind != kind or component.payload.axes != ("Y", "X"):
			raise ValueError(f"{component_name!r} must be a static {kind} (Y, X) component")
		if component.payload.shape != (physical.Y, physical.X):
			raise ValueError("white and dark components must cover the full detector")

	def profile(index: int | None) -> np.ndarray:
		result = np.empty(physical.Y, dtype=np.float64)
		for start in range(0, physical.Y, y_chunk):
			window = TomoWindow.from_bounds(physical, y=(start, min(start + y_chunk, physical.Y)))
			selection = window.axis_slice("Y")
			dark = build.read_component(
				model.dark_component, y_start=selection.start, y_stop=selection.stop,
			)
			values = build.read_component(
				model.white_component, y_start=selection.start, y_stop=selection.stop,
			) if index is None else source.read(index, y_start=selection.start, y_stop=selection.stop)
			gain = np.asarray(values, dtype=np.float64) - dark
			if not np.isfinite(gain).all():
				raise ValueError(f"non-finite dark-subtracted counts in frame {index}")
			np.maximum(gain, 1, out=gain)
			np.log(gain, out=gain)
			result[selection] = gain.mean(axis=1)
		return result - gaussian_filter1d(result, highpass_sigma, mode="reflect")

	reference = profile(None)
	if any(np.ptp(reference[value]) <= np.finfo(float).eps for value in rows):
		raise ValueError("every reference window needs beam texture to measure a shift")
	schedule = np.empty((physical.Theta, 2), dtype=np.float64)
	residual_sum = 0.0
	minimum_correlation = 1.0
	maximum_ambiguity = 0.0
	for index in range(physical.Theta):
		observed = profile(index)
		fits = [_profile_shift(
			reference, observed, int(np.ceil(np.max(np.abs(shifts)))), None, 1, 0, 1,
			reference_rows=value, search_shifts=shifts,
		) for value in rows]
		offsets = np.array([fit.measured_shift for fit in fits])
		if np.any(offsets <= shifts[0]) or np.any(offsets >= shifts[-1]):
			raise ValueError(f"frame {index} reaches the shift-search boundary; widen search_shifts")
		parameters, *_ = np.linalg.lstsq(design, offsets, rcond=None)
		schedule[index] = parameters[0], 1 + parameters[1]
		residual_sum += float(np.square(offsets - design @ parameters).sum())
		minimum_correlation = min(minimum_correlation, *(fit.zncc for fit in fits))
		maximum_ambiguity = max(maximum_ambiguity, *(fit.peak_ambiguity for fit in fits))
	if stretch_sigma:
		schedule[:, 1] = gaussian_filter1d(schedule[:, 1], stretch_sigma, mode="reflect")
	if np.any(schedule[:, 1] <= 0):
		raise ValueError("measured schedule has non-positive magnification")
	qc = ValidationResult(
		name="band_shift_schedule_fit", status="pass",
		message="six-window registration and linear shift/stretch fit; production parity unverified",
		metrics={
			"window_residual_rms_px": float(np.sqrt(residual_sum / (physical.Theta * len(rows)))),
			"frame_to_frame_dy_rms_px": float(np.sqrt(np.square(np.diff(schedule[:, 0])).mean())),
			"minimum_window_correlation": minimum_correlation,
			"maximum_window_peak_ambiguity": maximum_ambiguity,
		},
	)
	return build.write_array_component(
		name, (schedule,), shape=schedule.shape, dtype="float32",
		axes=("Theta", "Parameter"), kind="beam_schedule", coordinate_frame="beam",
		band_shift=model, interpolation="nearest",
		time_axis=TimeAxis(kind="projection_index", values=dataset.acquisition_indices.tolist()),
		validation_results=(qc,),
		metadata={
			"row_windows": [list(bounds) for bounds in row_windows],
			"highpass_sigma": highpass_sigma, "search_shifts": shifts.tolist(),
			"stretch_sigma": stretch_sigma, "smoothing_mode": "reflect",
			"profile": "mean_x(log(max(raw-dark, 1)))",
			"registration": "fixed_reference_window_zncc_with_parabolic_peak",
			"angles_radians": dataset.angles.tolist(),
			"y_chunk": y_chunk,
		},
	)
