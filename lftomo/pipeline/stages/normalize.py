"""Flat- and dark-field normalization stages."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from lftomo.calibration import (
	BeamSchedule,
	CalibrationBundle,
	CalibrationBundleRef,
	calibrate_schedule,
	diagnose_beam_change,
	optimal_transport_interp,
)
from lftomo.config import StageConfig
from lftomo.pipeline.core import Locality, PipelineState, ResourceHints, Stage, StageContract
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.pipeline.stages._layout import (
	configured_layout,
	n_projections,
	projection_view,
	resolve_layout,
)
from lftomo.prep.flat import _weighted_normalize_projection
from lftomo.util.shapes import ProjOrder, SinoOrder


class WeightedNormalizeStage(Stage):
	"""Blend pre/post flat fields and normalize a sinogram array in place.

	The working array is in ``(Y, Theta, X)`` order. The four flat fields are
	provided through ``Scan.metadata["flat_fields"]`` in pre-gain, post-gain,
	pre-dark, post-dark order.
	"""

	name = "normalize.weighted"
	contract = StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		produces=PRESERVE,
		accepts_domain=(DataDomain.RAW,),
		produces_domain=DataDomain.TRANSMISSION,
		locality=Locality.PROJECTION,
		accepts=(ProjOrder, SinoOrder),
		prefers=ProjOrder,
		mutates_input=True,
		window_axes=("Y",),
		required_metadata=("metadata.flat_fields",),
		resources=ResourceHints(memory_multiplier=1.0),
	)

	def __init__(self, pre_sub: bool = False, pre_subtracted: bool | None = None,
				timing_source: str | None = None, order: str | None = None):
		configured_layout(order)
		self.pre_sub = pre_sub if pre_subtracted is None else pre_subtracted
		self.timing_source = timing_source
		self.order = order

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("WeightedNormalizeStage currently requires the NumPy backend")

		working = np.asarray(state.working_array).astype(np.float32, copy=False)
		flats = np.asarray(state.scan.metadata["flat_fields"])
		layout = resolve_layout(state, self.order, self.name)
		if working.ndim != 3:
			raise ValueError("weighted-normalize expects a (Y, Theta, X) working array")
		y_size = working.shape[1] if layout is ProjOrder else working.shape[0]
		if flats.shape != (4, y_size, working.shape[2]):
			raise ValueError(
				"flat_fields must have shape (4, Y, X) matching the working array"
			)

		sinogram_view = working if layout is SinoOrder else np.transpose(working, (1, 0, 2))
		for projection in range(n_projections(working, layout)):
			_weighted_normalize_projection(
				sinogram_view,
				sinogram_view,
				flats,
				projection,
				n_projections(working, layout),
				self.pre_sub,
			)
		return state if working is state.working_array else state.with_array(working)


def _flat_fields(state: PipelineState, data: np.ndarray, layout: type) -> np.ndarray:
	flats = np.asarray(state.scan.metadata["flat_fields"])
	y_size = data.shape[1] if layout is ProjOrder else data.shape[0]
	if flats.shape != (4, y_size, data.shape[2]):
		raise ValueError("flat_fields must have shape (4, Y, X) matching the working array")
	return flats


def _normalize_with_maps(projection: np.ndarray, gain: np.ndarray, dark: np.ndarray,
				pre_subtracted: bool, floor_nonpositive: bool = False,
				floor_value: float = 1.0) -> np.ndarray:
	denominator = gain if pre_subtracted else gain - dark
	denominator = denominator.copy()
	numerator = projection - dark
	if floor_nonpositive:
		numerator = numerator.copy()
		numerator[numerator <= 0] = floor_value
		denominator[denominator <= 0] = floor_value
	zero = denominator == 0
	if zero.any():
		nonzero = np.abs(denominator[~zero])
		if nonzero.size == 0:
			raise ValueError("normalization gain map contains only zeros")
		denominator[zero] = nonzero.min()
	return np.divide(numerator, denominator)


def _calibration_y_window(state: PipelineState, y_size: int) -> tuple[int, int]:
	read = next((stage for stage in state.recon.pipeline if stage.kind == "read"), None)
	start = int(read.params.get("y_start", 0)) if read is not None else 0
	return start, start + y_size


def _tracking_fields(
	state: PipelineState, y_size: int, component_names: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	ref = state.artifacts.get("calibration_bundle")
	if isinstance(ref, CalibrationBundleRef):
		bundle = CalibrationBundle.open(ref.manifest_path)
		start, stop = _calibration_y_window(state, y_size)
		return tuple(
			bundle.read_component(component_names[name], y_start=start, y_stop=stop).astype(np.float32)
			for name in ("pre_gain", "post_gain", "pre_dark", "post_dark")
		)
	if "flat_fields" not in state.scan.metadata:
		raise ValueError("normalize.beam_tracking requires a calibration bundle or legacy flat_fields")
	fields = np.asarray(state.scan.metadata["flat_fields"])
	if fields.shape != (4, y_size, np.asarray(state.working_array).shape[2]):
		raise ValueError("legacy flat_fields must have shape (4, Y, X) for beam tracking")
	return tuple(fields)


class BeamTrackingNormalizeStage(Stage):
	"""Evaluate a calibrated two-flat beam model one projection at a time."""

	name = "normalize.beam_tracking"
	contract = StageContract(
		produces=PRESERVE, accepts=(ProjOrder, SinoOrder), prefers=ProjOrder,
		accepts_domain=(DataDomain.RAW,), produces_domain=DataDomain.TRANSMISSION,
		locality=Locality.PROJECTION, mutates_input=True, window_axes=("Y", "Theta"),
		produces_artifacts=("beam_tracking_qc",),
		resources=ResourceHints(memory_multiplier=2.0),
	)

	def __init__(
		self, *, transport_sigma: float = 12, schedule: dict | None = None,
		fit_schedule: bool = False, schedule_sample_count: int = 8,
		component_names: dict[str, str] | None = None, order: str | None = None,
		pre_subtracted: bool = False,
	):
		configured_layout(order)
		if schedule_sample_count < 2:
			raise ValueError("schedule_sample_count must be at least two")
		self.transport_sigma = transport_sigma
		self.schedule = BeamSchedule(**schedule) if schedule is not None else BeamSchedule()
		self.fit_schedule = fit_schedule
		self.schedule_sample_count = schedule_sample_count
		self.component_names = {
			"pre_gain": "pre_gain", "post_gain": "post_gain",
			"pre_dark": "pre_dark", "post_dark": "post_dark",
			**(component_names or {}),
		}
		self.order = order
		self.pre_subtracted = pre_subtracted

	@classmethod
	def from_config(cls, config: StageConfig) -> "BeamTrackingNormalizeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("BeamTrackingNormalizeStage currently requires the NumPy backend")
		data = np.asarray(state.working_array).astype(np.float32, copy=False)
		layout = resolve_layout(state, self.order, self.name)
		if data.ndim != 3:
			raise ValueError("normalize.beam_tracking expects a three-dimensional array")
		y_size = data.shape[1] if layout is ProjOrder else data.shape[0]
		pre_gain, post_gain, pre_dark, post_dark = _tracking_fields(
			state, y_size, self.component_names,
		)
		projection_count = int(state.artifacts.get("projection_count", state.scan.n_projections))
		projection_start = int(state.artifacts.get("projection_start", 0))
		batch_count = n_projections(data, layout)
		if batch_count != state.scan.n_projections and not {
			"projection_start", "projection_count",
		}.issubset(state.artifacts):
			raise ValueError(
				"Theta-windowed beam tracking requires projection_start and "
				"projection_count artifacts"
			)
		schedule = self.schedule
		fit = None
		if self.fit_schedule:
			indices = np.unique(np.linspace(0, batch_count - 1, self.schedule_sample_count).astype(int))
			fractions = (projection_start + indices) / max(projection_count - 1, 1)
			fit = calibrate_schedule(
				np.stack([projection_view(data, layout, int(index)) for index in indices]),
				fractions, pre_gain, post_gain,
			)
			schedule = fit.schedule
		gain_at = optimal_transport_interp(
			pre_gain, post_gain, transport_sigma=self.transport_sigma, schedule=schedule,
		)
		for local_index in range(batch_count):
			global_index = projection_start + local_index
			fraction = global_index / max(projection_count - 1, 1)
			beam_time = float(schedule(fraction))
			gain = gain_at(fraction)
			# Schedules may legitimately extrapolate beam structure. Dark fields do
			# not carry that motion model, so keep them within measured endpoints.
			dark_time = float(np.clip(beam_time, 0, 1))
			dark = (1 - dark_time) * pre_dark + dark_time * post_dark
			image = projection_view(data, layout, local_index)
			image[:] = _normalize_with_maps(
				image, gain, dark, self.pre_subtracted, floor_nonpositive=True,
			)
		state.artifacts["beam_tracking_qc"] = {
			"diagnostics": diagnose_beam_change(pre_gain, post_gain),
			"schedule": schedule.__dict__,
			"fit": None if fit is None else {
				"residual_std": fit.residual_std, "air_pixel_count": fit.air_pixel_count,
				"air_fraction": fit.air_fraction, "beam_times": fit.beam_times,
			},
			"projection_start": projection_start, "projection_count": projection_count,
		}
		return state if data is state.working_array else state.with_array(data)


class BeamKeyframesNormalizeStage(Stage):
	"""Normalize from the two physical-time keyframes bracketing each projection."""

	name = "normalize.beam_keyframes"
	contract = StageContract(
		produces=PRESERVE, accepts=(ProjOrder, SinoOrder), prefers=ProjOrder,
		accepts_domain=(DataDomain.RAW,), produces_domain=DataDomain.TRANSMISSION,
		locality=Locality.PROJECTION, mutates_input=True, window_axes=("Y", "Theta"),
		produces_artifacts=("beam_keyframe_qc",), resources=ResourceHints(memory_multiplier=2.0),
	)

	def __init__(self, *, component: str = "beam_keyframes", dark_component: str | None = None,
			transport_sigma: float = 12, maximum_gap: float | None = None,
			order: str | None = None):
		configured_layout(order)
		self.component = component
		self.dark_component = dark_component
		self.transport_sigma = transport_sigma
		self.maximum_gap = maximum_gap
		self.order = order

	@classmethod
	def from_config(cls, config: StageConfig) -> "BeamKeyframesNormalizeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		data = np.asarray(state.working_array).astype(np.float32, copy=False)
		layout = resolve_layout(state, self.order, self.name)
		y_size = data.shape[1] if layout is ProjOrder else data.shape[0]
		start, stop = _calibration_y_window(state, y_size)
		ref = state.artifacts.get("calibration_bundle")
		if not isinstance(ref, CalibrationBundleRef):
			raise ValueError("normalize.beam_keyframes requires a selected calibration bundle")
		bundle = CalibrationBundle.open(ref.manifest_path)
		component = bundle.component(self.component)
		if component.time_axis is None:
			raise ValueError("beam keyframes require an explicit time axis")
		projection_count = int(state.artifacts.get("projection_count", state.scan.n_projections))
		projection_start = int(state.artifacts.get("projection_start", 0))
		if component.time_axis.kind == "timestamp_seconds":
			times = state.artifacts.get("projection_times")
			if times is None:
				raise ValueError("physical-time keyframes require projection_times artifacts")
			times = np.asarray(times, dtype=float)
			if len(times) != n_projections(data, layout):
				raise ValueError("projection_times must match the current Theta window")
		else:
			times = projection_start + np.arange(n_projections(data, layout), dtype=float)
		dark = 0.0 if self.dark_component is None else bundle.read_component(
			self.dark_component, y_start=start, y_stop=stop,
		)
		maximum_seen_gap = 0.0
		for local_index, time in enumerate(times):
			bracket = bundle.read_time_bracket(
				self.component, float(time), y_start=start, y_stop=stop,
			)
			gap = bracket.after_time - bracket.before_time
			maximum_seen_gap = max(maximum_seen_gap, gap)
			if self.maximum_gap is not None and gap > self.maximum_gap:
				raise ValueError(
					f"projection time {time} lies in keyframe gap {gap}, exceeding {self.maximum_gap}"
				)
			average = 0.5 * (np.log(np.clip(bracket.before, 1e-6, None))
				+ np.log(np.clip(bracket.after, 1e-6, None)))
			transport = gaussian_filter(
				np.log(np.clip(bracket.after, 1e-6, None))
				- np.log(np.clip(bracket.before, 1e-6, None)), self.transport_sigma,
			)
			gain = np.exp(average + (bracket.weight - 0.5) * transport)
			image = projection_view(data, layout, local_index)
			image[:] = _normalize_with_maps(image, gain, dark, False, floor_nonpositive=True)
		state.artifacts["beam_keyframe_qc"] = {
			"component": self.component, "maximum_gap_seen": maximum_seen_gap,
			"projection_start": projection_start, "projection_count": projection_count,
			"observability": "events wholly inside a keyframe gap cannot be recovered",
		}
		return state if data is state.working_array else state.with_array(data)


class ThetaSwitchNormalizeStage(Stage):
	"""Use pre/post calibration fields with midpoint switching or linear blending."""

	name = "normalize.theta_switch"
	contract = StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		mutates_input=True,
		window_axes=("Y", "Theta"),
		required_metadata=("metadata.flat_fields",),
		resources=ResourceHints(memory_multiplier=2.0),
	)

	def __init__(self, gain_averaging: bool = False, pre_subtracted: bool = False,
				pre_gain: str | None = None, post_gain: str | None = None,
				pre_dark: str | None = None, post_dark: str | None = None,
				order: str | None = None, floor_nonpositive: bool = False,
				floor_value: float = 1.0):
		compatibility = configured_layout(order)
		shape = compatibility or SinoOrder
		self.contract = StageContract(
			expected_input=shape,
			produced_output=shape,
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			accepts_domain=(DataDomain.RAW,),
			produces_domain=DataDomain.TRANSMISSION,
			locality=Locality.PROJECTION,
			mutates_input=True,
			window_axes=("Y", "Theta"),
			required_metadata=("metadata.flat_fields",),
			resources=ResourceHints(memory_multiplier=2.0),
		)
		self.order = order
		self.floor_nonpositive = floor_nonpositive
		self.floor_value = floor_value
		self.gain_averaging = gain_averaging
		self.pre_subtracted = pre_subtracted
		self.calibration_paths = {
			"pre_gain": pre_gain,
			"post_gain": post_gain,
			"pre_dark": pre_dark,
			"post_dark": post_dark,
		}

	@classmethod
	def from_config(cls, config: StageConfig) -> "ThetaSwitchNormalizeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("ThetaSwitchNormalizeStage currently requires the NumPy backend")
		data = np.asarray(state.working_array).astype(np.float32, copy=False)
		layout = resolve_layout(state, self.order, self.name)
		if data.ndim != 3:
			raise ValueError("normalize.theta_switch expects a three-dimensional tomography array")
		flats = _flat_fields(state, data, layout)
		pre_gain, post_gain, pre_dark, post_dark = flats
		batch_count = n_projections(data, layout)
		total_count = int(state.artifacts.get("projection_count", state.scan.n_projections))
		projection_start = int(state.artifacts.get("projection_start", 0))
		for local_projection in range(batch_count):
			projection = projection_start + local_projection
			if self.gain_averaging:
				weight = projection / total_count
				gain = pre_gain * (1 - weight) + post_gain * weight
				dark = pre_dark * (1 - weight) + post_dark * weight
			elif projection > total_count / 2:
				gain, dark = post_gain, post_dark
			else:
				gain, dark = pre_gain, pre_dark
			image = projection_view(data, layout, local_projection)
			image[:] = _normalize_with_maps(
				image, gain, dark, self.pre_subtracted,
				self.floor_nonpositive, self.floor_value,
			)
		return state if data is state.working_array else state.with_array(data)


class TaperBlendedNormalizeStage(ThetaSwitchNormalizeStage):
	"""Phase-contrast lineage's linear pre/post gain and dark blending."""

	name = "normalize.taper_blended"

	def __init__(self, pre_subtracted: bool = False,
				pre_gain: str | None = None, post_gain: str | None = None,
			pre_dark: str | None = None, post_dark: str | None = None,
			pre_weight: str = "linear", post_weight: str = "linear",
				gain_averaging: bool = True, gain_movement_enabled: bool = False,
				order: str | None = None, floor_nonpositive: bool = False,
				floor_value: float = 1.0):
		if pre_weight != "linear" or post_weight != "linear":
			raise ValueError("taper-blended normalization currently supports linear weights")
		if not gain_averaging:
			raise ValueError("taper-blended normalization requires gain_averaging")
		if gain_movement_enabled:
			raise ValueError("gain movement correction must be a separate geometry Stage")
		super().__init__(
			gain_averaging=True,
			pre_subtracted=pre_subtracted,
			pre_gain=pre_gain,
			post_gain=post_gain,
			pre_dark=pre_dark,
			post_dark=post_dark,
			order=order,
			floor_nonpositive=floor_nonpositive,
			floor_value=floor_value,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "TaperBlendedNormalizeStage":
		return cls(**config.params)
