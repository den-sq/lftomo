"""Flat- and dark-field normalization stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Literal

from pydantic import Field, field_validator, model_validator

import numpy as np
from scipy.ndimage import gaussian_filter

from lftomo.calibration import (
	BeamSchedule,
	CalibrationBundle,
	CalibrationBundleRef,
	calibrate_schedule,
	diagnose_beam_change,
	fit_projection_beam,
	optimal_transport_interp,
)
from lftomo.pipeline.core import (
	BufferSpec,
	PipelineState,
	Stage,
	StageParams,
	StageContract,
	allocate_output,
	commit_output,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.storage import buffer_view
from lftomo.util.shapes import GenericOrder, ProjOrder, SinoOrder, Theta


def _normalization_output(state: PipelineState) -> tuple[object, np.ndarray]:
	"""Return a work-dtype output while the raw input remains live."""

	source = np.asarray(state.working_array)
	if state.source_window is None:
		raise RuntimeError("normalization requires PipelineState.source_window")
	work_dtype = np.dtype(state.source_window.internal.dtype)
	if state.storage is None and source.dtype == work_dtype:
		return state.working_array, source
	output, data = allocate_output(
		state, state.buffer_spec.shape, work_dtype, role="normalized",
		reuse_from=("normalized", "reconstruction"),
	)
	data[...] = source
	return output, data


def _finish_normalization(state: PipelineState, output: object) -> PipelineState:
	return commit_output(
		state, output, retire_as="raw_staging",
		buffer_spec=BufferSpec(state.buffer_spec.shape, np.dtype(output.dtype)),
	)


def _managed_flat_fields(state: PipelineState) -> np.ndarray:
	"""Publish legacy detector maps as one run-managed shared artifact."""

	existing = state.artifacts.get("flat_fields")
	if existing is not None:
		from lftomo.storage import buffer_view

		return buffer_view(existing)
	fields = np.asarray(state.scan.metadata["flat_fields"])
	if fields.ndim == 3:
		# Legacy flat_fields carries values but no source geometry. Until its
		# detector interval is explicit, infer one of the two historical origins
		# from Y extent so this consumer can derive the current padded-window view.
		# Binding the array to a DataRef would remove this inference and offset code.
		if state.source_window is None or state.dataset is None:
			raise RuntimeError("flat-field slicing requires canonical source geometry")
		source = state.source_window.with_pad(["Y"])
		selection = state.dataset.interval("Y")
		if fields.shape[1] == selection.length.Y:
			start = source.pos.Y - selection.start.Y
		elif fields.shape[1] == state.dataset.physical_shape.Y:
			start = source.pos.Y
		else:
			raise ValueError(
				"resolved flat_fields do not cover the canonical padded source window"
			)
		stop = start + source.shape.Y
		if start < 0 or stop > fields.shape[1]:
			raise ValueError(
				"selected flat_fields do not cover the canonical padded source window"
			)
		fields = fields[:, start:stop]
	if state.storage is None:
		state.artifacts["flat_fields"] = fields
		return fields
	artifact = state.storage.allocate(
		GenericOrder(*fields.shape), fields.dtype,
		role="flat_fields", reuse_from=("flat_fields",),
	)
	artifact.array()[...] = fields
	state.artifacts["flat_fields"] = artifact
	return artifact.array()


class WeightedNormalizeStage(Stage):
	"""Blend pre/post flat fields into a float32 managed output.

	The working array is in ``(Y, Theta, X)`` order. The four flat fields are
	provided through ``Scan.metadata["flat_fields"]`` in pre-gain, post-gain,
	pre-dark, post-dark order.
	"""

	name = "normalize.weighted"
	contract = StageContract(
		produces=PRESERVE,
		accepts_domain=(DataDomain.RAW,),
		produces_domain=DataDomain.TRANSMISSION,
		accepts=(ProjOrder, SinoOrder),
		prefers=ProjOrder,
		mutates_input=False,
		cacheable_prefix=True,
		window_axes=("Y",),
		required_metadata=("metadata.flat_fields",),
		produces_artifacts=("flat_fields",),
		managed_artifacts=("flat_fields",),
	)

	class Params(StageParams):
		pre_sub: bool = False
		timing_source: str | None = None

		@model_validator(mode="before")
		@classmethod
		def normalize_pre_subtracted(cls, value):
			if isinstance(value, dict) and "pre_subtracted" in value:
				value = dict(value)
				alias = value.pop("pre_subtracted")
				if alias is not None:
					value["pre_sub"] = alias
			return value


	def run(self, state: PipelineState) -> PipelineState:
		from lftomo.prep.flat import _weighted_normalize_projection

		if state.backend.name != "numpy":
			raise TypeError("WeightedNormalizeStage currently requires the NumPy backend")

		output, working = _normalization_output(state)
		flats = _managed_flat_fields(state)
		layout = type(state.buffer_spec.shape)
		if working.ndim != 3:
			raise ValueError("weighted-normalize expects a (Y, Theta, X) working array")
		y_size = working.shape[layout.axis_of("Y")]
		if flats.shape != (4, y_size, working.shape[layout.axis_of("X")]):
			raise ValueError(
				"flat_fields must have shape (4, Y, X) matching the working array"
			)

		sinogram_view = working if layout is SinoOrder else np.transpose(working, (1, 0, 2))
		if state.source_window is None or state.dataset is None:
			raise RuntimeError(f"{self.name} requires canonical source geometry")
		source_window = state.source_window.with_pad(["Y"])
		for local_index, global_index in enumerate(source_window.int_iter("Theta")):
			projection = source_window.rescope(
				pos=source_window.pos.merge(Theta(global_index), retain=False), shape=Theta(1),
			)
			image = sinogram_view[SinoOrder.point_index(Theta=local_index)]
			_weighted_normalize_projection(
				image,
				image,
				flats,
				projection,
				state.dataset.physical_shape,
				self.params.pre_sub,
			)
		return _finish_normalization(state, output)


def _flat_fields(state: PipelineState, data: np.ndarray, layout: type) -> np.ndarray:
	flats = _managed_flat_fields(state)
	y_size = data.shape[layout.axis_of("Y")]
	if flats.shape != (4, y_size, data.shape[layout.axis_of("X")]):
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


def _tracking_fields(
	state: PipelineState, y_size: int, component_names: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	ref = state.artifacts.get("calibration_bundle")
	if isinstance(ref, CalibrationBundleRef):
		bundle = CalibrationBundle.open(ref.manifest_path)
		if state.source_window is None:
			raise RuntimeError("calibration reads require PipelineState.source_window")
		source_window = state.source_window.with_pad(["Y"])
		start = source_window.pos.Y
		stop = start + source_window.shape.Y
		if stop - start != y_size:
			raise ValueError("calibration rows do not match the canonical source window")
		return tuple(
			bundle.read_component(component_names[name], y_start=start, y_stop=stop).astype(np.float32)
			for name in ("pre_gain", "post_gain", "pre_dark", "post_dark")
		)
	if "flat_fields" in state.artifacts or "flat_fields" in state.scan.metadata:
		fields = _managed_flat_fields(state)
	else:
		raise ValueError("normalize.beam_tracking requires a calibration bundle or legacy flat_fields")
	if fields.shape != (4, y_size, np.asarray(state.working_array).shape[2]):
		raise ValueError("legacy flat_fields must have shape (4, Y, X) for beam tracking")
	return tuple(fields)


class BeamTrackingNormalizeStage(Stage):
	"""Evaluate a calibrated two-flat beam model one projection at a time."""

	name = "normalize.beam_tracking"
	contract = StageContract(
		produces=PRESERVE, accepts=(ProjOrder, SinoOrder), prefers=ProjOrder,
		accepts_domain=(DataDomain.RAW,), produces_domain=DataDomain.TRANSMISSION,
		mutates_input=False, window_axes=("Y", "Theta"),
		cacheable_prefix=True,
		produces_artifacts=("beam_tracking_qc",),
	)

	class Params(StageParams):
		transport_sigma: float = Field(default=12, gt=0)
		schedule: BeamSchedule = Field(default_factory=BeamSchedule)
		fit_schedule: bool = False
		schedule_sample_count: int = Field(default=8, ge=2)
		component_names: dict[str, str] = Field(default_factory=dict, validate_default=True)
		pre_subtracted: bool = False

		@field_validator("schedule", mode="before")
		@classmethod
		def default_schedule(cls, value):
			return BeamSchedule() if value is None else value

		@field_validator("component_names", mode="before")
		@classmethod
		def merge_component_names(cls, value):
			if value is not None and not isinstance(value, Mapping):
				return value
			return {
				"pre_gain": "pre_gain", "post_gain": "post_gain",
				"pre_dark": "pre_dark", "post_dark": "post_dark",
				**(value or {}),
			}

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("BeamTrackingNormalizeStage currently requires the NumPy backend")
		output, data = _normalization_output(state)
		layout = type(state.buffer_spec.shape)
		if data.ndim != 3:
			raise ValueError("normalize.beam_tracking expects a three-dimensional array")
		y_size = data.shape[layout.axis_of("Y")]
		pre_gain, post_gain, pre_dark, post_dark = _tracking_fields(
			state, y_size, self.params.component_names,
		)
		if state.source_window is None or state.dataset is None:
			raise RuntimeError(f"{self.name} requires canonical source geometry")
		projection_count = state.dataset.physical_shape.Theta
		batch_count = state.buffer_spec.shape.Theta
		schedule = self.params.schedule
		fit = None
		prepared = getattr(self, "_dispatch_schedule_fit", None)
		if prepared is not None:
			fit = prepared
			schedule = fit.schedule
		elif self.params.fit_schedule:
			indices = np.unique(np.linspace(0, batch_count - 1, self.params.schedule_sample_count).astype(int))
			fractions = (
				state.source_window.pos.Theta
				+ indices * state.source_window.step.Theta
			) / max(projection_count - 1, 1)
			fit = calibrate_schedule(
				np.stack([data[layout.point_index(Theta=int(index))] for index in indices]),
				fractions, pre_gain, post_gain,
			)
			schedule = fit.schedule
		gain_at = optimal_transport_interp(
			pre_gain, post_gain, transport_sigma=self.params.transport_sigma, schedule=schedule,
		)
		for local_index in range(batch_count):
			global_index = (
				state.source_window.pos.Theta
				+ local_index * state.source_window.step.Theta
			)
			fraction = global_index / max(projection_count - 1, 1)
			beam_time = float(schedule(fraction))
			gain = gain_at(fraction)
			# Schedules may legitimately extrapolate beam structure. Dark fields do
			# not carry that motion model, so keep them within measured endpoints.
			dark_time = float(np.clip(beam_time, 0, 1))
			dark = (1 - dark_time) * pre_dark + dark_time * post_dark
			image = data[layout.point_index(Theta=local_index)]
			image[:] = _normalize_with_maps(
				image, gain, dark, self.params.pre_subtracted, floor_nonpositive=True,
			)
		state.artifacts["beam_tracking_qc"] = {
			"diagnostics": diagnose_beam_change(pre_gain, post_gain),
			"schedule": schedule.__dict__,
			"fit": None if fit is None else {
				"residual_std": fit.residual_std, "air_pixel_count": fit.air_pixel_count,
				"air_fraction": fit.air_fraction, "beam_times": fit.beam_times,
			},
		}
		return _finish_normalization(state, output)


class BeamKeyframesNormalizeStage(Stage):
	"""Normalize from the two physical-time keyframes bracketing each projection."""

	name = "normalize.beam_keyframes"
	contract = StageContract(
		produces=PRESERVE, accepts=(ProjOrder, SinoOrder), prefers=ProjOrder,
		accepts_domain=(DataDomain.RAW,), produces_domain=DataDomain.TRANSMISSION,
		mutates_input=False, window_axes=("Y", "Theta"),
		cacheable_prefix=True,
		produces_artifacts=("beam_keyframe_qc",),
	)

	class Params(StageParams):
		component: str = "beam_keyframes"
		dark_component: str | None = None
		transport_sigma: float = 12
		maximum_gap: float | None = None

	def run(self, state: PipelineState) -> PipelineState:
		output, data = _normalization_output(state)
		layout = type(state.buffer_spec.shape)
		y_size = data.shape[layout.axis_of("Y")]
		if state.source_window is None:
			raise RuntimeError(f"{self.name} requires PipelineState.source_window")
		source_window = state.source_window.with_pad(["Y"])
		start = source_window.pos.Y
		stop = start + source_window.shape.Y
		if stop - start != y_size:
			raise ValueError("calibration rows do not match the canonical source window")
		ref = state.artifacts.get("calibration_bundle")
		if not isinstance(ref, CalibrationBundleRef):
			raise ValueError("normalize.beam_keyframes requires a selected calibration bundle")
		bundle = CalibrationBundle.open(ref.manifest_path)
		component = bundle.component(self.params.component)
		if component.time_axis is None:
			raise ValueError("beam keyframes require an explicit time axis")
		if component.time_axis.kind == "timestamp_seconds":
			times = state.artifacts.get("projection_times")
			if times is None:
				raise ValueError("physical-time keyframes require projection_times artifacts")
			times = np.asarray(times, dtype=float)
			if len(times) != data.shape[layout.axis_of("Theta")]:
				raise ValueError("projection_times must match the current Theta window")
		else:
			times = (
				state.source_window.pos.Theta
				+ np.arange(state.buffer_spec.shape.Theta) * state.source_window.step.Theta
			).astype(float)
		dark = 0.0 if self.params.dark_component is None else bundle.read_component(
			self.params.dark_component, y_start=start, y_stop=stop,
		)
		maximum_seen_gap = 0.0
		for local_index, time in enumerate(times):
			bracket = bundle.read_time_bracket(
				self.params.component, float(time), y_start=start, y_stop=stop,
			)
			gap = bracket.after_time - bracket.before_time
			maximum_seen_gap = max(maximum_seen_gap, gap)
			if self.params.maximum_gap is not None and gap > self.params.maximum_gap:
				raise ValueError(
					f"projection time {time} lies in keyframe gap {gap}, exceeding {self.params.maximum_gap}"
				)
			average = 0.5 * (np.log(np.clip(bracket.before, 1e-6, None))
				+ np.log(np.clip(bracket.after, 1e-6, None)))
			transport = gaussian_filter(
				np.log(np.clip(bracket.after, 1e-6, None))
				- np.log(np.clip(bracket.before, 1e-6, None)), self.params.transport_sigma,
			)
			gain = np.exp(average + (bracket.weight - 0.5) * transport)
			image = data[layout.point_index(Theta=local_index)]
			image[:] = _normalize_with_maps(image, gain, dark, False, floor_nonpositive=True)
		state.artifacts["beam_keyframe_qc"] = {
			"component": self.params.component, "maximum_gap_seen": maximum_seen_gap,
			"observability": "events wholly inside a keyframe gap cannot be recovered",
		}
		return _finish_normalization(state, output)


class BeamBasisFitNormalizeStage(Stage):
	"""Fit a low-dimensional external beam library from air in each projection."""

	name = "normalize.beam_basis_fit"
	contract = StageContract(
		produces=PRESERVE, accepts=(ProjOrder, SinoOrder), prefers=ProjOrder,
		accepts_domain=(DataDomain.RAW,), produces_domain=DataDomain.TRANSMISSION,
		mutates_input=False, window_axes=("Y", "Theta"),
		cacheable_prefix=True,
		consumes_artifacts=("calibration_bundle",), produces_artifacts=("beam_basis_qc",),
	)

	class Params(StageParams):
		component: str = "beam_basis"
		detector_component: str | None = None
		air_mask_artifact: str = "air_mask"
		air_threshold: float = Field(default=0.95, gt=0, le=1)
		minimum_air_pixels: int = Field(default=100, gt=0)
		minimum_air_fraction: float = Field(default=0.005, gt=0, le=1)
		trim_sigma: float = Field(default=4.0, gt=0)
		erode: int = Field(default=1, ge=0)
		iterations: int = Field(default=4, ge=1)
		exposure_quantile: float = Field(default=0.2, ge=0, le=1)
		edge_quantile: float = Field(default=0.85, ge=0, le=1)
		maximum_fit_residual: float = Field(default=0.08, gt=0)
		maximum_out_of_basis: float = Field(default=0.75, gt=0)
		maximum_basis_condition: float = Field(default=1e6, gt=0)
		maximum_contamination_fraction: float = Field(default=0.5, gt=0)
		maximum_coefficient_extrapolation: float = Field(default=1.0, gt=0)
		fail_on_qc: bool = True

	def _mask(self, state: PipelineState, layout: type, index: int, shape: tuple[int, int]):
		value = state.artifacts.get(self.params.air_mask_artifact)
		if value is None:
			return None
		value = np.asarray(buffer_view(value), dtype=bool)
		if value.ndim == 2:
			mask = value
		elif value.ndim == 3:
			mask = value[layout.point_index(Theta=index)]
		else:
			raise ValueError(f"{self.params.air_mask_artifact} must be a 2-D or 3-D mask")
		if mask.shape != shape:
			raise ValueError(f"{self.params.air_mask_artifact} does not match the active detector window")
		return mask

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("BeamBasisFitNormalizeStage currently requires the NumPy backend")
		output, data = _normalization_output(state)
		layout = type(state.buffer_spec.shape)
		if data.ndim != 3:
			raise ValueError("normalize.beam_basis_fit expects a three-dimensional array")
		ref = state.artifacts.get("calibration_bundle")
		if not isinstance(ref, CalibrationBundleRef):
			raise ValueError("normalize.beam_basis_fit requires a selected calibration bundle")
		bundle = CalibrationBundle.open(ref.manifest_path)
		descriptor = bundle.component(self.params.component)
		if descriptor.kind != "beam_basis" or descriptor.coordinate_frame != "beam":
			raise ValueError("beam-basis component must have kind='beam_basis' in beam coordinates")
		y_size = data.shape[layout.axis_of("Y")]
		if state.source_window is None:
			raise RuntimeError(f"{self.name} requires PipelineState.source_window")
		source_window = state.source_window.with_pad(["Y"])
		start = source_window.pos.Y
		stop = start + source_window.shape.Y
		if stop - start != y_size:
			raise ValueError("calibration rows do not match the canonical source window")
		basis = bundle.read_component(self.params.component, y_start=start, y_stop=stop)
		if basis.shape[2] != data.shape[2]:
			raise ValueError("beam-basis X extent is incompatible with the active detector window")
		detector = None if self.params.detector_component is None else bundle.read_component(
			self.params.detector_component, y_start=start, y_stop=stop,
		)
		if self.params.detector_component is not None:
			detector_descriptor = bundle.component(self.params.detector_component)
			if (detector_descriptor.kind != "detector_static"
					or detector_descriptor.coordinate_frame != "detector"):
				raise ValueError(
					"detector component must have kind='detector_static' in detector coordinates"
				)
		metadata = descriptor.metadata
		training_min = np.asarray(metadata.get("training_coefficient_min", []), dtype=float)
		training_max = np.asarray(metadata.get("training_coefficient_max", []), dtype=float)
		qcs, coefficients, violations = [], [], []
		for index in range(data.shape[layout.axis_of("Theta")]):
			image = data[layout.point_index(Theta=index)]
			fit = fit_projection_beam(
				image, basis, detector=detector,
				air_mask=self._mask(state, layout, index, image.shape),
				air_threshold=self.params.air_threshold,
				minimum_air_pixels=self.params.minimum_air_pixels,
				minimum_air_fraction=self.params.minimum_air_fraction,
				trim_sigma=self.params.trim_sigma, erode=self.params.erode, n_iter=self.params.iterations,
				exposure_quantile=self.params.exposure_quantile,
				edge_quantile=self.params.edge_quantile,
			)
			coefficient_extrapolation = 0.0
			if training_min.shape == fit.coefficients.shape == training_max.shape:
				span = np.maximum(training_max - training_min, 1e-12)
				outside = np.maximum(training_min - fit.coefficients, fit.coefficients - training_max)
				coefficient_extrapolation = float(np.max(np.maximum(outside, 0) / span))
			contamination = max(fit.qc.rejected_fraction, fit.qc.edge_overlap_fraction)
			projection_violations = []
			for failed, label in (
				(fit.qc.fit_residual > self.params.maximum_fit_residual, "fit_residual"),
				(fit.qc.out_of_basis > self.params.maximum_out_of_basis, "out_of_basis"),
				(fit.qc.basis_condition > self.params.maximum_basis_condition, "basis_condition"),
				(contamination > self.params.maximum_contamination_fraction, "mask_contamination"),
				(coefficient_extrapolation > self.params.maximum_coefficient_extrapolation,
					"coefficient_extrapolation"),
			):
				if failed:
					projection_violations.append(label)
			if projection_violations and self.params.fail_on_qc:
				raise ValueError(
					f"beam-basis fit failed QC for projection {index}: "
					+ ", ".join(projection_violations)
				)
			image[:] = fit.transmission
			coefficients.append(fit.coefficients)
			violations.append(projection_violations)
			qcs.append({
				**asdict(fit.qc), "coefficient_extrapolation": coefficient_extrapolation,
				"mask_source": "auto" if self.params.air_mask_artifact not in state.artifacts else "supplied",
				"violations": projection_violations,
			})
		coefficient_array = np.asarray(coefficients)
		state.artifacts["beam_basis_qc"] = {
			"component": self.params.component,
			"training_coverage": descriptor.compatibility.training_coverage,
			"training_relative_residual": metadata.get("training_relative_residual"),
			"projections": qcs,
			"coefficient_std": coefficient_array.std(axis=0).tolist(),
			"maximum_coefficient_step": float(np.max(np.linalg.norm(
				np.diff(coefficient_array, axis=0), axis=1,
			))) if len(coefficient_array) > 1 else 0.0,
			"violating_projection_count": sum(bool(value) for value in violations),
			"basis_scope": "only beam states represented by training_coverage are supported",
			"mask_assumption": (
				"robust trimming assumes candidate-mask air pixels are the majority; "
				"fit and coefficient QC guard gross contamination"
			),
		}
		state.artifacts["_beam_basis_coefficients"] = coefficient_array
		state.artifacts["_beam_basis_violations"] = violations
		return _finish_normalization(state, output)


class ThetaSwitchNormalizeStage(Stage):
	"""Use pre/post calibration fields with midpoint switching or linear blending."""

	name = "normalize.theta_switch"
	contract = StageContract(
		produces=PRESERVE,
		accepts=(ProjOrder, SinoOrder),
		prefers=ProjOrder,
		accepts_domain=(DataDomain.RAW,),
		produces_domain=DataDomain.TRANSMISSION,
		mutates_input=False,
		cacheable_prefix=True,
		window_axes=("Y", "Theta"),
		required_metadata=("metadata.flat_fields",),
		produces_artifacts=("flat_fields",),
		managed_artifacts=("flat_fields",),
	)

	class Params(StageParams):
		gain_averaging: bool = False
		pre_subtracted: bool = False
		pre_gain: str | None = None
		post_gain: str | None = None
		pre_dark: str | None = None
		post_dark: str | None = None
		floor_nonpositive: bool = False
		floor_value: float = 1.0

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("ThetaSwitchNormalizeStage currently requires the NumPy backend")
		output, data = _normalization_output(state)
		layout = type(state.buffer_spec.shape)
		if data.ndim != 3:
			raise ValueError("normalize.theta_switch expects a three-dimensional tomography array")
		flats = _flat_fields(state, data, layout)
		pre_gain, post_gain, pre_dark, post_dark = flats
		if state.source_window is None or state.dataset is None:
			raise RuntimeError(f"{self.name} requires canonical source geometry")
		total_count = state.dataset.physical_shape.Theta
		for local_projection in range(state.buffer_spec.shape.Theta):
			projection = (
				state.source_window.pos.Theta
				+ local_projection * state.source_window.step.Theta
			)
			if self.params.gain_averaging:
				weight = projection / total_count
				gain = pre_gain * (1 - weight) + post_gain * weight
				dark = pre_dark * (1 - weight) + post_dark * weight
			elif projection > total_count / 2:
				gain, dark = post_gain, post_dark
			else:
				gain, dark = pre_gain, pre_dark
			image = data[layout.point_index(Theta=local_projection)]
			image[:] = _normalize_with_maps(
				image, gain, dark, self.params.pre_subtracted,
				self.params.floor_nonpositive, self.params.floor_value,
			)
		return _finish_normalization(state, output)


class TaperBlendedNormalizeStage(ThetaSwitchNormalizeStage):
	"""Phase-contrast lineage's linear pre/post gain and dark blending."""

	name = "normalize.taper_blended"

	class Params(ThetaSwitchNormalizeStage.Params):
		pre_weight: Literal["linear"] = "linear"
		post_weight: Literal["linear"] = "linear"
		gain_averaging: Literal[True] = True
		gain_movement_enabled: Literal[False] = False




def prepare_normalization_dispatch(stage: Stage, state: PipelineState) -> None:
	"""Perform normalization work that requires the complete projection window."""

	if not isinstance(stage, BeamTrackingNormalizeStage) or not stage.params.fit_schedule:
		return
	data = np.asarray(state.working_array)
	layout = type(state.buffer_spec.shape)
	y_size = data.shape[layout.axis_of("Y")]
	pre_gain, post_gain, _pre_dark, _post_dark = _tracking_fields(
		state, y_size, stage.params.component_names,
	)
	if state.source_window is None or state.dataset is None:
		raise RuntimeError(f"{stage.name} requires canonical source geometry")
	projection_count = state.dataset.physical_shape.Theta
	batch_count = state.buffer_spec.shape.Theta
	indices = np.unique(
		np.linspace(0, batch_count - 1, stage.params.schedule_sample_count).astype(int),
	)
	fractions = (
		state.source_window.pos.Theta
		+ indices * state.source_window.step.Theta
	) / max(projection_count - 1, 1)
	stage._dispatch_schedule_fit = calibrate_schedule(
		np.stack([data[layout.point_index(Theta=int(index))] for index in indices]),
		fractions, pre_gain, post_gain,
	)


def merge_normalization_dispatch(
	stage: Stage, state: PipelineState, results: list[dict],
) -> None:
	"""Merge ordered per-projection artifacts into the serial Stage contract."""

	if isinstance(stage, BeamBasisFitNormalizeStage) and results:
		qcs = [item["beam_basis_qc"] for item in results]
		coefficients = np.concatenate([
			np.asarray(item["_beam_basis_coefficients"]) for item in results
		], axis=0)
		violations = [
			value for item in results
			for value in item["_beam_basis_violations"]
		]
		merged = dict(qcs[0])
		merged["projections"] = [
			projection for qc in qcs for projection in qc["projections"]
		]
		merged["coefficient_std"] = coefficients.std(axis=0).tolist()
		merged["maximum_coefficient_step"] = (
			float(np.max(np.linalg.norm(np.diff(coefficients, axis=0), axis=1)))
			if len(coefficients) > 1 else 0.0
		)
		merged["violating_projection_count"] = sum(bool(value) for value in violations)
		state.artifacts["beam_basis_qc"] = merged
		return
	if isinstance(stage, BeamKeyframesNormalizeStage) and results:
		qcs = [item["beam_keyframe_qc"] for item in results]
		merged = dict(qcs[0])
		merged["maximum_gap_seen"] = max(item["maximum_gap_seen"] for item in qcs)
		state.artifacts["beam_keyframe_qc"] = merged
		return
	if isinstance(stage, BeamTrackingNormalizeStage) and results:
		merged = dict(results[0]["beam_tracking_qc"])
		state.artifacts["beam_tracking_qc"] = merged
		if hasattr(stage, "_dispatch_schedule_fit"):
			del stage._dispatch_schedule_fit
		return
	for artifacts in results:
		state.artifacts.update(artifacts)
