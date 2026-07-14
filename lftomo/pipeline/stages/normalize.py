"""Flat- and dark-field normalization stages."""

from __future__ import annotations

import numpy as np

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
