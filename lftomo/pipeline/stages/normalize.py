"""Flat- and dark-field normalization stages."""

from __future__ import annotations

import numpy as np

from lftomo.config import StageConfig
from lftomo.pipeline.core import PipelineState, ResourceHints, Stage, StageContract
from lftomo.prep.flat import _weighted_normalize_projection
from lftomo.util.shapes import SinoOrder


class WeightedNormalizeStage(Stage):
	"""Blend pre/post flat fields and normalize a sinogram array in place.

	The working array is in ``(Y, Theta, X)`` order. The four flat fields are
	provided through ``Scan.metadata["flat_fields"]`` in pre-gain, post-gain,
	pre-dark, post-dark order.
	"""

	name = "weighted-normalize"
	contract = StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		mutates_input=True,
		window_axes=("Y",),
		required_metadata=("metadata.flat_fields",),
		resources=ResourceHints(memory_multiplier=1.0),
	)

	def __init__(self, pre_sub: bool = False, pre_subtracted: bool | None = None,
				timing_source: str | None = None):
		self.pre_sub = pre_sub if pre_subtracted is None else pre_subtracted
		self.timing_source = timing_source

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("WeightedNormalizeStage currently requires the NumPy backend")

		working = np.asarray(state.working_array)
		flats = np.asarray(state.scan.metadata["flat_fields"])
		if working.ndim != 3:
			raise ValueError("weighted-normalize expects a (Y, Theta, X) working array")
		if flats.shape != (4, working.shape[0], working.shape[2]):
			raise ValueError(
				"flat_fields must have shape (4, Y, X) matching the working array"
			)

		for projection in range(working.shape[1]):
			_weighted_normalize_projection(
				working,
				working,
				flats,
				projection,
				working.shape[1],
				self.pre_sub,
			)
		return state


def _flat_fields(state: PipelineState, data: np.ndarray) -> np.ndarray:
	flats = np.asarray(state.scan.metadata["flat_fields"])
	if flats.shape != (4, data.shape[0], data.shape[2]):
		raise ValueError("flat_fields must have shape (4, Y, X) matching the working array")
	return flats


def _normalize_with_maps(projection: np.ndarray, gain: np.ndarray, dark: np.ndarray,
				pre_subtracted: bool) -> np.ndarray:
	denominator = gain if pre_subtracted else gain - dark
	denominator = denominator.copy()
	zero = denominator == 0
	if zero.any():
		nonzero = np.abs(denominator[~zero])
		if nonzero.size == 0:
			raise ValueError("normalization gain map contains only zeros")
		denominator[zero] = nonzero.min()
	return np.divide(projection - dark, denominator)


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
				pre_dark: str | None = None, post_dark: str | None = None):
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
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("normalize.theta_switch expects a (Y, Theta, X) sinogram array")
		flats = _flat_fields(state, data)
		pre_gain, post_gain, pre_dark, post_dark = flats
		count = data.shape[1]
		for projection in range(count):
			if self.gain_averaging:
				weight = projection / count
				gain = pre_gain * (1 - weight) + post_gain * weight
				dark = pre_dark * (1 - weight) + post_dark * weight
			elif projection > count / 2:
				gain, dark = post_gain, post_dark
			else:
				gain, dark = pre_gain, pre_dark
			data[:, projection, :] = _normalize_with_maps(
				data[:, projection, :], gain, dark, self.pre_subtracted
			)
		return state


class TaperBlendedNormalizeStage(ThetaSwitchNormalizeStage):
	"""Phase-contrast lineage's linear pre/post gain and dark blending."""

	name = "normalize.taper_blended"

	def __init__(self, pre_subtracted: bool = False,
				pre_gain: str | None = None, post_gain: str | None = None,
				pre_dark: str | None = None, post_dark: str | None = None,
				pre_weight: str = "linear", post_weight: str = "linear",
				gain_averaging: bool = True, gain_movement_enabled: bool = False):
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
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "TaperBlendedNormalizeStage":
		return cls(**config.params)
