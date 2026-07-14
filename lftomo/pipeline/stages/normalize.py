"""Flat- and dark-field normalization stages."""

from __future__ import annotations

import numpy as np

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

	def __init__(self, pre_sub: bool = False):
		self.pre_sub = pre_sub

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
