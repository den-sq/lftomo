"""Projection-domain image filtering stages."""

from __future__ import annotations

import cv2
import numpy as np

from lftomo.config import StageConfig
from lftomo.pipeline.core import PipelineState, ResourceHints, Stage, StageContract
from lftomo.util.shapes import ProjOrder, SinoOrder


class BilateralFilterStage(Stage):
	"""Apply OpenCV edge-preserving bilateral filtering per projection.

	OpenCV's bilateral implementation does not accept float64 images. Each
	projection is therefore evaluated at float32 precision, then assigned back
	into the original working array. A float64 working array remains float64,
	but the filtered values have float32 computational precision.
	"""

	name = "filter.bilateral"
	contract = StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		mutates_input=True,
		window_axes=("Y", "Theta"),
		resources=ResourceHints(memory_multiplier=2.0),
	)

	def __init__(self, diameter: int = 3, sigma_color: float = 1,
				sigma_space: float = 1, enabled: bool = True,
				border_type: int = cv2.BORDER_DEFAULT, order: str = "sinogram"):
		if diameter < 1:
			raise ValueError("bilateral diameter must be positive")
		if sigma_color <= 0 or sigma_space <= 0:
			raise ValueError("bilateral sigma values must be positive")
		if order not in ("projection", "sinogram"):
			raise ValueError("bilateral order must be 'projection' or 'sinogram'")
		shape = ProjOrder if order == "projection" else SinoOrder
		self.contract = StageContract(
			expected_input=shape, produced_output=shape, mutates_input=True,
			window_axes=("Y", "Theta"), resources=ResourceHints(memory_multiplier=2.0),
		)
		self.diameter = diameter
		self.sigma_color = sigma_color
		self.sigma_space = sigma_space
		self.enabled = enabled
		self.border_type = border_type
		self.order = order

	@classmethod
	def from_config(cls, config: StageConfig) -> "BilateralFilterStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("BilateralFilterStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("filter.bilateral expects a (Y, Theta, X) sinogram array")
		if not self.enabled:
			return state
		count = data.shape[0] if self.order == "projection" else data.shape[1]
		for projection in range(count):
			image = data[projection, :, :] if self.order == "projection" else data[:, projection, :]
			# This conversion is intentional and is part of the Stage's public
			# numerical behavior; cv2.bilateralFilter rejects float64 inputs.
			image[:] = cv2.bilateralFilter(
				image.astype(np.float32, copy=False),
				self.diameter,
				self.sigma_color,
				self.sigma_space,
				borderType=self.border_type,
			)
		return state
