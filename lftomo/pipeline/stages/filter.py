"""Projection-domain image filtering stages."""

from __future__ import annotations

import cv2
import numpy as np

from lftomo.config import StageConfig
from lftomo.pipeline.core import (
	AxisPad, Locality, PadRequirement, PipelineState, Stage,
	StageContract, working_view,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.pipeline.layout import configured_layout, n_projections, projection_view, resolve_layout
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
	)

	def __init__(self, diameter: int = 3, sigma_color: float = 1,
				sigma_space: float = 1, enabled: bool = True,
				border_type: int = cv2.BORDER_DEFAULT, order: str | None = None):
		if diameter < 1:
			raise ValueError("bilateral diameter must be positive")
		if sigma_color <= 0 or sigma_space <= 0:
			raise ValueError("bilateral sigma values must be positive")
		compatibility = configured_layout(order)
		shape = compatibility or SinoOrder
		self.contract = StageContract(
			expected_input=shape, produced_output=shape, mutates_input=enabled,
			accepts=(ProjOrder, SinoOrder), prefers=ProjOrder, produces=PRESERVE,
			accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
			produces_domain=PRESERVE, locality=Locality.PROJECTION,
			window_axes=("Y", "Theta"),
			required_pad=PadRequirement(
				fixed=AxisPad(y=diameter // 2, x=diameter // 2),
			),
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
		data = working_view(state)
		layout = resolve_layout(state, self.order, self.name)
		if data.ndim != 3:
			raise ValueError("filter.bilateral expects a (Y, Theta, X) sinogram array")
		if not self.enabled:
			return state
		count = n_projections(data, layout)
		for projection in range(count):
			image = projection_view(data, layout, projection)
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
