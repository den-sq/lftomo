"""Projection-domain image filtering stages."""

from __future__ import annotations

from functools import partial

import cv2
import numpy as np

from lftomo.config import StageConfig
from lftomo.pipeline.core import (
	Locality, PadRequirement, PipelineState, Stage,
	StageContract,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.util.shapes import Proj, ProjOrder, SinoOrder, SliceIter, Theta


class BilateralFilterStage(Stage):
	"""Apply OpenCV edge-preserving bilateral filtering per projection.

	OpenCV's bilateral implementation consumes the canonical float32 work buffer
	selected by the source window.
	"""

	name = "filter.bilateral"

	def __init__(self, diameter: int = 3, sigma_color: float = 1,
				sigma_space: float = 1, enabled: bool = True,
				border_type: int = cv2.BORDER_DEFAULT):
		if diameter < 1:
			raise ValueError("bilateral diameter must be positive")
		if sigma_color <= 0 or sigma_space <= 0:
			raise ValueError("bilateral sigma values must be positive")
		self.contract = StageContract(
			mutates_input=enabled,
			accepts=(ProjOrder, SinoOrder), prefers=ProjOrder, produces=PRESERVE,
			accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
			produces_domain=PRESERVE, locality=Locality.PROJECTION,
			window_axes=("Y", "Theta"),
			required_pad=PadRequirement(
				fixed=Proj(Y=diameter // 2, X=diameter // 2),
			),
		)
		self.diameter = diameter
		self.sigma_color = sigma_color
		self.sigma_space = sigma_space
		self.enabled = enabled
		self.border_type = border_type

	@classmethod
	def from_config(cls, config: StageConfig) -> "BilateralFilterStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("BilateralFilterStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		shape = state.buffer_spec.shape
		if data.ndim != 3:
			raise ValueError("filter.bilateral expects a (Y, Theta, X) sinogram array")
		if not self.enabled:
			return state
		if data.dtype != np.dtype(np.float32):
			raise TypeError("filter.bilateral requires the canonical float32 work dtype")
		for index in Theta.drange(0, shape.Theta, 1).get_iter(partial(SliceIter, shape=shape)):
			image = data[index]
			image[:] = cv2.bilateralFilter(
				image,
				self.diameter,
				self.sigma_color,
				self.sigma_space,
				borderType=self.border_type,
			)
		return state
