"""Projection-domain image filtering stages."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pydantic import Field

from lftomo.pipeline.core import (
	PadRequirement, PipelineState, ProjectionStage, StageContract, StageParams,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.util.shapes import Proj


class BilateralFilterStage(ProjectionStage):
	"""Apply OpenCV edge-preserving bilateral filtering per projection.

	OpenCV's bilateral implementation consumes the canonical float32 work buffer
	selected by the source window.
	"""

	name = "filter.bilateral"
	contract = replace(
		ProjectionStage.contract,
		mutates_input=True,
		accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
		produces_domain=PRESERVE,
		window_axes=("Y", "Theta"),
	)

	class Params(StageParams):
		diameter: int = Field(default=3, ge=1)
		sigma_color: float = Field(default=1, gt=0)
		sigma_space: float = Field(default=1, gt=0)
		enabled: bool = True
		border_type: int = 4  # OpenCV BORDER_DEFAULT; keep schema imports lightweight.

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		return replace(
			cls.contract,
			mutates_input=params.enabled,
			required_pad=PadRequirement(
				fixed=Proj(Y=params.diameter // 2, X=params.diameter // 2),
			),
		)

	def run_projection(self, image: np.ndarray, state: PipelineState) -> np.ndarray | None:
		if not self.params.enabled:
			return None
		if image.dtype != np.dtype(np.float32):
			raise TypeError("filter.bilateral requires the canonical float32 work dtype")
		import cv2

		return cv2.bilateralFilter(
			image,
			self.params.diameter,
			self.params.sigma_color,
			self.params.sigma_space,
			borderType=self.params.border_type,
		)
