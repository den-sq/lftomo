"""Projection-domain image filtering stages."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pydantic import Field

from lftomo.pipeline.core import (
	BoundedWorkRequest, BufferSpec, PadRequirement, PipelineState,
	ProjectionStage, StageContract, StageParams,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.util.shapes import Proj, ProjOrder, YSlice
from lftomo.util.window import TomoWindow


class ZingerFilterStage(ProjectionStage):
	"""Replace transmission pixels above 1.5 times their original 3x3 median.

	The median includes the centre pixel, uses only the current projection,
	and repeats physical edge samples. Real ROI neighbors come from the
	canonical source window's one-pixel halo.
	"""

	name = "filter.zinger"
	contract = replace(
		ProjectionStage.contract,
		mutates_input=True,
		accepts_domain=(DataDomain.TRANSMISSION,),
		produces_domain=PRESERVE,
		window_axes=("Y", "Theta"),
		required_pad=PadRequirement(fixed=Proj(Y=1, X=1)),
	)

	class Params(StageParams):
		enabled: bool = True

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		return replace(
			cls.contract, mutates_input=params.enabled,
			required_pad=cls.contract.required_pad if params.enabled else PadRequirement(),
		)

	def bounded_work_requests(
		self, input_spec: BufferSpec, window: TomoWindow | None = None,
	) -> tuple[BoundedWorkRequest, ...]:
		if not self.params.enabled:
			return ()
		produced = input_spec.shape if window is None else window.frange.length
		# SciPy's median_filter output must match its context-bearing input.
		# This unavoidable library plane is read-sized; threshold work and
		# replacements cover only the Stage's produced extent.
		read = input_spec.shape if window is None else window.with_pad().frange.length
		return tuple(BoundedWorkRequest(
			f"_zinger_{name}", BufferSpec(shape, dtype), f"zinger_{name}",
			required_for_fallback=True,
		) for name, shape, dtype in (
			("median", read.to(ProjOrder), input_spec.dtype),
			("threshold", produced.extract(YSlice), input_spec.dtype),
			("mask", produced.extract(YSlice), np.dtype(bool)),
		))

	def run_projection(self, image: np.ndarray, state: PipelineState) -> None:
		if not self.params.enabled:
			return
		from scipy.ndimage import median_filter

		work = tuple(
			state.artifacts[f"_zinger_{name}"][state.scratch_index(
				state.artifacts[f"_zinger_{name}"].shape, context=name == "median",
			)] for name in ("median", "threshold", "mask")
		)
		# Retain sliced handles while NumPy views use their shared mappings.
		median, threshold, mask = (handle.array()[0] for handle in work)
		# Compute from the unchanged projection so replacements cannot influence
		# their neighbors. SciPy writes directly into the declared median plane.
		median_filter(image, size=3, mode="nearest", output=median)
		selection = (
			(slice(None), slice(None)) if state.produced_window is None else
			state.read_window.packed_index(state.produced_window, Proj)
		)
		for row, local_median in zip(image[selection], median[selection]):
			np.multiply(local_median, 1.5, out=threshold)
			np.greater(row, threshold, out=mask)
			np.copyto(row, local_median, where=mask)


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
