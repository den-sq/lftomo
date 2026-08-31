"""Explicit full-volume layout conversion Stage."""

from __future__ import annotations

import numpy as np

from lftomo.config import StageConfig
from lftomo.pipeline.core import (
	BufferSpec, Locality, PipelineState, Stage, StageContract, allocate_output, commit_output,
)
from lftomo.pipeline.domains import PRESERVE
from lftomo.util.shapes import ProjOrder, SinoOrder


_LAYOUTS = {
	"projection": ProjOrder,
	"proj": ProjOrder,
	"sinogram": SinoOrder,
	"sino": SinoOrder,
}


class LayoutConvertStage(Stage):
	"""Materialize the working volume in the requested tomography layout."""

	name = "layout.convert"

	def __init__(self, target: str, reason: str):
		try:
			self.target = _LAYOUTS[target.lower()]
		except KeyError as error:
			raise ValueError("layout.convert target must be 'projection' or 'sinogram'") from error
		self.reason = reason.strip()
		if not self.reason:
			raise ValueError("layout.convert requires a non-empty reason")
		self.contract = StageContract(
			accepts=(ProjOrder, SinoOrder),
			produces=self.target,
			produces_domain=PRESERVE,
			locality=Locality.VOLUME,
			mutates_input=False,
			cacheable_prefix=True,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "LayoutConvertStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("layout.convert currently requires the NumPy backend")
		if state.layout not in (ProjOrder, SinoOrder):
			raise TypeError("layout.convert requires ProjOrder or SinoOrder input")
		if state.layout is self.target:
			raise ValueError("layout.convert target is already the current layout")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("layout.convert expects a three-dimensional tomography array")
		input_shape = state.buffer_spec.shape
		shape = input_shape.to(self.target)
		output, converted = allocate_output(
			state, shape, data.dtype, role=f"layout:{self.target.__name__}",
		)
		converted[...] = input_shape.transposed_view(data, self.target)
		return commit_output(
			state, output, retire_as=None, buffer_spec=BufferSpec(shape, data.dtype),
		)
