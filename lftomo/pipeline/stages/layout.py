"""Explicit full-volume layout conversion Stage."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
from pydantic import field_validator

from lftomo.pipeline.core import (
	PipelineState, Stage, StageContract, StageParams, allocate_output, commit_output,
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
	contract = StageContract(
		accepts=(ProjOrder, SinoOrder),
		produces_domain=PRESERVE,
		mutates_input=False,
		cacheable_prefix=True,
	)

	class Params(StageParams):
		target: Literal["projection", "sinogram"]
		reason: str

		@field_validator("target", mode="before")
		@classmethod
		def normalize_target(cls, value):
			if not isinstance(value, str):
				return value
			try:
				layout = _LAYOUTS[value.lower()]
			except KeyError as error:
				raise ValueError("layout.convert target must be 'projection' or 'sinogram'") from error
			return "projection" if layout is ProjOrder else "sinogram"

		@field_validator("reason")
		@classmethod
		def validate_reason(cls, value):
			value = value.strip()
			if not value:
				raise ValueError("layout.convert requires a non-empty reason")
			return value

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		return replace(cls.contract, produces=_LAYOUTS[params.target])

	def __init__(self, target: str, reason: str):
		# Preserve the public two-positional-argument form as an input adapter.
		super().__init__(target=target, reason=reason)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("layout.convert currently requires the NumPy backend")
		if state.layout not in (ProjOrder, SinoOrder):
			raise TypeError("layout.convert requires ProjOrder or SinoOrder input")
		target = self.contract.produces
		if state.layout is target:
			raise ValueError("layout.convert target is already the current layout")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("layout.convert expects a three-dimensional tomography array")
		input_shape = state.working_array.shape
		shape = input_shape.to(target)
		output, converted = allocate_output(
			state, shape, data.dtype, role=f"layout:{target.__name__}",
		)
		converted[state.work_index(shape)] = input_shape.transposed_view(
			data[state.work_index()], target,
		)
		return commit_output(
			state, output, retire_as=None,
		)
