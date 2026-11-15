"""Projection-domain geometry correction stages."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import Field, TypeAdapter, model_validator

import numpy as np
from scipy.ndimage import shift

from lftomo.pipeline.core import (
	BufferSpec, PadRequirement, PipelineState, Stage, StageContract, StageParams,
	allocate_output, commit_output,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.util.shapes import Proj, ProjOrder, SinoOrder, X
from lftomo.util.window import TomoWindow


def _contract(*, resamples_detector: bool = False, mutates_input: bool = True,
		required_pad: PadRequirement = PadRequirement()) -> StageContract:
	return StageContract(
		produces=PRESERVE,
		accepts=(ProjOrder, SinoOrder),
		prefers=ProjOrder,
		accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
		produces_domain=PRESERVE,
		resamples_detector=resamples_detector,
		mutates_input=mutates_input,
		window_axes=("Y", "Theta"),
		required_pad=required_pad,
	)


def _array(state: PipelineState, name: str) -> np.ndarray:
	if state.backend.name != "numpy":
		raise TypeError(f"{name} currently requires the NumPy backend")
	data = np.asarray(state.working_array)
	if data.ndim != 3:
		raise ValueError(f"{name} expects a (Y, Theta, X) sinogram array")
	return data


_INTERPOLATION = {
	"nearest": "INTER_NEAREST",
	"linear": "INTER_LINEAR",
	"cubic": "INTER_CUBIC",
	"lanczos": "INTER_LANCZOS4",
}

_BORDER_MODE = {
	"constant": "BORDER_CONSTANT",
	"edge": "BORDER_REPLICATE",
	"reflect": "BORDER_REFLECT_101",
	"symmetric": "BORDER_REFLECT",
}


def transform_affine_projection(
	stage: "AffineSkewMovementStage",
	image: np.ndarray,
	source_window: TomoWindow,
	local_projection: int,
	physical_shape,
	row_count: int,
	target: np.ndarray | None = None,
) -> np.ndarray:
	"""Return one fused affine projection for serial or process dispatch."""

	import cv2

	if stage.params.movement_enabled and (stage.params.movement_y != 0 or stage.params.movement_x != 0):
		fraction = source_window.source_index("Theta", local_projection) / physical_shape.Theta
		image = shift(
			image,
			(stage.params.movement_y * fraction, stage.params.movement_x * fraction),
			order=3, mode="constant", cval=0.0,
		)
	if target is None:
		width = stage.output_shape(ProjOrder(1, row_count, image.shape[1])).X
		target = np.empty((row_count, width), dtype=image.dtype)
	cv2.copyMakeBorder(
		image, 0, 0, stage.params.pad_x, stage.params.pad_x,
		getattr(cv2, _BORDER_MODE[stage.params.pad_mode]), dst=target, value=0,
	)
	if stage.params.skew_degrees != 0:
		width = target.shape[1]
		left_padding = (target.shape[1] - image.shape[1]) // 2
		center = (
			(image.shape[1] / 2 if stage.params.center_px is None else stage.params.center_px)
			+ left_padding,
			row_count / 2 if stage.params.center_y_px is None else stage.params.center_y_px,
		)
		matrix = cv2.getRotationMatrix2D(center, stage.params.skew_degrees, 1.0)
		try:
			interpolation = getattr(cv2, _INTERPOLATION[stage.params.interpolation])
		except KeyError as error:
			raise ValueError(
				f"unknown skew interpolation {stage.params.interpolation!r}"
			) from error
		# OpenCV does not support an affine transform whose source and destination
		# are the same array. Its result is the one unavoidable library temporary;
		# the canonical output allocation still owns the published buffer.
		target[...] = cv2.warpAffine(
			target, matrix, (width, row_count), flags=interpolation,
		)
	return target


class SkewCorrectStage(Stage):
	"""Rotate each projection image around the configured detector center."""

	name = "skew_correct.cv2_affine"
	contract = _contract()

	class Params(StageParams):
		degrees: float = 0.0
		center_px: float | None = None
		interpolation: Literal["nearest", "linear", "cubic", "lanczos"] = "cubic"
		enabled: bool = True
		center_y_px: float | None = None

		@model_validator(mode="before")
		@classmethod
		def normalize_degrees(cls, value):
			if isinstance(value, dict) and "skew_degrees" in value:
				value = dict(value)
				alias = value.pop("skew_degrees")
				if alias is not None:
					if value.get("degrees", 0) != 0:
						raise ValueError("provide degrees or skew_degrees, not both")
					value["degrees"] = alias
			return value

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		active = params.enabled and params.degrees != 0
		return replace(cls.contract, resamples_detector=active, mutates_input=active)

	def run(self, state: PipelineState) -> PipelineState:
		data = _array(state, self.name)
		layout = type(state.buffer_spec.shape)
		if not self.params.enabled or self.params.degrees == 0:
			return state
		import cv2

		height = data.shape[layout.axis_of("Y")]
		width = data.shape[layout.axis_of("X")]
		center = (
			width / 2 if self.params.center_px is None else self.params.center_px,
			height / 2 if self.params.center_y_px is None else self.params.center_y_px,
		)
		matrix = cv2.getRotationMatrix2D(center, self.params.degrees, 1.0)
		for projection in state.work_indices("Theta"):
			image = data[layout.point_index(Theta=projection)]
			image[:] = cv2.warpAffine(
				image,
				matrix,
				(width, height),
				flags=getattr(cv2, _INTERPOLATION[self.params.interpolation]),
			)
		return state


class MovementCorrectStage(Stage):
	"""Apply a linearly increasing detector shift across projection angle."""

	name = "movement_correct.linear_shift"
	contract = _contract()

	class Params(StageParams):
		y: float = 0.0
		x: float = 0.0
		enabled: bool = True
		order: int = Field(default=3, ge=0, le=5)
		mode: Literal[
			"reflect", "grid-mirror", "constant", "grid-constant", "nearest",
			"mirror", "grid-wrap", "wrap",
		] = "constant"
		cval: float = 0.0

		@model_validator(mode="before")
		@classmethod
		def normalize_movement(cls, value):
			if isinstance(value, dict):
				value = dict(value)
				for alias, canonical in (("movement_y", "y"), ("movement_x", "x")):
					if alias in value:
						amount = value.pop(alias)
						if amount is not None:
							value[canonical] = amount
			return value

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		active = params.enabled and (params.y != 0 or params.x != 0)
		return replace(
			cls.contract, resamples_detector=active, mutates_input=active,
			required_pad=PadRequirement(fixed=Proj(
				Y=int(np.ceil(abs(params.y))) if active else 0,
				X=int(np.ceil(abs(params.x))) if active else 0,
			)),
		)

	def run(self, state: PipelineState) -> PipelineState:
		data = _array(state, self.name)
		layout = type(state.buffer_spec.shape)
		if not self.params.enabled or (self.params.y == 0 and self.params.x == 0):
			return state
		if state.source_window is None or state.dataset is None:
			raise RuntimeError(f"{self.name} requires canonical source geometry")
		for projection in state.work_indices("Theta"):
			fraction = state.source_window.source_index("Theta", projection) / state.dataset.physical_shape.Theta
			image = data[layout.point_index(Theta=projection)]
			image[:] = shift(
				image,
				(self.params.y * fraction, self.params.x * fraction),
				order=self.params.order,
				mode=self.params.mode,
				cval=self.params.cval,
			)
		return state


class AffineSkewMovementStage(Stage):
	"""Legacy affine correction with an optional fused layout rewrite."""

	name = "geometry.affine_skew_movement"
	contract = _contract()

	class Params(StageParams):
		skew_degrees: float = 0.0
		movement_enabled: bool = False
		movement_y: float = 0.0
		movement_x: float = 0.0
		center_px: float | None = None
		interpolation: Literal["nearest", "linear", "cubic", "lanczos"] = "cubic"
		transition_to: Literal["projection", "sinogram"] | None = None
		pad_x: int = Field(default=0, ge=0)
		pad_mode: Literal["edge", "constant", "reflect", "symmetric"] = "edge"
		center_y_px: float | None = None

		@model_validator(mode="before")
		@classmethod
		def normalize_tilt(cls, value):
			if isinstance(value, dict) and "tilt_radians" in value:
				value = dict(value)
				tilt = value.pop("tilt_radians")
				if tilt is not None:
					if value.get("skew_degrees", 0) != 0:
						raise ValueError("provide skew_degrees or tilt_radians, not both")
					# recon_minimal's rotate_xy corrects the recorded detector tilt.
					value["skew_degrees"] = float(np.degrees(-TypeAdapter(float).validate_python(tilt)))
			return value

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		active = params.skew_degrees != 0 or (
			params.movement_enabled and (params.movement_y != 0 or params.movement_x != 0)
		)
		pad = PadRequirement(fixed=Proj(
			Y=int(np.ceil(abs(params.movement_y))) if params.movement_enabled else 0,
			X=int(np.ceil(abs(params.movement_x))) if params.movement_enabled else 0,
		))
		if params.transition_to is None:
			return replace(
				cls.contract, resamples_detector=active, mutates_input=active,
				required_pad=pad,
			)
		target = ProjOrder if params.transition_to == "projection" else SinoOrder
		source = SinoOrder if target is ProjOrder else ProjOrder
		return replace(
			cls.contract, accepts=(source,), prefers=source, produces=target,
			fused_transition=True, resamples_detector=active, mutates_input=False,
			window_axes=(), required_pad=pad,
		)

	@property
	def target_layout(self):
		return self.contract.produces if self.contract.fused_transition else None

	def output_window(self, source: TomoWindow) -> TomoWindow:
		"""Return the published output canvas for the fused affine transition."""

		return source.with_output_padding(X(self.params.pad_x))

	def output_shape(self, source_shape) -> ProjOrder | SinoOrder:
		"""Return the one canonical target-layout affine allocation shape."""

		expanded = source_shape + X(self.params.pad_x) + X(self.params.pad_x)
		return expanded.to(self.target_layout)

	def run(self, state: PipelineState) -> PipelineState:
		if self.target_layout is None:
			movement = MovementCorrectStage(
				enabled=self.params.movement_enabled,
				y=self.params.movement_y, x=self.params.movement_x,
			)
			skew = SkewCorrectStage(
				degrees=self.params.skew_degrees, center_px=self.params.center_px,
				center_y_px=self.params.center_y_px, interpolation=self.params.interpolation,
			)
			return skew.run(movement.run(state))
		if state.backend.name != "numpy":
			raise TypeError("geometry.affine_skew_movement currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("geometry.affine_skew_movement expects a three-dimensional array")
		input_layout = state.layout
		if input_layout not in (ProjOrder, SinoOrder):
			raise ValueError("fused affine transition requires a known tomography layout")
		if input_layout is self.target_layout:
			raise ValueError("fused affine transition must change layout")
		if state.source_window is None or state.dataset is None:
			raise RuntimeError(f"{self.name} requires canonical source geometry")
		source_shape = state.buffer_spec.shape
		row_count = source_shape.Y
		shape = self.output_shape(source_shape)
		output_buffer, output = allocate_output(
			state, shape, data.dtype, role="fused_geometry",
		)
		for projection in state.work_indices("Theta"):
			image = data[input_layout.point_index(Theta=projection)]
			transform_affine_projection(
				self, image, state.source_window, projection,
				state.dataset.physical_shape, row_count,
				target=output[self.target_layout.point_index(Theta=projection)],
			)
		result = commit_output(
			state, output_buffer, retire_as=None,
			buffer_spec=BufferSpec(shape, data.dtype),
		)
		if state.source_window is not None:
			result.source_window = self.output_window(state.source_window)
		result.artifacts["fused_layout_transition"] = {
			"from": input_layout.__name__, "to": self.target_layout.__name__,
			"pad_x": self.params.pad_x, "skew_degrees": self.params.skew_degrees,
		}
		return result
