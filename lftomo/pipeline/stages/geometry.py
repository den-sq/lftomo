"""Projection-domain geometry correction stages."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import Field, TypeAdapter, model_validator

import numpy as np
from scipy.ndimage import shift

from lftomo.pipeline.core import (
	BoundedWorkRequest, BufferSpec, PadRequirement, PipelineState, ProjectionStage,
	StageContract, StageParams,
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
_AFFINE_CANVAS = "_affine_canvas"


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
	if stage.target_layout is not None:
		if target is None:
			raise RuntimeError("fused affine correction requires its bounded canvas")
		cv2.copyMakeBorder(
			image, 0, 0, stage.params.pad_x, stage.params.pad_x,
			getattr(cv2, _BORDER_MODE[stage.params.pad_mode]), dst=target, value=0,
		)
	else:
		target = image
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
		# OpenCV cannot rotate in place. Its returned numerical temporary covers
		# the established padded input; only the produced view is published.
		return cv2.warpAffine(
			target, matrix, (width, row_count), flags=interpolation,
		)
	return target


class SkewCorrectStage(ProjectionStage):
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
		return replace(
			cls.contract, resamples_detector=active, mutates_input=active,
			extent_sensitive_axes=("Y", "X") if active else (),
		)

	def run_projection(self, image: np.ndarray, state: PipelineState) -> np.ndarray | None:
		if not self.params.enabled or self.params.degrees == 0:
			return None
		import cv2

		height, width = image.shape
		center = (
			width / 2 if self.params.center_px is None else self.params.center_px,
			height / 2 if self.params.center_y_px is None else self.params.center_y_px,
		)
		matrix = cv2.getRotationMatrix2D(center, self.params.degrees, 1.0)
		return cv2.warpAffine(
			image, matrix, (width, height),
			flags=getattr(cv2, _INTERPOLATION[self.params.interpolation]),
		)


class MovementCorrectStage(ProjectionStage):
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
			extent_sensitive_axes=("Y", "X") if active else (),
			required_pad=PadRequirement(fixed=Proj(
				Y=int(np.ceil(abs(params.y))) if active else 0,
				X=int(np.ceil(abs(params.x))) if active else 0,
			)),
		)

	def run_projection(self, image: np.ndarray, state: PipelineState) -> np.ndarray | None:
		if not self.params.enabled or (self.params.y == 0 and self.params.x == 0):
			return None
		if state.source_window is None or state.dataset is None:
			raise RuntimeError(f"{self.name} requires canonical source geometry")
		projection = next(state.work_indices("Theta"))
		fraction = state.source_window.source_index("Theta", projection) / state.dataset.physical_shape.Theta
		return shift(
			image, (self.params.y * fraction, self.params.x * fraction),
			order=self.params.order, mode=self.params.mode, cval=self.params.cval,
		)


class AffineSkewMovementStage(ProjectionStage):
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
				required_pad=pad, extent_sensitive_axes=("Y", "X") if active else (),
			)
		target = ProjOrder if params.transition_to == "projection" else SinoOrder
		source = SinoOrder if target is ProjOrder else ProjOrder
		return replace(
			cls.contract, accepts=(source,), prefers=source, produces=target,
			fused_transition=True, resamples_detector=active, mutates_input=False,
			window_axes=(), required_pad=pad,
			extent_sensitive_axes=("Y", "X") if active else (),
		)

	@property
	def target_layout(self):
		return self.contract.produces if self.contract.fused_transition else None

	def output_window(self, source: TomoWindow, scan=None) -> TomoWindow:
		"""Return the published output canvas for the fused affine transition."""

		return source if self.target_layout is None else source.with_output_padding(X(self.params.pad_x))

	def output_shape(self, source_shape) -> ProjOrder | SinoOrder:
		"""Return the one canonical target-layout affine allocation shape."""

		window = TomoWindow.from_bounds(source_shape)
		return self.output_window(window).frange.length.to(self.target_layout)

	def bounded_work_requests(self, input_spec: BufferSpec,
			window: TomoWindow | None = None) -> tuple[BoundedWorkRequest, ...]:
		if self.target_layout is None:
			return ()
		if window is None:
			zero = input_spec.shape.make_with(0)
			window = TomoWindow(
				input_spec.shape, zero, zero, input_spec.shape.make_with(1), input_spec.dtype,
			)
		canvas = self.output_window(window.with_pad())
		return (BoundedWorkRequest(
			_AFFINE_CANVAS, BufferSpec(canvas.frange.length.to(ProjOrder), input_spec.dtype),
			"affine_canvas", required_for_fallback=True,
		),)

	def run_projection(self, image: np.ndarray, state: PipelineState) -> np.ndarray | None:
		if self.params.skew_degrees == 0 and (
			not self.params.movement_enabled
			or (self.params.movement_y == 0 and self.params.movement_x == 0)
		):
			return None
		moving = self.params.movement_enabled and (
			self.params.movement_y != 0 or self.params.movement_x != 0
		)
		if moving and (state.source_window is None or state.dataset is None):
			raise RuntimeError(f"{self.name} requires canonical source geometry")
		height, _width = image.shape
		return transform_affine_projection(
			self, image, state.source_window, next(state.work_indices("Theta")),
			state.working_array.shape if state.dataset is None else state.dataset.physical_shape,
			height,
		)

	def fill_output(self, state: PipelineState, output_buffer) -> None:
		"""Calculate on the canonical padded canvas and publish its produced view."""
		from lftomo.pipeline.execution import _work_states

		read_output = self.output_window(state.read_window)
		produced_output = self.output_window(state.produced_window)
		crop = read_output.packed_index(produced_output, Proj)
		destination = self.output_window(state.source_window).with_pad().packed_index(
			produced_output, self.target_layout,
		)
		output = output_buffer.array()
		for image, projection_state in _work_states(
			state, "Theta", work_keys=(_AFFINE_CANVAS,), context=True,
		):
			projection = next(projection_state.work_indices("Theta"))
			canvas = projection_state.artifacts[_AFFINE_CANVAS]
			transformed = transform_affine_projection(
				self, image, state.source_window, projection,
				state.dataset.physical_shape, state.read_window.frange.length.Y,
				target=canvas.array()[0],
			)
			index = list(destination)
			index[self.target_layout.axis_of("Theta")] = projection
			output[tuple(index)] = transformed[crop]

	def run(self, state: PipelineState) -> PipelineState:
		if self.target_layout is None:
			return super().run(state)
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
		source_shape = state.working_array.shape
		shape = self.output_shape(source_shape)
		from lftomo.pipeline.execution import _prepare_bounded_work, _release_bounded_work

		requests = _prepare_bounded_work(state, (self,), "Theta")
		try:
			output_buffer, _output = allocate_output(
				state, shape, data.dtype, role="fused_geometry",
			)
			self.fill_output(state, output_buffer)
		finally:
			_release_bounded_work(state, requests)
		result = commit_output(
			state, output_buffer, retire_as=None,
		)
		if state.source_window is not None:
			result.source_window = self.output_window(state.source_window)
		if state.stage_window is not None:
			result.stage_window = self.output_window(state.stage_window)
		result.artifacts["fused_layout_transition"] = {
			"from": input_layout.__name__, "to": self.target_layout.__name__,
			"pad_x": self.params.pad_x, "skew_degrees": self.params.skew_degrees,
		}
		return result
