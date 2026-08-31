"""Projection-domain geometry correction stages."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import shift

from lftomo.config import StageConfig
from lftomo.pipeline.core import (
	BufferSpec, Locality, PadRequirement, PipelineState, Stage, StageContract,
	allocate_output, commit_output,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.util.shapes import Proj, ProjOrder, SinoOrder, X
from lftomo.util.window import TomoWindow


def _contract(*, resamples_detector: bool = False, mutates_input: bool = True,
		required_pad: PadRequirement = PadRequirement()) -> StageContract:
	return StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		produces=PRESERVE,
		accepts=(ProjOrder, SinoOrder),
		prefers=ProjOrder,
		accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
		produces_domain=PRESERVE,
		locality=Locality.PROJECTION,
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
	"nearest": cv2.INTER_NEAREST,
	"linear": cv2.INTER_LINEAR,
	"cubic": cv2.INTER_CUBIC,
	"lanczos": cv2.INTER_LANCZOS4,
}


def transform_affine_projection(
	stage: "AffineSkewMovementStage",
	image: np.ndarray,
	projection: int,
	projection_count: int,
	row_count: int,
) -> np.ndarray:
	"""Return one fused affine projection for serial or process dispatch."""

	if stage.movement_enabled and (stage.movement_y != 0 or stage.movement_x != 0):
		fraction = projection / projection_count
		image = shift(
			image,
			(stage.movement_y * fraction, stage.movement_x * fraction),
			order=3, mode="constant", cval=0.0,
		)
	padded = np.pad(
		image, ((0, 0), (stage.pad_x, stage.pad_x)), mode=stage.pad_mode,
	)
	if stage.skew_degrees != 0:
		width = padded.shape[1]
		left_padding = (padded.shape[1] - image.shape[1]) // 2
		center = (
			(image.shape[1] / 2 if stage.center_px is None else stage.center_px)
			+ left_padding,
			row_count / 2 if stage.center_y_px is None else stage.center_y_px,
		)
		matrix = cv2.getRotationMatrix2D(center, stage.skew_degrees, 1.0)
		try:
			interpolation = _INTERPOLATION[stage.interpolation_name]
		except KeyError as error:
			raise ValueError(
				f"unknown skew interpolation {stage.interpolation_name!r}"
			) from error
		padded = cv2.warpAffine(
			padded, matrix, (width, row_count), flags=interpolation,
		)
	return padded


class SkewCorrectStage(Stage):
	"""Rotate each projection image around the configured detector center."""

	name = "skew_correct.cv2_affine"
	contract = _contract()

	def __init__(self, degrees: float = 0.0, center_px: float | None = None,
				interpolation: str = "cubic", enabled: bool = True,
				skew_degrees: float | None = None, center_y_px: float | None = None):
		if skew_degrees is not None:
			if degrees != 0:
				raise ValueError("provide degrees or skew_degrees, not both")
			degrees = skew_degrees
		try:
			self.interpolation = _INTERPOLATION[interpolation]
		except KeyError as error:
			raise ValueError(f"unknown skew interpolation {interpolation!r}") from error
		self.degrees = degrees
		self.center_px = center_px
		self.center_y_px = center_y_px
		self.enabled = enabled
		active = enabled and degrees != 0
		self.contract = _contract(
			resamples_detector=active, mutates_input=active,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "SkewCorrectStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		data = _array(state, self.name)
		layout = type(state.buffer_spec.shape)
		if not self.enabled or self.degrees == 0:
			return state
		height = data.shape[layout.axis_of("Y")]
		width = data.shape[layout.axis_of("X")]
		center = (
			width / 2 if self.center_px is None else self.center_px,
			height / 2 if self.center_y_px is None else self.center_y_px,
		)
		matrix = cv2.getRotationMatrix2D(center, self.degrees, 1.0)
		count = data.shape[layout.axis_of("Theta")]
		for projection in range(count):
			image = data[layout.point_index(Theta=projection)]
			image[:] = cv2.warpAffine(
				image,
				matrix,
				(width, height),
				flags=self.interpolation,
			)
		return state


class MovementCorrectStage(Stage):
	"""Apply a linearly increasing detector shift across projection angle."""

	name = "movement_correct.linear_shift"
	contract = _contract()

	def __init__(self, y: float = 0.0, x: float = 0.0, enabled: bool = True,
				order: int = 3, mode: str = "constant", cval: float = 0.0,
				movement_y: float | None = None, movement_x: float | None = None):
		if movement_y is not None:
			y = movement_y
		if movement_x is not None:
			x = movement_x
		if not 0 <= order <= 5:
			raise ValueError("movement interpolation order must be between 0 and 5")
		self.y = y
		self.x = x
		self.enabled = enabled
		self.order = order
		self.mode = mode
		self.cval = cval
		active = enabled and (y != 0 or x != 0)
		self.contract = _contract(
			resamples_detector=active, mutates_input=active,
			required_pad=PadRequirement(fixed=Proj(
				Y=int(np.ceil(abs(y))) if active else 0,
				X=int(np.ceil(abs(x))) if active else 0,
			)),
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "MovementCorrectStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		data = _array(state, self.name)
		layout = type(state.buffer_spec.shape)
		if not self.enabled or (self.y == 0 and self.x == 0):
			return state
		count = data.shape[layout.axis_of("Theta")]
		projection_start = int(state.artifacts.get("projection_start", 0))
		projection_count = int(state.artifacts.get("projection_count", count))
		for projection in range(count):
			fraction = (projection_start + projection) / projection_count
			image = data[layout.point_index(Theta=projection)]
			image[:] = shift(
				image,
				(self.y * fraction, self.x * fraction),
				order=self.order,
				mode=self.mode,
				cval=self.cval,
			)
		return state


class AffineSkewMovementStage(Stage):
	"""Legacy affine correction with an optional fused layout rewrite."""

	name = "geometry.affine_skew_movement"
	contract = _contract()

	def __init__(self, skew_degrees: float = 0.0, movement_enabled: bool = False,
				movement_y: float = 0.0, movement_x: float = 0.0,
				center_px: float | None = None, interpolation: str = "cubic",
				transition_to: str | None = None, pad_x: int = 0,
				pad_mode: str = "edge", center_y_px: float | None = None,
				tilt_radians: float | None = None):
		if tilt_radians is not None:
			if skew_degrees != 0:
				raise ValueError("provide skew_degrees or tilt_radians, not both")
			# recon_minimal's rotate_xy corrects the recorded detector tilt.
			skew_degrees = float(np.degrees(-tilt_radians))
		if pad_x < 0:
			raise ValueError("fused affine pad_x cannot be negative")
		if pad_mode not in {"edge", "constant", "reflect", "symmetric"}:
			raise ValueError("unsupported fused affine pad_mode")
		try:
			target = {None: None, "projection": ProjOrder, "sinogram": SinoOrder}[
				transition_to
			]
		except KeyError as error:
			raise ValueError("transition_to must be 'projection' or 'sinogram'")
		self.skew_degrees = skew_degrees
		self.movement_enabled = movement_enabled
		self.movement_y = movement_y
		self.movement_x = movement_x
		self.center_px = center_px
		self.center_y_px = center_y_px
		self.interpolation_name = interpolation
		self.transition_to = transition_to
		self.target_layout = target
		self.pad_x = pad_x
		self.pad_mode = pad_mode
		self.movement = MovementCorrectStage(
			enabled=movement_enabled,
			movement_y=movement_y,
			movement_x=movement_x,
		)
		self.skew = SkewCorrectStage(
			skew_degrees=skew_degrees,
			center_px=center_px,
			center_y_px=center_y_px,
			interpolation=interpolation,
		)
		active_resample = (
			skew_degrees != 0
			or (movement_enabled and (movement_y != 0 or movement_x != 0))
		)
		if target is None:
			self.contract = _contract(
				resamples_detector=active_resample,
				mutates_input=active_resample,
				required_pad=PadRequirement(fixed=Proj(
					Y=int(np.ceil(abs(movement_y))) if movement_enabled else 0,
					X=int(np.ceil(abs(movement_x))) if movement_enabled else 0,
				)),
			)
		else:
			source = SinoOrder if target is ProjOrder else ProjOrder
			self.contract = StageContract(
				expected_input=source, produced_output=target,
				accepts=(source,), prefers=source, produces=target,
				accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
				produces_domain=PRESERVE, locality=Locality.VOLUME,
				fused_transition=True, resamples_detector=active_resample,
				mutates_input=False,
				required_pad=PadRequirement(fixed=Proj(
					Y=int(np.ceil(abs(movement_y))) if movement_enabled else 0,
					X=int(np.ceil(abs(movement_x))) if movement_enabled else 0,
				)),
			)

	@classmethod
	def from_config(cls, config: StageConfig) -> "AffineSkewMovementStage":
		return cls(**config.params)

	def output_window(self, source: TomoWindow) -> TomoWindow:
		"""Return the published output canvas for the fused affine transition."""

		return source.with_output_padding(X(self.pad_x))

	def output_shape(self, source_shape) -> ProjOrder | SinoOrder:
		"""Return the one canonical target-layout affine allocation shape."""

		expanded = source_shape + X(self.pad_x) + X(self.pad_x)
		return expanded.to(self.target_layout)

	def run(self, state: PipelineState) -> PipelineState:
		if self.target_layout is None:
			return self.skew.run(self.movement.run(state))
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
		source_shape = state.buffer_spec.shape
		projection_count = source_shape.Theta
		row_count = source_shape.Y
		shape = self.output_shape(source_shape)
		output_buffer, output = allocate_output(
			state, shape, data.dtype, role="fused_geometry",
		)
		for projection in range(projection_count):
			image = data[input_layout.point_index(Theta=projection)]
			output[self.target_layout.point_index(Theta=projection)] = (
				transform_affine_projection(
					self, image, projection, projection_count, row_count,
				)
			)
		result = commit_output(
			state, output_buffer, retire_as=None,
			buffer_spec=BufferSpec(shape, data.dtype),
		)
		if state.source_window is not None:
			result.source_window = self.output_window(state.source_window)
		result.artifacts["fused_layout_transition"] = {
			"from": input_layout.__name__, "to": self.target_layout.__name__,
			"pad_x": self.pad_x, "skew_degrees": self.skew_degrees,
		}
		return result
