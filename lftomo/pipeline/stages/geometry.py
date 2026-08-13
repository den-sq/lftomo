"""Projection-domain geometry correction stages."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import shift

from lftomo.config import StageConfig
from lftomo.pipeline.core import (
	Locality, PadRequirement, PipelineState, Stage, StageContract,
	allocate_output, commit_output,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.pipeline.layout import (
	configured_layout,
	n_projections,
	n_rows,
	projection_view,
	resolve_layout,
)
from lftomo.util.shapes import DataShape, Proj, ProjOrder, SinoOrder, X
from lftomo.util.window import TomoWindow


def _contract(order: str | None = None, *, resamples_detector: bool = False,
		mutates_input: bool = True,
		required_pad: PadRequirement = PadRequirement()) -> StageContract:
	compatibility = configured_layout(order)
	shape = compatibility or SinoOrder
	return StageContract(
		expected_input=shape,
		produced_output=shape,
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
	pad_before_x: int,
	pad_after_x: int,
	output_width: int,
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
		image, ((0, 0), (pad_before_x, pad_after_x)), mode=stage.pad_mode,
	)
	if stage.skew_degrees != 0:
		center = (
			(image.shape[1] / 2 if stage.center_px is None else stage.center_px)
			+ pad_before_x,
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
			padded, matrix, (output_width, row_count), flags=interpolation,
		)
	return padded


class SkewCorrectStage(Stage):
	"""Rotate each projection image around the configured detector center."""

	name = "skew_correct.cv2_affine"
	contract = _contract()

	def __init__(self, degrees: float = 0.0, center_px: float | None = None,
				interpolation: str = "cubic", enabled: bool = True,
				skew_degrees: float | None = None, center_y_px: float | None = None,
				order: str | None = None):
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
		self.order = order
		active = enabled and degrees != 0
		self.contract = _contract(
			order, resamples_detector=active, mutates_input=active,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "SkewCorrectStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		data = _array(state, self.name)
		layout = resolve_layout(state, self.order, self.name)
		if not self.enabled or self.degrees == 0:
			return state
		height = n_rows(data, layout)
		width = data.shape[2]
		center = (
			width / 2 if self.center_px is None else self.center_px,
			height / 2 if self.center_y_px is None else self.center_y_px,
		)
		matrix = cv2.getRotationMatrix2D(center, self.degrees, 1.0)
		count = n_projections(data, layout)
		for projection in range(count):
			image = projection_view(data, layout, projection)
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
				movement_y: float | None = None, movement_x: float | None = None,
				array_order: str | None = None):
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
		self.array_order = array_order
		active = enabled and (y != 0 or x != 0)
		self.contract = _contract(
			array_order, resamples_detector=active, mutates_input=active,
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
		layout = resolve_layout(state, self.array_order, self.name)
		if not self.enabled or (self.y == 0 and self.x == 0):
			return state
		count = n_projections(data, layout)
		projection_start = int(state.artifacts.get("projection_start", 0))
		projection_count = int(state.artifacts.get("projection_count", count))
		for projection in range(count):
			fraction = (projection_start + projection) / projection_count
			image = projection_view(data, layout, projection)
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
		target = configured_layout(transition_to)
		if transition_to is not None and target is None:
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

	def input_window(
		self,
		input_shape: DataShape,
		input_layout: type[DataShape],
		dtype,
		*,
		source_window: TomoWindow | None = None,
	) -> TomoWindow:
		"""Bind an affine working allocation to canonical source geometry."""

		if input_layout not in (ProjOrder, SinoOrder):
			raise ValueError("fused affine transition requires a known tomography layout")
		if source_window is None:
			canonical_shape = input_shape.to(ProjOrder)
			zero = canonical_shape.make_with(0)
			input_window = TomoWindow(
				canonical_shape,
				zero,
				zero,
				canonical_shape.make_with(1),
				dtype,
				dtype,
			)
		else:
			input_window = source_window
			expected = input_window.allocation_shape.to(input_layout)
			if expected != input_shape:
				raise ValueError(
					"source_window allocation does not match the affine input buffer"
				)
		return input_window

	def output_window(
		self, input_window: TomoWindow,
	) -> TomoWindow:
		"""Derive fused output allocation and coordinates through TomoWindow."""

		return input_window.expanded(X(self.pad_x))

	@staticmethod
	def output_padding(
		input_window: TomoWindow, output_window: TomoWindow,
	) -> tuple[int, int]:
		"""Return numerical X fill derived from canonical source extents."""

		before = int(input_window.source_pos.X - output_window.source_pos.X)
		input_stop = input_window.source_pos.X + input_window.source_shape.X
		output_stop = output_window.source_pos.X + output_window.source_shape.X
		return before, int(output_stop - input_stop)

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
		projection_count = n_projections(data, input_layout)
		row_count = n_rows(data, input_layout)
		input_window = self.input_window(
			input_layout(*data.shape), input_layout, data.dtype,
			source_window=state.source_window,
		)
		output_window = self.output_window(input_window)
		pad_before_x, pad_after_x = self.output_padding(input_window, output_window)
		shape = output_window.allocation_shape.to(self.target_layout)
		output_buffer, output = allocate_output(
			state, shape, data.dtype, role="fused_geometry",
		)
		for projection in range(projection_count):
			image = projection_view(data, input_layout, projection)
			projection_view(output, self.target_layout, projection)[:] = (
				transform_affine_projection(
					self, image, projection, projection_count, row_count,
					pad_before_x, pad_after_x, shape.X,
				)
			)
		result = commit_output(
			state, output_buffer, retire_as=None, layout=self.target_layout,
		)
		result.source_window = output_window
		result.artifacts["fused_layout_transition"] = {
			"from": input_layout.__name__, "to": self.target_layout.__name__,
			"pad_x": self.pad_x, "skew_degrees": self.skew_degrees,
		}
		return result
