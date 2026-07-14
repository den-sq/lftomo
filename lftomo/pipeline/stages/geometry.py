"""Projection-domain geometry correction stages."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import shift

from lftomo.config import StageConfig
from lftomo.pipeline.core import PipelineState, ResourceHints, Stage, StageContract
from lftomo.util.shapes import SinoOrder


def _contract() -> StageContract:
	return StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		mutates_input=True,
		window_axes=("Y", "Theta"),
		resources=ResourceHints(memory_multiplier=2.0),
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


class SkewCorrectStage(Stage):
	"""Rotate each projection image around the configured detector center."""

	name = "skew_correct.cv2_affine"
	contract = _contract()

	def __init__(self, degrees: float = 0.0, center_px: float | None = None,
				interpolation: str = "cubic", enabled: bool = True,
				skew_degrees: float | None = None):
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
		self.enabled = enabled

	@classmethod
	def from_config(cls, config: StageConfig) -> "SkewCorrectStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		data = _array(state, self.name)
		if not self.enabled or self.degrees == 0:
			return state
		height, _, width = data.shape
		center = (width / 2 if self.center_px is None else self.center_px, height / 2)
		matrix = cv2.getRotationMatrix2D(center, self.degrees, 1.0)
		for projection in range(data.shape[1]):
			data[:, projection, :] = cv2.warpAffine(
				data[:, projection, :],
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

	@classmethod
	def from_config(cls, config: StageConfig) -> "MovementCorrectStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		data = _array(state, self.name)
		if not self.enabled or (self.y == 0 and self.x == 0):
			return state
		count = data.shape[1]
		for projection in range(count):
			fraction = projection / count
			data[:, projection, :] = shift(
				data[:, projection, :],
				(self.y * fraction, self.x * fraction),
				order=self.order,
				mode=self.mode,
				cval=self.cval,
			)
		return state


class AffineSkewMovementStage(Stage):
	"""Legacy R2.54 compatibility Stage: movement followed by skew."""

	name = "geometry.affine_skew_movement"
	contract = _contract()

	def __init__(self, skew_degrees: float = 0.0, movement_enabled: bool = False,
				movement_y: float = 0.0, movement_x: float = 0.0,
				center_px: float | None = None, interpolation: str = "cubic"):
		self.movement = MovementCorrectStage(
			enabled=movement_enabled,
			movement_y=movement_y,
			movement_x=movement_x,
		)
		self.skew = SkewCorrectStage(
			skew_degrees=skew_degrees,
			center_px=center_px,
			interpolation=interpolation,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "AffineSkewMovementStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		return self.skew.run(self.movement.run(state))
