"""Helical-acquisition correction stages."""

from __future__ import annotations

import numpy as np

from lftomo.config import StageConfig
from lftomo.pipeline.core import (
	BufferSpec, Locality, PipelineState, Stage, StageContract,
	allocate_output, commit_output,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.util.shapes import ProjOrder, SinoOrder


class HelicalIndexRemapStage(Stage):
	"""Remap detector rows along the helical trajectory for each projection.

	The rounded linear row selection is adapted from
	``dan_scripts/berkeley_2021/allinwonderful_v1.5_A1_Helical.py`` at
	``8d324b270ab6c569acde246a01ea4572924cbab0``. The Stage generalizes that
	producer's one-row-at-a-time reader to an already-loaded sinogram window.
	"""

	name = "helical_correct.index_remap"
	contract = StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		window_axes=("Y", "Theta"),
	)

	def __init__(self, pitch_px_per_projection: float | None = None,
				enabled: bool = True, going_up: bool = False,
				projection_start: int = 0, crop_valid: bool = True,
				fill_value: float = 0.0, vertical_velocity_px: float | None = None):
		if pitch_px_per_projection is not None and vertical_velocity_px is not None:
			raise ValueError("provide pitch_px_per_projection or vertical_velocity_px, not both")
		if vertical_velocity_px is not None:
			pitch_px_per_projection = vertical_velocity_px
		if projection_start < 0:
			raise ValueError("helical projection_start cannot be negative")
		self.contract = StageContract(
			expected_input=SinoOrder, produced_output=SinoOrder,
			window_axes=("Y", "Theta"),
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder), prefers=ProjOrder,
			accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
			produces_domain=PRESERVE,
			locality=Locality.PROJECTION,
			requires_row_identity=enabled,
			requires_full_axes=("Y",) if enabled else (),
		)
		self.pitch_px_per_projection = pitch_px_per_projection
		self.enabled = enabled
		self.going_up = going_up
		self.projection_start = projection_start
		self.crop_valid = crop_valid
		self.fill_value = fill_value

	@classmethod
	def from_config(cls, config: StageConfig) -> "HelicalIndexRemapStage":
		return cls(**config.params)

	def _pitch(self, state: PipelineState) -> float:
		if self.pitch_px_per_projection is not None:
			return self.pitch_px_per_projection
		pitch_um = state.scan.helical.pitch_um_per_projection
		if pitch_um is None:
			raise ValueError(
				"helical index remap requires pitch_px_per_projection or "
				"Scan.helical.pitch_um_per_projection"
			)
		return pitch_um / state.scan.pixel_size_um

	def output_geometry(
		self, source_shape: ProjOrder | SinoOrder, pitch: float,
	) -> tuple[ProjOrder | SinoOrder, np.ndarray, int, int]:
		"""Return the canonical output shape, shifts, and valid source-row bounds."""

		direction = -1 if self.going_up else 1
		indices = self.projection_start + np.arange(source_shape.Theta)
		shifts = np.rint(direction * pitch * indices).astype(int)
		if not self.crop_valid:
			return source_shape, shifts, 0, source_shape.Y
		start = max(0, -int(shifts.min()))
		stop = min(source_shape.Y, source_shape.Y - int(shifts.max()))
		if start >= stop:
			raise ValueError("helical pitch leaves no detector rows valid across all projections")
		shape = type(source_shape)(**{**source_shape.as_dict(), "Y": stop - start})
		return shape, shifts, start, stop

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("HelicalIndexRemapStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		layout = type(state.buffer_spec.shape)
		if data.ndim != 3:
			raise ValueError("helical_correct.index_remap expects a (Y, Theta, X) sinogram array")
		if not self.enabled:
			return state
		source_shape = state.buffer_spec.shape
		count = source_shape.Theta
		y_size = source_shape.Y
		shape, shifts, start, stop = self.output_geometry(source_shape, self._pitch(state))

		if self.crop_valid:
			output_rows = np.arange(start, stop)
			output, remapped = allocate_output(
				state, shape, data.dtype, role="helical_remap",
			)
			for projection, shift_px in enumerate(shifts):
				remapped[layout.point_index(Theta=projection)] = (
					data[layout.point_index(Theta=projection)][output_rows + shift_px]
				)
		else:
			output, remapped = allocate_output(
				state, source_shape, data.dtype, role="helical_remap",
			)
			remapped.fill(self.fill_value)
			rows = np.arange(y_size)
			for projection, shift_px in enumerate(shifts):
				source = rows + shift_px
				valid = (source >= 0) & (source < y_size)
				remapped[layout.point_index(Theta=projection)][valid] = (
					data[layout.point_index(Theta=projection)][source[valid]]
				)
		result = commit_output(
			state, output, retire_as=None, buffer_spec=BufferSpec(shape, data.dtype),
		)
		result.artifacts["helical_row_shifts"] = shifts
		return result
