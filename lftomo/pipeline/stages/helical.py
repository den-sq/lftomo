"""Helical-acquisition correction stages."""

from __future__ import annotations

import numpy as np

from lftomo.config import StageConfig
from lftomo.pipeline.core import Locality, PipelineState, ResourceHints, Stage, StageContract
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
		resources=ResourceHints(memory_multiplier=2.0),
	)

	def __init__(self, pitch_px_per_projection: float | None = None,
				enabled: bool = True, going_up: bool = False,
				projection_start: int = 0, crop_valid: bool = True,
				fill_value: float = 0.0, vertical_velocity_px: float | None = None,
				order: str = "sinogram"):
		if pitch_px_per_projection is not None and vertical_velocity_px is not None:
			raise ValueError("provide pitch_px_per_projection or vertical_velocity_px, not both")
		if vertical_velocity_px is not None:
			pitch_px_per_projection = vertical_velocity_px
		if projection_start < 0:
			raise ValueError("helical projection_start cannot be negative")
		if order not in ("projection", "sinogram"):
			raise ValueError("helical order must be 'projection' or 'sinogram'")
		shape = ProjOrder if order == "projection" else SinoOrder
		self.contract = StageContract(
			expected_input=shape, produced_output=shape, window_axes=("Y", "Theta"),
			produces=PRESERVE,
			accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
			produces_domain=PRESERVE,
			locality=Locality.PROJECTION,
			requires_row_identity=enabled,
			resources=ResourceHints(memory_multiplier=2.0),
		)
		self.pitch_px_per_projection = pitch_px_per_projection
		self.enabled = enabled
		self.going_up = going_up
		self.projection_start = projection_start
		self.crop_valid = crop_valid
		self.fill_value = fill_value
		self.order = order

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

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("HelicalIndexRemapStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("helical_correct.index_remap expects a (Y, Theta, X) sinogram array")
		if not self.enabled:
			return state
		pitch = self._pitch(state)
		direction = -1 if self.going_up else 1
		count = data.shape[0] if self.order == "projection" else data.shape[1]
		y_size = data.shape[1] if self.order == "projection" else data.shape[0]
		projection_indices = self.projection_start + np.arange(count)
		shifts = np.rint(direction * pitch * projection_indices).astype(int)

		if self.crop_valid:
			start = max(0, -int(shifts.min()))
			stop = min(y_size, y_size - int(shifts.max()))
			if start >= stop:
				raise ValueError("helical pitch leaves no detector rows valid across all projections")
			output_rows = np.arange(start, stop)
			shape = (count, len(output_rows), data.shape[2]) if self.order == "projection" else (len(output_rows), count, data.shape[2])
			remapped = np.empty(shape, dtype=data.dtype)
			for projection, shift_px in enumerate(shifts):
				if self.order == "projection":
					remapped[projection, :, :] = data[projection, output_rows + shift_px, :]
				else:
					remapped[:, projection, :] = data[output_rows + shift_px, projection, :]
		else:
			remapped = np.full_like(data, self.fill_value)
			rows = np.arange(y_size)
			for projection, shift_px in enumerate(shifts):
				source = rows + shift_px
				valid = (source >= 0) & (source < y_size)
				if self.order == "projection":
					remapped[projection, valid, :] = data[projection, source[valid], :]
				else:
					remapped[valid, projection, :] = data[source[valid], projection, :]
		result = state.with_array(remapped)
		result.artifacts["helical_row_shifts"] = shifts
		return result
