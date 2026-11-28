"""Helical-acquisition correction stages."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pydantic import model_validator

from lftomo.pipeline.core import (
	PipelineState, Stage, StageContract, StageParams,
	allocate_output, commit_output,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.util.shapes import ProjOrder, SinoOrder, Y
from lftomo.util.window import TomoWindow


class HelicalIndexRemapStage(Stage):
	"""Remap detector rows along the helical trajectory for each projection.

	The rounded linear row selection is adapted from
	``dan_scripts/berkeley_2021/allinwonderful_v1.5_A1_Helical.py`` at
	``8d324b270ab6c569acde246a01ea4572924cbab0``. The Stage generalizes that
	producer's one-row-at-a-time reader to an already-loaded sinogram window.
	"""

	name = "helical_correct.index_remap"
	contract = StageContract(
		window_axes=("Y", "Theta"),
		produces=PRESERVE,
		accepts=(ProjOrder, SinoOrder), prefers=ProjOrder,
		accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
		produces_domain=PRESERVE,
		requires_row_identity=True,
		requires_full_axes=("Y",),
	)

	class Params(StageParams):
		pitch_px_per_projection: float | None = None
		enabled: bool = True
		going_up: bool = False
		crop_valid: bool = True
		fill_value: float = 0.0

		@model_validator(mode="before")
		@classmethod
		def normalize_pitch_alias(cls, values):
			values = dict(values)
			velocity = values.pop("vertical_velocity_px", None)
			if values.get("pitch_px_per_projection") is not None and velocity is not None:
				raise ValueError("provide pitch_px_per_projection or vertical_velocity_px, not both")
			if velocity is not None:
				values["pitch_px_per_projection"] = velocity
			return values

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		return replace(
			cls.contract, requires_row_identity=params.enabled,
			requires_full_axes=("Y",) if params.enabled else (),
		)

	def _pitch(self, scan) -> float:
		if self.params.pitch_px_per_projection is not None:
			return self.params.pitch_px_per_projection
		pitch_um = None if scan is None else scan.helical.pitch_um_per_projection
		if pitch_um is None:
			raise ValueError(
				"helical index remap requires pitch_px_per_projection or "
				"Scan.helical.pitch_um_per_projection"
			)
		return pitch_um / scan.pixel_size_um

	def output_window(self, source: TomoWindow, scan=None) -> TomoWindow:
		if not self.params.enabled:
			return source
		window, _shifts, _start, _stop = self.output_geometry(
			source.with_pad().frange.length, self._pitch(scan), source,
		)
		return window

	def output_geometry(
		self, source_shape: ProjOrder | SinoOrder, pitch: float, source_window: TomoWindow,
	) -> tuple[TomoWindow, np.ndarray, int, int]:
		"""Return the output canvas, shifts, and valid numerical source bounds."""

		direction = -1 if self.params.going_up else 1
		start = int(source_window.pos.Theta)
		covered = int(source_window.shape.Theta)
		step = int(source_window.step.Theta)
		indices = np.arange(start, start + covered, step, dtype=np.int64)
		if len(indices) != source_shape.Theta:
			raise ValueError("helical source window does not match the current Theta extent")
		shifts = np.rint(direction * pitch * indices).astype(int)
		if not self.params.crop_valid:
			return source_window, shifts, 0, source_shape.Y
		start = max(0, -int(shifts.min()))
		stop = min(source_shape.Y, source_shape.Y - int(shifts.max()))
		if start >= stop:
			raise ValueError("helical pitch leaves no detector rows valid across all projections")
		# Existing output rows retain the input canvas's numbering even when
		# the numerical source interval begins later. Shorten its tail, consuming
		# trailing halo before the published core, without shifting that origin.
		crop = Y((source_shape.Y - (stop - start)) * source_window.step.Y)
		canvas = source_window.with_pad().with_crop(Y(0), crop)
		after = type(source_window.pad_after).param_max(
			source_window.pad_after.make_with(0), source_window.pad_after - crop,
		)
		window = canvas.rescope(pad=source_window.pad_before, pad_after=after)
		return window, shifts, start, stop

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("HelicalIndexRemapStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		layout = state.layout
		if data.ndim != 3:
			raise ValueError("helical_correct.index_remap expects a (Y, Theta, X) sinogram array")
		if not self.params.enabled:
			return state
		source_shape = state.working_array.shape
		y_size = source_shape.Y
		if state.source_window is None:
			raise RuntimeError("helical remap requires PipelineState.source_window")
		window, shifts, start, stop = self.output_geometry(
			source_shape, self._pitch(state.scan), state.source_window,
		)
		shape = window.with_pad().frange.length.to(layout)
		produced = window if state.stage_window is None else self.output_window(state.stage_window, state.scan)
		selection = window.with_pad().packed_index(produced, layout)
		output_rows = np.arange(start, stop)[selection[layout.axis_of("Y")]]
		output, remapped = allocate_output(
			state, shape, data.dtype, role="helical_remap",
		)

		for projection in state.work_indices("Theta"):
			target_index = list(selection)
			target_index[layout.axis_of("Theta")] = projection
			target = remapped[tuple(target_index)]
			image = data[state.projection_index(projection, context=True)]
			source = output_rows + shifts[projection]
			if self.params.crop_valid:
				target[...] = image[source]
			else:
				target.fill(self.params.fill_value)
				valid = (source >= 0) & (source < y_size)
				target[valid] = image[source[valid]]
		result = commit_output(
			state, output, retire_as=None,
		)
		result.source_window = window
		result.stage_window = produced
		result.artifacts["helical_row_shifts"] = shifts
		return result
