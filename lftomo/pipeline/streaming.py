"""Runner-level window planning and retained/fresh transition contracts."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

import numpy as np

from lftomo.pipeline.core import AxisPad, PipelineState, Stage
from lftomo.storage import ManagedBuffer
from lftomo.util.shapes import ProjOrder, Y
from lftomo.util.window import CTFov, TomoDataSet, TomoWindow


LOGGER = logging.getLogger(__name__)


class WindowPlanError(ValueError):
	"""A pipeline cannot satisfy its contracts under the selected window plan."""


@dataclass(frozen=True)
class WindowConfig:
	"""Machine/runner window selection, deliberately outside Recon YAML."""

	rows: int | None = None
	extra_pad: AxisPad = AxisPad()

	def __post_init__(self):
		if self.rows is not None and self.rows < 1:
			raise ValueError("window rows must be positive")


@dataclass(frozen=True)
class RetainedFreshRegion:
	"""Explicit transition seam consumed later by retention and prefetch."""

	adjacent: bool
	retained: tuple[int, int] | None
	fresh: tuple[tuple[int, int], ...]
	prefix_valid: bool


@dataclass(frozen=True)
class StreamingPlan:
	"""Resolved padding and canonical dataset windows plus transition metadata."""

	selection: tuple[int, int]
	window_rows: int
	required_pad: AxisPad
	extra_pad: AxisPad
	effective_pad: AxisPad
	dataset: TomoDataSet
	windows: tuple[TomoWindow, ...]
	transitions: tuple[RetainedFreshRegion, ...]
	warnings: tuple[str, ...] = ()
	center_prepass: str | None = None

	@property
	def whole_volume(self) -> bool:
		return len(self.windows) == 1 and _y_bounds(self.windows[0]) == self.selection

	def record(self) -> dict[str, Any]:
		return {
			"selection": list(self.selection),
			"window_rows": self.window_rows,
			"required_pad": self.required_pad.as_dict(),
			"extra_pad": self.extra_pad.as_dict(),
			"effective_pad": self.effective_pad.as_dict(),
			"whole_volume": self.whole_volume,
			"center_prepass": self.center_prepass,
			"warnings": list(self.warnings),
			"windows": [{
				"index": index,
				"core": list(_y_bounds(window)),
				"source": list(_y_bounds(window.with_pad(["Y"]))),
				"interior": list(_y_bounds(window.internal)),
				"transition": {
					"adjacent": transition.adjacent,
					"retained": transition.retained,
					"fresh": transition.fresh,
					"prefix_valid": transition.prefix_valid,
				},
			} for index, (window, transition) in enumerate(
				zip(self.windows, self.transitions, strict=True)
			)],
		}


def _y_bounds(window: TomoWindow) -> tuple[int, int]:
	start = int(window.pos.Y)
	return start, start + int(window.shape.Y)


def _fresh_regions(
	start: int, stop: int, retained: tuple[int, int] | None,
) -> tuple[tuple[int, int], ...]:
	if retained is None:
		return ((start, stop),)
	regions = []
	if start < retained[0]:
		regions.append((start, retained[0]))
	if retained[1] < stop:
		regions.append((retained[1], stop))
	return tuple(regions)


def resolve_streaming_plan(
	stages: Sequence[Stage], scan: Any, config: WindowConfig | None = None,
) -> StreamingPlan:
	"""Resolve composed Stage padding and global row windows before allocation."""

	config = config or WindowConfig()
	rows = scan.detector.rows
	columns = scan.detector.columns
	if rows is None or columns is None:
		raise WindowPlanError("window planning requires detector rows and columns")
	read = next((stage for stage in stages if stage.name.startswith("read.")), None)
	selection_start = int(getattr(read, "y_start", 0))
	selection_stop = rows if read is None or getattr(read, "y_stop", None) is None \
		else int(read.y_stop)
	if not 0 <= selection_start < selection_stop <= rows:
		raise WindowPlanError("read Stage has an invalid detector-row selection")
	selected_rows = selection_stop - selection_start
	window_rows = selected_rows if config.rows is None else min(config.rows, selected_rows)
	theta_count = scan.n_projections
	input_shape = ProjOrder(theta_count, window_rows, columns)
	required = AxisPad()
	for stage in stages:
		required = required + stage.contract.required_pad.resolve(scan, input_shape)
	effective = required.maximum(config.extra_pad)
	multiple = window_rows < selected_rows
	if multiple:
		for stage in stages:
			if "Y" in stage.contract.requires_full_axes:
				raise WindowPlanError(
					f"{stage.name} requires the full Y axis and cannot run with "
					f"window_rows={window_rows}; select whole-volume mode",
				)
	warnings = []
	if multiple:
		sensitive = tuple(dict.fromkeys(
			stage.name for stage in stages
			if stage.name in {
				"phase.paganin", "phase.paganin_filter",
				"phase.paganin_phase_absorption", "phase.bac",
				"phase.bac_correction", "phase.bronnikov_aided",
			}
		))
		if sensitive:
			message = (
				"FFT-extent-sensitive windowed Stages use fixed window+padding parity, "
				"not whole-volume equivalence: " + ", ".join(sensitive)
			)
			warnings.append(message)
			LOGGER.warning(message)
	dataset = TomoDataSet(
		CTFov(theta_count, scan.degrees_per_fov, 1),
		ProjOrder(theta_count, rows, columns),
		Y.drange(selection_start, selection_stop, window_rows),
	)
	windows = tuple(dataset.iter_with(
		wsize=Y(window_rows),
		pad=ProjOrder(effective.theta, effective.y, effective.x),
	))
	transitions = []
	previous: TomoWindow | None = None
	for window in windows:
		core_start, core_stop = _y_bounds(window)
		source_start, source_stop = _y_bounds(window.with_pad(["Y"]))
		retained = None
		if previous is not None:
			previous_source_start, previous_source_stop = _y_bounds(previous.with_pad(["Y"]))
			start = max(previous_source_start, source_start)
			stop = min(previous_source_stop, source_stop)
			retained = (start, stop) if start < stop else None
		transitions.append(RetainedFreshRegion(
			adjacent=previous is not None and _y_bounds(previous)[1] == core_start,
			retained=retained,
			fresh=_fresh_regions(source_start, source_stop, retained),
			# #180 may make this true after it proves retained prefix state.
			prefix_valid=False,
		))
		previous = window
	center_prepass = (
		"replay_processed_prefix_per_window"
		if any(stage.name.startswith("center.") and stage.name != "center.manual" for stage in stages)
		else None
	)
	return StreamingPlan(
		selection=(selection_start, selection_stop), window_rows=window_rows,
		required_pad=required, extra_pad=config.extra_pad,
		effective_pad=effective, dataset=dataset, windows=windows,
		transitions=tuple(transitions),
		warnings=tuple(warnings), center_prepass=center_prepass,
	)


def crop_reconstruction_to_interior(
	state: PipelineState, window: TomoWindow, plan: StreamingPlan, *, index: int | None = None,
) -> PipelineState:
	"""Publish the core reconstruction rows and global source coordinates."""

	interior_start, interior_stop = _y_bounds(window.internal)
	working = state.working_array
	view = working[
		interior_start:interior_stop, :, :
	] if isinstance(working, ManagedBuffer) else np.asarray(working)[
		interior_start:interior_stop, :, :
	]
	result = state.with_array(view)
	source = state.source_window
	if source is not None:
		shape = ProjOrder(source.shape.Theta, window.shape.Y, source.shape.X)
		zero = ProjOrder(0, 0, 0)
		position = ProjOrder(source.pos.Theta, window.pos.Y, source.pos.X)
		step = ProjOrder(source.step.Theta, 1, source.step.X)
		result.source_window = TomoWindow(
			shape, zero, position, step, source.dtype, np.float32,
		)
	core_start, core_stop = _y_bounds(window)
	if index is None:
		index = next(i for i, planned in enumerate(plan.windows) if planned is window)
	result.artifacts["streaming_window"] = {
		"index": index,
		"core_start": core_start,
		"core_stop": core_stop,
		"output_start": core_start - plan.selection[0],
		"output_stop": core_stop - plan.selection[0],
		"total_rows": plan.selection[1] - plan.selection[0],
		"first": index == 0,
		"last": index == len(plan.windows) - 1,
	}
	return result
