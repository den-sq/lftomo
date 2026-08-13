"""Runner-level window planning and retained/fresh transition contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Sequence

import numpy as np

from lftomo.pipeline.core import PipelineState, Stage
from lftomo.storage import ManagedBuffer
from lftomo.util.shapes import DataRange, DataShape, ProjOrder, Y
from lftomo.util.window import CTFov, TomoDataSet, TomoWindow


LOGGER = logging.getLogger(__name__)


class WindowPlanError(ValueError):
	"""A pipeline cannot satisfy its contracts under the selected window plan."""


def _canonical_pad(value: DataShape) -> ProjOrder:
	if not set(asdict(value)).issubset({"Theta", "Y", "X"}):
		raise ValueError("window padding may use only Theta, Y, and X axes")
	if any(component < 0 for component in asdict(value).values()):
		raise ValueError("window padding cannot be negative")
	return ProjOrder(Theta=0, Y=0, X=0) + value


def _pad_record(value: DataShape) -> dict[str, int]:
	canonical = _canonical_pad(value)
	return {
		"theta": int(canonical.Theta),
		"y": int(canonical.Y),
		"x": int(canonical.X),
	}


@dataclass(frozen=True)
class WindowConfig:
	"""Machine/runner window selection, deliberately outside Recon YAML."""

	rows: int | None = None
	extra_pad: DataShape = field(default_factory=lambda: Y(0))

	def __post_init__(self):
		if self.rows is not None and self.rows < 1:
			raise ValueError("window rows must be positive")
		_canonical_pad(self.extra_pad)


@dataclass(frozen=True)
class RetainedFreshRegion:
	"""Explicit transition seam consumed later by retention and prefetch."""

	adjacent: bool
	retained: tuple[int, int] | None
	fresh: tuple[tuple[int, int], ...]
	prefix_valid: bool


@dataclass(frozen=True)
class StreamingPlan:
	"""Resolved window requirements and canonical source windows."""

	dataset: TomoDataSet
	selection: tuple[int, int]
	window_rows: int
	required_pad: ProjOrder
	extra_pad: ProjOrder
	effective_pad: ProjOrder
	windows: tuple[TomoWindow, ...]
	transitions: tuple[RetainedFreshRegion, ...]
	warnings: tuple[str, ...] = ()
	center_prepass: str | None = None

	@property
	def whole_volume(self) -> bool:
		window = self.windows[0]
		return len(self.windows) == 1 \
			and int(window.pos.Y) == self.selection[0] \
			and int(window.pos.Y + window.shape.Y) == self.selection[1]

	def record(self) -> dict[str, Any]:
		return {
			"selection": list(self.selection),
			"window_rows": self.window_rows,
			"required_pad": _pad_record(self.required_pad),
			"extra_pad": _pad_record(self.extra_pad),
			"effective_pad": _pad_record(self.effective_pad),
			"whole_volume": self.whole_volume,
			"center_prepass": self.center_prepass,
			"warnings": list(self.warnings),
			"windows": [{
				"index": index,
				"core": [
					int(window.pos.Y), int(window.pos.Y + window.shape.Y),
				],
				"source": [
					int(window.source_pos.Y),
					int(window.source_pos.Y + window.source_shape.Y),
				],
				"interior": [
					int(window.internal.pos.Y),
					int(window.internal.pos.Y + window.internal.shape.Y),
				],
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
	"""Resolve Stage halos into canonical windows before allocation."""

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
	theta_start = int(getattr(read, "theta_start", 0))
	theta_stop = scan.n_projections \
		if read is None or getattr(read, "theta_stop", None) is None \
		else int(read.theta_stop)
	theta_stride = int(getattr(read, "theta_stride", 1))
	if not 0 <= theta_start < theta_stop <= scan.n_projections or theta_stride < 1:
		raise WindowPlanError("read Stage has an invalid projection selection")
	selected_theta = len(range(theta_start, theta_stop, theta_stride))
	selected_rows = selection_stop - selection_start
	window_rows = selected_rows if config.rows is None else min(config.rows, selected_rows)
	input_shape = ProjOrder(selected_theta, window_rows, columns)
	required = ProjOrder(Theta=0, Y=0, X=0)
	for stage in stages:
		required = required + stage.contract.required_pad.resolve(scan, input_shape)
	extra = _canonical_pad(config.extra_pad)
	effective = ProjOrder.param_max(required, extra)
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
		CTFov(scan.n_projections, scan.degrees_per_fov, 1),
		ProjOrder(scan.n_projections, rows, columns),
		Y(1),
		dtype=np.dtype(scan.metadata.get("raw_dtype", np.uint16)),
		work_dtype=np.float32,
	)
	bounds = DataRange(
		ProjOrder(theta_start, selection_start, 0),
		ProjOrder(theta_stop, selection_stop, columns),
		ProjOrder(1, 1, 1),
	)
	windows = []
	transitions = []
	previous: TomoWindow | None = None
	for core_start in range(selection_start, selection_stop, window_rows):
		core_stop = min(selection_stop, core_start + window_rows)
		window = dataset.window(
			ProjOrder(theta_stop - theta_start, core_stop - core_start, columns),
			pos=ProjOrder(theta_start, core_start, 0),
			pad=effective,
			step=ProjOrder(theta_stride, 1, 1),
			bounds=bounds,
		)
		source_start = int(window.source_pos.Y)
		source_stop = int(window.source_pos.Y + window.source_shape.Y)
		retained = None
		if previous is not None:
			start = max(int(previous.source_pos.Y), source_start)
			stop = min(int(previous.source_pos.Y + previous.source_shape.Y), source_stop)
			retained = (start, stop) if start < stop else None
		transitions.append(RetainedFreshRegion(
			adjacent=(
				previous is not None
				and int(previous.pos.Y + previous.shape.Y) == core_start
			),
			retained=retained,
			fresh=_fresh_regions(source_start, source_stop, retained),
			# #180 may make this true after it proves retained prefix state.
			prefix_valid=False,
		))
		windows.append(window)
		previous = window
	center_prepass = (
		"replay_processed_prefix_per_window"
		if any(stage.name.startswith("center.") and stage.name != "center.manual" for stage in stages)
		else None
	)
	return StreamingPlan(
		dataset=dataset,
		selection=(selection_start, selection_stop), window_rows=window_rows,
		required_pad=required, extra_pad=extra, effective_pad=effective,
		windows=tuple(windows), transitions=tuple(transitions),
		warnings=tuple(warnings), center_prepass=center_prepass,
	)


def crop_reconstruction_to_interior(
	state: PipelineState, window: TomoWindow, plan: StreamingPlan, *, index: int = 0,
) -> PipelineState:
	"""Publish logical core rows using canonical buffer-local coordinates."""

	source = state.source_window or window
	internal = source.internal
	start = int(internal.pos.Y)
	stop = start + int(internal.shape.Y)
	working = state.working_array
	view = working[start:stop, :, :] if isinstance(working, ManagedBuffer) \
		else np.asarray(working)[start:stop, :, :]
	result = state.with_array(view)
	result.source_window = source.rescope(pad=source.shape.make_with(0))
	core_start = int(window.pos.Y)
	core_stop = core_start + int(window.shape.Y)
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
