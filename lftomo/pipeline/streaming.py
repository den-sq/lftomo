"""Runner-level planning through canonical tomography windows."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

from lftomo.pipeline.core import AxisPad, Stage
from lftomo.util.shapes import ProjOrder, Y
from lftomo.util.window import CTFov, TomoDataSet


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
class StreamingPlan:
	"""Resolved padding plus the canonical iterable physical dataset."""

	selection: tuple[int, int]
	window_rows: int
	required_pad: AxisPad
	extra_pad: AxisPad
	effective_pad: AxisPad
	dataset: TomoDataSet
	warnings: tuple[str, ...] = ()
	center_prepass: str | None = None

	def __iter__(self):
		return self.dataset.iter_with(
			wsize=Y(self.window_rows),
			pad=ProjOrder(
				self.effective_pad.theta, self.effective_pad.y, self.effective_pad.x,
			),
		)

	@property
	def window_count(self) -> int:
		return len(self.dataset.frange)

	@property
	def whole_volume(self) -> bool:
		return self.window_count == 1 and (
			int(self.dataset.pos.Y), int(self.dataset.pos.Y + self.dataset.shape.Y)
		) == self.selection

	def record(self) -> dict[str, Any]:
		"""Serialize derived plan geometry for the run manifest."""

		record = {
			"selection": list(self.selection),
			"window_rows": self.window_rows,
			"required_pad": self.required_pad.as_dict(),
			"extra_pad": self.extra_pad.as_dict(),
			"effective_pad": self.effective_pad.as_dict(),
			"whole_volume": self.whole_volume,
			"center_prepass": self.center_prepass,
			"warnings": list(self.warnings),
			"windows": [],
		}
		for index, window in enumerate(self):
			core = window.index[1]
			source = window.with_pad(["Y"]).index[1]
			interior = window.internal.index[1]
			record["windows"].append({
				"index": index,
				"core": [core.start, core.stop],
				"source": [source.start, source.stop],
				"interior": [interior.start, interior.stop],
			})
		return record


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
	center_prepass = (
		"replay_processed_prefix_per_window"
		if any(stage.name.startswith("center.") and stage.name != "center.manual" for stage in stages)
		else None
	)
	return StreamingPlan(
		selection=(selection_start, selection_stop), window_rows=window_rows,
		required_pad=required, extra_pad=config.extra_pad,
		effective_pad=effective, dataset=dataset,
		warnings=tuple(warnings), center_prepass=center_prepass,
	)
