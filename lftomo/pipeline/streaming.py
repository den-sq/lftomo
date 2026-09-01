"""Runner-level planning through canonical tomography windows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from typing import Any, Iterator, Sequence

import numpy as np

from lftomo.pipeline.core import _canonical_pad, Stage
from lftomo.util.shapes import DataShape, ProjOrder, Y
from lftomo.util.window import CTFov, TomoDataSet, TomoWindow


LOGGER = logging.getLogger(__name__)


class WindowPlanError(ValueError):
	"""A pipeline cannot satisfy its contracts under the selected window plan."""


@dataclass(frozen=True)
class WindowConfig:
	"""Machine/runner window selection, deliberately outside Recon YAML.

	``extra_pad`` is symmetric unless ``extra_pad_after`` is supplied.
	"""

	rows: int | None = None
	extra_pad: DataShape = field(default_factory=lambda: Y(0))
	extra_pad_after: DataShape | None = None

	def __post_init__(self):
		if self.rows is not None and self.rows < 1:
			raise ValueError("window rows must be positive")
		_canonical_pad(self.extra_pad)
		if self.extra_pad_after is not None:
			_canonical_pad(self.extra_pad_after)


def iter_streaming_windows(
	dataset: TomoDataSet, source_window: TomoWindow,
) -> Iterator[TomoWindow]:
	"""Apply a dataset's Y schedule to its selected source-coordinate window."""

	for scheduled in dataset.iter_with(substep=source_window.step):
		y = scheduled.axis_slice("Y")
		yield scheduled.rescope(
			pos=replace(source_window.pos, Y=y.start),
			shape=replace(source_window.shape, Y=y.stop - y.start),
		)


def resolve_streaming_plan(
	stages: Sequence[Stage], scan: Any, config: WindowConfig | None = None,
) -> tuple[TomoDataSet, TomoWindow]:
	"""Resolve composed Stage padding and global row windows before allocation."""

	config = config or WindowConfig()
	read = next((stage for stage in stages if stage.name.startswith("read.")), None)
	if read is None:
		from lftomo.pipeline.stages.read import SourceMetadata, _source_angles

		physical_shape = scan.projection_shape()
		indices, angles = _source_angles(scan, physical_shape.Theta)
		metadata = SourceMetadata(
			physical_shape=physical_shape,
			dtype=np.dtype(scan.metadata.get("raw_dtype", "uint16")),
			angles=angles,
			acquisition_indices=indices,
		)
	else:
		from lftomo.pipeline.stages.read import probe_source_metadata

		metadata = probe_source_metadata(scan, read)
		physical_shape = metadata.physical_shape
	rows = physical_shape.Y
	columns = physical_shape.X
	selection_start = int(getattr(read, "y_start", 0))
	selection_stop = rows if read is None or getattr(read, "y_stop", None) is None \
		else int(read.y_stop)
	if not 0 <= selection_start < selection_stop <= rows:
		raise WindowPlanError("read Stage has an invalid detector-row selection")
	selected_rows = selection_stop - selection_start
	window_rows = selected_rows if config.rows is None else min(config.rows, selected_rows)
	theta_start = int(getattr(read, "theta_start", 0))
	theta_stop = (
		physical_shape.Theta
		if read is None or getattr(read, "theta_stop", None) is None
		else int(read.theta_stop)
	)
	theta_stride = int(getattr(read, "theta_stride", 1))
	source_window = TomoWindow.from_bounds(
		physical_shape,
		y=(selection_start, selection_stop),
		theta=(theta_start, theta_stop, theta_stride),
		dtype=metadata.dtype,
		work_dtype=np.float32,
	)
	theta_count = source_window.frange.length.Theta
	input_shape = ProjOrder(theta_count, window_rows, columns)
	required = ProjOrder(0, 0, 0)
	required_after = ProjOrder(0, 0, 0)
	for stage in stages:
		stage_before, stage_after = stage.contract.required_pad.resolve(scan, input_shape)
		required = required + stage_before
		required_after = required_after + stage_after
	extra = _canonical_pad(config.extra_pad)
	extra_after = extra if config.extra_pad_after is None \
		else _canonical_pad(config.extra_pad_after)
	effective = ProjOrder.param_max(required, extra)
	effective_after = ProjOrder.param_max(required_after, extra_after)
	multiple = window_rows < selected_rows
	if multiple:
		for stage in stages:
			if "Y" in stage.contract.requires_full_axes:
				raise WindowPlanError(
					f"{stage.name} requires the full Y axis and cannot run with "
					f"window_rows={window_rows}; select whole-volume mode",
				)
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
			LOGGER.warning(message)
	dataset = TomoDataSet(
		CTFov(physical_shape.Theta, scan.degrees_per_fov, 1),
		physical_shape,
		Y.drange(selection_start, selection_stop, window_rows),
		dtype=metadata.dtype,
		work_dtype=np.float32,
		angles=metadata.angles,
		acquisition_indices=metadata.acquisition_indices,
		pad=effective,
		pad_after=effective_after,
	)
	return dataset, source_window
