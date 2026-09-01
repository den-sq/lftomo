"""Runner-level planning through canonical tomography windows."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Sequence

import numpy as np

from lftomo.pipeline.core import _canonical_pad, Stage
from lftomo.util.shapes import DataRange, DataShape, ProjOrder, Y
from lftomo.util.window import CTFov, TomoDataSet


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


def resolve_streaming_plan(
	stages: Sequence[Stage], scan: Any, config: WindowConfig | None = None,
) -> TomoDataSet:
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
	if not 0 <= theta_start < theta_stop <= physical_shape.Theta or theta_stride < 1:
		raise WindowPlanError("read Stage has an invalid projection selection")
	theta_count = len(range(theta_start, theta_stop, theta_stride))
	input_shape = ProjOrder(theta_count, window_rows, columns)
	pad_before = ProjOrder(0, 0, 0)
	pad_after = ProjOrder(0, 0, 0)
	multiple = window_rows < selected_rows
	sensitive = []
	seen_sensitive = set()
	for stage in stages:
		stage_before, stage_after = stage.contract.required_pad.resolve(scan, input_shape)
		pad_before = pad_before + stage_before
		pad_after = pad_after + stage_after
		if multiple and "Y" in stage.contract.requires_full_axes:
			raise WindowPlanError(
				f"{stage.name} requires the full Y axis and cannot run with "
				f"window_rows={window_rows}; select whole-volume mode",
			)
		if multiple and stage.name in {
			"phase.paganin", "phase.paganin_filter",
			"phase.paganin_phase_absorption", "phase.bac",
			"phase.bac_correction", "phase.bronnikov_aided",
		} and stage.name not in seen_sensitive:
			sensitive.append(stage.name)
			seen_sensitive.add(stage.name)
	extra = _canonical_pad(config.extra_pad)
	extra_after = extra if config.extra_pad_after is None \
		else _canonical_pad(config.extra_pad_after)
	pad_before = ProjOrder.param_max(pad_before, extra)
	pad_after = ProjOrder.param_max(pad_after, extra_after)
	if sensitive:
		message = (
			"FFT-extent-sensitive windowed Stages use fixed window+padding parity, "
			"not whole-volume equivalence: " + ", ".join(sensitive)
		)
		LOGGER.warning(message)
	selection_start_shape = ProjOrder(theta_start, selection_start, 0)
	selection_stop_shape = ProjOrder(theta_stop, selection_stop, columns)
	dataset = TomoDataSet.from_range(
		CTFov(physical_shape.Theta, scan.degrees_per_fov, 1),
		physical_shape,
		DataRange(selection_start_shape, selection_stop_shape, Y(window_rows)),
		dtype=metadata.dtype,
		work_dtype=np.float32,
		angles=metadata.angles,
		acquisition_indices=metadata.acquisition_indices,
		pad=pad_before,
		pad_after=pad_after,
		sample_step=ProjOrder(theta_stride, 1, 1),
	)
	return dataset
