"""Runner-level planning through canonical tomography windows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from lftomo.pipeline.core import _canonical_pad, Stage
from lftomo.util.shapes import DataRange, DataShape, ProjOrder, Theta, X, Y
from lftomo.util.window import CTFov, TomoDataSet, TomoWindow


LOGGER = logging.getLogger(__name__)
_AXES = {"Theta": Theta, "Y": Y, "X": X}


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
	theta: int | None = None
	columns: int | None = None
	axis: Literal["Theta", "Y", "X"] = "Y"
	intermediate_root: Path | None = None
	retain_intermediates: bool = False

	def __post_init__(self):
		for name in ("rows", "theta", "columns"):
			value = getattr(self, name)
			if value is not None and (type(value) is not int or value < 1):
				raise ValueError(f"window {name} must be positive")
		if self.axis not in _AXES:
			raise ValueError("window axis must be Theta, Y, or X")
		if self.intermediate_root is not None:
			object.__setattr__(self, "intermediate_root", Path(self.intermediate_root))
		_canonical_pad(self.extra_pad)
		if self.extra_pad_after is not None:
			_canonical_pad(self.extra_pad_after)


@dataclass(frozen=True)
class StreamingSegment:
	"""Adjacent authored Stages using one canonical directional window plan."""

	start: int
	stop: int
	dataset: TomoDataSet

	@property
	def axis(self) -> str:
		return self.dataset.step.names[0]


def _read_source_window(
	stage: Stage | None, physical_shape: ProjOrder, *, dtype=None,
) -> TomoWindow:
	"""Resolve the source selection declared by a read Stage."""

	params = getattr(stage, "params", None)
	return TomoWindow.from_bounds(
		physical_shape,
		y=(
			getattr(params, "y_start", 0),
			physical_shape.Y if getattr(params, "y_stop", None) is None else params.y_stop,
		),
		theta=(
			getattr(params, "theta_start", 0),
			physical_shape.Theta
			if getattr(params, "theta_stop", None) is None else params.theta_stop,
			getattr(params, "theta_stride", 1),
		),
		dtype=dtype,
	)


def _source_geometry(
	stages: Sequence[Stage], scan: Any, source_window: TomoWindow | None = None,
):
	"""Probe the acquisition once before creating any directional datasets."""

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
	try:
		selected = source_window if source_window is not None else _read_source_window(
			read, physical_shape, dtype=metadata.dtype,
		)
	except ValueError as error:
		raise WindowPlanError(str(error)) from error
	return metadata, selected


def _window_samples(config: WindowConfig, axis: str, selected: TomoWindow) -> int:
	limit = {"Y": config.rows, "Theta": config.theta, "X": config.columns}[axis]
	if limit is None and axis == "Theta" and any(
		value is not None for value in (config.rows, config.theta, config.columns)
	):
		# Match the configured Y batch; with only an X limit the Y batch is
		# the complete selected height. Explicit Theta limits take precedence.
		limit = config.rows if config.rows is not None else selected.frange.length.Y
	available = getattr(selected.frange.length, axis)
	return available if limit is None else min(limit, available)


def _dataset_for_axis(
	stages: Sequence[Stage], scan: Any, config: WindowConfig,
	metadata: Any, selected: TomoWindow, axis: str,
	*, whole_direction_preferences: bool = False,
) -> TomoDataSet:
	"""Resolve a segment's requirements into the sole canonical window model."""

	window_samples = _window_samples(config, axis, selected)
	selected_samples = selected.frange.length
	input_shape = replace(selected_samples, **{axis: window_samples})
	pad_before = ProjOrder(0, 0, 0)
	pad_after = ProjOrder(0, 0, 0)
	multiple = window_samples < getattr(selected_samples, axis)
	sensitive = []
	seen_sensitive = set()
	for stage in stages:
		if not whole_direction_preferences and stage.contract.required_window_axis not in (None, axis):
			raise WindowPlanError(
				f"{stage.name} requires window axis {stage.contract.required_window_axis}, "
				f"but the selected direction is {axis}",
			)
		if multiple and stage.contract.window_axes and axis not in stage.contract.window_axes:
			raise WindowPlanError(f"{stage.name} does not support {axis} windows")
		requirement = stage.contract.required_pad
		# A selected detector interval remains an ROI even when it is not split
		# in Y. Its phase context must come from real neighboring detector rows.
		stage_before, stage_after = requirement.resolve(
			scan, input_shape, physical_shape=metadata.physical_shape,
		)
		pad_before = pad_before + stage_before
		pad_after = pad_after + stage_after
		if multiple and axis in stage.contract.requires_full_axes:
			raise WindowPlanError(
				f"{stage.name} requires the full {axis} axis and cannot run with "
				f"{axis} window size {window_samples}; select whole-volume mode",
			)
		if (
			multiple and axis in (stage.contract.extent_sensitive_axes or ())
			and stage.name not in seen_sensitive
		):
			sensitive.append(stage.name)
			seen_sensitive.add(stage.name)
	extra = _canonical_pad(config.extra_pad)
	extra_after = extra if config.extra_pad_after is None \
		else _canonical_pad(config.extra_pad_after)
	pad_before = ProjOrder.param_max(pad_before, extra, _canonical_pad(selected.pad_before))
	pad_after = ProjOrder.param_max(pad_after, extra_after, _canonical_pad(selected.pad_after))
	if sensitive:
		message = (
			"Extent-sensitive windowed Stages use fixed window+padding parity, "
			"not whole-volume equivalence: " + ", ".join(sensitive)
		)
		LOGGER.warning(message)
	return TomoDataSet.from_range(
		CTFov(metadata.physical_shape.Theta, scan.degrees_per_fov, 1),
		metadata.physical_shape,
		DataRange(
			selected.pos, selected.pos + selected.shape,
			_AXES[axis](window_samples * getattr(selected.step, axis)),
		),
		dtype=metadata.dtype,
		work_dtype=np.float32,
		angles=metadata.angles,
		acquisition_indices=metadata.acquisition_indices,
		pad=pad_before,
		pad_after=pad_after,
		sample_step=selected.step,
	)


def resolve_streaming_plan(
	stages: Sequence[Stage], scan: Any, config: WindowConfig | None = None,
	*, source_window: TomoWindow | None = None,
) -> TomoDataSet:
	"""Resolve one explicitly selected direction, retaining the legacy Y API."""

	config = config or WindowConfig()
	metadata, selected = _source_geometry(stages, scan, source_window)
	return _dataset_for_axis(stages, scan, config, metadata, selected, config.axis)


def _stage_axis(stage: Stage, config: WindowConfig) -> str:
	contract = stage.contract
	if contract.required_window_axis is not None:
		return contract.required_window_axis
	if contract.preferred_window_axis is not None:
		return contract.preferred_window_axis
	# Empty declarations predate directional planning, including metadata-only
	# Stages. They impose no direction preference or additional capability claim.
	available = contract.window_axes or tuple(_AXES)
	available = tuple(axis for axis in available if axis not in contract.requires_full_axes)
	if config.axis in available:
		return config.axis
	if available:
		return available[0]
	if all(value is None for value in (config.rows, config.theta, config.columns)):
		return config.axis
	raise WindowPlanError(f"{stage.name} has no admissible streaming axis")


def _validate_boundary(stages: Sequence[Stage], stop: int, selected: TomoWindow) -> None:
	"""Reject boundaries whose geometry or published artifacts cannot be replayed."""

	produced = set()
	for stage in stages[:stop]:
		output = stage.output_window(selected)
		if any(
			getattr(output, name) != getattr(selected, name)
			for name in ("shape", "pos", "step", "pad_before", "pad_after")
		):
			raise WindowPlanError(
				f"{stage.name} changes output geometry before a streaming direction "
				"boundary; transformed intermediate geometry is not supported",
			)
		produced.update(stage.contract.produces_artifacts)
	for stage in stages[stop:]:
		needed = produced.intersection(stage.contract.consumes_artifacts)
		if needed:
			raise WindowPlanError(
				f"{stage.name} consumes window artifacts across a streaming direction "
				"boundary: " + ", ".join(sorted(needed)),
			)
		produced.difference_update(stage.contract.produces_artifacts)


def resolve_streaming_segments(
	stages: Sequence[Stage], scan: Any, config: WindowConfig | None = None,
	*, source_window: TomoWindow | None = None,
) -> tuple[StreamingSegment, ...]:
	"""Plan direction changes and propagate materialized halo coverage backward.

	Each segment's dataset owns its output selection and its source halo. The
	previous segment publishes that complete required source interval, so later
	reads never silently clip a requested halo at a materialization boundary.
	"""

	if not stages:
		return ()
	config = config or WindowConfig()
	metadata, selected = _source_geometry(stages, scan, source_window)
	if all(value is None for value in (config.rows, config.theta, config.columns)):
		# With no partitioning, every Stage already has all selected axes.
		# Direction preferences must not introduce unnecessary disk boundaries or
		# reject a pre-existing whole-volume geometry/artifact lifecycle.
		return (StreamingSegment(0, len(stages), _dataset_for_axis(
			stages, scan, config, metadata, selected, config.axis,
			whole_direction_preferences=True,
		)),)
	groups = []
	for index, stage in enumerate(stages):
		axis = _stage_axis(stage, config)
		if groups and groups[-1][2] == axis:
			groups[-1] = (groups[-1][0], index + 1, axis)
		else:
			groups.append((index, index + 1, axis))
	for _start, stop, _axis in groups[:-1]:
		_validate_boundary(stages, stop, selected)
	segments = []
	for start, stop, axis in reversed(groups):
		dataset = _dataset_for_axis(
			stages[start:stop], scan, config, metadata, selected, axis,
		)
		segments.append(StreamingSegment(start, stop, dataset))
		selected = dataset.full_window().with_pad()
	return tuple(reversed(segments))
