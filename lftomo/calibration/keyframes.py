"""Bounded flat-series digest construction and keyframe interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from natsort import natsorted
import numpy as np
import tifffile as tf

from .build import CalibrationBuildSession
from .models import CalibrationComponent, CompatibilityMetadata, TimeAxis, ValidationResult


class TIFFRandomFrameSource:
	"""Random-access single-frame TIFF series without loading the full series."""

	def __init__(self, path: str | Path, pattern: str = "*.tif*"):
		path = Path(path).expanduser().resolve()
		self.path = path
		self.pattern = pattern
		self.paths = natsorted(path.glob(pattern)) if path.is_dir() else [path]
		if not self.paths:
			raise FileNotFoundError(f"no TIFF frames match {path / pattern}")
		with tf.TiffFile(self.paths[0]) as tif:
			shape = tuple(tif.series[0].shape)
			self.dtype = np.dtype(tif.series[0].dtype)
		if len(self.paths) == 1 and len(shape) == 3:
			self._stack = True
			self.frame_count, self.rows, self.columns = map(int, shape)
		else:
			self._stack = False
			if len(shape) != 2:
				raise ValueError("TIFF digest input must contain 2-D frames")
			self.frame_count = len(self.paths)
			self.rows, self.columns = map(int, shape)

	def read(self, index: int, *, y_start: int = 0, y_stop: int | None = None,
			x_start: int = 0, x_stop: int | None = None) -> np.ndarray:
		if not 0 <= index < self.frame_count:
			raise IndexError(index)
		if self._stack:
			try:
				return np.asarray(tf.memmap(
					self.paths[0], mode="r",
				)[index, y_start:y_stop, x_start:x_stop])
			except (OSError, ValueError):
				with tf.TiffFile(self.paths[0]) as tif:
					return tif.pages[index].asarray()[y_start:y_stop, x_start:x_stop]
		try:
			return np.asarray(tf.memmap(
				self.paths[index], mode="r",
			)[y_start:y_stop, x_start:x_stop])
		except (OSError, ValueError):
			return np.asarray(tf.imread(self.paths[index]))[y_start:y_stop, x_start:x_stop]


@dataclass(frozen=True)
class FlatSeriesDigest:
	component: CalibrationComponent
	sample_centres: tuple[int, ...]
	times: tuple[float, ...]
	trajectory: tuple[dict[str, float], ...]
	maximum_gap: float


def _bin_frames(frames: np.ndarray, factor: int) -> np.ndarray:
	if factor == 1:
		return frames
	count, rows, columns = frames.shape
	rows -= rows % factor; columns -= columns % factor
	return frames[:, :rows, :columns].reshape(
		count, rows // factor, factor, columns // factor, factor,
	).mean(axis=(2, 4))


def build_flat_series_digest(
	build: CalibrationBuildSession,
	source: TIFFRandomFrameSource,
	*,
	keep: int = 120,
	median_window: int = 5,
	y_chunk: int = 64,
	crop: tuple[int, int, int, int] | None = None,
	binning: int = 1,
	frame_times: Sequence[float] | None = None,
	time_kind: str = "projection_index",
	time_unit: str = "frame",
	time_origin: Any | None = None,
	camera_segment: str,
	name: str = "beam_keyframes",
) -> FlatSeriesDigest:
	"""Sample interval centres and write short-median snapshots by Y window."""

	if not 2 <= keep <= source.frame_count:
		raise ValueError("keep must be between two and the source frame count")
	if median_window < 1 or y_chunk < 1 or binning < 1:
		raise ValueError("median_window, y_chunk, and binning must be positive")
	if crop is None:
		row0, row1, column0, column1 = 0, source.rows, 0, source.columns
	else:
		row0, row1, column0, column1 = map(int, crop)
	if not 0 <= row0 < row1 <= source.rows or not 0 <= column0 < column1 <= source.columns:
		raise ValueError("digest crop exceeds the source detector")
	output_rows = (row1 - row0) // binning
	output_columns = (column1 - column0) // binning
	if output_rows < 1 or output_columns < 1:
		raise ValueError("crop is smaller than the requested binning")
	edges = np.linspace(0, source.frame_count, keep + 1)
	centres = ((edges[:-1] + edges[1:]) / 2).astype(int)
	centres = np.clip(centres, 0, source.frame_count - 1)
	if len(np.unique(centres)) != keep:
		raise ValueError("sample centres are not unique; reduce keep")
	all_times = np.arange(source.frame_count, dtype=float) if frame_times is None else np.asarray(frame_times, float)
	if all_times.shape != (source.frame_count,) or np.any(np.diff(all_times) <= 0):
		raise ValueError("frame_times must be strictly increasing and match the source")
	selected_times = all_times[centres]
	time_axis = TimeAxis(
		kind=time_kind, unit=time_unit, origin=time_origin,
		values=[float(value) for value in selected_times],
	)
	writer = build.create_array_component(
		name, shape=(keep, output_rows, output_columns), dtype="float32",
		axes=("T", "Y", "X"), append_axis=1, kind="beam_keyframes",
		coordinate_frame="beam", time_axis=time_axis, interpolation="linear",
		compatibility=CompatibilityMetadata(camera_segment=camera_segment),
		metadata={
			"sample_centres": centres.tolist(), "median_window": median_window,
			"source_frame_count": source.frame_count, "crop": [row0, row1, column0, column1],
			"binning": binning, "camera_segment_boundary": "hard",
			"source": {
				"path": str(source.path), "pattern": source.pattern,
				"frame_count": source.frame_count,
				"first_frame": str(source.paths[0]), "last_frame": str(source.paths[-1]),
			},
		},
	)
	totals = np.zeros(keep); row_moments = np.zeros(keep); column_moments = np.zeros(keep)
	half = median_window // 2
	raw_chunk = y_chunk * binning
	for raw_start in range(row0, row0 + output_rows * binning, raw_chunk):
		raw_stop = min(row0 + output_rows * binning, raw_start + raw_chunk)
		snapshots = []
		for centre in centres:
			start, stop = max(0, centre - half), min(source.frame_count, centre + half + 1)
			frames = np.stack([
				source.read(index, y_start=raw_start, y_stop=raw_stop,
					x_start=column0, x_stop=column0 + output_columns * binning)
				for index in range(start, stop)
			])
			snapshots.append(np.median(frames, axis=0, overwrite_input=True))
		chunk = _bin_frames(np.asarray(snapshots), binning).astype(np.float32)
		writer.append(chunk)
		output_start = (raw_start - row0) // binning
		rows = np.arange(output_start, output_start + chunk.shape[1])
		columns = np.arange(output_columns)
		positive = np.clip(chunk, 0, None).astype(np.float64)
		totals += positive.sum(axis=(1, 2))
		row_moments += np.sum(positive.sum(axis=2) * rows[None], axis=1)
		column_moments += np.sum(positive.sum(axis=1) * columns[None], axis=1)
	trajectory = []
	for index in range(keep):
		trajectory.append({
			"frame_index": float(centres[index]), "time": float(selected_times[index]),
			"intensity": float(totals[index]),
			"centroid_row": float(row_moments[index] / totals[index]),
			"centroid_column": float(column_moments[index] / totals[index]),
		})
	maximum_gap = float(np.max(np.diff(selected_times)))
	writer.metadata["trajectory"] = trajectory
	writer.metadata["maximum_keyframe_gap"] = maximum_gap
	writer.add_validation_result(ValidationResult(
		name="keyframe_observability", status="warning" if maximum_gap > 2 * np.median(np.diff(selected_times)) else "pass",
		message="events wholly inside a keyframe gap are not observable",
		metrics={"maximum_gap": maximum_gap, "median_gap": float(np.median(np.diff(selected_times)))},
	))
	component = writer.close()
	return FlatSeriesDigest(
		component=component, sample_centres=tuple(map(int, centres)),
		times=tuple(map(float, selected_times)), trajectory=tuple(trajectory), maximum_gap=maximum_gap,
	)
