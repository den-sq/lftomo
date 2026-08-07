"""Memory-bounded generation of conventional four-map session medians."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Literal, Protocol, Sequence

import h5py
import numpy as np
from pydantic import Field
import tifffile as tf

from lftomo.calibration import DetectorRegion
from lftomo.config.models import ConfigModel


FOUR_MAP_ORDER = ("pre_gain", "post_gain", "pre_dark", "post_dark")
Orientation = Literal["identity", "flip_x", "flip_y", "rotate_180", "transpose"]
_TIFF_FILE_ERROR = getattr(tf, "TiffFileError", ValueError)
_TIFF_SOURCE_ERRORS = (OSError, ValueError, IndexError, _TIFF_FILE_ERROR)
_TIFF_MEMMAP_ERRORS = (OSError, ValueError, _TIFF_FILE_ERROR)


class FlatMedianError(ValueError):
	"""Base error for invalid sources or incomplete four-map builds."""


class FlatMedianSourceError(FlatMedianError):
	"""Raised when a frame series is empty, corrupt, or inconsistent."""


class FrameSeriesRecord(ConfigModel):
	format: Literal["hdf5", "tiff"]
	paths: list[Path]
	dataset: str | None = None
	frame_start: int = 0
	frame_stop: int
	frame_count: int = Field(gt=0)
	shape: tuple[int, int, int]
	dtype: str


class FlatMedianRecord(ConfigModel):
	schema_version: Literal["1.0"] = "1.0"
	kind: Literal["session_four_map_medians"] = "session_four_map_medians"
	created_at: datetime
	flat_fields: list[Path]
	output_shape: tuple[int, int]
	output_dtype: Literal["float32"] = "float32"
	region: DetectorRegion
	y_chunk: int
	sources: dict[str, FrameSeriesRecord]
	nonfinite_counts: dict[str, int] = Field(default_factory=dict)
	build_parameters: dict[str, Any] = Field(default_factory=dict)


class FrameSeries(Protocol):
	frame_count: int
	rows: int
	columns: int
	dtype: np.dtype

	def read_oriented_window(
		self, y_start: int, y_stop: int, orientation: Orientation,
	) -> np.ndarray: ...

	def record(self) -> FrameSeriesRecord: ...


def _oriented_shape(rows: int, columns: int, orientation: Orientation) -> tuple[int, int]:
	return (columns, rows) if orientation == "transpose" else (rows, columns)


def _orient_window(
	read_raw,
	rows: int,
	columns: int,
	y_start: int,
	y_stop: int,
	orientation: Orientation,
) -> np.ndarray:
	if orientation == "identity":
		return read_raw(slice(y_start, y_stop), slice(None))
	if orientation == "flip_x":
		return read_raw(slice(y_start, y_stop), slice(None))[:, :, ::-1]
	if orientation == "flip_y":
		return read_raw(slice(rows - y_stop, rows - y_start), slice(None))[:, ::-1]
	if orientation == "rotate_180":
		return read_raw(slice(rows - y_stop, rows - y_start), slice(None))[:, ::-1, ::-1]
	if orientation == "transpose":
		return np.transpose(
			read_raw(slice(None), slice(y_start, y_stop)), (0, 2, 1),
		)
	raise ValueError(f"unsupported detector orientation {orientation!r}")


class HDF5FrameSeries:
	"""A non-empty frame range in one three-dimensional HDF5 dataset."""

	def __init__(
		self, path: str | Path, dataset: str, *,
		frame_start: int = 0, frame_stop: int | None = None,
	):
		self.path = Path(path).expanduser().resolve()
		self.dataset = dataset
		try:
			with h5py.File(self.path, "r") as handle:
				if dataset not in handle:
					raise FlatMedianSourceError(f"HDF5 dataset {dataset!r} is missing from {self.path}")
				value = handle[dataset]
				if value.ndim != 3:
					raise FlatMedianSourceError(f"HDF5 dataset {dataset!r} must have (N, Y, X) axes")
				count, self.rows, self.columns = map(int, value.shape)
				self.dtype = np.dtype(value.dtype)
		except OSError as error:
			raise FlatMedianSourceError(f"cannot open HDF5 source {self.path}: {error}") from error
		self.frame_start = int(frame_start)
		self.frame_stop = count if frame_stop is None else int(frame_stop)
		if not 0 <= self.frame_start < self.frame_stop <= count:
			raise FlatMedianSourceError(
				f"invalid or empty frame range [{self.frame_start}, {self.frame_stop}) for {dataset!r}"
			)
		self.frame_count = self.frame_stop - self.frame_start

	def read_oriented_window(
		self, y_start: int, y_stop: int, orientation: Orientation,
	) -> np.ndarray:
		with h5py.File(self.path, "r") as handle:
			data = handle[self.dataset]
			return np.asarray(_orient_window(
				lambda ys, xs: data[self.frame_start:self.frame_stop, ys, xs],
				self.rows, self.columns, y_start, y_stop, orientation,
			))

	def record(self) -> FrameSeriesRecord:
		return FrameSeriesRecord(
			format="hdf5", paths=[self.path], dataset=self.dataset,
			frame_start=self.frame_start, frame_stop=self.frame_stop,
			frame_count=self.frame_count,
			shape=(self.frame_count, self.rows, self.columns), dtype=self.dtype.name,
		)


class TIFFFrameSeries:
	"""A multi-page TIFF or a sequence of single-frame TIFF files."""

	def __init__(self, paths: str | Path | Sequence[str | Path]):
		values = [paths] if isinstance(paths, (str, Path)) else list(paths)
		if not values:
			raise FlatMedianSourceError("TIFF frame series is empty")
		self.paths = [Path(path).expanduser().resolve() for path in values]
		try:
			with tf.TiffFile(self.paths[0]) as tif:
				shape = tuple(tif.series[0].shape)
				self.dtype = np.dtype(tif.series[0].dtype)
		except _TIFF_SOURCE_ERRORS as error:
			raise FlatMedianSourceError(f"cannot open TIFF source {self.paths[0]}: {error}") from error
		if len(self.paths) == 1 and len(shape) == 3:
			self._multipage = True
			self.frame_count, self.rows, self.columns = map(int, shape)
		elif len(shape) == 2:
			self._multipage = False
			self.frame_count = len(self.paths)
			self.rows, self.columns = map(int, shape)
			for path in self.paths[1:]:
				try:
					with tf.TiffFile(path) as tif:
						other_shape = tuple(tif.series[0].shape)
						other_dtype = np.dtype(tif.series[0].dtype)
				except _TIFF_SOURCE_ERRORS as error:
					raise FlatMedianSourceError(f"cannot open TIFF source {path}: {error}") from error
				if other_shape != shape or other_dtype != self.dtype:
					raise FlatMedianSourceError(
						f"TIFF frame {path} has shape/dtype {other_shape}/{other_dtype.name}; "
						f"expected {shape}/{self.dtype.name}"
					)
		else:
			raise FlatMedianSourceError("TIFF source must be a 2-D frame or 3-D frame stack")

	def _read_raw(self, y_slice: slice, x_slice: slice) -> np.ndarray:
		if self._multipage:
			try:
				mapped = tf.memmap(self.paths[0])
			except _TIFF_MEMMAP_ERRORS:
				with tf.TiffFile(self.paths[0]) as tif:
					return np.stack([page.asarray()[y_slice, x_slice] for page in tif.pages])
			return np.asarray(mapped[:, y_slice, x_slice])
		frames = []
		for path in self.paths:
			try:
				frame = tf.memmap(path)
			except _TIFF_MEMMAP_ERRORS:
				frame = tf.imread(path)
			frames.append(np.asarray(frame)[y_slice, x_slice])
		return np.stack(frames)

	def read_oriented_window(
		self, y_start: int, y_stop: int, orientation: Orientation,
	) -> np.ndarray:
		return _orient_window(
			self._read_raw, self.rows, self.columns, y_start, y_stop, orientation,
		)

	def record(self) -> FrameSeriesRecord:
		return FrameSeriesRecord(
			format="tiff", paths=self.paths, frame_start=0, frame_stop=self.frame_count,
			frame_count=self.frame_count,
			shape=(self.frame_count, self.rows, self.columns), dtype=self.dtype.name,
		)


class FourMapSources(ConfigModel):
	"""Pydantic-free source container is constructed through ``from_series``."""

	# Arbitrary reader objects are deliberately kept out of YAML sidecars.
	model_config = {"arbitrary_types_allowed": True}
	pre_gain: Any
	post_gain: Any
	pre_dark: Any
	post_dark: Any

	@classmethod
	def from_series(cls, pre_gain, post_gain, pre_dark, post_dark) -> "FourMapSources":
		return cls(
			pre_gain=pre_gain, post_gain=post_gain, pre_dark=pre_dark, post_dark=post_dark,
		)


class FourMapMedianResult(ConfigModel):
	"""Paths in the exact order consumed by ``Scan.metadata['flat_fields']``."""

	output_dir: Path
	pre_gain: Path
	post_gain: Path
	pre_dark: Path
	post_dark: Path
	metadata_path: Path

	@property
	def flat_fields(self) -> tuple[Path, Path, Path, Path]:
		return self.pre_gain, self.post_gain, self.pre_dark, self.post_dark

	def scan_metadata(self) -> dict[str, list[str]]:
		return {"flat_fields": [str(path) for path in self.flat_fields]}


def _validate_sources(
	sources: FourMapSources, orientation: Orientation, region: DetectorRegion | None,
) -> tuple[DetectorRegion, tuple[int, int]]:
	readers = [getattr(sources, name) for name in FOUR_MAP_ORDER]
	for name, reader in zip(FOUR_MAP_ORDER, readers, strict=True):
		if not all(hasattr(reader, attr) for attr in ("frame_count", "rows", "columns", "dtype", "read_oriented_window", "record")):
			raise TypeError(f"{name} does not implement the FrameSeries contract")
		if reader.frame_count < 1:
			raise FlatMedianSourceError(f"{name} frame series is empty")
	shapes = [_oriented_shape(reader.rows, reader.columns, orientation) for reader in readers]
	if any(shape != shapes[0] for shape in shapes[1:]):
		raise FlatMedianSourceError(f"four-map source shapes do not match after orientation: {shapes}")
	rows, columns = shapes[0]
	if region is None:
		region = DetectorRegion(
			y_start=0, x_start=0, height=rows, width=columns, orientation=orientation,
		)
	elif region.orientation != orientation:
		raise FlatMedianSourceError("region orientation does not match the requested transform")
	if region.y_start + region.height > rows or region.x_start + region.width > columns:
		raise FlatMedianSourceError(
			f"detector region {region.model_dump()} exceeds oriented source shape {(rows, columns)}"
		)
	return region, (region.height, region.width)


def generate_four_map_medians(
	sources: FourMapSources,
	output_dir: str | Path,
	*,
	y_chunk: int = 64,
	orientation: Orientation = "identity",
	region: DetectorRegion | None = None,
	build_parameters: dict[str, Any] | None = None,
) -> FourMapMedianResult:
	"""Median-combine four frame series with memory proportional to one Y chunk."""

	if y_chunk < 1:
		raise ValueError("y_chunk must be positive")
	region, output_shape = _validate_sources(sources, orientation, region)
	target = Path(output_dir).expanduser().resolve()
	if target.exists():
		raise FileExistsError(f"four-map output directory already exists: {target}")
	target.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent))
	try:
		completed_sources: dict[str, tuple[Path, int]] = {}
		nonfinite_counts: dict[str, int] = {}
		for name in FOUR_MAP_ORDER:
			reader = getattr(sources, name)
			source_key = reader.record().model_dump_json()
			if source_key in completed_sources:
				completed_path, nonfinite_count = completed_sources[source_key]
				shutil.copyfile(completed_path, staging / f"{name}.tif")
				nonfinite_counts[name] = nonfinite_count
				continue
			output = tf.memmap(
				staging / f"{name}.tif", shape=output_shape, dtype=np.float32,
			)
			nonfinite_count = 0
			for local_start in range(0, region.height, y_chunk):
				local_stop = min(region.height, local_start + y_chunk)
				frames = reader.read_oriented_window(
					region.y_start + local_start,
					region.y_start + local_stop,
					orientation,
				)[:, :, region.x_start:region.x_start + region.width]
				if frames.shape != (reader.frame_count, local_stop - local_start, region.width):
					raise FlatMedianSourceError(
						f"{name} returned invalid window shape {frames.shape}"
					)
				finite = np.isfinite(frames)
				invalid = int(frames.size - np.count_nonzero(finite))
				nonfinite_count += invalid
				if invalid:
					missing = np.all(~finite, axis=0)
					if missing.any():
						raise FlatMedianSourceError(
							f"{name} has {np.count_nonzero(missing)} pixel(s) with no finite frame value"
						)
					working = np.array(
						frames, dtype=np.result_type(frames.dtype, np.float32), copy=True,
					)
					working[~finite] = np.nan
					median = np.nanmedian(working, axis=0, overwrite_input=True)
				else:
					# A reader may return a writable disk-backed view. Never let NumPy's
					# in-place selection/sort mutate calibration source files.
					median = np.median(frames, axis=0, overwrite_input=False)
				output[local_start:local_stop] = median.astype(np.float32, copy=False)
			output.flush()
			del output
			nonfinite_counts[name] = nonfinite_count
			completed_sources[source_key] = staging / f"{name}.tif", nonfinite_count
		record = FlatMedianRecord(
			created_at=datetime.now(timezone.utc),
			flat_fields=[Path(f"{name}.tif") for name in FOUR_MAP_ORDER],
			output_shape=output_shape,
			region=region,
			y_chunk=y_chunk,
			sources={name: getattr(sources, name).record() for name in FOUR_MAP_ORDER},
			nonfinite_counts=nonfinite_counts,
			build_parameters=build_parameters or {},
		)
		metadata_path = staging / "flat_medians.yaml"
		metadata_path.write_text(record.to_yaml(), encoding="utf-8")
		if target.exists():
			raise FileExistsError(f"four-map output directory appeared during build: {target}")
		os.replace(staging, target)
	except Exception as error:
		shutil.rmtree(staging, ignore_errors=True)
		if isinstance(error, (FlatMedianError, FileExistsError, ValueError)):
			raise
		raise FlatMedianError(f"four-map median build failed: {error}") from error
	paths = {name: target / f"{name}.tif" for name in FOUR_MAP_ORDER}
	return FourMapMedianResult(
		output_dir=target,
		metadata_path=target / "flat_medians.yaml",
		**paths,
	)


def hdf5_four_map_sources(
	path: str | Path,
	*,
	pre_gain_dataset: str = "exchange/data_white",
	post_gain_dataset: str = "exchange/data_white",
	pre_dark_dataset: str = "exchange/data_dark",
	post_dark_dataset: str = "exchange/data_dark",
) -> FourMapSources:
	"""Create explicit HDF5 readers; repeated paths intentionally reuse one series."""

	return FourMapSources.from_series(
		HDF5FrameSeries(path, pre_gain_dataset),
		HDF5FrameSeries(path, post_gain_dataset),
		HDF5FrameSeries(path, pre_dark_dataset),
		HDF5FrameSeries(path, post_dark_dataset),
	)
