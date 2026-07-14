"""Projection input stages."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from natsort import natsorted
import h5py
import numpy as np
import tifffile as tf

from lftomo.config import HDF5DxchangeProjections, StageConfig, TIFFStackProjections
from lftomo.pipeline.core import PipelineState, ResourceHints, Stage, StageContract
from lftomo.pipeline.domains import DataDomain
from lftomo.util.io import distribute_tiff_read
from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import ProjOrder, SinoOrder


def _output_contract(order: str, *, required_metadata: tuple[str, ...], memory_multiplier: float) -> StageContract:
	if order not in ("projection", "sinogram"):
		raise ValueError("read order must be 'projection' or 'sinogram'")
	return StageContract(
		produced_output=ProjOrder if order == "projection" else SinoOrder,
		produces_domain=DataDomain.RAW,
		window_axes=("Theta", "Y"),
		required_metadata=required_metadata,
		resources=ResourceHints(memory_multiplier=memory_multiplier),
	)


class HDF5DxchangeReadStage(Stage):
	"""Read projections and optional calibration fields from dxchange HDF5."""

	name = "read.hdf5_dxchange"

	def __init__(self, order: str = "projection", theta_start: int = 0,
				theta_stop: int | None = None, y_start: int = 0,
				y_stop: int | None = None, calibration_policy: str = "optional",
				theta_count: int | None = None, theta_start_degrees: float | None = None,
				theta_stop_degrees: float | None = None, window_size: int | None = None):
		if calibration_policy not in ("ignore", "optional", "required"):
			raise ValueError("calibration_policy must be 'ignore', 'optional', or 'required'")
		self.order = order
		self.theta_start = theta_start
		self.theta_stop = theta_count if theta_stop is None else theta_stop
		self.y_start = y_start
		self.y_stop = y_stop
		self.calibration_policy = calibration_policy
		self.theta_start_degrees = theta_start_degrees
		self.theta_stop_degrees = theta_stop_degrees
		self.window_size = window_size
		self.contract = _output_contract(
			order,
			required_metadata=("projections.path", "projections.data_path"),
			memory_multiplier=1.0,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "HDF5DxchangeReadStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("HDF5DxchangeReadStage currently requires the NumPy backend")
		source = state.scan.projections
		if not isinstance(source, HDF5DxchangeProjections):
			raise TypeError("read.hdf5_dxchange requires HDF5DxchangeProjections scan metadata")

		with h5py.File(source.path, "r") as handle:
			if source.data_path not in handle:
				raise ValueError(
					f"dxchange projection dataset {source.data_path!r} is missing from {source.path}"
				)
			data = handle[source.data_path]
			if data.ndim != 3:
				raise ValueError(f"{source.data_path!r} must have (Theta, Y, X) dimensions")
			theta_count, rows, _columns = data.shape
			if theta_count != state.scan.n_projections:
				raise ValueError(
					f"scan declares {state.scan.n_projections} projections but HDF5 contains {theta_count}"
				)
			theta_stop = theta_count if self.theta_stop is None else self.theta_stop
			y_stop = rows if self.y_stop is None else self.y_stop
			if not 0 <= self.theta_start < theta_stop <= theta_count:
				raise ValueError("invalid HDF5 Theta window")
			if not 0 <= self.y_start < y_stop <= rows:
				raise ValueError("invalid HDF5 Y window")
			working = data[self.theta_start:theta_stop, self.y_start:y_stop, :]

			artifacts = dict(state.artifacts)
			if self.calibration_policy != "ignore":
				missing = [
					path for path in (source.white_path, source.dark_path) if path not in handle
				]
				if missing and self.calibration_policy == "required":
					raise ValueError(
						"dxchange calibration dataset(s) missing: " + ", ".join(missing)
					)
				if not missing:
					artifacts["white_fields"] = handle[source.white_path][:, self.y_start:y_stop, :]
					artifacts["dark_fields"] = handle[source.dark_path][:, self.y_start:y_stop, :]

		working = np.asarray(working)
		if self.order == "sinogram":
			working = np.transpose(working, (1, 0, 2)).copy()
		layout = ProjOrder if self.order == "projection" else SinoOrder
		result = state.with_array(working, layout=layout, domain=DataDomain.RAW)
		result.artifacts = artifacts
		return result


class TIFFStackReadStage(Stage):
	"""Read a naturally sorted TIFF stack through lftomo's direct-byte reader."""

	name = "read.tiff_stack"

	def __init__(self, order: str = "projection", threads: int | None = None,
				y_start: int = 0, y_stop: int | None = None,
				theta_start: int = 0, theta_stop: int | None = None,
				pattern: str | None = None, projection_count: int | None = None,
				sinogram_chunk_size: int | None = None, chunk_overlap: int = 0):
		if order not in ("projection", "sinogram"):
			raise ValueError("TIFF read order must be 'projection' or 'sinogram'")
		self.order = order
		self.threads = threads
		self.y_start = y_start
		self.y_stop = y_stop
		self.theta_start = theta_start
		self.theta_stop = theta_stop
		self.pattern = pattern
		self.projection_count = projection_count
		self.sinogram_chunk_size = sinogram_chunk_size
		self.chunk_overlap = chunk_overlap
		self.contract = _output_contract(
			order,
			required_metadata=("projections.path", "projections.pattern"),
			memory_multiplier=2.0,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "TIFFStackReadStage":
		return cls(**config.params)

	def _source_files(self, source: TIFFStackProjections) -> list[Path]:
		path = Path(source.path)
		pattern = self.pattern or source.pattern
		files = natsorted(path.glob(pattern)) if path.is_dir() else [path]
		if not files:
			raise FileNotFoundError(f"no TIFF projections match {path / source.pattern}")
		return files

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("TIFFStackReadStage currently requires the NumPy backend")
		if not isinstance(state.scan.projections, TIFFStackProjections):
			raise TypeError("read.tiff_stack requires TIFFStackProjections scan metadata")

		files = self._source_files(state.scan.projections)
		with tf.TiffFile(files[0]) as tif:
			page = tif.pages[0]
			dtype = page.dtype
			rows, columns = page.shape
			count = len(files) if len(files) > 1 else len(tif.pages)
		if count != state.scan.n_projections:
			raise ValueError(
				f"scan declares {state.scan.n_projections} projections but TIFF source contains {count}"
			)
		if self.projection_count is not None and count != self.projection_count:
			raise ValueError(
				f"read stage declares {self.projection_count} projections but TIFF source contains {count}"
			)

		stop = rows if self.y_stop is None else self.y_stop
		if not 0 <= self.y_start < stop <= rows:
			raise ValueError(f"invalid TIFF Y window [{self.y_start}, {stop}) for {rows} rows")
		theta_stop = count if self.theta_stop is None else self.theta_stop
		if not 0 <= self.theta_start < theta_stop <= count:
			raise ValueError(
				f"invalid TIFF Theta window [{self.theta_start}, {theta_stop}) for {count} projections"
			)
		selected_files = files[self.theta_start:theta_stop]
		if len(files) == 1 and (self.theta_start != 0 or theta_stop != count):
			raise ValueError("Theta-windowed TIFF reads currently require one file per projection")
		shape = ProjOrder(Theta=theta_stop - self.theta_start, Y=stop - self.y_start, X=columns)
		memory = SharedNPArray(f"tiff-read-{uuid4().hex}", shape, dtype, allocate=True)
		try:
			kwargs = {} if self.threads is None else {"thread_max": self.threads}
			distribute_tiff_read(
				selected_files, memory, y_start=self.y_start, y_stop=stop, **kwargs
			)
			with memory as values:
				working = values.copy()
		finally:
			memory.shutdown()

		if self.order == "sinogram":
			working = np.transpose(working, (1, 0, 2)).copy()
		layout = ProjOrder if self.order == "projection" else SinoOrder
		return state.with_array(working, layout=layout, domain=DataDomain.RAW)
