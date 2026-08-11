"""Projection input stages."""

from __future__ import annotations

from pathlib import Path

from natsort import natsorted
import h5py
import numpy as np
import tifffile as tf

from lftomo.config import HDF5DxchangeProjections, StageConfig, TIFFStackProjections
from lftomo.pipeline.core import PipelineState, Stage, StageContract
from lftomo.pipeline.domains import DataDomain
from lftomo.util.io import distribute_tiff_read, distribute_tiff_sinogram_read
from lftomo.util.shapes import ProjOrder, SinoOrder, Y
from lftomo.util.window import CTFov, TomoDataSet, TomoWindow


def _output_contract(order: str, *, required_metadata: tuple[str, ...],
		produces_artifacts: tuple[str, ...] = ()) -> StageContract:
	if order not in ("projection", "sinogram"):
		raise ValueError("read order must be 'projection' or 'sinogram'")
	return StageContract(
		produced_output=ProjOrder if order == "projection" else SinoOrder,
		cacheable_prefix=True,
		produces_domain=DataDomain.RAW,
		window_axes=("Theta", "Y"),
		required_metadata=required_metadata,
		produces_artifacts=produces_artifacts,
		managed_artifacts=produces_artifacts,
	)


def _storage(state: PipelineState):
	if state.storage is None:
		raise RuntimeError("read stages require an active PipelineRunContext")
	return state.storage


def _source_angles(state: PipelineState, count: int) -> np.ndarray:
	return np.linspace(
		np.deg2rad(state.scan.acquisition.rotation_start_deg),
		np.deg2rad(state.scan.acquisition.rotation_end_deg),
		count,
		endpoint=False,
		dtype=np.float32,
	)


def _bind_source_state(
	state: PipelineState,
	working,
	*,
	full_shape: ProjOrder,
	dtype,
	layout,
	theta_start: int,
	theta_stop: int,
	theta_stride: int,
	y_start: int,
	y_stop: int,
) -> PipelineState:
	angles = _source_angles(state, full_shape.Theta)
	dataset = TomoDataSet(
		CTFov(full_shape.Theta, state.scan.degrees_per_fov, 1),
		full_shape,
		Y(1),
		dtype=dtype,
		work_dtype=np.float32,
		angles=angles,
	)
	zero = ProjOrder(Theta=0, Y=0, X=0)
	# A TomoWindow describes the covered source-coordinate span.  With a
	# strided Theta selection its shape is therefore ``stop - start`` while the
	# working buffer contains ``len(range(start, stop, stride))`` samples; step
	# carries the mapping between those two extents.
	source_window = TomoWindow(
		ProjOrder(Theta=theta_stop - theta_start, Y=y_stop - y_start, X=full_shape.X),
		zero,
		ProjOrder(Theta=theta_start, Y=y_start, X=0),
		ProjOrder(Theta=theta_stride, Y=1, X=1),
		dtype,
		np.float32,
	)
	result = state.with_array(working, layout=layout, domain=DataDomain.RAW)
	result.dataset = dataset
	result.source_window = source_window
	return result


class HDF5DxchangeReadStage(Stage):
	"""Read projections and optional calibration fields from dxchange HDF5."""

	name = "read.hdf5_dxchange"

	def __init__(self, order: str = "projection", theta_start: int = 0,
				theta_stop: int | None = None, y_start: int = 0,
				y_stop: int | None = None, calibration_policy: str = "optional",
				theta_count: int | None = None, theta_start_degrees: float | None = None,
				theta_stop_degrees: float | None = None,
				theta_stride: int = 1):
		if calibration_policy not in ("ignore", "optional", "required"):
			raise ValueError("calibration_policy must be 'ignore', 'optional', or 'required'")
		self.order = order
		self.theta_start = theta_start
		self.theta_stop = theta_count if theta_stop is None else theta_stop
		if theta_stride < 1:
			raise ValueError("theta_stride must be positive")
		self.theta_stride = theta_stride
		self.y_start = y_start
		self.y_stop = y_stop
		self.calibration_policy = calibration_policy
		self.theta_start_degrees = theta_start_degrees
		self.theta_stop_degrees = theta_stop_degrees
		self.contract = _output_contract(
			order,
			required_metadata=("projections.path", "projections.data_path"),
			produces_artifacts=(
				("white_fields", "dark_fields") if calibration_policy == "required" else ()
			),
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
			theta_count, rows, columns = data.shape
			dtype = data.dtype
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
			selected_count = len(range(self.theta_start, theta_stop, self.theta_stride))
			layout = ProjOrder if self.order == "projection" else SinoOrder
			shape = (
				ProjOrder(Theta=selected_count, Y=y_stop - self.y_start, X=columns)
				if layout is ProjOrder else
				SinoOrder(Y=y_stop - self.y_start, Theta=selected_count, X=columns)
			)
			memory = _storage(state).allocate(
				shape, dtype, role="raw_staging",
				reuse_from=("raw_staging", "reconstruction"),
			)
			target = memory.array()
			selection = np.s_[
				self.theta_start:theta_stop:self.theta_stride, self.y_start:y_stop, :,
			]
			if layout is ProjOrder and data.chunks is None and data.compression is None:
				data.read_direct(target, source_sel=selection)
				read_strategy = "direct"
			else:
				read_strategy = "bounded_scratch"
				chunk = min(32, selected_count)
				for offset in range(0, selected_count, chunk):
					count = min(chunk, selected_count - offset)
					start = self.theta_start + offset * self.theta_stride
					stop = start + count * self.theta_stride
					scratch = np.empty((count, y_stop - self.y_start, columns), dtype=dtype)
					data.read_direct(
						scratch,
						source_sel=np.s_[start:stop:self.theta_stride, self.y_start:y_stop, :],
					)
					if layout is ProjOrder:
						target[offset:offset + count] = scratch
					else:
						target[:, offset:offset + count, :] = np.transpose(scratch, (1, 0, 2))

			artifacts = dict(state.artifacts)
			artifacts["hdf5_read_strategy"] = read_strategy
			if self.calibration_policy != "ignore":
				missing = [
					path for path in (source.white_path, source.dark_path) if path not in handle
				]
				if missing and self.calibration_policy == "required":
					raise ValueError(
						"dxchange calibration dataset(s) missing: " + ", ".join(missing)
					)
				if not missing:
					for artifact_name, path in (
						("white_fields", source.white_path), ("dark_fields", source.dark_path),
					):
						dataset = handle[path]
						artifact_shape = (
							dataset.shape[0], y_stop - self.y_start, dataset.shape[2],
						)
						artifact = _storage(state).allocate(
							artifact_shape, dataset.dtype, role=artifact_name,
						)
						dataset.read_direct(
							artifact.array(),
							source_sel=np.s_[:, self.y_start:y_stop, :],
						)
						artifacts[artifact_name] = artifact
			if self.theta_start != 0 or theta_stop != theta_count or self.theta_stride != 1:
				all_angles = np.linspace(
					np.deg2rad(state.scan.acquisition.rotation_start_deg),
					np.deg2rad(state.scan.acquisition.rotation_end_deg),
					theta_count,
					endpoint=False,
					dtype=np.float32,
				)
				artifacts["angles"] = all_angles[
					self.theta_start:theta_stop:self.theta_stride
				]

		result = _bind_source_state(
			state,
			memory,
			full_shape=ProjOrder(Theta=theta_count, Y=rows, X=columns),
			dtype=dtype,
			layout=layout,
			theta_start=self.theta_start,
			theta_stop=theta_stop,
			theta_stride=self.theta_stride,
			y_start=self.y_start,
			y_stop=y_stop,
		)
		result.artifacts = artifacts
		return result


class TIFFStackReadStage(Stage):
	"""Read a naturally sorted TIFF stack through lftomo's direct-byte reader."""

	name = "read.tiff_stack"

	def __init__(self, order: str = "projection", threads: int | None = None,
				y_start: int = 0, y_stop: int | None = None,
				theta_start: int = 0, theta_stop: int | None = None,
				pattern: str | None = None, projection_count: int | None = None,
				theta_stride: int = 1):
		if order not in ("projection", "sinogram"):
			raise ValueError("TIFF read order must be 'projection' or 'sinogram'")
		self.order = order
		self.threads = threads
		self.y_start = y_start
		self.y_stop = y_stop
		self.theta_start = theta_start
		self.theta_stop = theta_stop
		if theta_stride < 1:
			raise ValueError("theta_stride must be positive")
		self.theta_stride = theta_stride
		self.pattern = pattern
		self.projection_count = projection_count
		self.contract = _output_contract(
			order,
			required_metadata=("projections.path", "projections.pattern"),
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "TIFFStackReadStage":
		params = dict(config.params)
		params.pop("sinogram_chunk_size", None)
		params.pop("chunk_overlap", None)
		return cls(**params)

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
		selected_files = files[self.theta_start:theta_stop:self.theta_stride]
		if len(files) == 1 and (
			self.theta_start != 0 or theta_stop != count or self.theta_stride != 1
		):
			raise ValueError(
				"Theta-windowed or strided TIFF reads currently require one file per projection"
			)
		selected_count = count if len(files) == 1 else len(selected_files)
		layout = ProjOrder if self.order == "projection" else SinoOrder
		shape = (
			ProjOrder(Theta=selected_count, Y=stop - self.y_start, X=columns)
			if layout is ProjOrder else
			SinoOrder(Y=stop - self.y_start, Theta=selected_count, X=columns)
		)
		memory = _storage(state).allocate(
			shape, dtype, role="raw_staging",
			reuse_from=("raw_staging", "reconstruction"),
		)
		kwargs = {} if self.threads is None else {"thread_max": self.threads}
		if layout is ProjOrder:
			distribute_tiff_read(
				selected_files, memory, y_start=self.y_start, y_stop=stop, **kwargs
			)
		else:
			distribute_tiff_sinogram_read(
				files,
				memory,
				y_start=self.y_start,
				y_stop=stop,
				theta_start=self.theta_start,
				theta_stop=theta_stop,
				theta_stride=self.theta_stride,
				**kwargs,
			)
		result = _bind_source_state(
			state,
			memory,
			full_shape=ProjOrder(Theta=count, Y=rows, X=columns),
			dtype=dtype,
			layout=layout,
			theta_start=self.theta_start,
			theta_stop=theta_stop,
			theta_stride=self.theta_stride,
			y_start=self.y_start,
			y_stop=stop,
		)
		if self.theta_start != 0 or theta_stop != count or self.theta_stride != 1:
			all_angles = np.linspace(
				np.deg2rad(state.scan.acquisition.rotation_start_deg),
				np.deg2rad(state.scan.acquisition.rotation_end_deg),
				count,
				endpoint=False,
				dtype=np.float32,
			)
			result.artifacts = dict(result.artifacts)
			result.artifacts["angles"] = all_angles[
				self.theta_start:theta_stop:self.theta_stride
			]
		return result
