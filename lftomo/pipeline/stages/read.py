"""Projection input stages."""

from __future__ import annotations

from pathlib import Path

from natsort import natsorted
import h5py
import numpy as np
import tifffile as tf

from lftomo.angles import (
	angles_from_indices,
	hardware_timestamps,
	indices_from_angles,
	indices_from_hardware_timestamps,
	validate_acquisition_indices,
)
from lftomo.config import (
	ExplicitAngleSource,
	HDF5AngleSource,
	HDF5DxchangeProjections,
	HardwareTimestampAngleSource,
	StageConfig,
	TIFFStackProjections,
	UniformAngleSource,
)
from lftomo.pipeline.core import PipelineState, Stage, StageContract
from lftomo.pipeline.domains import DataDomain
from lftomo.util.io import distribute_tiff_read, distribute_tiff_sinogram_read
from lftomo.util.shapes import ProjOrder, SinoOrder, Y
from lftomo.util.window import CTFov, TomoDataSet, TomoWindow


def _output_contract(order: str, *, required_metadata: tuple[str, ...],
		produces_artifacts: tuple[str, ...] = ()) -> StageContract:
	if order not in ("projection", "sinogram"):
		raise ValueError("read order must be 'projection' or 'sinogram'")
	angle_artifacts = ("angles", "acquisition_indices", "angle_provenance")
	return StageContract(
		produced_output=ProjOrder if order == "projection" else SinoOrder,
		cacheable_prefix=True,
		produces_domain=DataDomain.RAW,
		window_axes=("Theta", "Y"),
		required_metadata=required_metadata,
		produces_artifacts=angle_artifacts + produces_artifacts,
		managed_artifacts=produces_artifacts,
	)


def _storage(state: PipelineState):
	if state.storage is None:
		raise RuntimeError("read stages require an active PipelineRunContext")
	return state.storage


def _source_angle_metadata(
	state: PipelineState,
	count: int,
	*,
	hdf5_handle=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
	acquisition = state.scan.acquisition
	planned = acquisition.n_projections
	source = acquisition.angle_source
	angles = None
	if isinstance(source, UniformAngleSource):
		indices = np.arange(count, dtype=np.int64)
		provenance = {"kind": "uniform"}
	elif isinstance(source, ExplicitAngleSource):
		indices = source.acquisition_indices
		angles = (
			None if source.angles_degrees is None
			else np.deg2rad(np.asarray(source.angles_degrees, dtype=np.float64))
		)
		provenance = {"kind": "explicit"}
	elif isinstance(source, HardwareTimestampAngleSource):
		timestamps = hardware_timestamps(
			source.path,
			column=source.timestamp_column,
			delimiter=source.delimiter,
			header_rows=source.header_rows,
		)
		indices = indices_from_hardware_timestamps(
			timestamps, recorded_count=count, planned_count=planned,
		)
		provenance = {
			"kind": "hardware_timestamps",
			"path": str(source.path),
			"timestamp_column": source.timestamp_column,
			"nominal_interval": (
				float(timestamps[-1] - timestamps[0]) / (planned - 1)
				if planned > 1 else None
			),
		}
	elif isinstance(source, HDF5AngleSource):
		if hdf5_handle is None:
			raise ValueError("hdf5 angle source requires an HDF5 projection source")
		if source.path not in hdf5_handle:
			raise ValueError(f"HDF5 angle dataset {source.path!r} is missing")
		angles = np.asarray(hdf5_handle[source.path], dtype=np.float64)
		if source.units == "degrees":
			angles = np.deg2rad(angles)
		indices = None
		provenance = {
			"kind": "hdf5",
			"path": source.path,
			"source_units": source.units,
		}
	else:  # pragma: no cover - discriminated config union is exhaustive
		raise TypeError(f"unsupported angle source: {type(source).__name__}")

	if indices is None:
		indices = indices_from_angles(
			angles,
			recorded_count=count,
			planned_count=planned,
			rotation_start_deg=acquisition.rotation_start_deg,
			rotation_end_deg=acquisition.rotation_end_deg,
		)
	else:
		indices = validate_acquisition_indices(
			indices, recorded_count=count, planned_count=planned,
		)
	expected_angles = angles_from_indices(
		indices,
		planned_count=planned,
		rotation_start_deg=acquisition.rotation_start_deg,
		rotation_end_deg=acquisition.rotation_end_deg,
	)
	if angles is not None:
		# Recovering indices above validates angle/index consistency. Explicit
		# sources that supplied both still need that same check.
		resolved = indices_from_angles(
			angles,
			recorded_count=count,
			planned_count=planned,
			rotation_start_deg=acquisition.rotation_start_deg,
			rotation_end_deg=acquisition.rotation_end_deg,
		)
		if not np.array_equal(resolved, indices):
			raise ValueError("angles and acquisition indices select different slots")
		resolved_angles = np.asarray(angles, dtype=np.float32)
	else:
		resolved_angles = expected_angles
	provenance.update({
		"recorded_projection_count": count,
		"planned_slot_count": planned,
		"rotation_start_deg": acquisition.rotation_start_deg,
		"rotation_end_deg": acquisition.rotation_end_deg,
		"endpoint": False,
	})
	return indices, resolved_angles, provenance


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
	angles: np.ndarray,
	acquisition_indices: np.ndarray,
) -> PipelineState:
	dataset = TomoDataSet(
		CTFov(full_shape.Theta, state.scan.degrees_per_fov, 1),
		full_shape,
		Y(1),
		dtype=dtype,
		work_dtype=np.float32,
		angles=angles,
		acquisition_indices=acquisition_indices,
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
			all_indices, all_angles, angle_provenance = _source_angle_metadata(
				state, theta_count, hdf5_handle=handle,
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
			angle_selection = slice(self.theta_start, theta_stop, self.theta_stride)
			artifacts["acquisition_indices"] = all_indices[angle_selection]
			artifacts["angles"] = all_angles[angle_selection]
			artifacts["angle_provenance"] = {
				**angle_provenance,
				"selection": {
					"start": self.theta_start,
					"stop": theta_stop,
					"stride": self.theta_stride,
				},
			}
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
			angles=all_angles,
			acquisition_indices=all_indices,
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
		all_indices, all_angles, angle_provenance = _source_angle_metadata(state, count)

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
			angles=all_angles,
			acquisition_indices=all_indices,
		)
		angle_selection = slice(self.theta_start, theta_stop, self.theta_stride)
		result.artifacts = dict(result.artifacts)
		result.artifacts["acquisition_indices"] = all_indices[angle_selection]
		result.artifacts["angles"] = all_angles[angle_selection]
		result.artifacts["angle_provenance"] = {
			**angle_provenance,
			"selection": {
				"start": self.theta_start,
				"stop": theta_stop,
				"stride": self.theta_stride,
			},
		}
		return result
