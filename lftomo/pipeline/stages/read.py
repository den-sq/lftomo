"""Projection input stages."""

from __future__ import annotations

from dataclasses import dataclass
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
from lftomo.util.shapes import GenericOrder, ProjOrder, SinoOrder
from lftomo.util.window import TomoWindow


def _output_contract(order: str, *, required_metadata: tuple[str, ...],
		produces_artifacts: tuple[str, ...] = ()) -> StageContract:
	if order not in ("projection", "sinogram"):
		raise ValueError("read order must be 'projection' or 'sinogram'")
	return StageContract(
		produces=ProjOrder if order == "projection" else SinoOrder,
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


def _source_angles(
	scan,
	count: int,
	*,
	hdf5_handle=None,
) -> tuple[np.ndarray, np.ndarray]:
	acquisition = scan.acquisition
	planned = acquisition.n_projections
	source = acquisition.angle_source
	angles = None
	if isinstance(source, UniformAngleSource):
		indices = np.arange(count, dtype=np.int64)
	elif isinstance(source, ExplicitAngleSource):
		indices = source.acquisition_indices
		angles = (
			None if source.angles_degrees is None
			else np.deg2rad(np.asarray(source.angles_degrees, dtype=np.float64))
		)
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
	elif isinstance(source, HDF5AngleSource):
		if hdf5_handle is None:
			raise ValueError("hdf5 angle source requires an HDF5 projection source")
		if source.path not in hdf5_handle:
			raise ValueError(f"HDF5 angle dataset {source.path!r} is missing")
		angles = np.asarray(hdf5_handle[source.path], dtype=np.float64)
		if source.units == "degrees":
			angles = np.deg2rad(angles)
		indices = None
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
	return indices, resolved_angles


@dataclass(frozen=True)
class SourceMetadata:
	"""One source-header probe, ready to become the run's TomoDataSet."""

	physical_shape: ProjOrder
	dtype: np.dtype
	angles: np.ndarray
	acquisition_indices: np.ndarray


def _validated_source_metadata(
	scan,
	stage: Stage,
	*,
	shape,
	dtype,
	hdf5_handle=None,
) -> SourceMetadata:
	if len(shape) != 3:
		raise ValueError("projection source must have (Theta, Y, X) dimensions")
	physical_shape = ProjOrder(*map(int, shape))
	declared = scan.projection_shape()
	if physical_shape != declared:
		if physical_shape.Theta != declared.Theta:
			raise ValueError(
				f"scan declares {declared.Theta} projections but source contains "
				f"{physical_shape.Theta}"
			)
		raise ValueError(
			f"scan declares projection shape {declared} but source header contains {physical_shape}"
		)
	if (
		isinstance(stage, TIFFStackReadStage)
		and stage.projection_count is not None
		and physical_shape.Theta != stage.projection_count
	):
		raise ValueError(
			f"read stage declares {stage.projection_count} projections but TIFF source "
			f"contains {physical_shape.Theta}"
		)
	indices, angles = _source_angles(
		scan, physical_shape.Theta, hdf5_handle=hdf5_handle,
	)
	return SourceMetadata(
		physical_shape=physical_shape,
		dtype=np.dtype(dtype),
		angles=angles,
		acquisition_indices=indices,
	)


def probe_source_metadata(scan, stage: Stage) -> SourceMetadata:
	"""Open one source file once and resolve the complete acquisition metadata."""

	source = scan.projections
	if isinstance(stage, HDF5DxchangeReadStage):
		if not isinstance(source, HDF5DxchangeProjections):
			raise TypeError("read.hdf5_dxchange requires HDF5DxchangeProjections scan metadata")
		with h5py.File(source.path, "r") as handle:
			if source.data_path not in handle:
				raise ValueError(
					f"dxchange projection dataset {source.data_path!r} is missing from {source.path}"
				)
			data = handle[source.data_path]
			return _validated_source_metadata(
				scan, stage, shape=data.shape, dtype=data.dtype, hdf5_handle=handle,
			)
	if isinstance(stage, TIFFStackReadStage):
		if not isinstance(source, TIFFStackProjections):
			raise TypeError("read.tiff_stack requires TIFFStackProjections scan metadata")
		files = stage._source_files(source)
		with tf.TiffFile(files[0]) as handle:
			page = handle.pages[0]
			count = len(files) if len(files) > 1 else len(handle.pages)
			return _validated_source_metadata(
				scan, stage, shape=(count, *page.shape), dtype=page.dtype,
			)
	raise TypeError(f"unsupported read stage for source probing: {stage.name}")


def _read_window(state: PipelineState, stage: Stage) -> TomoWindow:
	if state.dataset is None or state.source_window is None:
		raise RuntimeError("read stages require the plan-time dataset and source window")
	dataset = state.dataset
	planned_source = state.source_window.with_pad(["Y"])
	theta_stop = (
		dataset.physical_shape.Theta
		if getattr(stage, "theta_stop", None) is None else stage.theta_stop
	)
	y_stop = (
		dataset.physical_shape.Y
		if getattr(stage, "y_stop", None) is None else stage.y_stop
	)
	expected = TomoWindow.from_bounds(
		dataset.physical_shape,
		y=(getattr(stage, "y_start", 0), y_stop),
		theta=(
			getattr(stage, "theta_start", 0), theta_stop,
			getattr(stage, "theta_stride", 1),
		),
		dtype=dataset.dtype,
		work_dtype=np.float32,
	)
	if planned_source.frange != expected.frange:
		raise ValueError("read selection does not match the planned canonical window")
	return planned_source


def _bind_source_state(
	state: PipelineState,
	working,
	read_window: TomoWindow,
) -> PipelineState:
	if state.dataset is None or state.source_window is None:
		raise RuntimeError("read stages require the plan-time dataset and source window")
	shape = read_window.frange.length.to(type(working.shape))
	if working.shape != shape or np.dtype(working.dtype) != np.dtype(state.dataset.dtype):
		raise ValueError("read allocation disagrees with the planned source window")
	result = state.with_array(working, domain=DataDomain.RAW)
	if result.dataset is not state.dataset or result.source_window is not state.source_window:
		raise AssertionError("read replaced canonical plan geometry")
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
		read_window = _read_window(state, self)
		dataset = state.dataset
		layout = ProjOrder if self.order == "projection" else SinoOrder
		shape = read_window.frange.length.to(layout)

		with h5py.File(source.path, "r") as handle:
			if source.data_path not in handle:
				raise ValueError(
					f"dxchange projection dataset {source.data_path!r} is missing from {source.path}"
				)
			data = handle[source.data_path]
			if data.ndim != 3:
				raise ValueError(f"{source.data_path!r} must have (Theta, Y, X) dimensions")
			if ProjOrder(*data.shape) != dataset.physical_shape:
				raise ValueError(
					"HDF5 source shape changed after the plan-time metadata probe"
				)
			if np.dtype(data.dtype) != np.dtype(dataset.dtype):
				raise ValueError("HDF5 source dtype changed after the plan-time metadata probe")
			memory = _storage(state).allocate(
				shape, dataset.dtype, role="raw_staging",
				reuse_from=("raw_staging", "reconstruction"),
			)
			target = memory.array()
			selection = read_window.index
			if layout is ProjOrder and data.chunks is None and data.compression is None:
				data.read_direct(target, source_sel=selection)
				read_strategy = "direct"
			else:
				read_strategy = "bounded_scratch"
				selected_count = shape.Theta
				chunk = min(32, selected_count)
				theta_slice = read_window.axis_slice("Theta")
				y_slice = read_window.axis_slice("Y")
				for offset in range(0, selected_count, chunk):
					count = min(chunk, selected_count - offset)
					start = theta_slice.start + offset * theta_slice.step
					stop = start + count * theta_slice.step
					scratch = np.empty(
						(count, shape.Y, dataset.physical_shape.X), dtype=dataset.dtype,
					)
					data.read_direct(
						scratch,
						source_sel=np.s_[start:stop:theta_slice.step, y_slice, :],
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
					y_slice = read_window.axis_slice("Y")
					for artifact_name, path in (
						("white_fields", source.white_path), ("dark_fields", source.dark_path),
					):
						calibration = handle[path]
						artifact_shape = GenericOrder(
							calibration.shape[0], shape.Y, calibration.shape[2],
						)
						artifact = _storage(state).allocate(
							artifact_shape, calibration.dtype, role=artifact_name,
						)
						calibration.read_direct(
							artifact.array(),
							source_sel=np.s_[:, y_slice, :],
						)
						artifacts[artifact_name] = artifact
		result = _bind_source_state(state, memory, read_window)
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
		read_window = _read_window(state, self)
		dataset = state.dataset
		layout = ProjOrder if self.order == "projection" else SinoOrder
		shape = read_window.frange.length.to(layout)
		memory = _storage(state).allocate(
			shape, dataset.dtype, role="raw_staging",
			reuse_from=("raw_staging", "reconstruction"),
		)
		kwargs = {} if self.threads is None else {"thread_max": self.threads}
		if layout is ProjOrder:
			distribute_tiff_read(files, memory, read_window, **kwargs)
		else:
			distribute_tiff_sinogram_read(files, memory, read_window, **kwargs)
		return _bind_source_state(state, memory, read_window)
