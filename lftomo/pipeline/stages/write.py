"""Reconstruction output stages."""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tifffile as tf
import h5py

from lftomo.config import StageConfig
from lftomo.pipeline.core import (
	PipelineState,
	RuntimeRequirement,
	Stage,
	StageContract,
	working_view,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.util.io import np_convert
from lftomo.util.shapes import ReconOrder


_CLOUDVOLUME_RUNTIME_REQUIREMENT = RuntimeRequirement(
	capability="neuroglancer",
	package="cloud-volume",
	module="cloudvolume",
	probe_attributes=("CloudVolume",),
	corrective_action="install lftomo with the neuroglancer extra",
)


_OUTPUT_DTYPES = {
	"float32": np.dtype(np.float32),
	"int16": np.dtype(np.int16),
	"uint16": np.dtype(np.uint16),
}


def _write_array_slice(array: np.ndarray, path: Path, dtype: np.dtype) -> None:
	"""Write one independent slice; suitable for multiprocessing workers."""
	values = array.astype(np.float32, copy=False) if dtype == np.dtype(np.float32) else np_convert(dtype, array)
	tf.imwrite(path, values)


class TIFFSlicesWriteStage(Stage):
	"""Write one TIFF per first-axis reconstruction slice."""

	name = "write.tiff_slices"
	contract = StageContract(
		expected_input=ReconOrder,
		produced_output=ReconOrder,
		produces=PRESERVE,
		accepts_domain=(DataDomain.RECON,),
		produces_domain=PRESERVE,
		window_axes=("Y",),
		produces_artifacts=("output_files",),
	)

	def __init__(self, dtype: str = "float32",
				naming_template: str = "{sample_id}_{slice:05d}",
				processes: int = 1, sample_id: str | None = None,
				slice_start: int = 0, write_full: bool = True,
				write_bin2: bool = False, write_bin4: bool = False):
		try:
			self.dtype = _OUTPUT_DTYPES[dtype]
		except KeyError as error:
			raise ValueError("TIFF dtype must be float32, int16, or uint16") from error
		if processes < 1:
			raise ValueError("processes must be positive")
		self.naming_template = naming_template
		self.processes = processes
		self.sample_id = sample_id
		self.slice_start = slice_start
		if not write_full or write_bin2 or write_bin4:
			raise ValueError("TIFF writer currently supports full-resolution output only")

	@classmethod
	def from_config(cls, config: StageConfig) -> "TIFFSlicesWriteStage":
		return cls(**config.params)

	def _sample_id(self, state: PipelineState) -> str:
		return str(
			self.sample_id
			or state.recon.metadata.get("sample_id")
			or state.scan.metadata.get("sample_id")
			or "recon"
		)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("TIFFSlicesWriteStage currently requires the NumPy backend")
		if state.recon.output_path is None:
			raise ValueError("write.tiff_slices requires Recon.output_path")
		working = working_view(state)
		if working.ndim != 3:
			raise ValueError("write.tiff_slices expects a (Y, Z, X) reconstruction array")

		output = Path(state.recon.output_path)
		output.mkdir(parents=True, exist_ok=True)
		sample_id = self._sample_id(state)
		source_y = 0 if state.source_window is None else int(state.source_window.pos.Y)
		paths = []
		for local_index in range(working.shape[0]):
			slice_index = self.slice_start + source_y + local_index
			name = self.naming_template.format(sample_id=sample_id, slice=slice_index)
			paths.append(output / (name if name.lower().endswith((".tif", ".tiff")) else f"{name}.tif"))

		calls = [(working[index], path, self.dtype) for index, path in enumerate(paths)]
		if self.processes == 1:
			for call in calls:
				_write_array_slice(*call)
		else:
			with Pool(self.processes) as pool:
				pool.starmap(_write_array_slice, calls)
		state.artifacts["output_files"] = paths
		return state


class HDF5VolumeWriteStage(Stage):
	"""Write the complete reconstruction array to one HDF5 dataset."""

	name = "write.hdf5_volume"
	contract = StageContract(
		expected_input=ReconOrder,
		produced_output=ReconOrder,
		produces=PRESERVE,
		accepts_domain=(DataDomain.RECON,),
		produces_domain=PRESERVE,
		window_axes=("Y",),
		produces_artifacts=("output_files",),
	)

	def __init__(self, path: str | Path | None = None, dataset: str = "reconstruction",
				compression: str | None = "gzip", compression_opts: int | None = 4,
				chunks: bool | tuple[int, int, int] = True, overwrite: bool = False):
		self.path = None if path is None else Path(path)
		self.dataset = dataset
		self.compression = compression
		self.compression_opts = compression_opts
		self.chunks = chunks
		self.overwrite = overwrite

	@classmethod
	def from_config(cls, config: StageConfig) -> "HDF5VolumeWriteStage":
		return cls(**config.params)

	def _path(self, state: PipelineState) -> Path:
		path = self.path or state.recon.output_path
		if path is None:
			raise ValueError("write.hdf5_volume requires a path or Recon.output_path")
		path = Path(path)
		return path if path.suffix.lower() in (".h5", ".hdf5") else path / "reconstruction.h5"

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("HDF5VolumeWriteStage currently requires the NumPy backend")
		data = working_view(state)
		if data.ndim != 3:
			raise ValueError("write.hdf5_volume expects a (Y, Z, X) reconstruction array")
		path = self._path(state)
		path.parent.mkdir(parents=True, exist_ok=True)
		if path.exists() and not self.overwrite:
			raise FileExistsError(f"HDF5 output already exists: {path}")
		with h5py.File(path, "w") as handle:
			handle.create_dataset(
				self.dataset,
				data=data,
				compression=self.compression,
				compression_opts=self.compression_opts if self.compression else None,
				chunks=self.chunks,
			)
		state.artifacts["output_files"] = [path]
		return state


class NeuroglancerWriteStage(Stage):
	"""Write a Neuroglancer precomputed mip-0 image layer.

	Adapted with attribution from ``ReconChengLab/Neuroglancer/ct_to_ng.py``.
	Optional mip generation remains outside this Stage so writing has no hidden
	task-queue side effects.
	"""

	name = "write.neuroglancer"
	contract = StageContract(
		expected_input=ReconOrder,
		produced_output=ReconOrder,
		produces=PRESERVE,
		accepts_domain=(DataDomain.RECON,),
		produces_domain=PRESERVE,
		window_axes=("Y",),
		produces_artifacts=("output_files", "neuroglancer_source"),
		runtime_requirements=(_CLOUDVOLUME_RUNTIME_REQUIREMENT,),
	)

	def __init__(self, path: str | Path | None = None, layer_name: str = "reconstruction",
				resolution_nm_xyz: tuple[int, int, int] | list[int] | None = None,
				chunk_xyz: tuple[int, int, int] | list[int] = (64, 64, 64),
				block_slices: int = 16, overwrite: bool = False,
				data_type: str | None = None):
		if block_slices < 1 or any(int(value) < 1 for value in chunk_xyz):
			raise ValueError("Neuroglancer block and chunk sizes must be positive")
		self.path = None if path is None else Path(path)
		self.layer_name = layer_name
		self.resolution_nm_xyz = None if resolution_nm_xyz is None else tuple(resolution_nm_xyz)
		self.chunk_xyz = tuple(chunk_xyz)
		self.block_slices = block_slices
		self.overwrite = overwrite
		self.data_type = data_type

	@classmethod
	def from_config(cls, config: StageConfig) -> "NeuroglancerWriteStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("NeuroglancerWriteStage currently requires the NumPy backend")
		data = working_view(state)
		if data.ndim != 3:
			raise ValueError("write.neuroglancer expects a (Y, Z, X) reconstruction array")
		root = self.path or state.recon.output_path
		if root is None:
			raise ValueError("write.neuroglancer requires a path or Recon.output_path")
		layer = Path(root) / self.layer_name
		if layer.exists():
			if not self.overwrite:
				raise FileExistsError(f"Neuroglancer layer already exists: {layer}")
			import shutil

			shutil.rmtree(layer)
		layer.mkdir(parents=True)

		try:
			from cloudvolume import CloudVolume
		except ImportError as error:
			raise RuntimeError("write.neuroglancer requires the neuroglancer optional dependencies") from error
		resolution = self.resolution_nm_xyz or (int(state.scan.pixel_size_um * 1000),) * 3
		volume_size = (data.shape[2], data.shape[1], data.shape[0])
		info = CloudVolume.create_new_info(
			num_channels=1,
			layer_type="image",
			data_type=self.data_type or data.dtype.name,
			encoding="raw",
			resolution=list(resolution),
			voxel_offset=[0, 0, 0],
			chunk_size=list(self.chunk_xyz),
			volume_size=list(volume_size),
		)
		cloudpath = f"precomputed://file://{layer.as_posix()}"
		volume = CloudVolume(
			cloudpath, info=info, mip=0, compress=False, progress=False,
			parallel=1, cache=False, fill_missing=False, bounded=True,
		)
		volume.commit_info()
		for start in range(0, data.shape[0], self.block_slices):
			stop = min(data.shape[0], start + self.block_slices)
			volume[:, :, start:stop] = np.transpose(data[start:stop], (2, 1, 0))
		state.artifacts["output_files"] = [layer / "info"]
		state.artifacts["neuroglancer_source"] = cloudpath
		return state
