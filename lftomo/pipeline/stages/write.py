"""Reconstruction output stages."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from lftomo.pipeline.core import (
	PipelineState,
	RuntimeRequirement,
	Stage,
	StageContract,
	StageParams,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.storage import FileBackedNPArray, ManagedBuffer
from lftomo.util.shapes import IntIter, ReconOrder, ReverseReconOrder, Y


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
	import tifffile as tf
	from lftomo.util.io import np_convert

	values = array.astype(np.float32, copy=False) if dtype == np.dtype(np.float32) else np_convert(dtype, array)
	tf.imwrite(path, values)


def _record_outputs(state: PipelineState, paths: list[Path]) -> None:
	"""Append durable files to the run's output ledger without duplicates."""

	existing = list(state.artifacts.get("output_files", ()))
	state.artifacts["output_files"] = existing + [path for path in paths if path not in existing]


def _volume_output_region(state: PipelineState) -> tuple[ReconOrder, slice]:
	"""Derive packed volume rows at the output-format boundary."""

	shape = state.working_array.shape
	if state.dataset is None or state.source_window is None:
		return shape, slice(0, shape.Y)
	full = state.dataset.full_window()
	# Projection/canvas axes may change during reconstruction; only Y places
	# reconstructed slices within the selected output volume.
	selected = full.rescope(
		pos=state.source_window.pos.extract(Y), shape=state.source_window.shape.extract(Y),
	)
	selected = replace(selected, step=replace(full.step, Y=state.source_window.step.Y))
	rows = full.packed_index(selected, Y)[0]
	return replace(shape, Y=full.packed_index(full, Y)[0].stop), rows


class TIFFSlicesWriteStage(Stage):
	"""Write one TIFF per first-axis reconstruction slice."""

	name = "write.tiff_slices"
	contract = StageContract(
		accepts=(ReconOrder,),
		produces=PRESERVE,
		accepts_domain=(DataDomain.RECON,),
		produces_domain=PRESERVE,
		window_axes=("Y",),
		produces_artifacts=("output_files",),
	)

	class Params(StageParams):
		dtype: Literal["float32", "int16", "uint16"] = "float32"
		naming_template: str = "{sample_id}_{slice:05d}"
		processes: int = Field(default=1, ge=1)
		sample_id: str | None = None
		slice_start: int = 0
		write_full: Literal[True] = True
		write_bin2: Literal[False] = False
		write_bin4: Literal[False] = False

	def _sample_id(self, state: PipelineState) -> str:
		return str(
			self.params.sample_id
			or state.recon.metadata.get("sample_id")
			or state.scan.metadata.get("sample_id")
			or "recon"
		)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("TIFFSlicesWriteStage currently requires the NumPy backend")
		if state.recon.output_path is None:
			raise ValueError("write.tiff_slices requires Recon.output_path")
		working = np.asarray(state.working_array)
		dtype = _OUTPUT_DTYPES[self.params.dtype]
		if working.ndim != 3:
			raise ValueError("write.tiff_slices expects a (Y, Z, X) reconstruction array")

		output = Path(state.recon.output_path)
		output.mkdir(parents=True, exist_ok=True)
		sample_id = self._sample_id(state)

		rows = list(
			state.source_window.int_iter("Y") if state.source_window is not None
			else Y.drange(0, working.shape[0], 1).get_iter(IntIter)
		)
		paths = []
		for source_row in rows:
			slice_index = self.params.slice_start + source_row
			name = self.params.naming_template.format(sample_id=sample_id, slice=slice_index)
			paths.append(output / (name if name.lower().endswith((".tif", ".tiff")) else f"{name}.tif"))

		executor = state.executor or getattr(state.storage, "pipeline_executor", None)
		if isinstance(state.working_array, ManagedBuffer):
			from lftomo.pipeline.execution import write_tiff_slice_kernel

			calls = [
				(
					state.working_array[state.layout.point_index(
						Y=slice(local, local + 1),
					)], path, dtype,
				)
				for local, path in enumerate(paths)
			]
			if self.params.processes > 1 and executor is not None:
				executor.submit_all(write_tiff_slice_kernel, calls)
			else:
				for call in calls:
					write_tiff_slice_kernel(*call)
		else:
			# ndarray-only hermetic tests remain serial; slices are never pickled.
			for local, path in enumerate(paths):
				index = state.layout.point_index(Y=slice(local, local + 1))
				_write_array_slice(working[index][0], path, dtype)
		_record_outputs(state, paths)
		return state


class HDF5VolumeWriteStage(Stage):
	"""Write the complete reconstruction array to one HDF5 dataset."""

	name = "write.hdf5_volume"
	contract = StageContract(
		accepts=(ReconOrder,),
		produces=PRESERVE,
		accepts_domain=(DataDomain.RECON,),
		produces_domain=PRESERVE,
		window_axes=("Y",),
		produces_artifacts=("output_files",),
	)

	class Params(StageParams):
		path: Path | None = None
		dataset: str = "reconstruction"
		compression: str | None = "gzip"
		compression_opts: int | None = 4
		chunks: bool | tuple[int, int, int] = True
		overwrite: bool = False

	def _path(self, state: PipelineState) -> Path:
		path = self.params.path or state.recon.output_path
		if path is None:
			raise ValueError("write.hdf5_volume requires a path or Recon.output_path")
		path = Path(path)
		return path if path.suffix.lower() in (".h5", ".hdf5") else path / "reconstruction.h5"

	def run(self, state: PipelineState) -> PipelineState:
		import h5py

		if state.backend.name != "numpy":
			raise TypeError("HDF5VolumeWriteStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("write.hdf5_volume expects a (Y, Z, X) reconstruction array")
		path = self._path(state)
		path.parent.mkdir(parents=True, exist_ok=True)
		if state.dataset is None or state.source_window is None:
			if path.exists() and not self.params.overwrite:
				raise FileExistsError(f"HDF5 output already exists: {path}")
			with h5py.File(path, "w") as handle:
				handle.create_dataset(
					self.params.dataset,
					data=data,
					compression=self.params.compression,
					compression_opts=self.params.compression_opts if self.params.compression else None,
					chunks=self.params.chunks,
				)
		else:
			shape, rows = _volume_output_region(state)
			first = rows.start == 0
			if first and path.exists() and not self.params.overwrite:
				raise FileExistsError(f"HDF5 output already exists: {path}")
			mode = "w" if first else "r+"
			with h5py.File(path, mode) as handle:
				if first:
					handle.create_dataset(
						self.params.dataset,
						shape=shape.as_tuple(),
						dtype=data.dtype,
						compression=self.params.compression,
						compression_opts=self.params.compression_opts if self.params.compression else None,
						chunks=self.params.chunks,
					)
				handle[self.params.dataset][rows] = data
		_record_outputs(state, [path])
		return state


class NPYVolumeWriteStage(Stage):
	"""Write a durable float32 NumPy volume, placing streamed rows in one file."""

	name = "write.npy_volume"
	contract = HDF5VolumeWriteStage.contract

	class Params(StageParams):
		path: Path | None = None
		overwrite: bool = False

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("NPYVolumeWriteStage currently requires the NumPy backend")
		root = self.params.path or state.recon.output_path
		if root is None:
			raise ValueError("write.npy_volume requires a path or Recon.output_path")
		path = Path(root)
		if path.suffix.lower() != ".npy":
			path = path / "reconstruction.npy"
		path.parent.mkdir(parents=True, exist_ok=True)
		shape, rows = _volume_output_region(state)
		if rows.start == 0:
			if path.exists() and self.params.overwrite:
				path.unlink()
			output = FileBackedNPArray.create(path, shape, np.float32)
		else:
			output = FileBackedNPArray(path, shape, mode="r+")
		with output as values:
			if output.dtype != np.dtype(np.float32):
				raise ValueError("write.npy_volume requires a float32 output volume")
			values[rows] = np.asarray(state.working_array)
		_record_outputs(state, [path])
		return state


class NeuroglancerWriteStage(Stage):
	"""Write a Neuroglancer precomputed mip-0 image layer.

	Adapted with attribution from ``ReconChengLab/Neuroglancer/ct_to_ng.py``.
	Optional mip generation remains outside this Stage so writing has no hidden
	task-queue side effects.
	"""

	name = "write.neuroglancer"
	contract = StageContract(
		accepts=(ReconOrder,),
		produces=PRESERVE,
		accepts_domain=(DataDomain.RECON,),
		produces_domain=PRESERVE,
		window_axes=("Y",),
		produces_artifacts=("output_files", "neuroglancer_source"),
		runtime_requirements=(_CLOUDVOLUME_RUNTIME_REQUIREMENT,),
	)

	class Params(StageParams):
		path: Path | None = None
		layer_name: str = "reconstruction"
		resolution_nm_xyz: tuple[int, ...] | None = None
		chunk_xyz: tuple[Annotated[int, Field(ge=1)], ...] = (64, 64, 64)
		block_slices: int = Field(default=16, ge=1)
		overwrite: bool = False
		data_type: str | None = None

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("NeuroglancerWriteStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("write.neuroglancer expects a (Y, Z, X) reconstruction array")
		root = self.params.path or state.recon.output_path
		if root is None:
			raise ValueError("write.neuroglancer requires a path or Recon.output_path")
		layer = Path(root) / self.params.layer_name
		shape, rows = _volume_output_region(state)
		output_start = rows.start
		first = output_start == 0
		if first and layer.exists():
			if not self.params.overwrite:
				raise FileExistsError(f"Neuroglancer layer already exists: {layer}")
			import shutil

			shutil.rmtree(layer)
		if first:
			layer.mkdir(parents=True, exist_ok=True)

		try:
			from cloudvolume import CloudVolume
		except ImportError as error:
			raise RuntimeError("write.neuroglancer requires the neuroglancer optional dependencies") from error
		resolution = self.params.resolution_nm_xyz or (int(state.scan.pixel_size_um * 1000),) * 3
		cloudpath = f"precomputed://file://{layer.as_posix()}"
		if first:
			volume_size = (shape.X, shape.Z, shape.Y)
			info = CloudVolume.create_new_info(
				num_channels=1,
				layer_type="image",
				data_type=self.params.data_type or data.dtype.name,
				encoding="raw",
				resolution=list(resolution),
				voxel_offset=[0, 0, 0],
				chunk_size=list(self.params.chunk_xyz),
				volume_size=list(volume_size),
			)
			volume = CloudVolume(
				cloudpath, info=info, mip=0, compress=False, progress=False,
				parallel=1, cache=False, fill_missing=False, bounded=True,
			)
			volume.commit_info()
		else:
			volume = CloudVolume(
				cloudpath, mip=0, compress=False, progress=False,
				parallel=1, cache=False, fill_missing=False, bounded=True,
			)
		for position in Y.drange(0, data.shape[0], self.params.block_slices):
			block = data[position.Y:position.Y + self.params.block_slices]
			start = output_start + position.Y
			volume[:, :, start:start + block.shape[0]] = ReconOrder(*block.shape).transposed_view(
				block, ReverseReconOrder,
			)
		_record_outputs(state, [layer / "info"])
		state.artifacts["neuroglancer_source"] = cloudpath
		return state
