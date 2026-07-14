"""Reconstruction output stages."""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tifffile as tf

from lftomo.config import StageConfig
from lftomo.pipeline.core import PipelineState, ResourceHints, Stage, StageContract
from lftomo.util.io import np_convert
from lftomo.util.shapes import ReconOrder


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
		window_axes=("Y",),
		produces_artifacts=("output_files",),
		resources=ResourceHints(memory_multiplier=1.0),
	)

	def __init__(self, dtype: str = "float32",
				naming_template: str = "{sample_id}_{slice:05d}",
				processes: int = 1, sample_id: str | None = None,
				slice_start: int = 0):
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
		working = np.asarray(state.working_array)
		if working.ndim != 3:
			raise ValueError("write.tiff_slices expects a (Y, Z, X) reconstruction array")

		output = Path(state.recon.output_path)
		output.mkdir(parents=True, exist_ok=True)
		sample_id = self._sample_id(state)
		paths = []
		for local_index in range(working.shape[0]):
			slice_index = self.slice_start + local_index
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
