"""Projection input stages."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from natsort import natsorted
import numpy as np
import tifffile as tf

from lftomo.config import StageConfig, TIFFStackProjections
from lftomo.pipeline.core import PipelineState, ResourceHints, Stage, StageContract
from lftomo.util.io import distribute_tiff_read
from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import ProjOrder, SinoOrder


class TIFFStackReadStage(Stage):
	"""Read a naturally sorted TIFF stack through lftomo's direct-byte reader."""

	name = "read.tiff_stack"

	def __init__(self, order: str = "projection", threads: int | None = None,
				y_start: int = 0, y_stop: int | None = None):
		if order not in ("projection", "sinogram"):
			raise ValueError("TIFF read order must be 'projection' or 'sinogram'")
		self.order = order
		self.threads = threads
		self.y_start = y_start
		self.y_stop = y_stop
		output = ProjOrder if order == "projection" else SinoOrder
		self.contract = StageContract(
			produced_output=output,
			window_axes=("Y",),
			required_metadata=("projections.path", "projections.pattern"),
			resources=ResourceHints(memory_multiplier=2.0),
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "TIFFStackReadStage":
		return cls(**config.params)

	def _source_files(self, source: TIFFStackProjections) -> list[Path]:
		path = Path(source.path)
		files = natsorted(path.glob(source.pattern)) if path.is_dir() else [path]
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

		stop = rows if self.y_stop is None else self.y_stop
		if not 0 <= self.y_start < stop <= rows:
			raise ValueError(f"invalid TIFF Y window [{self.y_start}, {stop}) for {rows} rows")
		shape = ProjOrder(Theta=count, Y=stop - self.y_start, X=columns)
		memory = SharedNPArray(f"tiff-read-{uuid4().hex}", shape, dtype, allocate=True)
		try:
			kwargs = {} if self.threads is None else {"thread_max": self.threads}
			distribute_tiff_read(
				files, memory, y_start=self.y_start, y_stop=stop, **kwargs
			)
			with memory as values:
				working = values.copy()
		finally:
			memory.shutdown()

		if self.order == "sinogram":
			working = np.transpose(working, (1, 0, 2)).copy()
		return state.with_array(working)
