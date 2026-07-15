"""AC-7 three-arm benchmark for sinogram-local layout strategies.

Run one arm per process so peak RSS and large memmap storage remain isolated.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
import time

import numpy as np
import psutil

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import DataDomain, Pipeline, PipelineState
from lftomo.pipeline.stages import VoDeadStripeStage, VoLargeStripeStage, VoSortStripeStage
from lftomo.util.shapes import ProjOrder, SinoOrder


@dataclass(frozen=True)
class BenchmarkResult:
	arm: str
	shape: tuple[int, int, int]
	bytes: int
	workers: int
	storage: str
	seconds: float
	peak_rss_bytes: int
	peak_rss_delta_bytes: int
	fingerprint: tuple[float, float, float]


class _PeakRSS:
	def __enter__(self):
		self.process = psutil.Process()
		self.baseline = self.process.memory_info().rss
		self.peak = self.baseline
		self.stop = threading.Event()
		def sample():
			while not self.stop.wait(0.01):
				self.peak = max(self.peak, self.process.memory_info().rss)
		self.thread = threading.Thread(target=sample, daemon=True)
		self.thread.start()
		return self

	def __exit__(self, *_args):
		self.stop.set()
		self.thread.join()
		self.peak = max(self.peak, self.process.memory_info().rss)


def _scan(theta: int) -> Scan:
	return Scan(
		energy_keV=20, pixel_size_um=1, n_projections=theta, degrees_per_fov=180,
		detector=Detector(pixel_size_um=1),
		acquisition=Acquisition(n_projections=theta, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="benchmark"),
	)


def _stages():
	return [
		VoDeadStripeStage(snr=3, size=51, norm=False),
		VoLargeStripeStage(snr=3, size=51, drop_ratio=0.1, norm=True),
		VoSortStripeStage(size=21),
	]


def _allocate(shape: tuple[int, int, int], scratch: Path | None, arm: str) -> np.ndarray:
	byte_count = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
	if scratch is None and byte_count > 1024 ** 3:
		raise MemoryError(
			"arrays over 1 GiB require --scratch so the benchmark uses memmap storage"
		)
	if scratch is None:
		array = np.empty(shape, dtype=np.float32)
	else:
		scratch.mkdir(parents=True, exist_ok=True)
		array = np.memmap(scratch / f"{arm}-projection.float32", mode="w+", dtype=np.float32, shape=shape)
	rng = np.random.default_rng(91)
	for theta in range(shape[0]):
		array[theta] = 1 + 0.02 * rng.standard_normal(shape[1:], dtype=np.float32)
		array[theta, :, shape[2] // 3] *= 3
	if isinstance(array, np.memmap):
		array.flush()
	return array


def _state(array: np.ndarray, layout: type) -> PipelineState:
	theta = array.shape[0] if layout is ProjOrder else array.shape[1]
	return PipelineState(
		working_array=array, scan=_scan(theta),
		recon=Recon(pipeline=[StageConfig(kind="benchmark")]),
		layout=layout, domain=DataDomain.ATTENUATION,
	)


def _fingerprint(array: np.ndarray, layout: type) -> tuple[float, float, float]:
	"""Sample identical projection-order coordinates without copying the volume."""

	canonical_shape = array.shape if layout is ProjOrder else (
		array.shape[1], array.shape[0], array.shape[2],
	)
	value_count = int(np.prod(canonical_shape, dtype=np.int64))
	indices = np.linspace(0, value_count - 1, min(4096, value_count)).astype(np.int64)
	theta, row, column = np.unravel_index(indices, canonical_shape)
	values = array[theta, row, column] if layout is ProjOrder else array[row, theta, column]
	sample = np.asarray(values, dtype=np.float64)
	return float(sample.mean()), float(sample.std()), float(np.linalg.norm(sample))


def _run_direct_rows(
	array: np.ndarray,
	layout: type,
	stages: list,
	state: PipelineState,
	workers: int,
) -> None:
	"""Run a sinogram-local chain with equal row concurrency for direct arms."""

	row_count = array.shape[1] if layout is ProjOrder else array.shape[0]

	def process(row: int) -> None:
		sinogram = array[:, row, :] if layout is ProjOrder else array[row]
		for stage in stages:
			stage.run_sinogram(sinogram, state)

	if workers == 1:
		for row in range(row_count):
			process(row)
	else:
		with ThreadPoolExecutor(max_workers=min(workers, row_count)) as pool:
			list(pool.map(process, range(row_count)))


def _close_memmap(array: np.ndarray) -> None:
	"""Release obsolete disk-backed source pages after a layout transition."""

	if isinstance(array, np.memmap):
		array._mmap.close()


def run(arm: str, shape: tuple[int, int, int], workers: int = 1,
		scratch: Path | None = None) -> BenchmarkResult:
	"""Run one isolated AC-7 arm; ``shape`` is ``(Theta, Y, X)``."""

	if arm not in {"bounded", "transition", "strided"}:
		raise ValueError("arm must be bounded, transition, or strided")
	if len(shape) != 3 or min(shape) < 1:
		raise ValueError("shape must contain three positive dimensions")
	if workers < 1:
		raise ValueError("workers must be positive")
	projection = _allocate(shape, scratch, arm)
	stages = _stages()
	with _PeakRSS() as memory:
		started = time.perf_counter()
		if arm == "bounded":
			Pipeline(
				stages, scan=_scan(shape[0]), initial_layout=ProjOrder,
				initial_domain=DataDomain.ATTENUATION, bounded_workers=workers,
			).run(_state(projection, ProjOrder))
			result = projection
		elif arm == "transition":
			if scratch is None:
				sinogram = np.transpose(projection, (1, 0, 2)).copy()
			else:
				sinogram = np.memmap(
					scratch / f"{arm}-sinogram.float32", mode="w+", dtype=np.float32,
					shape=(shape[1], shape[0], shape[2]),
				)
				for theta in range(shape[0]):
					sinogram[:, theta, :] = projection[theta]
				sinogram.flush()
			# The source volume is obsolete after an explicit transition. Closing its
			# mapping keeps the processing phase to one resident volume while the RSS
			# sampler still records the materialization peak above.
			_close_memmap(projection)
			del projection
			state = _state(sinogram, SinoOrder)
			_run_direct_rows(sinogram, SinoOrder, stages, state, workers)
			result = sinogram
		else:
			state = _state(projection, ProjOrder)
			_run_direct_rows(projection, ProjOrder, stages, state, workers)
			result = projection
		seconds = time.perf_counter() - started
	return BenchmarkResult(
		arm=arm, shape=shape, bytes=int(np.prod(shape)) * 4, workers=workers,
		storage="memmap" if scratch is not None else "memory",
		seconds=seconds, peak_rss_bytes=memory.peak,
		peak_rss_delta_bytes=memory.peak - memory.baseline,
		fingerprint=_fingerprint(result, SinoOrder if arm == "transition" else ProjOrder),
	)


def _shape(value: str) -> tuple[int, int, int]:
	try:
		shape = tuple(int(item) for item in value.split(","))
	except ValueError as error:
		raise argparse.ArgumentTypeError("shape must be Theta,Y,X") from error
	if len(shape) != 3 or min(shape) < 1:
		raise argparse.ArgumentTypeError("shape must contain three positive dimensions")
	return shape


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--arm", required=True, choices=("bounded", "transition", "strided"))
	parser.add_argument("--shape", required=True, type=_shape, help="Theta,Y,X")
	parser.add_argument("--workers", type=int, default=1)
	parser.add_argument("--scratch", type=Path)
	args = parser.parse_args()
	print(json.dumps(asdict(run(args.arm, args.shape, args.workers, args.scratch)), indent=2))


if __name__ == "__main__":
	main()
