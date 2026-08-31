"""Opt-in SciPy/Bottleneck Vo benchmark for issue #209.

Both arms use the real manager-owned work planes and process dispatcher.
``--plan-only`` reports exact scratch bytes without allocating a potentially
large volume.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import threading
import time

import numpy as np
import psutil

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import DataDomain, Pipeline, PipelineRunContext, PipelineState
from lftomo.pipeline.stages import VoDeadStripeStage, VoLargeStripeStage, VoSortStripeStage
from lftomo.pipeline.stages.stripe import _vo_work_planes
from lftomo.prep.stripe import _remove_large_array, _remove_sort_array
from lftomo.util.shapes import ProjOrder


@dataclass(frozen=True)
class VoBenchmarkResult:
	variant: str
	operation: str
	shape: tuple[int, int, int]
	workers: int
	plan_only: bool
	input_bytes: int
	managed_scratch_bytes: int
	float_work_bytes: int
	index_work_bytes: int
	median_pad_work_bytes: int
	seconds: float | None
	private_bytes_before: int | None
	private_bytes_peak: int | None
	private_bytes_after: int | None
	fingerprint: tuple[float, float, float] | None


def scratch_report(
	shape: tuple[int, int, int], dtype=np.float32, *, median_pad_half_width: int = 0,
) -> dict[str, int]:
	"""Return exact rows-proportional Vo scratch allocation for ``Theta,Y,X``."""

	theta, rows, columns = shape
	items = int(theta) * int(rows) * int(columns)
	float_bytes = items * np.dtype(dtype).itemsize
	index_bytes = items * np.dtype(np.uint32).itemsize
	median_pad_bytes = (
		int(rows) * int(theta) * (int(columns) + 2 * median_pad_half_width)
		* np.dtype(dtype).itemsize
		if median_pad_half_width else 0
	)
	return {
		"input_bytes": float_bytes,
		"float_work_bytes": float_bytes,
		"index_work_bytes": index_bytes,
		"median_pad_work_bytes": median_pad_bytes,
		"managed_scratch_bytes": float_bytes + index_bytes + median_pad_bytes,
	}


def _private_bytes(process: psutil.Process) -> int:
	"""Sum USS for this process tree, falling back to RSS where unavailable."""

	total = 0
	for item in (process, *process.children(recursive=True)):
		try:
			info = item.memory_full_info()
			total += int(getattr(info, "uss", info.rss))
		except (psutil.AccessDenied, psutil.NoSuchProcess):
			continue
	return total


class _PeakPrivate:
	def __enter__(self):
		self.process = psutil.Process(os.getpid())
		self.before = _private_bytes(self.process)
		self.peak = self.before
		self.stop = threading.Event()

		def sample():
			while not self.stop.wait(0.02):
				self.peak = max(self.peak, _private_bytes(self.process))

		self.thread = threading.Thread(target=sample, daemon=True)
		self.thread.start()
		return self

	def __exit__(self, *_args):
		self.stop.set()
		self.thread.join()
		self.after = _private_bytes(self.process)
		self.peak = max(self.peak, self.after)


def _scan(shape: tuple[int, int, int]) -> Scan:
	theta, rows, columns = shape
	return Scan(
		energy_keV=20, pixel_size_um=1, n_projections=theta, degrees_per_fov=180,
		detector=Detector(pixel_size_um=1, rows=rows, columns=columns),
		acquisition=Acquisition(n_projections=theta, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="benchmark"),
	)


class _ScipyVoSortStage(VoSortStripeStage):
	"""Pre-#209 reusable-work stage retaining the SciPy line median."""

	@property
	def vo_line_median_half_width(self) -> int:
		return 0

	def run_sinogram(self, sinogram, state):
		work, index_work, _pad_work = _vo_work_planes(state, sinogram)
		_remove_sort_array(
			sinogram, size=self.size, dim=self.dim, work=work,
			index_work=index_work, use_bottleneck=False,
		)


class _ScipyVoLargeStage(VoLargeStripeStage):
	"""Pre-#209 reusable-work stage retaining the SciPy line median."""

	@property
	def vo_line_median_half_width(self) -> int:
		return 0

	def run_sinogram(self, sinogram, state):
		work, index_work, _pad_work = _vo_work_planes(state, sinogram)
		_remove_large_array(
			sinogram, snr=self.snr, size=self.size, drop_ratio=self.drop_ratio,
			norm=self.norm, work=work, index_work=index_work,
			use_bottleneck=False,
		)


def _stages(operation: str, *, bottleneck: bool):
	dead = VoDeadStripeStage
	large = VoLargeStripeStage if bottleneck else _ScipyVoLargeStage
	sort = VoSortStripeStage if bottleneck else _ScipyVoSortStage
	available = {
		"dead": [dead(snr=3, size=51, norm=False)],
		"large": [large(snr=3, size=51, drop_ratio=0.1, norm=True)],
		"sort": [sort(size=21)],
		"chain": [
			dead(snr=3, size=51, norm=False),
			large(snr=3, size=51, drop_ratio=0.1, norm=True),
			sort(size=21),
		],
	}
	try:
		stages = available[operation]
	except KeyError as error:
		raise ValueError("operation must be dead, large, sort, or chain") from error
	return stages


def _fill_fixture(values: np.ndarray) -> None:
	rng = np.random.default_rng(198)
	for theta in range(values.shape[0]):
		values[theta] = 1 + 0.02 * rng.standard_normal(values.shape[1:], dtype=np.float32)
		values[theta, :, values.shape[2] // 3] *= 3


def _fingerprint(values: np.ndarray) -> tuple[float, float, float]:
	count = int(np.prod(values.shape, dtype=np.int64))
	indices = np.linspace(0, count - 1, min(4096, count)).astype(np.int64)
	sample = np.asarray(values.reshape(-1)[indices], dtype=np.float64)
	return float(sample.mean()), float(sample.std()), float(np.linalg.norm(sample))


def run(
	variant: str,
	operation: str,
	shape: tuple[int, int, int],
	workers: int = 2,
	*,
	plan_only: bool = False,
) -> VoBenchmarkResult:
	"""Run or plan one isolated benchmark arm."""

	if variant not in {"scipy", "bottleneck"}:
		raise ValueError("variant must be scipy or bottleneck")
	if len(shape) != 3 or min(shape) < 1:
		raise ValueError("shape must contain positive Theta,Y,X dimensions")
	if workers < 1:
		raise ValueError("workers must be positive")
	stages = _stages(operation, bottleneck=variant == "bottleneck")
	median_half_width = max(
		(int(getattr(stage, "vo_line_median_half_width", 0)) for stage in stages),
		default=0,
	)
	bytes_report = scratch_report(
		shape, median_pad_half_width=median_half_width,
	)
	managed = bytes_report["managed_scratch_bytes"]
	if plan_only:
		return VoBenchmarkResult(
			variant, operation, shape, workers, True,
			bytes_report["input_bytes"], managed,
			bytes_report["float_work_bytes"], bytes_report["index_work_bytes"],
			bytes_report["median_pad_work_bytes"],
			None, None, None, None, None,
		)

	scan = _scan(shape)
	recon = Recon(pipeline=[StageConfig(kind="benchmark")], process_workers=workers)
	with PipelineRunContext(max_workers=workers) as context:
		working = context.storage.allocate(ProjOrder(*shape), np.float32, role="normalized")
		_fill_fixture(working.array())
		state = PipelineState(
			working, scan, recon, storage=context.storage, executor=context.executor,
			domain=DataDomain.ATTENUATION,
		)
		pipeline = Pipeline(
			stages, scan=scan,
			initial_layout=ProjOrder, initial_domain=DataDomain.ATTENUATION,
			bounded_workers=workers,
		)
		with _PeakPrivate() as memory:
			started = time.perf_counter()
			result = pipeline.run(state)
			seconds = time.perf_counter() - started
		fingerprint = _fingerprint(np.asarray(result.working_array))
	return VoBenchmarkResult(
		variant, operation, shape, workers, False,
		bytes_report["input_bytes"], managed,
		bytes_report["float_work_bytes"], bytes_report["index_work_bytes"],
		bytes_report["median_pad_work_bytes"],
		seconds, memory.before, memory.peak, memory.after, fingerprint,
	)


def _shape(value: str) -> tuple[int, int, int]:
	try:
		shape = tuple(int(item) for item in value.split(","))
	except ValueError as error:
		raise argparse.ArgumentTypeError("shape must be Theta,Y,X") from error
	if len(shape) != 3 or min(shape) < 1:
		raise argparse.ArgumentTypeError("shape must contain positive Theta,Y,X dimensions")
	return shape


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--variant", required=True, choices=("scipy", "bottleneck"))
	parser.add_argument("--operation", default="chain", choices=("dead", "large", "sort", "chain"))
	parser.add_argument("--shape", required=True, type=_shape, help="Theta,Y,X")
	parser.add_argument("--workers", type=int, default=2)
	parser.add_argument("--plan-only", action="store_true")
	args = parser.parse_args()
	print(json.dumps(asdict(run(
		args.variant, args.operation, args.shape, args.workers,
		plan_only=args.plan_only,
	)), indent=2))


if __name__ == "__main__":
	main()
