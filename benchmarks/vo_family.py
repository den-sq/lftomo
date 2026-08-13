"""Opt-in Vo-family kernel/scratch benchmark for issue #198.

The execution arm uses the real manager-owned buffer and process dispatcher.
Use ``--variant legacy`` to exercise allocation-per-kernel fallbacks or
``--variant reusable`` for the row-indexed managed work planes.  ``--plan-only``
reports exact scratch bytes without allocating a potentially large volume.
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
from scipy.ndimage import binary_dilation, median_filter, uniform_filter1d
from scipy.signal import medfilt2d

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import DataDomain, Pipeline, PipelineRunContext, PipelineState
from lftomo.pipeline.stages import VoDeadStripeStage, VoLargeStripeStage, VoSortStripeStage
from lftomo.prep.stripe import _detect_stripe
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
	seconds: float | None
	private_bytes_before: int | None
	private_bytes_peak: int | None
	private_bytes_after: int | None
	fingerprint: tuple[float, float, float] | None


def scratch_report(shape: tuple[int, int, int], dtype=np.float32) -> dict[str, int]:
	"""Return exact rows-proportional Vo scratch allocation for ``Theta,Y,X``."""

	theta, rows, columns = shape
	items = int(theta) * int(rows) * int(columns)
	float_bytes = items * np.dtype(dtype).itemsize
	index_bytes = items * np.dtype(np.uint32).itemsize
	return {
		"input_bytes": float_bytes,
		"float_work_bytes": float_bytes,
		"index_work_bytes": index_bytes,
		"managed_scratch_bytes": float_bytes + index_bytes,
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


class _BaselineVoSortStage(VoSortStripeStage):
	"""The allocation-per-call implementation from lftomo@dd4ce95."""

	uses_vo_work = False

	def run_sinogram(self, sinogram, _state):
		shape = (self.size, 1 if self.dim == 1 else self.size)
		modified = sinogram.T.copy()
		indices = modified.argsort(axis=1)
		modified[:] = np.take_along_axis(modified, indices, axis=1)
		indices[:] = indices.argsort(axis=1)
		modified[:] = medfilt2d(modified, shape)
		modified[:] = np.take_along_axis(modified, indices, axis=1)
		sinogram[:] = modified.T


class _BaselineVoLargeStage(VoLargeStripeStage):
	"""The allocation-per-call implementation from lftomo@dd4ce95."""

	uses_vo_work = False

	def run_sinogram(self, sinogram_base, _state):
		sinogram = sinogram_base.copy()
		drop_ratio = np.clip(self.drop_ratio, 0.0, 0.8)
		rows = sinogram.shape[0]
		drop = int(0.5 * drop_ratio * rows)
		kept = slice(drop, -drop if drop else None)
		indices = sinogram.argsort(axis=0)
		smoothed = np.take_along_axis(sinogram, indices, axis=0)
		sorted_mean = np.mean(smoothed[kept, :], axis=0)
		smoothed[:] = medfilt2d(smoothed, (1, self.size))
		smooth_mean = np.mean(smoothed[kept, :], axis=0)
		indices[:] = indices.argsort(axis=0)
		smoothed[:] = np.take_along_axis(smoothed, indices, axis=0)
		factor = np.divide(
			sorted_mean, smooth_mean, out=np.ones_like(sorted_mean),
			where=smooth_mean != 0,
		)
		mask = binary_dilation(_detect_stripe(factor, self.snr), iterations=1)
		if self.norm:
			np.divide(
				sinogram, np.tile(factor, (rows, 1)), out=sinogram_base,
			)
		sinogram_base[:, mask] = smoothed[:, mask]


class _BaselineVoDeadStage(VoDeadStripeStage):
	"""The allocation-per-call implementation from lftomo@dd4ce95."""

	uses_vo_work = False

	def run_sinogram(self, sinogram_base, state):
		mask_cut = np.s_[2:-2]
		sinogram = np.ascontiguousarray(sinogram_base, dtype=sinogram_base.dtype)
		smooth_diff = np.sum(
			np.abs(sinogram - np.apply_along_axis(uniform_filter1d, 0, sinogram, 10)),
			axis=0,
		)
		median_diff = median_filter(smooth_diff, self.size)
		factor = np.divide(
			smooth_diff, median_diff, out=np.ones_like(smooth_diff),
			where=median_diff != 0,
		)
		mask = np.full(factor.shape, False)
		mask[mask_cut] = binary_dilation(
			_detect_stripe(factor, self.snr), iterations=1,
		)[mask_cut]
		changed = mask.nonzero()[0]
		kept = np.invert(mask).nonzero()[0]
		if len(changed) > 0:
			for theta in range(sinogram.shape[0]):
				sinogram_base[theta, changed] = np.interp(
					changed, kept, sinogram[theta, kept],
				)
		if self.norm:
			_BaselineVoLargeStage(
				snr=self.snr, size=self.size, norm=True,
			).run_sinogram(sinogram_base, state)


def _stages(operation: str, *, reusable: bool):
	dead = VoDeadStripeStage if reusable else _BaselineVoDeadStage
	large = VoLargeStripeStage if reusable else _BaselineVoLargeStage
	sort = VoSortStripeStage if reusable else _BaselineVoSortStage
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

	if variant not in {"legacy", "reusable"}:
		raise ValueError("variant must be legacy or reusable")
	if len(shape) != 3 or min(shape) < 1:
		raise ValueError("shape must contain positive Theta,Y,X dimensions")
	if workers < 1:
		raise ValueError("workers must be positive")
	bytes_report = scratch_report(shape)
	managed = bytes_report["managed_scratch_bytes"] if variant == "reusable" else 0
	if plan_only:
		return VoBenchmarkResult(
			variant, operation, shape, workers, True,
			bytes_report["input_bytes"], managed,
			bytes_report["float_work_bytes"] if variant == "reusable" else 0,
			bytes_report["index_work_bytes"] if variant == "reusable" else 0,
			None, None, None, None, None,
		)

	scan = _scan(shape)
	recon = Recon(pipeline=[StageConfig(kind="benchmark")], process_workers=workers)
	with PipelineRunContext(max_workers=workers) as context:
		working = context.storage.allocate(ProjOrder(*shape), np.float32, role="normalized")
		_fill_fixture(working.array())
		state = PipelineState(
			working, scan, recon, storage=context.storage, executor=context.executor,
			layout=ProjOrder, domain=DataDomain.ATTENUATION,
		)
		pipeline = Pipeline(
			_stages(operation, reusable=variant == "reusable"), scan=scan,
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
		bytes_report["float_work_bytes"] if variant == "reusable" else 0,
		bytes_report["index_work_bytes"] if variant == "reusable" else 0,
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
	parser.add_argument("--variant", required=True, choices=("legacy", "reusable"))
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
