"""Current-lftomo B10 reconstruction driver.

This is the active-use port. The pre-port producer scripts under B10Scripts are
left untouched for snapshot reference generation against lftomo@ddfcdf6.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import tifffile as tf

from lftomo.prep.flat import FLAT, weighted_normalize
from lftomo.util.io import read_window, write_slice
from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import DataRange, GenericOrder, ProjOrder, SinoOrder, Y
from lftomo.util.window import CTFov, TomoDataSet, TomoWindow


Reconstructor = Callable[[np.ndarray, np.ndarray, float, int], np.ndarray]


@dataclass(frozen=True)
class B10ReconConfig:
	source: Path
	flats: Path
	output: Path
	center: float
	window_size: int = 16
	degrees: float = 180.0
	alpha: float = 1.6
	gamma: float = 1.2
	apply_bac: bool = True
	processes: int = 1
	y_start: int = 0
	y_stop: int | None = None


def _tomopy_gridrec(data: np.ndarray, angles: np.ndarray, center: float, processes: int) -> np.ndarray:
	import tomopy

	return tomopy.recon(
		data,
		angles,
		sinogram_order=False,
		center=center,
		algorithm="gridrec",
		ncore=processes,
	).astype(np.float32)


def _flat_path(root: Path, flat: FLAT) -> Path:
	matches = sorted(root.glob(f"{flat}*.tif*"))
	if not matches:
		raise FileNotFoundError(f"missing {flat} flat under {root}")
	return matches[0]


def _load_flats(root: Path, detector_x: int) -> np.ndarray:
	values = [tf.imread(_flat_path(root, flat)) for flat in FLAT]
	shapes = {value.shape for value in values}
	if len(shapes) != 1 or values[0].ndim != 2 or values[0].shape[1] != detector_x:
		raise ValueError(
			f"B10 flats must share a two-dimensional detector shape with X={detector_x}; "
			f"found {sorted(shapes)}"
		)
	flats = np.stack(values).astype(np.float32)
	flats[FLAT.PREGAIN.index] -= flats[FLAT.PREDARK.index]
	flats[FLAT.POSTGAIN.index] -= flats[FLAT.POSTDARK.index]
	return flats


def _apply_bac(memory: SharedNPArray, alpha: float, gamma: float) -> None:
	from hotopy.holo import BronnikovAidedCorrection

	filter_bac = BronnikovAidedCorrection((memory.shape.Y, memory.shape.X), alpha=alpha, gamma=gamma)
	with memory as data:
		for projection in range(memory.shape.Theta):
			data[:, projection, :] = filter_bac(data[:, projection, :])


def run(config: B10ReconConfig, reconstructor: Reconstructor = _tomopy_gridrec) -> list[Path]:
	"""Run the current-API B10 path and return the written reconstruction slices."""
	config.output.mkdir(parents=True, exist_ok=True)
	source = Path(config.source)
	source_files = [source] if source.is_file() else sorted(source.glob("**/*.tif*"))
	if config.window_size < 1:
		raise ValueError("B10 window size must be positive")
	if config.y_start < 0:
		raise ValueError("B10 y_start cannot be negative")
	if config.y_start and config.y_stop is None:
		raise ValueError("B10 y_stop is required when y_start is nonzero")
	if config.y_stop is not None and config.y_stop <= config.y_start:
		raise ValueError("B10 y_stop must be greater than y_start")
	over = (
		Y(config.window_size) if config.y_start == 0 and config.y_stop is None else
		DataRange(Y(config.y_start), Y(config.y_stop), Y(config.window_size))
	)
	dataset = TomoDataSet.from_disk(
		source,
		CTFov(Theta=None, Degrees=config.degrees, Count=1),
		ProjOrder,
		over,
		work_dtype=np.float32,
	)
	flat_values = _load_flats(config.flats, dataset.shape.X)
	flat_memory = SharedNPArray(
		f"b10-flats-{uuid4().hex}", GenericOrder(*flat_values.shape), np.float32, allocate=True
	)
	written = []

	try:
		with flat_memory as target:
			target[:] = flat_values

		for window in dataset.iter_with(substep=Y(1)):
			proj_memory = SharedNPArray(
				f"b10-proj-{uuid4().hex}", window.shape, dataset.dtype, allocate=True
			)
			sino_shape = window.shape.to(SinoOrder)
			read_sino = SharedNPArray(
				f"b10-read-sino-{uuid4().hex}", sino_shape, dataset.dtype, allocate=True
			)
			normalized = SharedNPArray(
				f"b10-normalized-{uuid4().hex}", sino_shape, np.float32, allocate=True
			)
			recon_shape = window.recon_shape()
			recon_memory = SharedNPArray(
				f"b10-recon-{uuid4().hex}", recon_shape, np.float32, allocate=True
			)

			try:
				read_window(window, source_files, proj_memory, thread_max=max(config.processes, 1))
				with proj_memory as projections, read_sino as sinograms:
					sinograms[:] = np.transpose(projections, (1, 0, 2))

				local_window = TomoWindow(
					sino_shape, Y(0), window.pos, Y(1), dataset.dtype, np.float32
				)
				for projection in range(sino_shape.Theta):
					weighted_normalize(
						normalized, flat_memory, read_sino, local_window, projection, pre_sub=True
					)

				if config.apply_bac:
					_apply_bac(normalized, config.alpha, config.gamma)

				with normalized as sinograms:
					recon_input = np.transpose(sinograms, (1, 0, 2)).copy()
				reconstruction = reconstructor(
					recon_input, dataset.angles, config.center, config.processes
				).astype(np.float32)
				with recon_memory as target:
					target[:] = reconstruction

				for local_y in range(window.shape.Y):
					global_y = window.pos.Y + local_y
					path = config.output / f"{global_y:05d}.tif"
					write_slice(recon_memory, path, local_y)
					written.append(path)
			finally:
				proj_memory.shutdown()
				read_sino.shutdown()
				normalized.shutdown()
				recon_memory.shutdown()
	finally:
		flat_memory.shutdown()

	return written


def main() -> None:
	parser = ArgumentParser(description=__doc__)
	parser.add_argument("source", type=Path)
	parser.add_argument("flats", type=Path)
	parser.add_argument("output", type=Path)
	parser.add_argument("--center", type=float, required=True)
	parser.add_argument("--window-size", type=int, default=16)
	parser.add_argument("--alpha", type=float, default=1.6)
	parser.add_argument("--gamma", type=float, default=1.2)
	parser.add_argument("--no-bac", action="store_true")
	parser.add_argument("--processes", type=int, default=1)
	parser.add_argument("--y-start", type=int, default=0)
	parser.add_argument("--y-stop", type=int)
	args = parser.parse_args()
	run(B10ReconConfig(
		source=args.source,
		flats=args.flats,
		output=args.output,
		center=args.center,
		window_size=args.window_size,
		alpha=args.alpha,
		gamma=args.gamma,
		apply_bac=not args.no_bac,
		processes=args.processes,
		y_start=args.y_start,
		y_stop=args.y_stop,
	))


if __name__ == "__main__":
	main()
