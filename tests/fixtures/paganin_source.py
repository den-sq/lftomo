"""Source-derived Paganin fixtures pinned to dan_scripts@f601fc995ffbfc8757447c992cbb3f5fd1e36b19.

Only the audited ``tp=False`` reciprocal-grid/filter behavior is reproduced.
The helper deliberately does not import the legacy module, whose process,
shared-memory, logging, and terminal-negative-log code are outside issue #199.
"""

from __future__ import annotations

import numpy as np


PLANCK_CONSTANT = 6.58211928e-19  # keV s
SPEED_OF_LIGHT = 299792458e2  # cm / s
SOURCE_COMMIT = "f601fc995ffbfc8757447c992cbb3f5fd1e36b19"


def source_reciprocal_coordinate(pixel_size_cm: float, count: int) -> np.ndarray:
	"""Reproduce ``prep/phase.py::_reciprocal_coord(..., tp=False)``."""

	n = count - 1
	return (
		np.arange(-n, count, 2, dtype=np.float32)
		* 2 * np.pi / (n * pixel_size_cm)
	)


def source_transfer(
	shape: tuple[int, int], *, alpha: float, energy_kev: float,
	pixel_size_um: float, ssd_cm: float,
) -> np.ndarray:
	"""Reproduce the source's centered, shifted, max-normalized transfer."""

	pixel_size_cm = pixel_size_um / 10_000
	ky = source_reciprocal_coordinate(pixel_size_cm, shape[0])
	kx = source_reciprocal_coordinate(pixel_size_cm, shape[1])
	w2 = np.add.outer(np.square(ky), np.square(kx))
	wavelength = 2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy_kev
	values = 1 / (1 + ssd_cm * alpha * wavelength * w2 / (4 * np.pi))
	shifted = np.fft.fftshift(values)
	return shifted / shifted.max()


def source_filter(image: np.ndarray, **parameters) -> np.ndarray:
	"""Apply the audited source discretization to one transmission image."""

	return np.real(np.fft.ifft2(np.fft.fft2(image) * source_transfer(
		image.shape, **parameters,
	)))


def phase_fixture(shape: tuple[int, int]) -> np.ndarray:
	"""Deterministic smooth/edge/frequency content for characterization."""

	y, x = np.indices(shape, dtype=np.float64)
	y /= max(shape[0] - 1, 1)
	x /= max(shape[1] - 1, 1)
	image = (
		0.82
		- 0.27 * np.exp(-((x - 0.43) ** 2 + (y - 0.55) ** 2) / 0.018)
		+ 0.035 * np.sin(7 * np.pi * x)
		+ 0.025 * np.cos(5 * np.pi * y)
	)
	image[:, shape[1] // 2:] -= 0.08
	return np.clip(image, 0.05, 0.99).astype(np.float32)
