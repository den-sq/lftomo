"""Tier-1 synthetic tomography fixtures.

These helpers intentionally generate tiny arrays in caller-provided temporary
directories. They are unit-test fixtures, not performance or producer-reference
datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tf


@dataclass(frozen=True)
class Tier1Stack:
	"""Paths and arrays for a generated Tier-1 projection stack."""

	root: Path
	projections: np.ndarray
	projection_paths: list[Path]
	flats: np.ndarray | None = None
	darks: np.ndarray | None = None
	flat_paths: list[Path] | None = None
	dark_paths: list[Path] | None = None


def shepp_logan(size: int = 64) -> np.ndarray:
	"""Generate a small Shepp-Logan-like phantom with numpy ellipses."""
	if size <= 0:
		raise ValueError("size must be positive")

	y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
	phantom = np.zeros((size, size), dtype=np.float32)

	for y0, x0, yr, xr, value in (
		(0.0, 0.0, 0.92, 0.69, 1.0),
		(-0.02, 0.0, 0.84, 0.62, -0.8),
		(-0.22, -0.22, 0.25, 0.16, -0.2),
		(-0.22, 0.22, 0.25, 0.16, -0.2),
		(0.35, 0.0, 0.20, 0.50, 0.35),
		(0.10, -0.35, 0.12, 0.10, 0.2),
		(0.10, 0.35, 0.12, 0.10, 0.2),
	):
		mask = ((y - y0) / yr) ** 2 + ((x - x0) / xr) ** 2 <= 1
		phantom[mask] += value

	phantom -= phantom.min()
	phantom /= phantom.max() or 1.0
	return phantom.astype(np.float32)


def projection_stack(size: int = 64, n_projections: int = 16, misalignment: int = 0) -> np.ndarray:
	"""Generate a tiny projection stack by rolling a phantom across angles."""
	if n_projections <= 0:
		raise ValueError("n_projections must be positive")

	base = shepp_logan(size)
	projections = []
	for i in range(n_projections):
		shift = int(round(i * size / n_projections)) + misalignment
		rolled = np.roll(base, shift=shift, axis=1)
		projections.append(rolled)
	return np.stack(projections).astype(np.float32)


def striped_sinogram(size: int = 64, n_projections: int = 16, stripe_strength: float = 0.35) -> np.ndarray:
	"""Generate projection data with deterministic vertical stripe artifacts."""
	stack = projection_stack(size=size, n_projections=n_projections)
	stripes = np.ones((size,), dtype=np.float32)
	stripes[::8] += stripe_strength
	stripes[4::8] -= stripe_strength / 2
	return (stack * stripes[None, None, :]).astype(np.float32)


def flat_dark_frames(size: int = 64, n_flats: int = 4, gain: float = 1.25, dark_level: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
	"""Generate synthetic flat and dark frames."""
	if n_flats <= 0:
		raise ValueError("n_flats must be positive")
	y_gradient = np.linspace(0.95, 1.05, size, dtype=np.float32)[:, None]
	x_gradient = np.linspace(0.98, 1.02, size, dtype=np.float32)[None, :]
	flat = gain * y_gradient * x_gradient
	dark = np.full((size, size), dark_level, dtype=np.float32)
	return np.repeat(flat[None, :, :], n_flats, axis=0), np.repeat(dark[None, :, :], n_flats, axis=0)


def write_projection_stack(
	root: Path,
	size: int = 64,
	n_projections: int = 16,
	striped: bool = False,
	misalignment: int = 0,
) -> Tier1Stack:
	"""Write a synthetic projection TIFF stack under ``root/projections``."""
	root = Path(root)
	proj_dir = root / "projections"
	proj_dir.mkdir(parents=True, exist_ok=True)

	stack = striped_sinogram(size, n_projections) if striped else projection_stack(size, n_projections, misalignment)
	paths = []
	for i, image in enumerate(stack):
		path = proj_dir / f"projection_{i:04d}.tif"
		tf.imwrite(path, image.astype(np.float32))
		paths.append(path)
	return Tier1Stack(root=root, projections=stack, projection_paths=paths)


def write_flat_dark_stack(root: Path, size: int = 64, n_flats: int = 4) -> Tier1Stack:
	"""Write synthetic flat and dark TIFF frames under ``root/flats``."""
	root = Path(root)
	flat_dir = root / "flats"
	flat_dir.mkdir(parents=True, exist_ok=True)

	flats, darks = flat_dark_frames(size=size, n_flats=n_flats)
	flat_paths = []
	dark_paths = []
	for i, image in enumerate(flats):
		path = flat_dir / f"flat_{i:04d}.tif"
		tf.imwrite(path, image.astype(np.float32))
		flat_paths.append(path)
	for i, image in enumerate(darks):
		path = flat_dir / f"dark_{i:04d}.tif"
		tf.imwrite(path, image.astype(np.float32))
		dark_paths.append(path)
	return Tier1Stack(root=root, projections=np.empty((0, size, size)), projection_paths=[], flats=flats, darks=darks,
					flat_paths=flat_paths, dark_paths=dark_paths)
