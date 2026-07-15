"""Memory-bounded beam-basis construction and robust projection fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_gradient_magnitude

from .build import CalibrationBuildSession
from .models import (
	CalibrationComponent, CompatibilityMetadata, DetectorRegion, ValidationResult,
)
from .static_dynamic import _clean


class HDF5RandomFrameSource:
	"""Bounded random access to a frame range in an ``(N, Y, X)`` dataset."""

	def __init__(
		self, path: str | Path, dataset: str = "exchange/data_white", *,
		frame_start: int = 0, frame_stop: int | None = None,
	):
		self.path = Path(path).expanduser().resolve()
		self.dataset = dataset
		with h5py.File(self.path, "r") as handle:
			if dataset not in handle:
				raise ValueError(f"HDF5 dataset {dataset!r} is missing from {self.path}")
			value = handle[dataset]
			if value.ndim != 3:
				raise ValueError(f"HDF5 dataset {dataset!r} must have (N, Y, X) axes")
			count, self.rows, self.columns = map(int, value.shape)
			self.dtype = np.dtype(value.dtype)
		self.frame_start = int(frame_start)
		self.frame_stop = count if frame_stop is None else int(frame_stop)
		if not 0 <= self.frame_start < self.frame_stop <= count:
			raise ValueError(
				f"invalid frame range [{self.frame_start}, {self.frame_stop}) for {dataset!r}"
			)
		self.frame_count = self.frame_stop - self.frame_start

	def read(self, index: int, *, y_start: int = 0, y_stop: int | None = None,
			x_start: int = 0, x_stop: int | None = None) -> np.ndarray:
		if not 0 <= index < self.frame_count:
			raise IndexError(index)
		with h5py.File(self.path, "r") as handle:
			return np.asarray(handle[self.dataset][
				self.frame_start + index, y_start:y_stop, x_start:x_stop,
			])


class ConcatenatedFrameSource:
	"""Present compatible frame sources as one training library."""

	def __init__(self, sources: list[Any] | tuple[Any, ...]):
		if not sources:
			raise ValueError("a concatenated frame source cannot be empty")
		self.sources = tuple(sources)
		first = self.sources[0]
		self.rows, self.columns, self.dtype = first.rows, first.columns, first.dtype
		for source in self.sources[1:]:
			if (source.rows, source.columns) != (self.rows, self.columns):
				raise ValueError("concatenated frame sources must have matching detector shapes")
		self._offsets = np.cumsum([0, *(source.frame_count for source in self.sources)])
		self.frame_count = int(self._offsets[-1])

	def read(self, index: int, **window) -> np.ndarray:
		if not 0 <= index < self.frame_count:
			raise IndexError(index)
		source_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
		return self.sources[source_index].read(
			index - int(self._offsets[source_index]), **window,
		)


@dataclass(frozen=True)
class BeamBasisBuild:
	component: CalibrationComponent
	explained_fraction: tuple[float, ...]
	training_residual: float


def build_beam_basis(
	build: CalibrationBuildSession,
	source: Any,
	*,
	n_components: int = 3,
	y_chunk: int = 64,
	crop: tuple[int, int, int, int] | None = None,
	camera_segment: str | None = None,
	training_coverage: list[str] | None = None,
	name: str = "beam_basis",
) -> BeamBasisBuild:
	"""Build log-beam PCA with two source passes and one small Gram matrix."""

	count = source.frame_count
	if count < 2 or not 1 <= n_components < count or y_chunk < 1:
		raise ValueError("beam basis requires 2+ frames, 1..N-1 components, and positive y_chunk")
	if not training_coverage:
		raise ValueError("beam basis requires identifiers for its training_coverage")
	if crop is None:
		row0, row1, column0, column1 = 0, source.rows, 0, source.columns
	else:
		row0, row1, column0, column1 = map(int, crop)
	if not 0 <= row0 < row1 <= source.rows or not 0 <= column0 < column1 <= source.columns:
		raise ValueError("beam-basis crop exceeds the source detector")
	rows, columns = row1 - row0, column1 - column0
	gram = np.zeros((count, count), dtype=np.float64)
	for start in range(row0, row1, y_chunk):
		stop = min(row1, start + y_chunk)
		logs = np.stack([
			np.log(_clean(source.read(
				index, y_start=start, y_stop=stop, x_start=column0, x_stop=column1,
			))).astype(np.float32)
			for index in range(count)
		])
		centered = (logs - logs.mean(axis=0)).reshape(count, -1)
		gram += centered @ centered.T
	eigenvalues, eigenvectors = np.linalg.eigh(gram)
	order = np.argsort(eigenvalues)[::-1][:n_components]
	selected = np.maximum(eigenvalues[order], 0)
	if selected[-1] <= max(float(np.max(selected)), 1.0) * np.finfo(np.float32).eps:
		raise ValueError("requested components exceed the numerical rank of the training flats")
	explained = selected / max(float(np.sum(np.maximum(eigenvalues, 0))), 1e-12)
	training_coefficients = eigenvectors[:, order] * np.sqrt(selected)[None]
	writer = build.create_array_component(
		name, shape=(n_components + 1, rows, columns), dtype="float32",
		axes=("Basis", "Y", "X"), append_axis=1, kind="beam_basis",
		coordinate_frame="beam",
		compatibility=CompatibilityMetadata(
			camera_segment=camera_segment, training_coverage=training_coverage,
			region=DetectorRegion(
				y_start=row0, x_start=column0, height=rows, width=columns,
			),
		),
		metadata={
			"mean_index": 0, "component_indices": list(range(1, n_components + 1)),
			"explained_fraction": explained.tolist(), "training_frame_count": count,
			"training_coefficient_min": training_coefficients.min(axis=0).tolist(),
			"training_coefficient_max": training_coefficients.max(axis=0).tolist(),
			"crop": [row0, row1, column0, column1], "y_chunk": y_chunk,
		},
	)
	residual_sum = 0.0; value_sum = 0.0
	for start in range(row0, row1, y_chunk):
		stop = min(row1, start + y_chunk)
		logs = np.stack([
			np.log(_clean(source.read(
				index, y_start=start, y_stop=stop, x_start=column0, x_stop=column1,
			))).astype(np.float32)
			for index in range(count)
		])
		mean = logs.mean(axis=0)
		centered = (logs - mean).reshape(count, -1)
		components = []
		for eigen_index, eigenvalue in zip(order, selected, strict=True):
			vector = centered.T @ eigenvectors[:, eigen_index]
			components.append((vector / max(np.sqrt(eigenvalue), 1e-12)).reshape(stop - start, columns))
		component_array = np.asarray(components, dtype=np.float32)
		writer.append(np.concatenate([mean[None], component_array], axis=0))
		reconstructed = training_coefficients @ component_array.reshape(n_components, -1)
		residual_sum += float(np.sum((centered - reconstructed) ** 2))
		value_sum += float(np.sum(centered ** 2))
	training_residual = float(np.sqrt(residual_sum / max(value_sum, 1e-12)))
	writer.metadata["training_relative_residual"] = training_residual
	writer.add_validation_result(ValidationResult(
		name="basis_training_coverage", status="pass",
		message="training flats projected into the retained log-beam basis",
		metrics={"relative_residual": training_residual, "explained_fraction": explained.tolist()},
	))
	return BeamBasisBuild(
		component=writer.close(), explained_fraction=tuple(map(float, explained)),
		training_residual=training_residual,
	)


@dataclass(frozen=True)
class BeamFitQC:
	air_pixel_count: int
	air_fraction: float
	rejected_fraction: float
	edge_overlap_fraction: float
	fit_residual: float
	coefficient_norm: float
	out_of_basis: float
	basis_rank: int
	basis_condition: float
	iterations: int


@dataclass(frozen=True)
class ProjectionBeamFit:
	beam: np.ndarray
	transmission: np.ndarray
	air_mask: np.ndarray
	coefficients: np.ndarray
	qc: BeamFitQC


def fit_projection_beam(
	projection: Any,
	basis: Any,
	*,
	detector: Any | None = None,
	air_mask: np.ndarray | None = None,
	air_threshold: float = 0.95,
	minimum_air_pixels: int = 100,
	minimum_air_fraction: float = 0.005,
	trim_sigma: float = 4.0,
	erode: int = 1,
	n_iter: int = 4,
	exposure_quantile: float = 0.2,
	edge_quantile: float = 0.85,
) -> ProjectionBeamFit:
	"""Robustly fit one projection and report mask contamination/coverage QC.

	Residual trimming assumes the candidate mask contains a majority of true air
	pixels. Fit residual and coefficient-range QC remain required backstops when
	gross contamination violates that assumption.
	"""

	if (
		not 0 < air_threshold <= 1 or not 0 < minimum_air_fraction <= 1
		or not 0 <= exposure_quantile <= 1 or not 0 <= edge_quantile <= 1
	):
		raise ValueError(
			"air_threshold and minimum_air_fraction must lie in (0, 1], and mask "
			"quantiles must lie in [0, 1]"
		)
	if minimum_air_pixels < 1 or trim_sigma <= 0 or erode < 0 or n_iter < 1:
		raise ValueError("fit counts must be positive and erosion cannot be negative")
	projection = _clean(projection)
	basis = np.asarray(basis, dtype=np.float64)
	if basis.ndim != 3 or len(basis) < 2 or basis.shape[1:] != projection.shape:
		raise ValueError("beam basis must have shape (mean+components, Y, X) matching projection")
	working = projection if detector is None else projection / _clean(detector)
	log_projection = np.log(working)
	mean, components = basis[0], basis[1:]
	auto = air_mask is None
	exposed = mean > np.quantile(mean, exposure_quantile)
	mask = (
		(log_projection - mean > np.log(air_threshold)) & exposed
		if auto else np.asarray(air_mask, bool).copy()
	)
	if mask.shape != projection.shape:
		raise ValueError("air mask must match the projection")
	edges = gaussian_gradient_magnitude(log_projection - mean, 1)
	edge_limit = np.quantile(edges, edge_quantile)
	edge_pixels = edges >= edge_limit
	edge_overlap = float(np.count_nonzero(mask & edge_pixels) / max(np.count_nonzero(mask), 1))
	mask &= ~edge_pixels
	if erode:
		mask = binary_erosion(mask, iterations=erode)
	initial_count = int(np.count_nonzero(mask))
	minimum = max(minimum_air_pixels, int(np.ceil(minimum_air_fraction * mask.size)))
	if initial_count < minimum:
		raise ValueError(f"insufficient trustworthy air pixels: {initial_count} < {minimum}")
	coefficients = np.zeros(len(components)); iterations = 0
	basis_rank = 0; basis_condition = float("inf")
	for iterations in range(1, max(n_iter, 1) + 1):
		flat_mask = mask.ravel()
		matrix = components.reshape(len(components), -1)[:, flat_mask].T
		target = (log_projection - mean).ravel()[flat_mask]
		basis_rank = int(np.linalg.matrix_rank(matrix))
		basis_condition = float(np.linalg.cond(matrix))
		if basis_rank < len(components) or not np.isfinite(basis_condition):
			raise ValueError("air pixels do not cover the retained beam basis")
		coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
		residual = target - matrix @ coefficients
		median = np.median(residual); mad = 1.4826 * np.median(np.abs(residual - median))
		keep = np.abs(residual - median) <= trim_sigma * max(mad, 1e-9)
		if keep.all():
			break
		indices = np.flatnonzero(flat_mask)
		new_mask = np.zeros(mask.size, bool); new_mask[indices[keep]] = True
		new_mask = new_mask.reshape(mask.shape)
		if np.count_nonzero(new_mask) < minimum:
			raise ValueError("robust fit rejected too many contaminated air pixels")
		if np.array_equal(new_mask, mask):
			break
		mask = new_mask
	flat_mask = mask.ravel()
	matrix = components.reshape(len(components), -1)[:, flat_mask].T
	target = (log_projection - mean).ravel()[flat_mask]
	basis_rank = int(np.linalg.matrix_rank(matrix))
	basis_condition = float(np.linalg.cond(matrix))
	if basis_rank < len(components) or not np.isfinite(basis_condition):
		raise ValueError("final air pixels do not cover the retained beam basis")
	coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
	log_beam = mean + np.tensordot(coefficients, components, axes=1)
	beam = np.exp(log_beam)
	transmission = working / beam
	air_target = (log_projection - mean)[mask]
	air_residual = (log_projection - log_beam)[mask]
	fit_residual = float(np.sqrt(np.mean(air_residual ** 2)))
	out_of_basis = fit_residual / max(float(np.sqrt(np.mean(air_target ** 2))), 1e-12)
	final_count = int(np.count_nonzero(mask))
	qc = BeamFitQC(
		air_pixel_count=final_count, air_fraction=final_count / mask.size,
		rejected_fraction=1 - final_count / initial_count,
		edge_overlap_fraction=edge_overlap, fit_residual=fit_residual,
		coefficient_norm=float(np.linalg.norm(coefficients)), out_of_basis=out_of_basis,
		basis_rank=basis_rank, basis_condition=basis_condition, iterations=iterations,
	)
	return ProjectionBeamFit(
		beam=beam.astype(np.float32), transmission=transmission.astype(np.float32),
		air_mask=mask, coefficients=coefficients, qc=qc,
	)
