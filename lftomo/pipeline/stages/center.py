"""Center-of-rotation stages."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
import numpy as np
from skimage.measure import shannon_entropy
import tifffile as tf

from lftomo.config import StageConfig
from lftomo.pipeline.core import (
	Locality,
	PipelineState,
	RuntimeRequirement,
	Stage,
	StageContract,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.pipeline.runtime import tomopy_requirement
from lftomo.util.shapes import ProjOrder, SinoOrder


LOGGER = logging.getLogger(__name__)


def _contract(runtime_requirements: tuple[RuntimeRequirement, ...] = ()) -> StageContract:
	return StageContract(
		produces=PRESERVE,
		accepts=(SinoOrder,), prefers=SinoOrder,
		bounded_views_from=(ProjOrder,), locality=Locality.SINOGRAM,
		accepts_domain=(DataDomain.ATTENUATION,),
		produces_domain=PRESERVE,
		window_axes=("Y",),
		produces_artifacts=("center",),
		runtime_requirements=runtime_requirements,
	)


def _sinograms(state: PipelineState) -> tuple[np.ndarray, type]:
	if state.backend.name != "numpy":
		raise TypeError("center Stages currently require the NumPy backend")
	data = np.asarray(state.working_array)
	if data.ndim != 3:
		raise ValueError("center Stages expect a (Y, Theta, X) sinogram array")
	return data, type(state.buffer_spec.shape)


def _angles(state: PipelineState, theta_count: int) -> np.ndarray:
	angles = None
	if state.dataset is not None and state.source_window is not None:
		angles = state.dataset.angles[state.source_window.axis_slice("Theta")]
	if angles is None:
		acquisition = state.scan.acquisition
		angles = np.linspace(
			np.deg2rad(acquisition.rotation_start_deg),
			np.deg2rad(acquisition.rotation_end_deg),
			theta_count,
			endpoint=False,
			dtype=np.float32,
		)
	angles = np.asarray(angles, dtype=np.float32)
	if angles.ndim != 1 or len(angles) != theta_count:
		raise ValueError(f"center Stage requires {theta_count} one-dimensional projection angles")
	return angles


def _store_center(state: PipelineState, center: float | np.ndarray) -> PipelineState:
	values = np.asarray(center, dtype=np.float32)
	state.artifacts["center"] = float(values) if values.ndim == 0 else values
	return state


class ManualCenterStage(Stage):
	"""Provide a fixed center, optionally varying linearly with detector row."""

	name = "center.manual"
	contract = _contract()

	def __init__(
		self,
		value_px: float,
		tilt: float = 0.0,
		tilt_radians: float | None = None,
		reference_y: float = 0.0,
		find_center: bool = False,
		pc_fallback: bool = False,
		vo_fallback: bool = False,
		center_dump: bool = False,
		center_samples_per_job: int | None = None,
	):
		if tilt_radians is not None and tilt != 0:
			raise ValueError("center.manual accepts either tilt or tilt_radians, not both")
		if find_center or pc_fallback or vo_fallback:
			raise ValueError("automatic center finding must use a center.vo or center.pc Stage")
		if center_samples_per_job is not None and center_samples_per_job < 1:
			raise ValueError("center_samples_per_job must be positive")
		self.contract = StageContract(
			window_axes=("Y",),
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			accepts_domain=(DataDomain.ATTENUATION,),
			produces_domain=PRESERVE,
			produces_artifacts=("center",),
		)
		self.value_px = value_px
		self.slope = np.tan(tilt_radians) if tilt_radians is not None else tilt
		self.reference_y = reference_y
		self.center_dump = center_dump

	@classmethod
	def from_config(cls, config: StageConfig) -> "ManualCenterStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("center.manual currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		layout = type(state.buffer_spec.shape)
		if data.ndim != 3:
			raise ValueError("center.manual expects a three-dimensional tomography array")
		if self.slope == 0:
			return _store_center(state, self.value_px)
		y_size = data.shape[layout.axis_of("Y")]
		source_y = 0 if state.source_window is None else int(state.source_window.pos.Y)
		rows = source_y + np.arange(y_size, dtype=np.float32)
		return _store_center(state, self.value_px + (rows - self.reference_y) * self.slope)


class EntropyLocalCenterStage(Stage):
	"""Select the lowest-entropy gridrec result from a local center sweep."""

	name = "center.entropy_local"
	contract = _contract(
		runtime_requirements=(tomopy_requirement("circ_mask", "recon"),),
	)

	def __init__(
		self,
		start: float | None = None,
		range: float = 5.0,
		step: float = 0.25,
		slice_index: int | None = None,
		filter_name: str = "butterworth",
		filter_parameters: list[float] | tuple[float, ...] = (0.2, 2),
		ncore: int | None = None,
		circle_mask_ratio: float = 1.0,
		max_expansions: int = 3,
		diagnostic_images: bool = False,
		diagnostic_subdirectory: str = "center-diagnostics",
	):
		if range < 0:
			raise ValueError("center.entropy_local range cannot be negative")
		if step <= 0:
			raise ValueError("center.entropy_local step must be positive")
		if ncore is not None and ncore < 1:
			raise ValueError("center.entropy_local ncore must be positive")
		if not 0 < circle_mask_ratio <= 1:
			raise ValueError("center.entropy_local circle_mask_ratio must be in (0, 1]")
		if max_expansions < 0:
			raise ValueError("center.entropy_local max_expansions cannot be negative")
		diagnostic_path = Path(diagnostic_subdirectory)
		if (
			diagnostic_path.is_absolute()
			or not diagnostic_path.parts
			or ".." in diagnostic_path.parts
		):
			raise ValueError(
				"center.entropy_local diagnostic_subdirectory must be a safe relative path"
			)
		self.start = start
		self.range = range
		self.step = step
		self.slice_index = slice_index
		self.filter_name = filter_name
		self.filter_parameters = filter_parameters
		self.ncore = ncore
		self.circle_mask_ratio = circle_mask_ratio
		self.max_expansions = max_expansions
		self.diagnostic_images = diagnostic_images
		self.diagnostic_subdirectory = diagnostic_path
		produces = ("center", "center_entropy")
		if diagnostic_images:
			produces += ("output_files",)
		self.contract = replace(type(self).contract, produces_artifacts=produces)

	@classmethod
	def from_config(cls, config: StageConfig) -> "EntropyLocalCenterStage":
		return cls(**config.params)

	def _diagnostic_root(self, state: PipelineState) -> Path:
		output = state.recon.output_path
		if output is None:
			raise ValueError(
				"center.entropy_local diagnostic_images requires Recon.output_path"
			)
		output = Path(output)
		if output.suffix.lower() in {".h5", ".hdf5"}:
			root = output.parent / f"{output.stem}-{self.diagnostic_subdirectory}"
		else:
			root = output / self.diagnostic_subdirectory
		root.mkdir(parents=True, exist_ok=True)
		return root

	def _diagnostic_path(
		self, root: Path, source_slice: int, search: int,
		candidate_index: int, center: float,
	) -> Path:
		return root / (
			f"slice-{source_slice:05d}-search-{search:02d}-"
			f"candidate-{candidate_index:03d}-center-{center:.4f}.tif"
		)

	def run(self, state: PipelineState) -> PipelineState:
		data, layout = _sinograms(state)
		row_count = data.shape[layout.axis_of("Y")]
		index = row_count // 2 if self.slice_index is None else self.slice_index
		if not 0 <= index < row_count:
			raise ValueError("center.entropy_local slice_index is outside the sinogram window")
		current_start = data.shape[layout.axis_of("X")] / 2 if self.start is None else self.start
		angles = _angles(state, data.shape[layout.axis_of("Theta")])
		sinogram = np.ascontiguousarray(data[layout.point_index(Y=index)])
		source_y = 0 if state.source_window is None else int(state.source_window.pos.Y)
		source_slice = source_y + index
		diagnostic_root = self._diagnostic_root(state) if self.diagnostic_images else None

		import tomopy

		searches = []
		all_candidates = []
		all_entropies = []
		diagnostic_files = []
		previous_boundary = None
		termination = "cap-reached"
		converged = False
		selected_center = float(current_start)
		for search in range(self.max_expansions + 1):
			candidates = np.arange(
				current_start - self.range,
				current_start + self.range + self.step / 2,
				self.step,
				dtype=np.float32,
			)
			entropies = []
			for candidate_index, center in enumerate(candidates):
				reconstruction = tomopy.recon(
					sinogram[np.newaxis, :, :],
					angles,
					center=float(center),
					sinogram_order=True,
					algorithm="gridrec",
					filter_name=self.filter_name,
					filter_par=self.filter_parameters,
					ncore=self.ncore,
				)
				reconstruction = tomopy.circ_mask(
					reconstruction,
					axis=0,
					ratio=self.circle_mask_ratio,
					ncore=self.ncore,
				)
				entropies.append(float(shannon_entropy(reconstruction)))
				if diagnostic_root is not None:
					path = self._diagnostic_path(
						diagnostic_root, source_slice, search,
						candidate_index, float(center),
					)
					tf.imwrite(path, np.asarray(reconstruction[0], dtype=np.float32))
					diagnostic_files.append(path)
			entropies_array = np.asarray(entropies, dtype=np.float64)
			best = int(np.argmin(entropies_array))
			selected_center = float(candidates[best])
			left = best == 0
			right = best == len(candidates) - 1
			boundary = "left" if left else "right" if right else None
			searches.append({
				"iteration": search,
				"start": float(current_start),
				"candidates": candidates,
				"entropies": entropies_array,
				"best_index": best,
				"best_center": selected_center,
				"boundary": boundary,
			})
			all_candidates.append(candidates)
			all_entropies.append(entropies_array)
			if boundary is None:
				termination = "converged"
				converged = True
				break
			if len(candidates) == 1 or self.range == 0 or (
				previous_boundary is not None and boundary != previous_boundary
			):
				termination = "boundary-persisted"
				break
			if search == self.max_expansions:
				termination = "cap-reached"
				break
			current_start += -2 * self.range if boundary == "left" else 2 * self.range
			previous_boundary = boundary
		state.artifacts["center_entropy"] = {
			"candidates": np.concatenate(all_candidates),
			"entropies": np.concatenate(all_entropies),
			"slice_index": index,
			"source_slice_index": source_slice,
			"searches": searches,
			"termination": termination,
			"converged": converged,
			"expansions": len(searches) - 1,
			"max_expansions": self.max_expansions,
			"selected_center": selected_center,
			"diagnostic_files": diagnostic_files,
		}
		if diagnostic_files:
			existing = list(state.artifacts.get("output_files", ()))
			state.artifacts["output_files"] = existing + [
				path for path in diagnostic_files if path not in existing
			]
		return _store_center(state, selected_center)


class VoCenterStage(Stage):
	"""Find center with TomoPy's Vo method, optionally fitting row tilt."""

	name = "center.vo"
	contract = _contract(
		runtime_requirements=(
			tomopy_requirement("circ_mask", "find_center_vo", "recon"),
		),
	)

	def __init__(
		self,
		ind: int | None = None,
		smin: float = -50,
		smax: float = 50,
		srad: float = 6,
		step: float = 0.25,
		ratio: float = 0.5,
		drop: int = 20,
		ncore: int | None = None,
		tilt_fit: str | None = None,
		tilt_samples: int = 5,
		fit_degree: int = 1,
		tilt_outlier_sigma: float = 3.5,
		tilt_max_iterations: int = 5,
		tilt_min_inliers: int | None = None,
		tilt_residual_cutoff_px: float | None = None,
		tilt_qc_policy: str = "reject",
		tilt_quantization_px: float | None = None,
		fallback: str | None = None,
		fallback_params: dict | None = None,
	):
		if tilt_fit not in (None, "polynomial"):
			raise ValueError("center.vo tilt_fit must be 'polynomial' or omitted")
		if tilt_samples < 2:
			raise ValueError("center.vo tilt_samples must be at least two")
		if fit_degree < 1:
			raise ValueError("center.vo fit_degree must be positive")
		if tilt_outlier_sigma <= 0:
			raise ValueError("center.vo tilt_outlier_sigma must be positive")
		if tilt_max_iterations < 1:
			raise ValueError("center.vo tilt_max_iterations must be positive")
		if tilt_min_inliers is not None and tilt_min_inliers < fit_degree + 1:
			raise ValueError(
				"center.vo tilt_min_inliers must exceed the polynomial degree"
			)
		if tilt_residual_cutoff_px is not None and tilt_residual_cutoff_px < 0:
			raise ValueError("center.vo tilt_residual_cutoff_px cannot be negative")
		if tilt_qc_policy not in ("reject", "force"):
			raise ValueError("center.vo tilt_qc_policy must be 'reject' or 'force'")
		if tilt_quantization_px is not None and tilt_quantization_px <= 0:
			raise ValueError("center.vo tilt_quantization_px must be positive")
		if fallback not in (None, "entropy_local"):
			raise ValueError("center.vo fallback must be entropy_local or omitted")
		self.ind = ind
		self.kwargs = {
			"smin": smin,
			"smax": smax,
			"srad": srad,
			"step": step,
			"ratio": ratio,
			"drop": drop,
			"ncore": ncore,
		}
		self.tilt_fit = tilt_fit
		self.tilt_samples = tilt_samples
		self.fit_degree = fit_degree
		self.tilt_outlier_sigma = tilt_outlier_sigma
		self.tilt_max_iterations = tilt_max_iterations
		self.tilt_min_inliers = tilt_min_inliers
		self.tilt_residual_cutoff_px = tilt_residual_cutoff_px
		self.tilt_qc_policy = tilt_qc_policy
		self.tilt_quantization_px = tilt_quantization_px
		self.fallback = fallback
		self.fallback_params = fallback_params or {}

	@classmethod
	def from_config(cls, config: StageConfig) -> "VoCenterStage":
		return cls(**config.params)

	def _find(self, sinogram: np.ndarray) -> float:
		import tomopy

		center = float(tomopy.find_center_vo(sinogram, **self.kwargs))
		if not np.isfinite(center):
			raise ValueError("center.vo returned a non-finite center")
		return center

	def _robust_fit(
		self, rows: np.ndarray, centers: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
		"""Iteratively reject center residuals using a median/MAD scale."""

		minimum = self.tilt_min_inliers
		if minimum is None:
			minimum = min(len(rows), max(self.fit_degree + 1, 3))
		if len(rows) < minimum:
			raise ValueError(
				f"center.vo tilt fit requires at least {minimum} unique sampled rows"
			)
		mask = np.ones(len(rows), dtype=bool)
		clipping_limited = False
		iterations = 0
		for iterations in range(1, self.tilt_max_iterations + 1):
			coefficients = np.polyfit(rows[mask], centers[mask], self.fit_degree)
			residuals = centers - np.polyval(coefficients, rows)
			inlier_residuals = residuals[mask]
			residual_center = float(np.median(inlier_residuals))
			deviations = np.abs(residuals - residual_center)
			mad = float(np.median(np.abs(inlier_residuals - residual_center)))
			threshold = max(1e-6, self.tilt_outlier_sigma * 1.4826 * mad)
			updated = mask & (deviations <= threshold)
			if np.count_nonzero(updated) < minimum:
				clipping_limited = True
				break
			if np.array_equal(updated, mask):
				break
			mask = updated
		coefficients = np.polyfit(rows[mask], centers[mask], self.fit_degree)
		residuals = centers - np.polyval(coefficients, rows)
		return coefficients, residuals, mask, iterations, clipping_limited

	def _tilt_center(self, state: PipelineState, data: np.ndarray, layout: type) -> np.ndarray:
		row_count = data.shape[layout.axis_of("Y")]
		count = min(self.tilt_samples, row_count)
		local_rows = np.unique(np.linspace(0, row_count - 1, count).round().astype(int))
		if len(local_rows) <= self.fit_degree:
			raise ValueError("center.vo needs more sampled rows than the polynomial degree")
		source_start = 0
		source_step = 1
		if state.source_window is not None:
			source_start = int(state.source_window.pos.Y)
			source_step = int(state.source_window.step.Y)
		source_rows = source_start + local_rows * source_step
		centers = np.asarray([
			self._find(np.ascontiguousarray(data[layout.point_index(Y=int(index))]))
			for index in local_rows
		], dtype=np.float64)
		coefficients, residuals, inliers, iterations, clipping_limited = self._robust_fit(
			source_rows.astype(np.float64), centers,
		)
		evaluation_rows = source_start + np.arange(row_count) * source_step
		unquantized = np.polyval(coefficients, evaluation_rows)
		center = unquantized
		if self.tilt_quantization_px is not None:
			center = (
				np.round(center / self.tilt_quantization_px) * self.tilt_quantization_px
			)
		inlier_residuals = residuals[inliers]
		residual_rms = float(np.sqrt(np.mean(np.square(inlier_residuals))))
		residual_max = float(np.max(np.abs(inlier_residuals)))
		qc_reasons = []
		if clipping_limited:
			qc_reasons.append("outlier clipping would leave too few inliers")
		if (
			self.tilt_residual_cutoff_px is not None
			and residual_rms > self.tilt_residual_cutoff_px
		):
			qc_reasons.append("inlier residual RMS exceeds cutoff")
		qc_passed = not qc_reasons
		status = "accepted" if qc_passed else (
			"forced" if self.tilt_qc_policy == "force" else "rejected"
		)
		state.artifacts["center_fit"] = {
			"sampled_local_rows": local_rows,
			"sampled_source_rows": source_rows,
			"sampled_centers": centers,
			"coefficients": coefficients,
			"residuals_px": residuals,
			"inlier_mask": inliers,
			"outlier_source_rows": source_rows[~inliers],
			"residual_rms_px": residual_rms,
			"residual_max_abs_px": residual_max,
			"residual_cutoff_px": self.tilt_residual_cutoff_px,
			"row_span": int(source_rows.max() - source_rows.min()),
			"fit_degree": self.fit_degree,
			"outlier_sigma": self.tilt_outlier_sigma,
			"fit_iterations": iterations,
			"quantization_px": self.tilt_quantization_px,
			"unquantized_center": unquantized,
			"qc_passed": qc_passed,
			"qc_policy": self.tilt_qc_policy,
			"qc_status": status,
			"qc_reasons": qc_reasons,
		}
		if not qc_passed and self.tilt_qc_policy == "reject":
			raise ValueError("center.vo tilt fit rejected: " + "; ".join(qc_reasons))
		return np.asarray(center, dtype=np.float32)

	def run(self, state: PipelineState) -> PipelineState:
		data, layout = _sinograms(state)
		row_count = data.shape[layout.axis_of("Y")]
		try:
			if self.tilt_fit == "polynomial":
				center = self._tilt_center(state, data, layout)
			else:
				index = row_count // 2 if self.ind is None else self.ind
				if not 0 <= index < row_count:
					raise ValueError("center.vo ind is outside the sinogram window")
				center = self._find(np.ascontiguousarray(data[layout.point_index(Y=index)]))
		except (ValueError, RuntimeError, ArithmeticError) as error:
			if self.fallback != "entropy_local":
				raise
			LOGGER.warning(
				"center.vo failed with %s; using center.entropy_local fallback",
				error,
				exc_info=True,
			)
			return EntropyLocalCenterStage(**self.fallback_params).run(state)
		return _store_center(state, center)


class PhaseCorrelationCenterStage(Stage):
	"""Find center from the first and last projections by phase correlation."""

	name = "center.pc"
	contract = _contract(
		runtime_requirements=(tomopy_requirement("find_center_pc"),),
	)

	def __init__(self, tol: float = 0.5, rotc_guess: float | None = None):
		if tol <= 0:
			raise ValueError("center.pc tol must be positive")
		self.tol = tol
		self.rotc_guess = rotc_guess
		self.contract = StageContract(
			accepts=(ProjOrder, SinoOrder), prefers=ProjOrder, produces=PRESERVE,
			accepts_domain=(DataDomain.ATTENUATION,), produces_domain=PRESERVE,
			locality=Locality.PROJECTION, produces_artifacts=("center",),
			runtime_requirements=type(self).contract.runtime_requirements,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "PhaseCorrelationCenterStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		data, layout = _sinograms(state)
		projection_count = data.shape[layout.axis_of("Theta")]
		if projection_count < 2:
			raise ValueError("center.pc requires at least two projections")

		import tomopy

		center = tomopy.find_center_pc(
			data[layout.point_index(Theta=0)],
			data[layout.point_index(Theta=projection_count - 1)],
			tol=self.tol,
			rotc_guess=self.rotc_guess,
		)
		if not np.isfinite(center):
			raise ValueError("center.pc returned a non-finite center")
		return _store_center(state, center)
