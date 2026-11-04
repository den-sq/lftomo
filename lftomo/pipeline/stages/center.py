"""Center-of-rotation stages."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator
import numpy as np
from skimage.measure import shannon_entropy
import tifffile as tf

from lftomo.pipeline.core import (
	PipelineState,
	RuntimeRequirement,
	Stage,
	StageParams,
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
		bounded_views_from=(ProjOrder,),
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
	contract = StageContract(
		window_axes=("Y",), produces=PRESERVE, accepts=(ProjOrder, SinoOrder),
		accepts_domain=(DataDomain.ATTENUATION,), produces_domain=PRESERVE,
		produces_artifacts=("center",),
	)

	class Params(StageParams):
		value_px: float
		tilt: float = 0.0
		reference_y: float = 0.0
		find_center: Literal[False] = False
		pc_fallback: Literal[False] = False
		vo_fallback: Literal[False] = False
		center_dump: bool = False
		center_samples_per_job: int | None = Field(default=None, ge=1)

		@model_validator(mode="wrap")
		@classmethod
		def normalize_tilt(cls, value, handler):
			radians_input = False
			if isinstance(value, dict) and "tilt_radians" in value:
				value = dict(value)
				tilt = value.pop("tilt_radians")
				if tilt is not None:
					if value.get("tilt", 0) != 0:
						raise ValueError("center.manual accepts either tilt or tilt_radians, not both")
					value["tilt"] = np.tan(TypeAdapter(float).validate_python(tilt))
					radians_input = True
			params = handler(value)
			if radians_input:
				# Preserve the original NumPy scalar promotion in the row arithmetic.
				# A Python float would round some float32 centers differently.
				object.__setattr__(params, "tilt", np.float64(params.tilt))
			return params

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("center.manual currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		layout = type(state.buffer_spec.shape)
		if data.ndim != 3:
			raise ValueError("center.manual expects a three-dimensional tomography array")
		if self.params.tilt == 0:
			return _store_center(state, self.params.value_px)
		if state.source_window is None:
			raise RuntimeError("center.manual tilt requires PipelineState.source_window")
		y_size = data.shape[layout.axis_of("Y")]
		source_y = int(state.source_window.pos.Y)
		rows = source_y + np.arange(y_size, dtype=np.float32)
		return _store_center(state, self.params.value_px + (rows - self.params.reference_y) * self.params.tilt)


class EntropyLocalCenterStage(Stage):
	"""Select the lowest-entropy gridrec result from a local center sweep."""

	name = "center.entropy_local"
	contract = replace(
		_contract(runtime_requirements=(tomopy_requirement("circ_mask", "recon"),)),
		produces_artifacts=("center", "center_entropy"),
	)

	class Params(StageParams):
		start: float | None = None
		range: float = Field(default=5.0, ge=0)
		step: float = Field(default=0.25, gt=0)
		slice_index: int | None = Field(default=None, ge=0)
		filter_name: str = "butterworth"
		filter_parameters: tuple[float, ...] = (0.2, 2)
		ncore: int | None = Field(default=None, ge=1)
		circle_mask_ratio: float = Field(default=1.0, gt=0, le=1)
		max_expansions: int = Field(default=3, ge=0)
		diagnostic_images: bool = False
		diagnostic_subdirectory: Path = Path("center-diagnostics")

		@field_validator("diagnostic_subdirectory")
		@classmethod
		def relative_diagnostic_path(cls, value):
			if value.is_absolute() or not value.parts or ".." in value.parts:
				raise ValueError(
					"center.entropy_local diagnostic_subdirectory must be a safe relative path"
				)
			return value

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		if params.diagnostic_images:
			return replace(
				cls.contract, produces_artifacts=cls.contract.produces_artifacts + ("output_files",),
			)
		return cls.contract

	def _diagnostic_root(self, state: PipelineState) -> Path:
		output = state.recon.output_path
		if output is None:
			raise ValueError(
				"center.entropy_local diagnostic_images requires Recon.output_path"
			)
		output = Path(output)
		if output.suffix.lower() in {".h5", ".hdf5"}:
			root = output.parent / f"{output.stem}-{self.params.diagnostic_subdirectory}"
		else:
			root = output / self.params.diagnostic_subdirectory
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
		index = row_count // 2 if self.params.slice_index is None else self.params.slice_index
		if not 0 <= index < row_count:
			raise ValueError("center.entropy_local slice_index is outside the sinogram window")
		if state.source_window is None:
			raise RuntimeError("center.entropy_local requires PipelineState.source_window")
		current_start = data.shape[layout.axis_of("X")] / 2 if self.params.start is None else self.params.start
		angles = _angles(state, data.shape[layout.axis_of("Theta")])
		window = state.source_window.with_pad()
		source_slice = int(window.pos.Y + index * window.step.Y)
		sinogram = data[layout.point_index(Y=index)]
		diagnostic_root = self._diagnostic_root(state) if self.params.diagnostic_images else None

		import tomopy

		searches = []
		all_candidates = []
		all_entropies = []
		diagnostic_files = []
		previous_boundary = None
		termination = "cap-reached"
		converged = False
		selected_center = float(current_start)
		for search in range(self.params.max_expansions + 1):
			candidates = np.arange(
				current_start - self.params.range,
				current_start + self.params.range + self.params.step / 2,
				self.params.step,
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
					filter_name=self.params.filter_name,
					filter_par=self.params.filter_parameters,
					ncore=self.params.ncore,
				)
				reconstruction = tomopy.circ_mask(
					reconstruction,
					axis=0,
					ratio=self.params.circle_mask_ratio,
					ncore=self.params.ncore,
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
			if len(candidates) == 1 or self.params.range == 0 or (
				previous_boundary is not None and boundary != previous_boundary
			):
				termination = "boundary-persisted"
				break
			if search == self.params.max_expansions:
				termination = "cap-reached"
				break
			current_start += -2 * self.params.range if boundary == "left" else 2 * self.params.range
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
			"max_expansions": self.params.max_expansions,
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

	class Params(StageParams):
		ind: int | None = Field(default=None, ge=0)
		smin: float = -50
		smax: float = 50
		srad: float = 6
		step: float = 0.25
		ratio: float = 0.5
		drop: int = 20
		ncore: int | None = Field(default=None, ge=1)
		tilt_fit: Literal["polynomial"] | None = None
		tilt_samples: int = Field(default=5, ge=2)
		fit_degree: int = Field(default=1, ge=1)
		tilt_outlier_sigma: float = Field(default=3.5, gt=0)
		tilt_max_iterations: int = Field(default=5, ge=1)
		tilt_min_inliers: int | None = None
		tilt_residual_cutoff_px: float | None = Field(default=None, ge=0)
		tilt_qc_policy: Literal["reject", "force"] = "reject"
		tilt_quantization_px: float | None = Field(default=None, gt=0)
		fallback: Literal["entropy_local"] | None = None
		fallback_params: EntropyLocalCenterStage.Params = Field(
			default_factory=EntropyLocalCenterStage.Params,
		)

		@field_validator("fallback_params", mode="before")
		@classmethod
		def default_fallback_params(cls, value):
			return {} if value is None else value

		@model_validator(mode="after")
		def enough_inliers(self):
			if self.tilt_min_inliers is not None and self.tilt_min_inliers < self.fit_degree + 1:
				raise ValueError("center.vo tilt_min_inliers must exceed the polynomial degree")
			return self

	def _find(self, sinogram: np.ndarray) -> float:
		import tomopy

		center = float(tomopy.find_center_vo(sinogram, **self.params.model_dump(
			include={"smin", "smax", "srad", "step", "ratio", "drop", "ncore"},
		)))
		if not np.isfinite(center):
			raise ValueError("center.vo returned a non-finite center")
		return center

	def _robust_fit(
		self, rows: np.ndarray, centers: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
		"""Iteratively reject center residuals using a median/MAD scale."""

		minimum = self.params.tilt_min_inliers
		if minimum is None:
			minimum = min(len(rows), max(self.params.fit_degree + 1, 3))
		if len(rows) < minimum:
			raise ValueError(
				f"center.vo tilt fit requires at least {minimum} unique sampled rows"
			)
		mask = np.ones(len(rows), dtype=bool)
		clipping_limited = False
		iterations = 0
		for iterations in range(1, self.params.tilt_max_iterations + 1):
			coefficients = np.polyfit(rows[mask], centers[mask], self.params.fit_degree)
			residuals = centers - np.polyval(coefficients, rows)
			inlier_residuals = residuals[mask]
			residual_center = float(np.median(inlier_residuals))
			deviations = np.abs(residuals - residual_center)
			mad = float(np.median(np.abs(inlier_residuals - residual_center)))
			threshold = max(1e-6, self.params.tilt_outlier_sigma * 1.4826 * mad)
			updated = mask & (deviations <= threshold)
			if np.count_nonzero(updated) < minimum:
				clipping_limited = True
				break
			if np.array_equal(updated, mask):
				break
			mask = updated
		coefficients = np.polyfit(rows[mask], centers[mask], self.params.fit_degree)
		residuals = centers - np.polyval(coefficients, rows)
		return coefficients, residuals, mask, iterations, clipping_limited

	def _tilt_center(self, state: PipelineState, data: np.ndarray, layout: type) -> np.ndarray:
		row_count = data.shape[layout.axis_of("Y")]
		count = min(self.params.tilt_samples, row_count)
		local_rows = np.unique(np.linspace(0, row_count - 1, count).round().astype(int))
		if len(local_rows) <= self.params.fit_degree:
			raise ValueError("center.vo needs more sampled rows than the polynomial degree")
		if state.source_window is None:
			raise RuntimeError("center.vo tilt fit requires PipelineState.source_window")
		window = state.source_window.with_pad()
		source_rows = window.pos.Y + local_rows * window.step.Y
		centers = np.asarray([
			self._find(data[layout.point_index(Y=local)])
			for local in local_rows
		], dtype=np.float64)
		coefficients, residuals, inliers, iterations, clipping_limited = self._robust_fit(
			source_rows.astype(np.float64), centers,
		)
		evaluation_rows = window.pos.Y + np.arange(row_count) * window.step.Y
		unquantized = np.polyval(coefficients, evaluation_rows)
		center = unquantized
		if self.params.tilt_quantization_px is not None:
			center = (
				np.round(center / self.params.tilt_quantization_px) * self.params.tilt_quantization_px
			)
		inlier_residuals = residuals[inliers]
		residual_rms = float(np.sqrt(np.mean(np.square(inlier_residuals))))
		residual_max = float(np.max(np.abs(inlier_residuals)))
		qc_reasons = []
		if clipping_limited:
			qc_reasons.append("outlier clipping would leave too few inliers")
		if (
			self.params.tilt_residual_cutoff_px is not None
			and residual_rms > self.params.tilt_residual_cutoff_px
		):
			qc_reasons.append("inlier residual RMS exceeds cutoff")
		qc_passed = not qc_reasons
		status = "accepted" if qc_passed else (
			"forced" if self.params.tilt_qc_policy == "force" else "rejected"
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
			"residual_cutoff_px": self.params.tilt_residual_cutoff_px,
			"row_span": int(source_rows.max() - source_rows.min()),
			"fit_degree": self.params.fit_degree,
			"outlier_sigma": self.params.tilt_outlier_sigma,
			"fit_iterations": iterations,
			"quantization_px": self.params.tilt_quantization_px,
			"unquantized_center": unquantized,
			"qc_passed": qc_passed,
			"qc_policy": self.params.tilt_qc_policy,
			"qc_status": status,
			"qc_reasons": qc_reasons,
		}
		if not qc_passed and self.params.tilt_qc_policy == "reject":
			raise ValueError("center.vo tilt fit rejected: " + "; ".join(qc_reasons))
		return np.asarray(center, dtype=np.float32)

	def run(self, state: PipelineState) -> PipelineState:
		data, layout = _sinograms(state)
		if state.source_window is None:
			raise RuntimeError("center.vo requires PipelineState.source_window")
		row_count = data.shape[layout.axis_of("Y")]
		try:
			if self.params.tilt_fit == "polynomial":
				center = self._tilt_center(state, data, layout)
			else:
				index = row_count // 2 if self.params.ind is None else self.params.ind
				if not 0 <= index < row_count:
					raise ValueError("center.vo ind is outside the sinogram window")
				center = self._find(data[layout.point_index(Y=index)])
		except (ValueError, RuntimeError, ArithmeticError) as error:
			if self.params.fallback != "entropy_local":
				raise
			LOGGER.warning(
				"center.vo failed with %s; using center.entropy_local fallback",
				error,
				exc_info=True,
			)
			return EntropyLocalCenterStage(**self.params.fallback_params.model_dump()).run(state)
		return _store_center(state, center)


class PhaseCorrelationCenterStage(Stage):
	"""Find center from the first and last projections by phase correlation."""

	name = "center.pc"
	contract = replace(
		_contract(runtime_requirements=(tomopy_requirement("find_center_pc"),)),
		accepts=(ProjOrder, SinoOrder), prefers=ProjOrder, bounded_views_from=(),
		window_axes=(),
	)

	class Params(StageParams):
		tol: float = Field(default=0.5, gt=0)
		rotc_guess: float | None = None

	def run(self, state: PipelineState) -> PipelineState:
		data, layout = _sinograms(state)
		projection_count = data.shape[layout.axis_of("Theta")]
		if projection_count < 2:
			raise ValueError("center.pc requires at least two projections")

		import tomopy

		center = tomopy.find_center_pc(
			data[layout.point_index(Theta=0)],
			data[layout.point_index(Theta=projection_count - 1)],
			tol=self.params.tol,
			rotc_guess=self.params.rotc_guess,
		)
		if not np.isfinite(center):
			raise ValueError("center.pc returned a non-finite center")
		return _store_center(state, center)
