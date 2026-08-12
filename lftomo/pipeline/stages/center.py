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
	working_view,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.pipeline.runtime import tomopy_requirement
from lftomo.pipeline.layout import (
	configured_layout,
	n_projections,
	n_rows,
	projection_view,
	resolve_layout,
	sinogram_scratch,
)
from lftomo.util.shapes import ProjOrder, SinoOrder


LOGGER = logging.getLogger(__name__)


def _contract(runtime_requirements: tuple[RuntimeRequirement, ...] = ()) -> StageContract:
	return StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
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
	data = working_view(state)
	if data.ndim != 3:
		raise ValueError("center Stages expect a (Y, Theta, X) sinogram array")
	return data, resolve_layout(state, None, "center")


def _angles(state: PipelineState, theta_count: int) -> np.ndarray:
	angles = state.artifacts.get("angles")
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
		order: str | None = None,
	):
		if tilt_radians is not None and tilt != 0:
			raise ValueError("center.manual accepts either tilt or tilt_radians, not both")
		if find_center or pc_fallback or vo_fallback:
			raise ValueError("automatic center finding must use a center.vo or center.pc Stage")
		if center_samples_per_job is not None and center_samples_per_job < 1:
			raise ValueError("center_samples_per_job must be positive")
		compatibility = configured_layout(order)
		shape = compatibility or SinoOrder
		self.contract = StageContract(
			expected_input=shape, produced_output=shape, window_axes=("Y",),
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
		self.order = order

	@classmethod
	def from_config(cls, config: StageConfig) -> "ManualCenterStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("center.manual currently requires the NumPy backend")
		data = working_view(state)
		layout = resolve_layout(state, self.order, self.name)
		if data.ndim != 3:
			raise ValueError("center.manual expects a three-dimensional tomography array")
		if self.slope == 0:
			return _store_center(state, self.value_px)
		y_size = n_rows(data, layout)
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
		row_count = n_rows(data, layout)
		index = row_count // 2 if self.slice_index is None else self.slice_index
		if not 0 <= index < row_count:
			raise ValueError("center.entropy_local slice_index is outside the sinogram window")
		current_start = data.shape[2] / 2 if self.start is None else self.start
		angles = _angles(state, n_projections(data, layout))
		sinogram = sinogram_scratch(data, layout, index)
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
		fallback: str | None = None,
		fallback_params: dict | None = None,
	):
		if tilt_fit not in (None, "polynomial"):
			raise ValueError("center.vo tilt_fit must be 'polynomial' or omitted")
		if tilt_samples < 2:
			raise ValueError("center.vo tilt_samples must be at least two")
		if fit_degree < 1:
			raise ValueError("center.vo fit_degree must be positive")
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

	def run(self, state: PipelineState) -> PipelineState:
		data, layout = _sinograms(state)
		row_count = n_rows(data, layout)
		try:
			if self.tilt_fit == "polynomial":
				count = min(self.tilt_samples, row_count)
				indices = np.unique(np.linspace(0, row_count - 1, count).round().astype(int))
				if len(indices) <= self.fit_degree:
					raise ValueError("center.vo needs more sampled rows than the polynomial degree")
				centers = np.asarray([
					self._find(sinogram_scratch(data, layout, int(index))) for index in indices
				])
				coefficients = np.polyfit(indices, centers, self.fit_degree)
				center = np.polyval(coefficients, np.arange(row_count)).astype(np.float32)
				state.artifacts["center_fit"] = {
					"indices": indices,
					"centers": centers,
					"coefficients": coefficients,
				}
			else:
				index = row_count // 2 if self.ind is None else self.ind
				if not 0 <= index < row_count:
					raise ValueError("center.vo ind is outside the sinogram window")
				center = self._find(sinogram_scratch(data, layout, index))
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
			expected_input=SinoOrder, produced_output=SinoOrder,
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
		projection_count = n_projections(data, layout)
		if projection_count < 2:
			raise ValueError("center.pc requires at least two projections")

		import tomopy

		center = tomopy.find_center_pc(
			projection_view(data, layout, 0),
			projection_view(data, layout, projection_count - 1),
			tol=self.tol,
			rotc_guess=self.rotc_guess,
		)
		if not np.isfinite(center):
			raise ValueError("center.pc returned a non-finite center")
		return _store_center(state, center)
