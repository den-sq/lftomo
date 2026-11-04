"""Propagation-based phase-retrieval stages."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pydantic import Field, model_validator

from lftomo.pipeline.core import (
	PadRequirement,
	PipelineState,
	ProjectionStage,
	ResourceHints,
	RuntimeRequirement,
	Stage,
	StageContract,
	StageParams,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.storage import buffer_view
from lftomo.util.shapes import ImgSlice


def _managed_phase_filter(
	state: PipelineState, values: np.ndarray, signature: tuple,
) -> np.ndarray:
	"""Publish the detector-sized Fourier filter as a shared run artifact."""

	if state.storage is None:
		state.artifacts["phase_filter"] = values
		state.artifacts["phase_filter_signature"] = signature
		return values
	previous = state.artifacts.get("phase_filter")
	artifact = state.storage.allocate(
		ImgSlice(*values.shape), values.dtype,
		role="phase_filter", reuse_from=("phase_filter",),
	)
	artifact.array()[...] = values
	state.artifacts["phase_filter"] = artifact
	state.artifacts["phase_filter_signature"] = signature
	if hasattr(previous, "name"):
		state.storage.retire(previous, role="phase_filter")
	return artifact.array()


def _resolved_phase_filter(
	state: PipelineState, shape: tuple[int, int], alpha: float,
) -> np.ndarray:
	signature = (
		"paganin", shape, float(alpha), float(state.scan.energy_keV),
		float(state.scan.pixel_size_um), float(state.scan.ssd_cm),
	)
	if state.artifacts.get("phase_filter_signature") == signature:
		return buffer_view(state.artifacts["phase_filter"])
	values = _paganin_transfer(
		shape,
		alpha=alpha,
		energy_kev=state.scan.energy_keV,
		pixel_size_um=state.scan.pixel_size_um,
		ssd_cm=state.scan.ssd_cm,
	)
	return _managed_phase_filter(state, values, signature)


def prepare_phase_filter(stage: Stage, state: PipelineState) -> None:
	"""Create one shared filter before projection workers are dispatched."""

	data = np.asarray(state.working_array)
	layout = type(state.buffer_spec.shape)
	_resolved_phase_filter(
		state, (
			data.shape[layout.axis_of("Y")], data.shape[layout.axis_of("X")],
		), stage.params.alpha,
	)


_BAC_RUNTIME_REQUIREMENT = RuntimeRequirement(
	capability="bac",
	package="hotopy",
	module="hotopy.holo",
	probe_attributes=("BronnikovAidedCorrection",),
	corrective_action="install lftomo with the bac extra in this interpreter",
)


PLANCK_CONSTANT = 6.58211928e-19  # keV s
SPEED_OF_LIGHT = 299792458e2  # cm / s


def _wavelength_cm(energy_kev: float) -> float:
	return 2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy_kev


def _angular_frequency_squared(shape: tuple[int, int], pixel_size_cm: float) -> np.ndarray:
	"""Return the intentional DFT-bin angular grid documented in issue #199."""

	ky = 2 * np.pi * np.fft.fftfreq(shape[0], d=pixel_size_cm)
	kx = 2 * np.pi * np.fft.fftfreq(shape[1], d=pixel_size_cm)
	return np.add.outer(np.square(ky), np.square(kx))


def _paganin_transfer(
	shape: tuple[int, int],
	*,
	alpha: float,
	energy_kev: float,
	pixel_size_um: float,
	ssd_cm: float,
) -> np.ndarray:
	"""Return the Paganin Fourier transfer array for one projection shape."""

	pixel_size_cm = pixel_size_um / 10_000
	wavelength = _wavelength_cm(energy_kev)
	w2 = _angular_frequency_squared(shape, pixel_size_cm)
	return 1 / (1 + ssd_cm * alpha * wavelength * w2 / (4 * np.pi))


def _paganin_filter(image: np.ndarray, transfer: np.ndarray) -> np.ndarray:
	"""Return the transmission-like result of the Paganin Fourier filter."""

	return np.real(np.fft.ifft2(np.fft.fft2(image) * transfer))


def _negative_log(
	image: np.ndarray,
	*,
	epsilon: float | None,
	maximum: float | None,
	invert: bool = False,
) -> np.ndarray:
	"""Apply the one terminal transmission-to-attenuation conversion."""

	if epsilon is None:
		if not np.isfinite(image).all() or np.any(image <= 0):
			raise ValueError(
				"phase.negative_log requires finite positive transmission when epsilon is null"
			)
		transmission = np.minimum(image, maximum) if maximum is not None else image
	else:
		transmission = np.clip(image, epsilon, maximum)
	result = -np.log(transmission)
	return -result if invert else result


def _bac_filter_kernel(shape: tuple[int, int], alpha: float) -> np.ndarray:
	"""Return HoToPy BAC's dimensionless regularized Laplace kernel."""

	qy = 2 * np.pi * np.fft.fftfreq(shape[0])
	qx = 2 * np.pi * np.fft.fftfreq(shape[1])
	w2 = np.add.outer(np.square(qy), np.square(qx))
	return w2 / (w2 + alpha)


def _bac_corrected_transmission(
	image: np.ndarray,
	*,
	alpha: float,
	gamma: float,
) -> np.ndarray:
	"""Return BAC's corrected transmission before HoToPy's terminal ``-log``."""

	kernel = _bac_filter_kernel(image.shape, alpha)
	correction = -np.real(np.fft.ifft2(np.fft.fft2(image - 1) * kernel))
	denominator = 1 - gamma * correction
	corrected = image / denominator
	if not np.isfinite(corrected).all() or np.any(corrected <= 0):
		raise ValueError(
			"phase.bac_correction produced non-positive or non-finite transmission; "
			"adjust alpha/gamma or the input-clipping policy"
		)
	return corrected


class PaganinFilterStage(ProjectionStage):
	"""Apply only Paganin's transmission-preserving Fourier filter."""

	name = "phase.paganin_filter"
	contract = replace(
		ProjectionStage.contract,
		extent_sensitive_axes=("Y", "X"),
		mutates_input=True,
		cacheable_prefix=True,
		window_axes=("Y", "Theta"),
		required_metadata=("energy_keV", "pixel_size_um", "ssd_cm"),
		accepts_domain=(DataDomain.TRANSMISSION,),
		produces_domain=PRESERVE,
		produces_artifacts=("phase_filter",),
		managed_artifacts=("phase_filter",),
		required_pad=PadRequirement(phase_fft_y=True),
	)

	class Params(StageParams):
		alpha: float = Field(default=0.06, gt=0)
		enabled: bool = True

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		if params.enabled:
			return cls.contract
		return replace(
			cls.contract,
			mutates_input=False,
			required_metadata=(),
			accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
			produces_domain=PRESERVE,
			produces_artifacts=(),
			managed_artifacts=(),
			required_pad=PadRequirement(),
		)

	def run_projection(self, image: np.ndarray, state: PipelineState) -> np.ndarray | None:
		if not self.params.enabled:
			return None
		transfer = _resolved_phase_filter(state, image.shape, self.params.alpha)
		return _paganin_filter(image, transfer)


class NegativeLogStage(ProjectionStage):
	"""Perform the single terminal conversion from transmission to attenuation."""

	name = "phase.negative_log"
	contract = replace(
		ProjectionStage.contract,
		extent_sensitive_axes=(),
		mutates_input=True,
		cacheable_prefix=True,
		window_axes=("Y", "Theta"),
		accepts_domain=(DataDomain.TRANSMISSION,),
		produces_domain=DataDomain.ATTENUATION,
	)

	class Params(StageParams):
		epsilon: float | None = Field(default=1e-8, gt=0)
		maximum: float | None = Field(default=None, gt=0)
		enabled: bool = True
		invert: bool = False

		@model_validator(mode="after")
		def validate_bounds(self):
			if self.epsilon is not None and self.maximum is not None and self.maximum < self.epsilon:
				raise ValueError("negative-log maximum must not be smaller than epsilon")
			return self

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		if params.enabled:
			return cls.contract
		return replace(
			cls.contract,
			mutates_input=False,
			accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
			produces_domain=PRESERVE,
		)

	def run_projection(self, image: np.ndarray, state: PipelineState) -> np.ndarray | None:
		if not self.params.enabled:
			return None
		return _negative_log(
			image, epsilon=self.params.epsilon, maximum=self.params.maximum,
			invert=self.params.invert,
		)


class BetaScaleStage(Stage):
	"""Apply the wavelength-dependent scale used by coupled phase/absorption."""

	name = "phase.beta_scale"
	contract = replace(
		ProjectionStage.contract,
		extent_sensitive_axes=(),
		mutates_input=True,
		cacheable_prefix=True,
		window_axes=("Y", "Theta"),
		required_metadata=("energy_keV",),
		accepts_domain=(DataDomain.ATTENUATION,),
		produces_domain=PRESERVE,
	)

	class Params(StageParams):
		enabled: bool = True

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		if params.enabled:
			return cls.contract
		return replace(
			cls.contract,
			mutates_input=False,
			required_metadata=(),
			accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
		)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("BetaScaleStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("phase.beta_scale expects a three-dimensional tomography array")
		if not self.params.enabled:
			return state
		data /= 4 * np.pi / _wavelength_cm(state.scan.energy_keV)
		return state


class PaganinStage(PaganinFilterStage):
	"""Standard single-distance Paganin phase-only retrieval."""

	name = "phase.paganin"
	contract = replace(
		PaganinFilterStage.contract, produces_domain=DataDomain.ATTENUATION,
	)

	class Params(PaganinFilterStage.Params):
		epsilon: float | None = 1e-8
		pad_y: bool = False

	def run_projection(self, image: np.ndarray, state: PipelineState) -> np.ndarray | None:
		if not self.params.enabled:
			return None
		transfer = _resolved_phase_filter(state, image.shape, self.params.alpha)
		filtered = _paganin_filter(image, transfer)
		return _negative_log(filtered, epsilon=self.params.epsilon, maximum=None)


class PaganinPhaseAbsorptionStage(ProjectionStage):
	"""Sorotassu's one-shot coupled phase/absorption Paganin retrieval."""

	name = "phase.paganin_phase_absorption"
	contract = replace(
		PaganinFilterStage.contract, produces_domain=DataDomain.ATTENUATION,
	)

	class Params(StageParams):
		alpha: float = Field(default=0.06, gt=0)
		epsilon: float | None = 1e-8

	def run_projection(self, image: np.ndarray, state: PipelineState) -> np.ndarray:
		wavelength = _wavelength_cm(state.scan.energy_keV)
		beta_scale = 4 * np.pi / wavelength
		transfer = _resolved_phase_filter(state, image.shape, self.params.alpha)
		# Corrected-paper branch from dan_scripts/prep/phase.py:22-144.
		transmission = _paganin_filter(image, transfer)
		absorption = _negative_log(
			transmission, epsilon=self.params.epsilon, maximum=0.99999999,
		)
		return absorption / beta_scale


class BronnikovAidedCorrectionFilterStage(ProjectionStage):
	"""Apply BAC while preserving its corrected transmission intermediate."""

	name = "phase.bac_correction"
	contract = replace(
		ProjectionStage.contract,
		extent_sensitive_axes=("Y", "X"),
		mutates_input=True,
		cacheable_prefix=True,
		accepts_domain=(DataDomain.TRANSMISSION,),
		produces_domain=PRESERVE,
		window_axes=("Y", "Theta"),
	)

	class Params(StageParams):
		alpha: float = Field(gt=0)
		gamma: float = Field(default=1.0, gt=0)
		enabled: bool = True
		processes: int = Field(default=1, ge=1)
		clip_input: bool = False
		clip_minimum: float = 1e-9
		clip_threshold: float = 2.0
		clip_high_value: float = 1.0

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		if params.enabled:
			return cls.contract
		return replace(
			cls.contract,
			mutates_input=False,
			accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
			produces_domain=PRESERVE,
		)

	def _correct(self, image: np.ndarray) -> np.ndarray:
		if self.params.clip_input:
			image[~np.isfinite(image)] = self.params.clip_minimum
			image[image <= 0] = self.params.clip_minimum
			image[image >= self.params.clip_threshold] = self.params.clip_high_value
		return _bac_corrected_transmission(
			image, alpha=self.params.alpha, gamma=self.params.gamma,
		)

	def run_projection(self, image: np.ndarray, state: PipelineState) -> np.ndarray | None:
		if not self.params.enabled:
			return None
		return self._correct(image)


class BronnikovAidedCorrectionStage(BronnikovAidedCorrectionFilterStage):
	"""Apply hotopy Bronnikov-aided correction to each projection image."""

	name = "phase.bac"
	contract = replace(
		BronnikovAidedCorrectionFilterStage.contract,
		produces_domain=DataDomain.ATTENUATION,
		resources=ResourceHints(cpu_thread_pool="torch"),
		runtime_requirements=(_BAC_RUNTIME_REQUIREMENT,),
	)

	class Params(BronnikovAidedCorrectionFilterStage.Params):
		invert: bool = False
		device: str | None = None

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		contract = super().contract_for(params)
		if params.enabled:
			return contract
		return replace(
			contract, resources=replace(contract.resources, cpu_thread_pool=None),
		)

	def _correct(self, image: np.ndarray) -> np.ndarray:
		try:
			from hotopy.holo import BronnikovAidedCorrection
		except ImportError as error:
			raise RuntimeError(
				"phase.bac requires the optional hotopy dependency; "
				"install lftomo with the 'bac' extra"
			) from error

		if self.params.clip_input:
			image[~np.isfinite(image)] = self.params.clip_minimum
			image[image <= 0] = self.params.clip_minimum
			image[image >= self.params.clip_threshold] = self.params.clip_high_value
		operator = BronnikovAidedCorrection(
			image.shape,
			alpha=self.params.alpha,
			gamma=self.params.gamma,
			device=self.params.device,
		)
		corrected = np.asarray(operator(image), dtype=np.float32)
		return -corrected if self.params.invert else corrected
