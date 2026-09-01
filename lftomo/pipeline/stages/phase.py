"""Propagation-based phase-retrieval stages."""

from __future__ import annotations

from functools import partial

import numpy as np

from lftomo.config import StageConfig
from lftomo.pipeline.core import (
	Locality,
	PadRequirement,
	PipelineState,
	RuntimeRequirement,
	Stage,
	StageContract,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.storage import buffer_view
from lftomo.util.shapes import ImgSlice, ProjOrder, SinoOrder, SliceIter, Theta


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
		), stage.alpha,
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


class PaganinFilterStage(Stage):
	"""Apply only Paganin's transmission-preserving Fourier filter."""

	name = "phase.paganin_filter"

	def __init__(self, alpha: float = 0.06, enabled: bool = True):
		if alpha <= 0:
			raise ValueError("Paganin alpha must be positive")
		self.alpha = alpha
		self.enabled = enabled
		self.contract = StageContract(
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
			cacheable_prefix=True,
			window_axes=("Y", "Theta"),
			required_metadata=("energy_keV", "pixel_size_um", "ssd_cm") if enabled else (),
			accepts_domain=(DataDomain.TRANSMISSION,) if enabled else (
				DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
			),
			produces_domain=PRESERVE,
			locality=Locality.PROJECTION,
			produces_artifacts=("phase_filter",) if enabled else (),
			managed_artifacts=("phase_filter",) if enabled else (),
			required_pad=PadRequirement(phase_fft_y=enabled),
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "PaganinFilterStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("PaganinFilterStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		shape = state.buffer_spec.shape
		layout = type(shape)
		if data.ndim != 3:
			raise ValueError("phase.paganin_filter expects a three-dimensional tomography array")
		if not self.enabled:
			return state
		transfer = _resolved_phase_filter(
			state, (
				data.shape[layout.axis_of("Y")], data.shape[layout.axis_of("X")],
			), self.alpha,
		)
		for index in Theta.drange(0, shape.Theta, 1).get_iter(partial(SliceIter, shape=shape)):
			image = data[index]
			image[:] = _paganin_filter(image, transfer)
		return state


class NegativeLogStage(Stage):
	"""Perform the single terminal conversion from transmission to attenuation."""

	name = "phase.negative_log"

	def __init__(self, epsilon: float | None = 1e-8, maximum: float | None = None,
			enabled: bool = True, invert: bool = False):
		if epsilon is not None and epsilon <= 0:
			raise ValueError("negative-log epsilon must be positive or null")
		if maximum is not None and maximum <= 0:
			raise ValueError("negative-log maximum must be positive or null")
		if epsilon is not None and maximum is not None and maximum < epsilon:
			raise ValueError("negative-log maximum must not be smaller than epsilon")
		self.epsilon = epsilon
		self.maximum = maximum
		self.enabled = enabled
		self.invert = invert
		self.contract = StageContract(
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
			cacheable_prefix=True,
			window_axes=("Y", "Theta"),
			accepts_domain=(DataDomain.TRANSMISSION,) if enabled else (
				DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
			),
			produces_domain=DataDomain.ATTENUATION if enabled else PRESERVE,
			locality=Locality.PROJECTION,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "NegativeLogStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("NegativeLogStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		shape = state.buffer_spec.shape
		if data.ndim != 3:
			raise ValueError("phase.negative_log expects a three-dimensional tomography array")
		if not self.enabled:
			return state
		for index in Theta.drange(0, shape.Theta, 1).get_iter(partial(SliceIter, shape=shape)):
			image = data[index]
			image[:] = _negative_log(
				image, epsilon=self.epsilon, maximum=self.maximum, invert=self.invert,
			)
		return state


class BetaScaleStage(Stage):
	"""Apply the wavelength-dependent scale used by coupled phase/absorption."""

	name = "phase.beta_scale"

	def __init__(self, enabled: bool = True):
		self.enabled = enabled
		self.contract = StageContract(
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
			cacheable_prefix=True,
			window_axes=("Y", "Theta"),
			required_metadata=("energy_keV",) if enabled else (),
			accepts_domain=(DataDomain.ATTENUATION,) if enabled else (
				DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
			),
			produces_domain=PRESERVE,
			locality=Locality.PROJECTION,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "BetaScaleStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("BetaScaleStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("phase.beta_scale expects a three-dimensional tomography array")
		if not self.enabled:
			return state
		data /= 4 * np.pi / _wavelength_cm(state.scan.energy_keV)
		return state


class PaganinStage(Stage):
	"""Standard single-distance Paganin phase-only retrieval."""

	name = "phase.paganin"
	contract = StageContract(
		produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,),
		produces_domain=DataDomain.ATTENUATION,
		locality=Locality.PROJECTION,
		accepts=(ProjOrder, SinoOrder),
		prefers=ProjOrder,
		mutates_input=True,
		cacheable_prefix=True,
		window_axes=("Y", "Theta"),
		required_metadata=("energy_keV", "pixel_size_um", "ssd_cm"),
		produces_artifacts=("phase_filter",),
		managed_artifacts=("phase_filter",),
	)

	def __init__(self, alpha: float = 0.06, epsilon: float = 1e-8,
				enabled: bool = True, pad_y: bool = False,
			):
		if alpha <= 0:
			raise ValueError("Paganin alpha must be positive")
		self.alpha = alpha
		self.epsilon = epsilon
		self.enabled = enabled
		self.pad_y = pad_y
		self.contract = StageContract(
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
			cacheable_prefix=True,
			window_axes=("Y", "Theta"),
			required_metadata=("energy_keV", "pixel_size_um", "ssd_cm") if enabled else (),
			accepts_domain=(DataDomain.TRANSMISSION,) if enabled else (
				DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
			),
			produces_domain=DataDomain.ATTENUATION if enabled else PRESERVE,
			locality=Locality.PROJECTION,
			produces_artifacts=("phase_filter",) if enabled else (),
			managed_artifacts=("phase_filter",) if enabled else (),
			required_pad=PadRequirement(phase_fft_y=enabled),
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "PaganinStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("PaganinStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		shape = state.buffer_spec.shape
		layout = type(shape)
		if data.ndim != 3:
			raise ValueError("phase.paganin expects a (Y, Theta, X) sinogram array")
		if not self.enabled:
			return state
		transfer = _resolved_phase_filter(
			state, (
				data.shape[layout.axis_of("Y")], data.shape[layout.axis_of("X")],
			), self.alpha,
		)
		for index in Theta.drange(0, shape.Theta, 1).get_iter(partial(SliceIter, shape=shape)):
			image = data[index]
			filtered = _paganin_filter(image, transfer)
			image[:] = _negative_log(filtered, epsilon=self.epsilon, maximum=None)
		return state


class PaganinPhaseAbsorptionStage(Stage):
	"""Sorotassu's one-shot coupled phase/absorption Paganin retrieval."""

	name = "phase.paganin_phase_absorption"
	contract = StageContract(
		produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,),
		produces_domain=DataDomain.ATTENUATION,
		locality=Locality.PROJECTION,
		accepts=(ProjOrder, SinoOrder),
		prefers=ProjOrder,
		mutates_input=True,
		cacheable_prefix=True,
		window_axes=("Y", "Theta"),
		required_metadata=("energy_keV", "pixel_size_um", "ssd_cm"),
		produces_artifacts=("phase_filter",),
		managed_artifacts=("phase_filter",),
	)

	def __init__(self, alpha: float = 0.06, epsilon: float = 1e-8,
			):
		if alpha <= 0:
			raise ValueError("phase/absorption alpha must be positive")
		self.alpha = alpha
		self.epsilon = epsilon
		self.contract = StageContract(
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=True,
			cacheable_prefix=True,
			window_axes=("Y", "Theta"),
			required_metadata=("energy_keV", "pixel_size_um", "ssd_cm"),
			accepts_domain=(DataDomain.TRANSMISSION,),
			produces_domain=DataDomain.ATTENUATION,
			locality=Locality.PROJECTION,
			produces_artifacts=("phase_filter",),
			managed_artifacts=("phase_filter",),
			required_pad=PadRequirement(phase_fft_y=True),
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "PaganinPhaseAbsorptionStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("PaganinPhaseAbsorptionStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		shape = state.buffer_spec.shape
		layout = type(shape)
		if data.ndim != 3:
			raise ValueError("phase.paganin_phase_absorption expects a (Y, Theta, X) sinogram array")
		wavelength = _wavelength_cm(state.scan.energy_keV)
		beta_scale = 4 * np.pi / wavelength
		transfer = _resolved_phase_filter(
			state, (
				data.shape[layout.axis_of("Y")], data.shape[layout.axis_of("X")],
			), self.alpha,
		)
		for index in Theta.drange(0, shape.Theta, 1).get_iter(partial(SliceIter, shape=shape)):
			image = data[index]
			# Corrected-paper branch from dan_scripts/prep/phase.py:22-144.
			transmission = _paganin_filter(image, transfer)
			absorption = _negative_log(
				transmission, epsilon=self.epsilon, maximum=0.99999999,
			)
			image[:] = absorption / beta_scale
		return state


class BronnikovAidedCorrectionFilterStage(Stage):
	"""Apply BAC while preserving its corrected transmission intermediate."""

	name = "phase.bac_correction"

	def __init__(self, alpha: float, gamma: float = 1.0, enabled: bool = True,
			processes: int = 1, clip_input: bool = False,
			clip_minimum: float = 1e-9, clip_threshold: float = 2.0,
			clip_high_value: float = 1.0):
		if alpha <= 0 or gamma <= 0:
			raise ValueError("BAC alpha and gamma must be positive")
		if processes < 1:
			raise ValueError("BAC processes must be positive")
		self.contract = StageContract(
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
			cacheable_prefix=True,
			accepts_domain=(DataDomain.TRANSMISSION,) if enabled else (
				DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
			),
			produces_domain=PRESERVE,
			locality=Locality.PROJECTION,
			window_axes=("Y", "Theta"),
		)
		self.alpha = alpha
		self.gamma = gamma
		self.enabled = enabled
		self.processes = processes
		self.clip_input = clip_input
		self.clip_minimum = clip_minimum
		self.clip_threshold = clip_threshold
		self.clip_high_value = clip_high_value

	@classmethod
	def from_config(cls, config: StageConfig) -> "BronnikovAidedCorrectionFilterStage":
		return cls(**config.params)

	def _correct(self, image: np.ndarray) -> np.ndarray:
		if self.clip_input:
			image[~np.isfinite(image)] = self.clip_minimum
			image[image <= 0] = self.clip_minimum
			image[image >= self.clip_threshold] = self.clip_high_value
		return _bac_corrected_transmission(
			image, alpha=self.alpha, gamma=self.gamma,
		)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError(
				"BronnikovAidedCorrectionFilterStage currently requires the NumPy backend"
			)
		data = np.asarray(state.working_array)
		shape = state.buffer_spec.shape
		if data.ndim != 3:
			raise ValueError("phase.bac_correction expects a three-dimensional tomography array")
		if not self.enabled:
			return state
		for index in Theta.drange(0, shape.Theta, 1).get_iter(partial(SliceIter, shape=shape)):
			image = data[index]
			image[:] = self._correct(image)
		return state


class BronnikovAidedCorrectionStage(Stage):
	"""Apply hotopy Bronnikov-aided correction to each projection image."""

	name = "phase.bac"
	contract = StageContract(
		mutates_input=True,
		cacheable_prefix=True,
		window_axes=("Y", "Theta"),
		runtime_requirements=(_BAC_RUNTIME_REQUIREMENT,),
	)

	def __init__(self, alpha: float, gamma: float = 1.0, enabled: bool = True,
				invert: bool = False, processes: int = 1, device: str | None = None,
				clip_input: bool = False,
				clip_minimum: float = 1e-9, clip_threshold: float = 2.0,
				clip_high_value: float = 1.0):
		if alpha <= 0 or gamma <= 0:
			raise ValueError("BAC alpha and gamma must be positive")
		if processes < 1:
			raise ValueError("BAC processes must be positive")
		self.contract = StageContract(
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
			cacheable_prefix=True,
			accepts_domain=(DataDomain.TRANSMISSION,) if enabled else (
				DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
			),
			produces_domain=DataDomain.ATTENUATION if enabled else PRESERVE,
			locality=Locality.PROJECTION,
			window_axes=("Y", "Theta"),
			runtime_requirements=type(self).contract.runtime_requirements,
		)
		self.alpha = alpha
		self.gamma = gamma
		self.enabled = enabled
		self.invert = invert
		self.processes = processes
		self.device = device
		self.clip_input = clip_input
		self.clip_minimum = clip_minimum
		self.clip_threshold = clip_threshold
		self.clip_high_value = clip_high_value

	@classmethod
	def from_config(cls, config: StageConfig) -> "BronnikovAidedCorrectionStage":
		return cls(**config.params)

	def _correct(self, image: np.ndarray) -> np.ndarray:
		try:
			from hotopy.holo import BronnikovAidedCorrection
		except ImportError as error:
			raise RuntimeError(
				"phase.bac requires the optional hotopy dependency; "
				"install lftomo with the 'bac' extra"
			) from error

		if self.clip_input:
			image[~np.isfinite(image)] = self.clip_minimum
			image[image <= 0] = self.clip_minimum
			image[image >= self.clip_threshold] = self.clip_high_value
		operator = BronnikovAidedCorrection(
			image.shape,
			alpha=self.alpha,
			gamma=self.gamma,
			device=self.device,
		)
		corrected = np.asarray(operator(image), dtype=np.float32)
		return -corrected if self.invert else corrected

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("BronnikovAidedCorrectionStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		shape = state.buffer_spec.shape
		if data.ndim != 3:
			raise ValueError("phase.bac expects a three-dimensional tomography array")
		if not self.enabled:
			return state
		for index in Theta.drange(0, shape.Theta, 1).get_iter(partial(SliceIter, shape=shape)):
			image = data[index]
			image[:] = self._correct(image)
		return state
