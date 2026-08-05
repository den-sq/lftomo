"""Propagation-based phase-retrieval stages."""

from __future__ import annotations

import numpy as np
from multiprocessing.pool import ThreadPool

from lftomo.config import StageConfig
from lftomo.pipeline.core import Locality, PipelineState, ResourceHints, Stage, StageContract
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.pipeline.stages._layout import (
	configured_layout,
	n_projections,
	n_rows,
	projection_view,
	resolve_layout,
)
from lftomo.util.shapes import ProjOrder, SinoOrder


PLANCK_CONSTANT = 6.58211928e-19  # keV s
SPEED_OF_LIGHT = 299792458e2  # cm / s


def _wavelength_cm(energy_kev: float) -> float:
	return 2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy_kev


def _angular_frequency_squared(shape: tuple[int, int], pixel_size_cm: float) -> np.ndarray:
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

	def __init__(self, alpha: float = 0.06, enabled: bool = True,
			order: str | None = None):
		if alpha <= 0:
			raise ValueError("Paganin alpha must be positive")
		self.alpha = alpha
		self.enabled = enabled
		self.order = order
		compatibility = configured_layout(order)
		self.contract = StageContract(
			expected_input=compatibility or SinoOrder,
			produced_output=compatibility or SinoOrder,
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
			window_axes=("Y", "Theta"),
			required_metadata=("energy_keV", "pixel_size_um", "ssd_cm") if enabled else (),
			resources=ResourceHints(memory_multiplier=2.0),
			accepts_domain=(DataDomain.TRANSMISSION,) if enabled else (
				DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
			),
			produces_domain=PRESERVE,
			locality=Locality.PROJECTION,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "PaganinFilterStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("PaganinFilterStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		layout = resolve_layout(state, self.order, self.name)
		if data.ndim != 3:
			raise ValueError("phase.paganin_filter expects a three-dimensional tomography array")
		if not self.enabled:
			return state
		transfer = _paganin_transfer(
			(n_rows(data, layout), data.shape[2]),
			alpha=self.alpha,
			energy_kev=state.scan.energy_keV,
			pixel_size_um=state.scan.pixel_size_um,
			ssd_cm=state.scan.ssd_cm,
		)
		for projection in range(n_projections(data, layout)):
			image = projection_view(data, layout, projection)
			image[:] = _paganin_filter(image, transfer)
		return state


class NegativeLogStage(Stage):
	"""Perform the single terminal conversion from transmission to attenuation."""

	name = "phase.negative_log"

	def __init__(self, epsilon: float | None = 1e-8, maximum: float | None = None,
			enabled: bool = True, invert: bool = False, order: str | None = None):
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
		self.order = order
		compatibility = configured_layout(order)
		self.contract = StageContract(
			expected_input=compatibility or SinoOrder,
			produced_output=compatibility or SinoOrder,
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
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
		layout = resolve_layout(state, self.order, self.name)
		if data.ndim != 3:
			raise ValueError("phase.negative_log expects a three-dimensional tomography array")
		if not self.enabled:
			return state
		for projection in range(n_projections(data, layout)):
			image = projection_view(data, layout, projection)
			image[:] = _negative_log(
				image, epsilon=self.epsilon, maximum=self.maximum, invert=self.invert,
			)
		return state


class BetaScaleStage(Stage):
	"""Apply the wavelength-dependent scale used by coupled phase/absorption."""

	name = "phase.beta_scale"

	def __init__(self, enabled: bool = True, order: str | None = None):
		self.enabled = enabled
		self.order = order
		compatibility = configured_layout(order)
		self.contract = StageContract(
			expected_input=compatibility or SinoOrder,
			produced_output=compatibility or SinoOrder,
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
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
		resolve_layout(state, self.order, self.name)
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
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,),
		produces_domain=DataDomain.ATTENUATION,
		locality=Locality.PROJECTION,
		accepts=(ProjOrder, SinoOrder),
		prefers=ProjOrder,
		mutates_input=True,
		window_axes=("Y", "Theta"),
		required_metadata=("energy_keV", "pixel_size_um", "ssd_cm"),
		resources=ResourceHints(memory_multiplier=2.0),
	)

	def __init__(self, alpha: float = 0.06, epsilon: float = 1e-8,
				enabled: bool = True, pad_y: bool = False,
				order: str | None = None):
		if alpha <= 0:
			raise ValueError("Paganin alpha must be positive")
		self.alpha = alpha
		self.epsilon = epsilon
		self.enabled = enabled
		self.pad_y = pad_y
		self.order = order
		compatibility = configured_layout(order)
		self.contract = StageContract(
			expected_input=compatibility or SinoOrder,
			produced_output=compatibility or SinoOrder,
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
			window_axes=("Y", "Theta"),
			required_metadata=("energy_keV", "pixel_size_um", "ssd_cm") if enabled else (),
			resources=ResourceHints(memory_multiplier=2.0),
			accepts_domain=(DataDomain.TRANSMISSION,) if enabled else (
				DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
			),
			produces_domain=DataDomain.ATTENUATION if enabled else PRESERVE,
			locality=Locality.PROJECTION,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "PaganinStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("PaganinStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		layout = resolve_layout(state, self.order, self.name)
		if data.ndim != 3:
			raise ValueError("phase.paganin expects a (Y, Theta, X) sinogram array")
		if not self.enabled:
			return state
		transfer = _paganin_transfer(
			(n_rows(data, layout), data.shape[2]),
			alpha=self.alpha,
			energy_kev=state.scan.energy_keV,
			pixel_size_um=state.scan.pixel_size_um,
			ssd_cm=state.scan.ssd_cm,
		)
		for projection in range(n_projections(data, layout)):
			image = projection_view(data, layout, projection)
			filtered = _paganin_filter(image, transfer)
			image[:] = _negative_log(filtered, epsilon=self.epsilon, maximum=None)
		return state


class PaganinPhaseAbsorptionStage(Stage):
	"""Sorotassu's one-shot coupled phase/absorption Paganin retrieval."""

	name = "phase.paganin_phase_absorption"
	contract = StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,),
		produces_domain=DataDomain.ATTENUATION,
		locality=Locality.PROJECTION,
		accepts=(ProjOrder, SinoOrder),
		prefers=ProjOrder,
		mutates_input=True,
		window_axes=("Y", "Theta"),
		required_metadata=("energy_keV", "pixel_size_um", "ssd_cm"),
		resources=ResourceHints(memory_multiplier=2.0),
	)

	def __init__(self, alpha: float = 0.06, epsilon: float = 1e-8,
				order: str | None = None):
		if alpha <= 0:
			raise ValueError("phase/absorption alpha must be positive")
		self.alpha = alpha
		self.epsilon = epsilon
		self.order = order
		compatibility = configured_layout(order)
		self.contract = StageContract(
			expected_input=compatibility or SinoOrder,
			produced_output=compatibility or SinoOrder,
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=True,
			window_axes=("Y", "Theta"),
			required_metadata=("energy_keV", "pixel_size_um", "ssd_cm"),
			resources=ResourceHints(memory_multiplier=2.0),
			accepts_domain=(DataDomain.TRANSMISSION,),
			produces_domain=DataDomain.ATTENUATION,
			locality=Locality.PROJECTION,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "PaganinPhaseAbsorptionStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("PaganinPhaseAbsorptionStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		layout = resolve_layout(state, self.order, self.name)
		if data.ndim != 3:
			raise ValueError("phase.paganin_phase_absorption expects a (Y, Theta, X) sinogram array")
		wavelength = _wavelength_cm(state.scan.energy_keV)
		beta_scale = 4 * np.pi / wavelength
		transfer = _paganin_transfer(
			(n_rows(data, layout), data.shape[2]),
			alpha=self.alpha,
			energy_kev=state.scan.energy_keV,
			pixel_size_um=state.scan.pixel_size_um,
			ssd_cm=state.scan.ssd_cm,
		)
		for projection in range(n_projections(data, layout)):
			image = projection_view(data, layout, projection)
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
			processes: int = 1, order: str | None = None, clip_input: bool = False,
			clip_minimum: float = 1e-9, clip_threshold: float = 2.0,
			clip_high_value: float = 1.0):
		if alpha <= 0 or gamma <= 0:
			raise ValueError("BAC alpha and gamma must be positive")
		if processes < 1:
			raise ValueError("BAC processes must be positive")
		compatibility = configured_layout(order)
		shape = compatibility or SinoOrder
		self.contract = StageContract(
			expected_input=shape,
			produced_output=shape,
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
			accepts_domain=(DataDomain.TRANSMISSION,) if enabled else (
				DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
			),
			produces_domain=PRESERVE,
			locality=Locality.PROJECTION,
			window_axes=("Y", "Theta"),
			resources=ResourceHints(memory_multiplier=2.0),
		)
		self.alpha = alpha
		self.gamma = gamma
		self.enabled = enabled
		self.processes = processes
		self.order = order
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
		layout = resolve_layout(state, self.order, self.name)
		if data.ndim != 3:
			raise ValueError("phase.bac_correction expects a three-dimensional tomography array")
		if not self.enabled:
			return state
		count = n_projections(data, layout)

		def image(index: int) -> np.ndarray:
			return projection_view(data, layout, index).copy()

		if self.processes == 1:
			corrected = (self._correct(image(index)) for index in range(count))
		else:
			with ThreadPool(self.processes) as pool:
				corrected = pool.imap(self._correct, (image(index) for index in range(count)))
				for projection, result in enumerate(corrected):
					projection_view(data, layout, projection)[:] = result
			return state
		for projection, result in enumerate(corrected):
			projection_view(data, layout, projection)[:] = result
		return state


class BronnikovAidedCorrectionStage(Stage):
	"""Apply hotopy Bronnikov-aided correction to each projection image."""

	name = "phase.bac"
	contract = StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		mutates_input=True,
		window_axes=("Y", "Theta"),
		resources=ResourceHints(memory_multiplier=2.0),
	)

	def __init__(self, alpha: float, gamma: float = 1.0, enabled: bool = True,
				invert: bool = False, processes: int = 1, device: str | None = None,
				order: str | None = None, clip_input: bool = False,
				clip_minimum: float = 1e-9, clip_threshold: float = 2.0,
				clip_high_value: float = 1.0):
		if alpha <= 0 or gamma <= 0:
			raise ValueError("BAC alpha and gamma must be positive")
		if processes < 1:
			raise ValueError("BAC processes must be positive")
		compatibility = configured_layout(order)
		shape = compatibility or SinoOrder
		self.contract = StageContract(
			expected_input=shape,
			produced_output=shape,
			produces=PRESERVE,
			accepts=(ProjOrder, SinoOrder),
			prefers=ProjOrder,
			mutates_input=enabled,
			accepts_domain=(DataDomain.TRANSMISSION,) if enabled else (
				DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
			),
			produces_domain=DataDomain.ATTENUATION if enabled else PRESERVE,
			locality=Locality.PROJECTION,
			window_axes=("Y", "Theta"),
			resources=ResourceHints(memory_multiplier=2.0),
		)
		self.alpha = alpha
		self.gamma = gamma
		self.enabled = enabled
		self.invert = invert
		self.processes = processes
		self.device = device
		self.order = order
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
		layout = resolve_layout(state, self.order, self.name)
		if data.ndim != 3:
			raise ValueError("phase.bac expects a three-dimensional tomography array")
		if not self.enabled:
			return state
		count = n_projections(data, layout)
		def image(index: int) -> np.ndarray:
			return projection_view(data, layout, index).copy()
		if self.processes == 1:
			corrected = (self._correct(image(index)) for index in range(count))
		else:
			with ThreadPool(self.processes) as pool:
				corrected = pool.imap(self._correct, (image(index) for index in range(count)))
				for projection, result in enumerate(corrected):
					projection_view(data, layout, projection)[:] = result
			return state
		for projection, result in enumerate(corrected):
			projection_view(data, layout, projection)[:] = result
		return state
