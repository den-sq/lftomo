"""Propagation-based phase-retrieval stages."""

from __future__ import annotations

import numpy as np
from multiprocessing.pool import ThreadPool

from lftomo.config import StageConfig
from lftomo.pipeline.core import PipelineState, ResourceHints, Stage, StageContract
from lftomo.util.shapes import ProjOrder, SinoOrder


PLANCK_CONSTANT = 6.58211928e-19  # keV s
SPEED_OF_LIGHT = 299792458e2  # cm / s


def _wavelength_cm(energy_kev: float) -> float:
	return 2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy_kev


def _angular_frequency_squared(shape: tuple[int, int], pixel_size_cm: float) -> np.ndarray:
	ky = 2 * np.pi * np.fft.fftfreq(shape[0], d=pixel_size_cm)
	kx = 2 * np.pi * np.fft.fftfreq(shape[1], d=pixel_size_cm)
	return np.add.outer(np.square(ky), np.square(kx))


class PaganinStage(Stage):
	"""Standard single-distance Paganin phase-only retrieval."""

	name = "phase.paganin"
	contract = StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		mutates_input=True,
		window_axes=("Y", "Theta"),
		required_metadata=("energy_keV", "pixel_size_um", "ssd_cm"),
		resources=ResourceHints(memory_multiplier=2.0),
	)

	def __init__(self, alpha: float = 0.06, epsilon: float = 1e-8,
				enabled: bool = True, pad_y: bool = False):
		if alpha <= 0:
			raise ValueError("Paganin alpha must be positive")
		self.alpha = alpha
		self.epsilon = epsilon
		self.enabled = enabled
		self.pad_y = pad_y

	@classmethod
	def from_config(cls, config: StageConfig) -> "PaganinStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("PaganinStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("phase.paganin expects a (Y, Theta, X) sinogram array")
		if not self.enabled:
			return state
		pixel_size_cm = state.scan.pixel_size_um / 10_000
		wavelength = _wavelength_cm(state.scan.energy_keV)
		w2 = _angular_frequency_squared((data.shape[0], data.shape[2]), pixel_size_cm)
		transfer = 1 / (1 + state.scan.ssd_cm * self.alpha * wavelength * w2 / (4 * np.pi))
		for projection in range(data.shape[1]):
			filtered = np.real(np.fft.ifft2(np.fft.fft2(data[:, projection, :]) * transfer))
			data[:, projection, :] = -np.log(np.clip(filtered, self.epsilon, None))
		return state


class PaganinPhaseAbsorptionStage(Stage):
	"""Sorotassu's one-shot coupled phase/absorption Paganin retrieval."""

	name = "phase.paganin_phase_absorption"
	contract = StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		mutates_input=True,
		window_axes=("Y", "Theta"),
		required_metadata=("energy_keV", "pixel_size_um", "ssd_cm"),
		resources=ResourceHints(memory_multiplier=2.0),
	)

	def __init__(self, alpha: float = 0.06, epsilon: float = 1e-8):
		if alpha <= 0:
			raise ValueError("phase/absorption alpha must be positive")
		self.alpha = alpha
		self.epsilon = epsilon

	@classmethod
	def from_config(cls, config: StageConfig) -> "PaganinPhaseAbsorptionStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("PaganinPhaseAbsorptionStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("phase.paganin_phase_absorption expects a (Y, Theta, X) sinogram array")
		pixel_size_cm = state.scan.pixel_size_um / 10_000
		wavelength = _wavelength_cm(state.scan.energy_keV)
		w2 = _angular_frequency_squared((data.shape[0], data.shape[2]), pixel_size_cm)
		# Corrected-paper branch from dan_scripts/prep/phase.py:22-144.
		filter_factor = 1 / (1 + state.scan.ssd_cm * self.alpha * wavelength * w2 / (4 * np.pi))
		beta_scale = 4 * np.pi / wavelength
		for projection in range(data.shape[1]):
			complex_image = np.fft.fft2(data[:, projection, :])
			complex_image *= filter_factor
			transmission = np.real(np.fft.ifft2(complex_image))
			absorption = -np.log(np.clip(transmission, self.epsilon, 0.99999999))
			data[:, projection, :] = absorption / beta_scale
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
				order: str = "sinogram", clip_input: bool = False,
				clip_minimum: float = 1e-9, clip_threshold: float = 2.0,
				clip_high_value: float = 1.0):
		if alpha <= 0 or gamma <= 0:
			raise ValueError("BAC alpha and gamma must be positive")
		if processes < 1:
			raise ValueError("BAC processes must be positive")
		if order not in ("projection", "sinogram"):
			raise ValueError("BAC order must be 'projection' or 'sinogram'")
		shape = ProjOrder if order == "projection" else SinoOrder
		self.contract = StageContract(
			expected_input=shape,
			produced_output=shape,
			mutates_input=True,
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
		if data.ndim != 3:
			raise ValueError("phase.bac expects a three-dimensional tomography array")
		if not self.enabled:
			return state
		count = data.shape[0] if self.order == "projection" else data.shape[1]
		def image(index: int) -> np.ndarray:
			values = data[index, :, :] if self.order == "projection" else data[:, index, :]
			return values.copy()
		if self.processes == 1:
			corrected = (self._correct(image(index)) for index in range(count))
		else:
			with ThreadPool(self.processes) as pool:
				corrected = pool.imap(self._correct, (image(index) for index in range(count)))
				for projection, result in enumerate(corrected):
					if self.order == "projection":
						data[projection, :, :] = result
					else:
						data[:, projection, :] = result
			return state
		for projection, result in enumerate(corrected):
			if self.order == "projection":
				data[projection, :, :] = result
			else:
				data[:, projection, :] = result
		return state
