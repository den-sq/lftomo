"""Propagation-based phase-retrieval stages."""

from __future__ import annotations

import numpy as np

from lftomo.config import StageConfig
from lftomo.pipeline.core import PipelineState, ResourceHints, Stage, StageContract
from lftomo.util.shapes import SinoOrder


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

	def __init__(self, alpha: float = 0.06, epsilon: float = 1e-8):
		if alpha <= 0:
			raise ValueError("Paganin alpha must be positive")
		self.alpha = alpha
		self.epsilon = epsilon

	@classmethod
	def from_config(cls, config: StageConfig) -> "PaganinStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("PaganinStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("phase.paganin expects a (Y, Theta, X) sinogram array")
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
