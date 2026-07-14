"""Vo-family stripe-removal stages."""

from __future__ import annotations

import numpy as np

from lftomo.config import StageConfig
from lftomo.pipeline.core import PipelineState, ResourceHints, Stage, StageContract
from lftomo.prep.stripe import (
	_remove_dead_array,
	_remove_large_array,
	_remove_sort_array,
	_remove_stripe_fw_array,
)
from lftomo.util.shapes import SinoOrder


def _contract(memory_multiplier: float) -> StageContract:
	return StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		mutates_input=True,
		window_axes=("Y",),
		resources=ResourceHints(memory_multiplier=memory_multiplier),
	)


class VoSortStripeStage(Stage):
	name = "stripe.vo_sort"
	contract = _contract(3.0)

	def __init__(self, size: int = 21, dim: int = 1):
		self.size = size
		self.dim = dim

	@classmethod
	def from_config(cls, config: StageConfig) -> "VoSortStripeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("VoSortStripeStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("stripe.vo_sort expects a (Y, Theta, X) sinogram array")
		for y_index in range(data.shape[0]):
			_remove_sort_array(data[y_index], size=self.size, dim=self.dim)
		return state


class VoDeadStripeStage(Stage):
	name = "stripe.vo_dead"
	contract = _contract(2.0)

	def __init__(self, snr: int = 3, size: int = 51, norm: bool = True):
		self.snr = snr
		self.size = size
		self.norm = norm

	@classmethod
	def from_config(cls, config: StageConfig) -> "VoDeadStripeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("VoDeadStripeStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("stripe.vo_dead expects a (Y, Theta, X) sinogram array")
		for y_index in range(data.shape[0]):
			_remove_dead_array(data[y_index], snr=self.snr, size=self.size)
			if self.norm:
				_remove_large_array(data[y_index], snr=self.snr, size=self.size, norm=True)
		return state


class VoLargeStripeStage(Stage):
	name = "stripe.vo_large"
	contract = _contract(3.0)

	def __init__(self, snr: int = 3, size: int = 51,
				drop_ratio: float = 0.1, norm: bool = True):
		self.snr = snr
		self.size = size
		self.drop_ratio = drop_ratio
		self.norm = norm

	@classmethod
	def from_config(cls, config: StageConfig) -> "VoLargeStripeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("VoLargeStripeStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("stripe.vo_large expects a (Y, Theta, X) sinogram array")
		for y_index in range(data.shape[0]):
			_remove_large_array(
				data[y_index], snr=self.snr, size=self.size,
				drop_ratio=self.drop_ratio, norm=self.norm,
			)
		return state


class FourierWaveletStripeStage(Stage):
	name = "stripe.fw"
	contract = _contract(2.0)

	def __init__(self, level: int | None = None, wname: str = "db5",
				sigma: float = 2.0, zero_pad: bool = False):
		if level is not None and level < 1:
			raise ValueError("Fourier-wavelet level must be positive")
		if sigma <= 0:
			raise ValueError("Fourier-wavelet sigma must be positive")
		self.level = level
		self.wname = wname
		self.sigma = sigma
		self.zero_pad = zero_pad

	@classmethod
	def from_config(cls, config: StageConfig) -> "FourierWaveletStripeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("FourierWaveletStripeStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("stripe.fw expects a (Y, Theta, X) sinogram array")
		level = self.level or int(np.ceil(np.log2(max(data.shape[1:]))))
		for y_index in range(data.shape[0]):
			_remove_stripe_fw_array(
				data[y_index], level=level, wname=self.wname,
				sigma=self.sigma, zero_pad=self.zero_pad,
			)
		return state
