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
from lftomo.util.shapes import ProjOrder, SinoOrder


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

	def __init__(self, size: int = 21, dim: int = 1, ncore: int | None = None):
		self.size = size
		self.dim = dim
		self.ncore = ncore

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


class VoFamilyStripeStage(Stage):
	"""Execute an explicitly ordered sequence of Vo stripe-removal methods."""

	name = "stripe.vo_family"
	contract = _contract(3.0)

	def __init__(self, order: list[str] | tuple[str, ...] = (
				"remove_dead_vo", "remove_large_vo", "remove_sort_vo"),
				size: int = 31, snr: float = 3.0, dim: int = 1,
				drop_ratio: float = 0.1, norm: bool = True):
		known = {"remove_dead_vo", "remove_large_vo", "remove_sort_vo"}
		unknown = set(order) - known
		if unknown:
			raise ValueError(f"unknown Vo-family operation(s): {sorted(unknown)}")
		self.order = tuple(order)
		self.size = size
		self.snr = snr
		self.dim = dim
		self.drop_ratio = drop_ratio
		self.norm = norm

	@classmethod
	def from_config(cls, config: StageConfig) -> "VoFamilyStripeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("VoFamilyStripeStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("stripe.vo_family expects a (Y, Theta, X) sinogram array")
		for y_index in range(data.shape[0]):
			for operation in self.order:
				if operation == "remove_dead_vo":
					_remove_dead_array(data[y_index], snr=self.snr, size=self.size)
				elif operation == "remove_large_vo":
					_remove_large_array(
						data[y_index], snr=self.snr, size=self.size,
						drop_ratio=self.drop_ratio, norm=self.norm,
					)
				else:
					_remove_sort_array(data[y_index], size=self.size, dim=self.dim)
		return state


class FourierWaveletStripeStage(Stage):
	name = "stripe.fw"
	contract = _contract(2.0)

	def __init__(self, level: int | None = None, wname: str = "db5",
				sigma: float = 2.0, zero_pad: bool = False,
				wavelet: str | None = None, ncore: int | None = None,
				order: str = "sinogram"):
		if level is not None and level < 1:
			raise ValueError("Fourier-wavelet level must be positive")
		if sigma <= 0:
			raise ValueError("Fourier-wavelet sigma must be positive")
		if order not in ("projection", "sinogram"):
			raise ValueError("Fourier-wavelet order must be 'projection' or 'sinogram'")
		shape = ProjOrder if order == "projection" else SinoOrder
		self.contract = StageContract(
			expected_input=shape, produced_output=shape, mutates_input=True,
			window_axes=("Y",), resources=ResourceHints(memory_multiplier=2.0),
		)
		self.level = level
		self.wname = wavelet or wname
		self.sigma = sigma
		self.zero_pad = zero_pad
		self.ncore = ncore
		self.order = order

	@classmethod
	def from_config(cls, config: StageConfig) -> "FourierWaveletStripeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("FourierWaveletStripeStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("stripe.fw expects a (Y, Theta, X) sinogram array")
		theta_size = data.shape[0] if self.order == "projection" else data.shape[1]
		y_size = data.shape[1] if self.order == "projection" else data.shape[0]
		level = self.level or int(np.ceil(np.log2(max(theta_size, data.shape[2]))))
		for y_index in range(y_size):
			sinogram = data[:, y_index, :] if self.order == "projection" else data[y_index]
			_remove_stripe_fw_array(
				sinogram, level=level, wname=self.wname,
				sigma=self.sigma, zero_pad=self.zero_pad,
			)
		return state


class AllOfTheStripeStage(Stage):
	"""Run TomoPy's combined Vo dead, large, and sorting algorithms."""

	name = "stripe.all_of_the_stripe"
	contract = _contract(3.0)

	def __init__(self, snr: float = 3, la_size: int = 61, sm_size: int = 21,
				dim: int = 1, ncore: int | None = None, nchunk: int | None = None):
		if snr <= 0:
			raise ValueError("all-of-the-stripe snr must be positive")
		if la_size < 1 or sm_size < 1:
			raise ValueError("all-of-the-stripe filter sizes must be positive")
		if dim not in (1, 2):
			raise ValueError("all-of-the-stripe dim must be 1 or 2")
		self.snr = snr
		self.la_size = la_size
		self.sm_size = sm_size
		self.dim = dim
		self.ncore = ncore
		self.nchunk = nchunk

	@classmethod
	def from_config(cls, config: StageConfig) -> "AllOfTheStripeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("AllOfTheStripeStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("stripe.all_of_the_stripe expects a (Y, Theta, X) sinogram array")

		import tomopy

		corrected = tomopy.remove_all_stripe(
			np.transpose(data, (1, 0, 2)),
			snr=self.snr,
			la_size=self.la_size,
			sm_size=self.sm_size,
			dim=self.dim,
			ncore=self.ncore,
			nchunk=self.nchunk,
		)
		data[:] = np.transpose(corrected, (1, 0, 2))
		return state
