"""Vo-family stripe-removal stages."""

from __future__ import annotations

import numpy as np

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
from lftomo.pipeline.stages._layout import (
	configured_layout,
	n_rows,
	resolve_layout,
	sinogram_scratch,
	sinogram_view,
)
from lftomo.prep.stripe import (
	_remove_dead_array,
	_remove_large_array,
	_remove_sort_array,
	_remove_stripe_fw_array,
)
from lftomo.util.shapes import ProjOrder, SinoOrder


def _contract(runtime_requirements: tuple[RuntimeRequirement, ...] = ()) -> StageContract:
	return StageContract(
		expected_input=SinoOrder,
		produced_output=SinoOrder,
		accepts=(SinoOrder,), prefers=SinoOrder, produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
		produces_domain=PRESERVE, locality=Locality.SINOGRAM,
		bounded_views_from=(ProjOrder,),
		mutates_input=True,
		cacheable_prefix=True,
		window_axes=("Y",),
		runtime_requirements=runtime_requirements,
	)


def _run_sinogram_stage(
	stage: Stage, state: PipelineState, configured: str | None = None,
) -> PipelineState:
	if state.backend.name != "numpy":
		raise TypeError(f"{stage.name} currently requires the NumPy backend")
	data = working_view(state)
	layout = resolve_layout(state, configured, stage.name)
	if data.ndim != 3:
		raise ValueError(f"{stage.name} expects a three-dimensional tomography array")
	for y_index in range(n_rows(data, layout)):
		if layout is ProjOrder:
			sinogram = sinogram_scratch(data, layout, y_index)
			stage.run_sinogram(sinogram, state)
			sinogram_view(data, layout, y_index)[:] = sinogram
		else:
			stage.run_sinogram(sinogram_view(data, layout, y_index), state)
	return state


class VoSortStripeStage(Stage):
	name = "stripe.vo_sort"
	contract = _contract()

	def __init__(self, size: int = 21, dim: int = 1, ncore: int | None = None):
		self.size = size
		self.dim = dim
		self.ncore = ncore

	@classmethod
	def from_config(cls, config: StageConfig) -> "VoSortStripeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		return _run_sinogram_stage(self, state)

	def run_sinogram(self, sinogram: np.ndarray, _state: PipelineState) -> None:
		_remove_sort_array(sinogram, size=self.size, dim=self.dim)


class VoDeadStripeStage(Stage):
	name = "stripe.vo_dead"
	contract = _contract()

	def __init__(self, snr: int = 3, size: int = 51, norm: bool = True):
		self.snr = snr
		self.size = size
		self.norm = norm

	@classmethod
	def from_config(cls, config: StageConfig) -> "VoDeadStripeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		return _run_sinogram_stage(self, state)

	def run_sinogram(self, sinogram: np.ndarray, _state: PipelineState) -> None:
		_remove_dead_array(sinogram, snr=self.snr, size=self.size)
		if self.norm:
			_remove_large_array(sinogram, snr=self.snr, size=self.size, norm=True)


class VoLargeStripeStage(Stage):
	name = "stripe.vo_large"
	contract = _contract()

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
		return _run_sinogram_stage(self, state)

	def run_sinogram(self, sinogram: np.ndarray, _state: PipelineState) -> None:
		_remove_large_array(
			sinogram, snr=self.snr, size=self.size,
			drop_ratio=self.drop_ratio, norm=self.norm,
		)


class VoFamilyStripeStage(Stage):
	"""Execute an explicitly ordered sequence of Vo stripe-removal methods."""

	name = "stripe.vo_family"
	contract = _contract()

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
		return _run_sinogram_stage(self, state)

	def run_sinogram(self, sinogram: np.ndarray, _state: PipelineState) -> None:
		for operation in self.order:
			if operation == "remove_dead_vo":
				_remove_dead_array(sinogram, snr=self.snr, size=self.size)
			elif operation == "remove_large_vo":
				_remove_large_array(
					sinogram, snr=self.snr, size=self.size,
					drop_ratio=self.drop_ratio, norm=self.norm,
				)
			else:
				_remove_sort_array(sinogram, size=self.size, dim=self.dim)


class FourierWaveletStripeStage(Stage):
	name = "stripe.fw"
	contract = _contract()

	def __init__(self, level: int | None = None, wname: str = "db5",
				sigma: float = 2.0, zero_pad: bool = False,
				wavelet: str | None = None, ncore: int | None = None,
				order: str | None = None):
		if level is not None and level < 1:
			raise ValueError("Fourier-wavelet level must be positive")
		if sigma <= 0:
			raise ValueError("Fourier-wavelet sigma must be positive")
		compatibility = configured_layout(order)
		shape = compatibility or SinoOrder
		self.contract = StageContract(
			expected_input=shape, produced_output=shape, mutates_input=True,
			cacheable_prefix=True,
			accepts=(SinoOrder,), prefers=SinoOrder, produces=PRESERVE,
			accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
			produces_domain=PRESERVE, locality=Locality.SINOGRAM,
			bounded_views_from=(ProjOrder,),
			window_axes=("Y",),
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
		return _run_sinogram_stage(self, state, self.order)

	def run_sinogram(self, sinogram: np.ndarray, _state: PipelineState) -> None:
		level = self.level or int(np.ceil(np.log2(max(sinogram.shape))))
		_remove_stripe_fw_array(
			sinogram, level=level, wname=self.wname,
			sigma=self.sigma, zero_pad=self.zero_pad,
		)


class AllOfTheStripeStage(Stage):
	"""Run TomoPy's combined Vo dead, large, and sorting algorithms."""

	name = "stripe.all_of_the_stripe"
	contract = _contract(
		runtime_requirements=(tomopy_requirement("remove_all_stripe"),),
	)

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
		self.contract = StageContract(
			expected_input=SinoOrder, produced_output=SinoOrder,
			accepts=(ProjOrder, SinoOrder), prefers=ProjOrder, produces=PRESERVE,
			accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
			produces_domain=PRESERVE, locality=Locality.SINOGRAM,
			mutates_input=True, cacheable_prefix=True, window_axes=("Y",),
			runtime_requirements=type(self).contract.runtime_requirements,
		)

	@classmethod
	def from_config(cls, config: StageConfig) -> "AllOfTheStripeStage":
		return cls(**config.params)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("AllOfTheStripeStage currently requires the NumPy backend")
		data = working_view(state)
		layout = resolve_layout(state, None, self.name)
		if data.ndim != 3:
			raise ValueError("stripe.all_of_the_stripe expects a (Y, Theta, X) sinogram array")

		import tomopy

		tomopy_input = data if layout is ProjOrder else np.transpose(data, (1, 0, 2))
		corrected = tomopy.remove_all_stripe(
			tomopy_input,
			snr=self.snr,
			la_size=self.la_size,
			sm_size=self.sm_size,
			dim=self.dim,
			ncore=self.ncore,
			nchunk=self.nchunk,
		)
		data[:] = corrected if layout is ProjOrder else np.transpose(corrected, (1, 0, 2))
		return state
