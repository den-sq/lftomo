"""Vo-family stripe-removal stages."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from lftomo.pipeline.core import (
	BoundedWorkRequest,
	BufferSpec,
	PipelineState,
	RuntimeRequirement,
	SinogramStage,
	Stage,
	StageContract,
	StageParams,
)
from lftomo.pipeline.domains import DataDomain
from lftomo.pipeline.runtime import tomopy_requirement
from lftomo.util.shapes import ProjOrder, SinoOrder, X
from lftomo.util.window import TomoWindow


VO_FLOAT_WORK_ARTIFACT = "_vo_float_work"
VO_INDEX_WORK_ARTIFACT = "_vo_index_work"
VO_MEDIAN_PAD_WORK_ARTIFACT = "_vo_median_pad_work"


def _vo_work_requests(
	input_spec: BufferSpec, median_half_width: int,
) -> tuple[BoundedWorkRequest, ...]:
	"""Own Vo dtypes, sorting capacity and optional zero-filled line padding."""

	shape = input_spec.shape.to(SinoOrder)
	dtype = input_spec.dtype
	if max(shape.Theta, shape.X) > np.iinfo(np.uint32).max:
		raise ValueError("Vo sorting indices exceed uint32 capacity")
	if not np.issubdtype(dtype, np.floating):
		raise TypeError("Vo reusable work requires floating-point pipeline data")
	requests = [
		BoundedWorkRequest(VO_FLOAT_WORK_ARTIFACT, BufferSpec(shape, dtype), "vo_float_work"),
		BoundedWorkRequest(VO_INDEX_WORK_ARTIFACT, BufferSpec(shape, np.uint32), "vo_index_work"),
	]
	if median_half_width and dtype in (np.dtype(np.float32), np.dtype(np.float64)) and dtype.isnative:
		zero = shape.make_with(0)
		window = TomoWindow(
			shape, zero + X(median_half_width), zero, shape.make_with(1), dtype,
		)
		requests.append(BoundedWorkRequest(
			VO_MEDIAN_PAD_WORK_ARTIFACT,
			BufferSpec(window.with_pad().shape, dtype), "vo_median_pad_work",
			fill_value=0, grow_axes=("X",),
		))
	return tuple(requests)


def _vo_work_planes(
	state: PipelineState, sinogram: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
	"""Open the row-indexed reusable planes prepared by bounded execution."""

	float_buffer = state.artifacts.get(VO_FLOAT_WORK_ARTIFACT)
	index_buffer = state.artifacts.get(VO_INDEX_WORK_ARTIFACT)
	median_pad_buffer = state.artifacts.get(VO_MEDIAN_PAD_WORK_ARTIFACT)
	if float_buffer is None and index_buffer is None and median_pad_buffer is None:
		return None, None, None
	if float_buffer is None or index_buffer is None:
		raise ValueError("Vo reusable work requires both floating-point and index planes")
	if state.assigned_range("Y").length.Y != 1:
		raise ValueError("Vo reusable work requires a dispatched Y unit")
	if float_buffer.shape.Y != 1 or index_buffer.shape.Y != 1:
		raise ValueError("Vo reusable work must be scoped to its dispatched Y unit")
	float_work = float_buffer.array()[0]
	index_work = index_buffer.array()[0]
	if float_work.shape != sinogram.shape or index_work.shape != sinogram.shape:
		raise ValueError("Vo reusable work does not match the dispatched sinogram shape")
	median_pad_work = None
	if median_pad_buffer is not None:
		if median_pad_buffer.shape.Y != 1:
			raise ValueError("Vo median-pad work must be scoped to its dispatched Y unit")
		median_pad_work = median_pad_buffer.array()[0]
		if (
			median_pad_work.shape[0] != sinogram.shape[0]
			or median_pad_work.shape[1] < sinogram.shape[1]
			or median_pad_work.dtype != sinogram.dtype
		):
			raise ValueError("Vo median-pad work does not match the dispatched sinogram")
	return float_work, index_work, median_pad_work


def _contract(runtime_requirements: tuple[RuntimeRequirement, ...] = ()) -> StageContract:
	return replace(
		SinogramStage.contract,
		accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
		cacheable_prefix=True,
		runtime_requirements=runtime_requirements,
	)


class VoSortStripeStage(SinogramStage):
	name = "stripe.vo_sort"
	contract = _contract()

	class Params(StageParams):
		size: int = 21
		dim: int = 1
		ncore: int | None = None

	@property
	def vo_line_median_half_width(self) -> int:
		from lftomo.prep.stripe import _bottleneck_line_half_width

		return _bottleneck_line_half_width(self.params.size) if self.params.dim == 1 else 0

	def bounded_work_requests(self, input_spec: BufferSpec) -> tuple[BoundedWorkRequest, ...]:
		return _vo_work_requests(input_spec, self.vo_line_median_half_width)

	def run_sinogram(self, sinogram: np.ndarray, state: PipelineState) -> None:
		from lftomo.prep.stripe import _remove_sort_array

		work, index_work, median_pad_work = _vo_work_planes(state, sinogram)
		_remove_sort_array(
			sinogram, size=self.params.size, dim=self.params.dim,
			work=work, index_work=index_work, median_pad_work=median_pad_work,
		)


class VoDeadStripeStage(SinogramStage):
	name = "stripe.vo_dead"
	contract = _contract()

	class Params(StageParams):
		snr: float = 3
		size: int = 51
		norm: bool = True

	@property
	def vo_line_median_half_width(self) -> int:
		from lftomo.prep.stripe import _bottleneck_line_half_width

		return _bottleneck_line_half_width(self.params.size) if self.params.norm else 0

	def bounded_work_requests(self, input_spec: BufferSpec) -> tuple[BoundedWorkRequest, ...]:
		return _vo_work_requests(input_spec, self.vo_line_median_half_width)

	def run_sinogram(self, sinogram: np.ndarray, state: PipelineState) -> None:
		from lftomo.prep.stripe import _remove_dead_array, _remove_large_array

		work, index_work, median_pad_work = _vo_work_planes(state, sinogram)
		_remove_dead_array(sinogram, snr=self.params.snr, size=self.params.size, work=work)
		if self.params.norm:
			_remove_large_array(
				sinogram, snr=self.params.snr, size=self.params.size, norm=True,
				work=work, index_work=index_work, median_pad_work=median_pad_work,
			)


class VoLargeStripeStage(SinogramStage):
	name = "stripe.vo_large"
	contract = _contract()

	class Params(StageParams):
		snr: float = 3
		size: int = 51
		drop_ratio: float = 0.1
		norm: bool = True

	@property
	def vo_line_median_half_width(self) -> int:
		from lftomo.prep.stripe import _bottleneck_line_half_width

		return _bottleneck_line_half_width(self.params.size)

	def bounded_work_requests(self, input_spec: BufferSpec) -> tuple[BoundedWorkRequest, ...]:
		return _vo_work_requests(input_spec, self.vo_line_median_half_width)

	def run_sinogram(self, sinogram: np.ndarray, state: PipelineState) -> None:
		from lftomo.prep.stripe import _remove_large_array

		work, index_work, median_pad_work = _vo_work_planes(state, sinogram)
		_remove_large_array(
			sinogram, snr=self.params.snr, size=self.params.size,
			drop_ratio=self.params.drop_ratio, norm=self.params.norm,
			work=work, index_work=index_work, median_pad_work=median_pad_work,
		)


class VoFamilyStripeStage(SinogramStage):
	"""Execute an explicitly ordered sequence of Vo stripe-removal methods."""

	name = "stripe.vo_family"
	contract = _contract()

	class Params(StageParams):
		order: tuple[Literal["remove_dead_vo", "remove_large_vo", "remove_sort_vo"], ...] = (
			"remove_dead_vo", "remove_large_vo", "remove_sort_vo",
		)
		size: int = 31
		snr: float = 3.0
		dim: int = 1
		drop_ratio: float = 0.1
		norm: bool = True

	@property
	def vo_line_median_half_width(self) -> int:
		from lftomo.prep.stripe import _bottleneck_line_half_width

		uses_large = "remove_large_vo" in self.params.order
		uses_sort = "remove_sort_vo" in self.params.order and self.params.dim == 1
		return _bottleneck_line_half_width(self.params.size) if uses_large or uses_sort else 0

	def bounded_work_requests(self, input_spec: BufferSpec) -> tuple[BoundedWorkRequest, ...]:
		return _vo_work_requests(input_spec, self.vo_line_median_half_width)

	def run_sinogram(self, sinogram: np.ndarray, state: PipelineState) -> None:
		from lftomo.prep.stripe import _remove_dead_array, _remove_large_array, _remove_sort_array

		work, index_work, median_pad_work = _vo_work_planes(state, sinogram)
		for operation in self.params.order:
			if operation == "remove_dead_vo":
				_remove_dead_array(
					sinogram, snr=self.params.snr, size=self.params.size, work=work,
				)
			elif operation == "remove_large_vo":
				_remove_large_array(
					sinogram, snr=self.params.snr, size=self.params.size,
					drop_ratio=self.params.drop_ratio, norm=self.params.norm,
					work=work, index_work=index_work, median_pad_work=median_pad_work,
				)
			else:
				_remove_sort_array(
					sinogram, size=self.params.size, dim=self.params.dim,
					work=work, index_work=index_work, median_pad_work=median_pad_work,
				)


class FourierWaveletStripeStage(SinogramStage):
	name = "stripe.fw"
	contract = _contract()

	class Params(StageParams):
		level: int | None = Field(default=None, ge=1)
		wname: str = "db5"
		sigma: float = Field(default=2.0, gt=0)
		zero_pad: bool = False
		ncore: int | None = None

		@model_validator(mode="before")
		@classmethod
		def resolve_wavelet_alias(cls, values):
			if isinstance(values, dict) and "wavelet" in values:
				values = dict(values)
				wavelet = values.pop("wavelet")
				if wavelet:
					values["wname"] = wavelet
			return values

	def run_sinogram(self, sinogram: np.ndarray, _state: PipelineState) -> None:
		from lftomo.prep.stripe import _remove_stripe_fw_array

		level = self.params.level or int(np.ceil(np.log2(max(sinogram.shape))))
		_remove_stripe_fw_array(
			sinogram, level=level, wname=self.params.wname,
			sigma=self.params.sigma, zero_pad=self.params.zero_pad,
		)


class AllOfTheStripeStage(Stage):
	"""Run TomoPy's combined Vo dead, large, and sorting algorithms."""

	name = "stripe.all_of_the_stripe"
	contract = replace(
		_contract(runtime_requirements=(tomopy_requirement("remove_all_stripe"),)),
		accepts=(ProjOrder, SinoOrder), prefers=ProjOrder, bounded_views_from=(),
	)

	class Params(StageParams):
		snr: float = Field(default=3, gt=0)
		la_size: int = Field(default=61, ge=1)
		sm_size: int = Field(default=21, ge=1)
		dim: Literal[1, 2] = 1
		ncore: int | None = None
		nchunk: int | None = None

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("AllOfTheStripeStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		layout = type(state.buffer_spec.shape)
		if data.ndim != 3:
			raise ValueError("stripe.all_of_the_stripe expects a (Y, Theta, X) sinogram array")

		import tomopy

		shape = state.buffer_spec.shape
		tomopy_input = shape.transposed_view(data, ProjOrder)
		# TomoPy copies this view into its own multiprocessing shared array and
		# returns that allocation. The final assignment is therefore the required
		# library-boundary handoff; no pipeline-side transpose is materialized.
		corrected = tomopy.remove_all_stripe(
			tomopy_input,
			snr=self.params.snr,
			la_size=self.params.la_size,
			sm_size=self.params.sm_size,
			dim=self.params.dim,
			ncore=self.params.ncore,
			nchunk=self.params.nchunk,
		)
		data[:] = shape.to(ProjOrder).transposed_view(corrected, layout)
		return state
