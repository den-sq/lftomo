"""Tomographic reconstruction stages."""

from __future__ import annotations

from dataclasses import replace
import os
from typing import Literal

import numpy as np
from pydantic import Field, PositiveInt, TypeAdapter, ValidationError, model_validator

from lftomo.pipeline.core import (
	BufferSpec,
	PipelineState,
	ResourceHints,
	Stage,
	StageContract,
	StageParams,
	allocate_output,
	commit_output,
)
from lftomo.pipeline.domains import DataDomain
from lftomo.pipeline.runtime import ASTRA_RUNTIME_REQUIREMENT, tomopy_requirement
from lftomo.util.shapes import ProjOrder, ReconOrder, SinoOrder, X
from lftomo.util.window import TomoWindow


class _ReconstructionParams(StageParams):
	center: float | list[float] | None = None
	circle_mask: bool = True
	circle_mask_ratio: float = Field(default=1.0, gt=0, le=1)
	circle_mask_value: float = 0.0
	filter_name: str = "shepp"
	filter_parameters: list[float] | tuple[float, ...] | None = None
	edge_buffer_left: int = Field(default=0, ge=0)
	edge_buffer_right: int = Field(default=0, ge=0)
	output_width: int | None = Field(default=None, ge=1)


class GridrecStage(Stage):
	"""Reconstruct projection or sinogram-order data with TomoPy gridrec."""

	name = "recon.gridrec"
	contract = StageContract(
		accepts=(ProjOrder, SinoOrder),
		prefers=SinoOrder,
		produces=ReconOrder,
		accepts_domain=(DataDomain.ATTENUATION,),
		produces_domain=DataDomain.RECON,
		window_axes=("Y",),
		consumes_artifacts=("center",),
		runtime_requirements=(tomopy_requirement("circ_mask", "recon"),),
	)

	class Params(_ReconstructionParams):
		ncore: PositiveInt | None = None
		iterations: Literal[1] | None = None
		edge_buffer_input: bool = False
		angle_endpoint: bool = False

		@model_validator(mode="before")
		@classmethod
		def resolve_machine_cores(cls, values):
			values = dict(values)
			if values.pop("ncore_from_machine", False):
				# Validate the supplied value before applying the existing override.
				if values.get("ncore") is not None:
					try:
						TypeAdapter(PositiveInt).validate_python(values["ncore"])
					except ValidationError as error:
						raise ValueError(f"ncore: {error.errors()[0]['msg']}") from error
				values["ncore"] = os.cpu_count() or 1
			return values

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		return replace(
			cls.contract,
			consumes_artifacts=() if params.center is not None else ("center",),
			resources=replace(cls.contract.resources, preferred_cpus=params.ncore),
		)

	def _angles(self, state: PipelineState, theta_count: int) -> np.ndarray:
		angles = None
		if state.dataset is not None and state.source_window is not None:
			angles = state.dataset.angles[state.source_window.axis_slice("Theta")]
		if angles is None:
			acquisition = state.scan.acquisition
			angles = np.linspace(
				np.deg2rad(acquisition.rotation_start_deg),
				np.deg2rad(acquisition.rotation_end_deg),
				theta_count,
				endpoint=getattr(self.params, "angle_endpoint", False),
				dtype=np.float32,
			)
		angles = np.asarray(angles, dtype=np.float32)
		if angles.ndim != 1 or len(angles) != theta_count:
			raise ValueError(f"gridrec requires {theta_count} one-dimensional projection angles")
		return angles

	def _center(
		self, state: PipelineState, slice_count: int, input_y: slice,
	) -> float | np.ndarray:
		center = self.params.center if self.params.center is not None else state.artifacts.get("center")
		if center is None:
			raise ValueError("recon.gridrec requires a center parameter or center artifact")
		values = np.asarray(center, dtype=np.float32)
		if values.ndim == 0:
			return float(values)
		if self.params.center is None and values.ndim == 1 \
				and len(values) == state.buffer_spec.shape.Y:
			values = values[input_y]
		if values.ndim != 1 or len(values) != slice_count:
			raise ValueError(f"gridrec center array must contain one value for each of {slice_count} slices")
		return values

	def _input_view(
		self, state: PipelineState, data: np.ndarray, layout, input_x: slice,
	) -> tuple[np.ndarray, slice]:
		input_y = (
			slice(None) if state.source_window is None
			else state.source_window.internal.axis_slice("Y")
		)
		return data[layout.point_index(Y=input_y, X=input_x)], input_y

	def output_geometry(
		self, source_shape: ProjOrder | SinoOrder,
		source_window: TomoWindow | None = None,
	) -> tuple[ReconOrder, ReconOrder, slice, slice]:
		"""Return allocation/output shapes and their input/output X views."""

		if source_window is None:
			width, input_offset = source_shape.X, 0
		else:
			width = source_window.recon_shape().X
			input_offset = source_window.internal.axis_slice("X").start
		edge_buffer_input = getattr(self.params, "edge_buffer_input", False)
		edge_stop = width - self.params.edge_buffer_right
		if self.params.edge_buffer_left >= edge_stop:
			raise ValueError("gridrec edge buffers remove the entire reconstruction width")
		input_start, input_stop = (
			(self.params.edge_buffer_left, edge_stop) if edge_buffer_input else (0, width)
		)
		output_start, output_stop = (
			(0, input_stop - input_start)
			if edge_buffer_input else (self.params.edge_buffer_left, edge_stop)
		)
		if self.params.output_width is not None:
			if self.params.output_width > output_stop - output_start:
				raise ValueError("gridrec output_width exceeds the reconstructed width")
			output_start += (output_stop - output_start - self.params.output_width) // 2
			output_stop = output_start + self.params.output_width

		if source_window is None:
			allocation = ReconOrder.of(
				source_shape - X(input_start + width - input_stop)
			)
			output = allocation - X(allocation.X - output_stop + output_start)
		else:
			input_window = source_window.with_crop(
				X(input_start), X(width - input_stop),
			)
			allocation = input_window.recon_shape()
			output_window = input_window.with_crop(
				X(output_start), X(input_window.shape.X - output_stop),
			)
			output = allocation - X(allocation.X - output_window.shape.X)
		input_offset = (
			0 if source_window is None else source_window.internal.axis_slice("X").start
		)
		return (
			allocation, output,
			slice(input_offset + input_start, input_offset + input_stop),
			slice(output_start, output_stop),
		)

	def _allocate_reconstruction(
		self, state: PipelineState, shape: ReconOrder,
	):
		current_role = getattr(state.working_array, "role", None)
		can_alias_input = state.layout is ProjOrder and current_role == "normalized"
		return allocate_output(
			state,
			shape,
			np.float32,
			role="reconstruction",
			reuse_from=("raw_staging", "reconstruction"),
			alias_from_current=can_alias_input,
			proof="tomopy_input_materialization" if can_alias_input else None,
		)

	def _finish_reconstruction(
		self, state: PipelineState, output, target: np.ndarray, tomopy,
		shape: ReconOrder, output_x: slice, *, ncore: int | None,
	) -> PipelineState:
		if self.params.circle_mask:
			tomopy.circ_mask(
				target,
				axis=0,
				ratio=self.params.circle_mask_ratio,
				val=self.params.circle_mask_value,
				ncore=ncore,
			)
		cropped = output[ReconOrder.point_index(X=output_x)]
		return commit_output(
			state, cropped, retire_as=None,
			buffer_spec=BufferSpec(shape, np.dtype(np.float32)),
		)

	def run(self, state: PipelineState) -> PipelineState:
		if state.backend.name != "numpy":
			raise TypeError("GridrecStage currently requires the NumPy backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError("recon.gridrec expects a three-dimensional tomography array")
		layout = type(state.buffer_spec.shape)
		allocation_shape, output_shape, input_x, output_x = self.output_geometry(
			state.buffer_spec.shape, state.source_window,
		)
		data, input_y = self._input_view(state, data, layout, input_x)
		sinogram_order = layout is SinoOrder
		angles = self._angles(state, data.shape[layout.axis_of("Theta")])
		center = self._center(state, data.shape[layout.axis_of("Y")], input_y)

		import tomopy

		kwargs = {"filter_name": self.params.filter_name}
		if self.params.filter_parameters is not None:
			kwargs["filter_par"] = self.params.filter_parameters
		output, target = self._allocate_reconstruction(state, allocation_shape)
		tomopy.recon(
			data,
			angles,
			center=center,
			sinogram_order=sinogram_order,
			algorithm="gridrec",
			ncore=self.params.ncore,
			init_recon=target,
			**kwargs,
		)
		return self._finish_reconstruction(
			state, output, target, tomopy, output_shape, output_x, ncore=self.params.ncore,
		)


class AstraCudaStage(GridrecStage):
	"""Base Stage for TomoPy's ASTRA CUDA bridge."""

	method = ""
	iterative = False
	contract = replace(
		GridrecStage.contract,
		resources=ResourceHints(preferred_cpus=1, requires_gpu=True),
		runtime_requirements=(
			tomopy_requirement("astra", "circ_mask", "recon"),
			ASTRA_RUNTIME_REQUIREMENT,
		),
	)

	class Params(_ReconstructionParams):
		iterations: int = Field(default=1, ge=1)
		proj_type: str = "cuda"
		extra_options: dict = Field(default_factory=dict)
		filter_name: str = "butterworth"
		filter_parameters: list[float] | tuple[float, ...] | None = (0.2, 2)

		@model_validator(mode="before")
		@classmethod
		def normalize_options(cls, values):
			values = dict(values)
			if values.get("extra_options") is None:
				values["extra_options"] = {}
			return values

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		return replace(cls.contract, consumes_artifacts=() if params.center is not None else ("center",))

	def run(self, state: PipelineState) -> PipelineState:
		if not state.backend.supports_gpu:
			raise TypeError(f"{self.name} requires a GPU-capable backend")
		data = np.asarray(state.working_array)
		if data.ndim != 3:
			raise ValueError(f"{self.name} expects a three-dimensional tomography array")
		layout = type(state.buffer_spec.shape)
		allocation_shape, output_shape, input_x, output_x = self.output_geometry(
			state.buffer_spec.shape, state.source_window,
		)
		data, input_y = self._input_view(state, data, layout, input_x)
		sinogram_order = layout is SinoOrder
		angles = self._angles(state, data.shape[layout.axis_of("Theta")])
		center = self._center(state, data.shape[layout.axis_of("Y")], input_y)

		import tomopy

		options = {
			"proj_type": self.params.proj_type,
			"method": self.method,
			"extra_options": dict(self.params.extra_options),
		}
		if self.iterative:
			options["num_iter"] = self.params.iterations
		kwargs = {"filter_name": self.params.filter_name}
		if self.params.filter_parameters is not None:
			kwargs["filter_par"] = self.params.filter_parameters
		output, target = self._allocate_reconstruction(state, allocation_shape)
		tomopy.recon(
			data,
			angles,
			center=center,
			sinogram_order=sinogram_order,
			algorithm=tomopy.astra,
			ncore=1,
			init_recon=target,
			options=options,
			**kwargs,
		)
		return self._finish_reconstruction(
			state, output, target, tomopy, output_shape, output_x, ncore=1,
		)


class FPCudaStage(AstraCudaStage):
	name = "recon.fp_cuda"
	method = "FP_CUDA"


class BPCudaStage(AstraCudaStage):
	name = "recon.bp_cuda"
	method = "BP_CUDA"


class FBPCudaStage(AstraCudaStage):
	name = "recon.fbp_cuda"
	method = "FBP_CUDA"


class SIRTCudaStage(AstraCudaStage):
	name = "recon.sirt_cuda"
	method = "SIRT_CUDA"
	iterative = True


class SARTCudaStage(AstraCudaStage):
	name = "recon.sart_cuda"
	method = "SART_CUDA"
	iterative = True


class CGLSCudaStage(AstraCudaStage):
	name = "recon.cgls_cuda"
	method = "CGLS_CUDA"
	iterative = True


class EMCudaStage(AstraCudaStage):
	name = "recon.em_cuda"
	method = "EM_CUDA"
	iterative = True
