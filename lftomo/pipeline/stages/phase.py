"""Propagation-based phase-retrieval stages."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pydantic import Field, model_validator

from lftomo.pipeline.core import (
	BoundedWorkRequest,
	BufferSpec,
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
from lftomo.prep.phase import (
	negative_log as _negative_log,
	phase_fft_window,
	phase_frequency_squared,
	single_material_filter,
	slope_denominator,
)
from lftomo.storage import buffer_view
from lftomo.util.shapes import ImgSlice, ProjOrder
from lftomo.util.window import TomoWindow


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

	if getattr(stage.params, "slope", None) is not None and not stage.params.enabled:
		return
	shape = (
		state.working_array.shape if state.read_window is None
		else state.read_window.frange.length
	)
	shape = (shape.Y, shape.X)
	if getattr(stage.params, "slope", None) is None:
		_resolved_phase_filter(state, shape, stage.params.alpha)
	else:
		_resolved_slope_filter(state, shape, slope=stage.params.slope, pad=stage.params.fft_pad)
		_prepare_material_envelope(stage, state)


def slope_filter_shape(shape, pad: bool) -> ImgSlice:
	"""Derive the shared real-frequency allocation from its canonical canvas."""
	canvas = phase_fft_window((shape.Y, shape.X), pad).with_pad().shape
	return ImgSlice(canvas.Y, canvas.X // 2 + 1)


def _resolved_slope_filter(
	state: PipelineState, shape: tuple[int, int], *, slope: float, pad: bool,
) -> np.ndarray:
	window = phase_fft_window(shape, pad)
	pixel_m, distance_m = state.scan.pixel_size_um * 1e-6, state.scan.ssd_cm * 1e-2
	signature = ("slope", window, pixel_m, distance_m, slope)
	previous = state.artifacts.get("phase_filter_signature")
	if (
		previous is not None and previous[0] == "slope"
		and previous[2:] == signature[2:]
		and previous[1].shape == window.shape
		and previous[1].pad_before == window.pad_before
		and previous[1].pad_after == window.pad_after
	):
		return buffer_view(state.artifacts["phase_filter"])
	return _managed_phase_filter(
		state, slope_denominator(window, pixel_m, distance_m, slope), signature,
	)


def _prepare_material_envelope(stage: Stage, state: PipelineState) -> None:
	"""Normalize the config boundary once into the window's owned envelope."""

	value = stage.params.envelope_m
	if value is None or stage.params.material_alpha == 0:
		return
	value = np.asarray(value)
	previous = state.artifacts.get("phase_envelope")
	if value.ndim == 0:
		state.artifacts["phase_envelope"] = float(value)
		if state.storage is not None and hasattr(previous, "name"):
			state.storage.retire(previous, role="phase_envelope")
		return
	window = state.read_window
	if window is None:
		selected = np.broadcast_to(value, state.working_array.shape.to(ProjOrder).as_tuple())
	else:
		physical = state.dataset.physical_shape if state.dataset is not None else state.scan.projection_shape()
		full = np.broadcast_to(value, physical.as_tuple())
		selected = full[tuple(window.axis_slice(axis) for axis in ("Theta", "Y", "X"))]
	selected = selected.transpose(tuple(ProjOrder.axis_of(axis) for axis in state.working_array.shape.names))
	if state.storage is None:
		state.artifacts["phase_envelope"] = np.asarray(selected, dtype=np.float32)
		return
	envelope = state.storage.allocate(
		state.working_array.shape if window is None else window.frange.length.to(state.layout),
		np.float32, role="phase_envelope", reuse_from=("phase_envelope",),
	)
	envelope.array()[...] = selected
	state.artifacts["phase_envelope"] = envelope
	if hasattr(previous, "name"):
		state.storage.retire(previous, role="phase_envelope")


def _material_envelope(state: PipelineState):
	value = state.artifacts.get("phase_envelope")
	if value is None or np.isscalar(value):
		return value
	return buffer_view(value)[state.layout.point_index(Theta=0)]


def _slope_filtered(stage: Stage, state: PipelineState, image: np.ndarray) -> np.ndarray:
	return single_material_filter(
		image, state.scan.pixel_size_um * 1e-6, state.scan.ssd_cm * 1e-2,
		stage.params.slope, material_alpha=stage.params.material_alpha,
		envelope=_material_envelope(state), pad=stage.params.fft_pad,
		denominator=_resolved_slope_filter(
			state, image.shape, slope=stage.params.slope, pad=stage.params.fft_pad,
		), workers=1,
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

	return phase_frequency_squared(
		phase_fft_window(shape, pad=False), pixel_size_cm, real=False, angular=True,
	)


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
	# Finiteness is checked for the completed held window by the Stage, which
	# can report every projection's count. Preserve the existing positive-
	# transmission requirement before the split negative-log consumer.
	with np.errstate(divide="ignore", invalid="ignore"):
		corrected = image / denominator
	if np.any(np.isfinite(corrected) & (corrected <= 0)):
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
		slope: float | None = Field(default=None, gt=0)
		material_alpha: float = 0.0
		envelope_m: float | list | None = None
		fft_pad: bool = True

		@model_validator(mode="after")
		def validate_material_line(self):
			if self.material_alpha != 0 and (self.slope is None or self.envelope_m is None):
				raise ValueError("nonzero material_alpha requires slope and envelope_m")
			if self.envelope_m is not None and self.slope is None:
				raise ValueError("envelope_m requires slope")
			return self

	@classmethod
	def contract_for(cls, params: Params) -> StageContract:
		if params.enabled:
			# The sampled FFT inverse has global support, including small
			# oscillatory tails beyond its continuum exponential decay length.
			# Retain physical detector rows so an ROI shares the full run's
			# Fourier canvas and detector-edge treatment, also for affine terms.
			required_pad = (
				PadRequirement(full_source_axes=("Y",))
				if params.slope is not None else cls.contract.required_pad
			)
			if params.material_alpha != 0:
				return replace(
					cls.contract,
					required_pad=required_pad,
					preferred_window_axis="Theta",
					produces_artifacts=cls.contract.produces_artifacts + ("phase_envelope",),
					managed_artifacts=cls.contract.managed_artifacts + (
						("phase_envelope",) if isinstance(params.envelope_m, list) else ()
					),
				)
			return replace(
				cls.contract,
				required_pad=required_pad,
				preferred_window_axis="Theta" if params.slope is not None else None,
			)
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

	def prepare_projections(self, state: PipelineState) -> None:
		if self.params.enabled:
			prepare_phase_filter(self, state)

	def phase_filter_shape(self, shape) -> ImgSlice:
		return (
			ImgSlice(shape.Y, shape.X) if self.params.slope is None
			else slope_filter_shape(shape, self.params.fft_pad)
		)

	def run_projection(self, image: np.ndarray, state: PipelineState) -> np.ndarray | None:
		if not self.params.enabled:
			return None
		if self.params.slope is not None:
			return _slope_filtered(self, state, image)
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
		window_axes=("Y", "Theta", "X"),
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
		window_axes=("Y", "Theta", "X"),
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
		data[state.work_index()] /= 4 * np.pi / _wavelength_cm(state.scan.energy_keV)
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

		@model_validator(mode="before")
		@classmethod
		def material_line_epsilon(cls, values):
			if values.get("slope") is not None and "epsilon" not in values:
				values = {**values, "epsilon": 1e-6}
			return values

	def run_projection(self, image: np.ndarray, state: PipelineState) -> np.ndarray | None:
		if not self.params.enabled:
			return None
		filtered = super().run_projection(image, state)
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

	def prepare_projections(self, state: PipelineState) -> None:
		prepare_phase_filter(self, state)

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


class BACNonFiniteError(ValueError):
	"""BAC failed without replacing output; ``qc`` retains the window's counts."""

	def __init__(self, stage_name: str, qc: dict):
		self.stage_name = stage_name
		self.qc = qc
		failed = [item for item in qc["projections"] if item["nonfinite_count"]]
		super().__init__(
			f"{stage_name} produced non-finite output: "
			+ ", ".join(
				f"projection {item['projection_index']} "
				f"(acquisition {item['acquisition_index']}): {item['nonfinite_count']} pixels"
				for item in failed
			)
		)

	def __reduce__(self):
		return type(self), (self.stage_name, self.qc)


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
		produces_artifacts=("bac_qc",),
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
			return replace(
				cls.contract, preferred_window_axis="Theta",
				# The two-dimensional Fourier operator has global support. Keep
				# real detector rows so a selected ROI shares the full-run canvas.
				required_pad=PadRequirement(full_source_axes=("Y",)),
			)
		return replace(
			cls.contract,
			mutates_input=False,
			accepts_domain=(DataDomain.TRANSMISSION, DataDomain.ATTENUATION),
			produces_domain=PRESERVE,
			produces_artifacts=(),
		)

	def finish_projections(self, state: PipelineState) -> PipelineState:
		if not self.params.enabled:
			return state
		data = np.asarray(state.working_array)
		projections = []
		for local_index in state.work_indices("Theta"):
			# Borrowed-array callers may omit acquisition geometry. Report their
			# assigned buffer ordinals explicitly, never invent detector indices.
			source_index = (
				state.source_window.source_index("Theta", local_index)
				if state.source_window is not None else local_index
			)
			image = data[state.projection_index(local_index)]
			# Count row by row: QC needs no additional detector-sized mask.
			finite = sum(int(np.count_nonzero(np.isfinite(row))) for row in image)
			projections.append({
				"projection_index": source_index,
				"acquisition_index": (
					int(state.dataset.acquisition_indices[source_index])
					if state.dataset is not None and state.source_window is not None else None
				),
				"nonfinite_count": int(image.size - finite),
			})
		qc = {
			"stage": self.name, "projections": projections,
			"index_space": "source" if state.source_window is not None else "buffer",
		}
		state.artifacts["bac_qc"] = qc
		if any(item["nonfinite_count"] for item in projections):
			raise BACNonFiniteError(self.name, qc)
		return state

	def bounded_work_requests(
		self, input_spec: BufferSpec, window: TomoWindow | None = None,
	) -> tuple[BoundedWorkRequest, ...]:
		if not self.params.enabled or not self.params.clip_input:
			return ()
		shape = input_spec.shape if window is None else window.with_pad().frange.length
		return (BoundedWorkRequest(
			"_bac_clip_work", BufferSpec(shape.to(ProjOrder), input_spec.dtype),
			"bac_clip_work", required_for_fallback=True,
		),)

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
		if self.params.clip_input:
			buffer = state.artifacts["_bac_clip_work"]
			assigned = buffer[state.scratch_index(buffer.shape, context=True)]
			clipped = buffer_view(assigned)[0]
			clipped[...] = image
			image = clipped
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
