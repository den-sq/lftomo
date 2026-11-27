"""Calibration-driven vertical beam texture normalization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter1d, spline_filter

from lftomo.calibration import CalibrationBundle, CalibrationBundleRef
from lftomo.pipeline.core import BoundedWorkRequest, BufferSpec, PipelineState, Stage, StageContract, StageParams, allocate_output
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.storage import ManagedBuffer, TypedBuffer, buffer_view
from lftomo.util.shapes import Proj, ProjOrder, SinoOrder, YSlice
from lftomo.util.window import TomoWindow
from .normalize import _finish_normalization, _normalization_output


_PREPARED = "_beam_band_shift_prepared"
_WARP = "_beam_band_shift_warp"
_LINE = "_beam_band_shift_line"


@dataclass
class _Prepared:
	"""Temporary derived fields; the bundle still owns all measured facts."""

	bundle: CalibrationBundle
	window: TomoWindow
	spline_window: TomoWindow
	buffers: dict[str, TypedBuffer] = field(default_factory=dict)


class BeamBandShiftNormalizeStage(Stage):
	"""Apply the measured dy/m schedule to an X-smoothed log white field.

	Full detector calibration planes preserve cubic spline boundary conditions
	across row selections. Projection reads need no additional context. SciPy
	filters write into managed buffers; their internal line work is library-owned.
	"""

	name = "normalize.beam_band_shift"
	contract = StageContract(
		produces=PRESERVE, accepts=(ProjOrder, SinoOrder), prefers=ProjOrder,
		accepts_domain=(DataDomain.RAW,), produces_domain=DataDomain.TRANSMISSION,
		mutates_input=False, window_axes=("Y", "Theta"), cacheable_prefix=True,
		consumes_artifacts=("calibration_bundle",),
	)

	class Params(StageParams):
		schedule_component: str = "beam_schedule"

	def bounded_work_requests(self, input_spec: BufferSpec) -> tuple[BoundedWorkRequest, ...]:
		shape = input_spec.shape.to(ProjOrder)
		return (
			BoundedWorkRequest(_WARP, BufferSpec(shape, np.float32), "beam_band_shift_warp", required_for_fallback=True),
			BoundedWorkRequest(_LINE, BufferSpec(shape.extract(YSlice), np.float32), "beam_band_shift_line", required_for_fallback=True),
		)

	def prepare(self, state: PipelineState) -> None:
		if state.backend.name != "numpy":
			raise TypeError(f"{self.name} requires the NumPy backend")
		if state.source_window is None or state.dataset is None:
			raise RuntimeError(f"{self.name} requires canonical source geometry")
		ref = state.artifacts.get("calibration_bundle")
		if not isinstance(ref, CalibrationBundleRef):
			raise ValueError(f"{self.name} requires a selected calibration bundle")
		bundle = CalibrationBundle.open(ref.manifest_path)
		descriptor = bundle.component(self.params.schedule_component)
		model = descriptor.band_shift
		if model is None:
			raise ValueError("beam schedule requires a typed band_shift model")
		plane = state.dataset.physical_shape.extract(Proj)
		window = TomoWindow(plane, plane.make_with(0), plane.make_with(0), plane.make_with(1), np.float32)
		# scipy.ndimage prefilters nearest-mode cubic interpolation after a
		# twelve-sample edge extension. Prepare the same coefficients once.
		spline_window = window.with_output_padding(Proj(12, 12))
		prepared = _Prepared(bundle, window, spline_window)
		state.artifacts[_PREPARED] = prepared
		try:
			def allocate(key, geometry=window, dtype=np.float32):
				buffer, values = allocate_output(
					state, geometry.shape, dtype, role=f"beam_band_shift_{key}",
					reuse_from=(f"beam_band_shift_{key}",),
				)
				prepared.buffers[key] = buffer
				return values

			def read_map(name, kind):
				component = bundle.component(name)
				if component.kind != kind or component.payload.axes != ("Y", "X"):
					raise ValueError(f"{name!r} must be a {kind} (Y, X) component")
				region = bundle.compatibility(name).region
				if region is not None and (region.x_start != 0 or region.width != plane.X):
					raise ValueError("band-shift calibration must cover the detector columns")
				values = bundle.read_component(name, y_start=window.pos.Y, y_stop=window.interval("Y").stop.Y)
				if values.shape != plane.as_tuple() or not all(np.isfinite(row).all() for row in values):
					raise ValueError(f"{name!r} must be a finite full-detector field")
				return values

			dark = allocate("dark")
			dark[...] = read_map(model.dark_component, "dark")
			gain = allocate("gain")
			gain[...] = read_map(model.white_component, "detector_static")
			np.subtract(gain, dark, out=gain)
			np.maximum(gain, 1, out=gain)
			texture = allocate("texture")
			np.log(gain, out=texture)
			gaussian_filter1d(texture, model.sigma_x, axis=1, output=texture, mode="reflect")
			coefficients = allocate("spline", spline_window, np.float64)
			# Generated padding belongs to spline_window; edge replication itself
			# is only a numerical fill of that canonical allocation.
			inner = window.rescope(pos=window.pos - spline_window.pos).index
			coefficients[inner] = texture
			y_slice, x_slice = inner
			coefficients[:y_slice.start, x_slice] = texture[:1]
			coefficients[y_slice.stop:, x_slice] = texture[-1:]
			coefficients[:, :x_slice.start] = coefficients[:, x_slice.start:x_slice.start + 1]
			coefficients[:, x_slice.stop:] = coefficients[:, x_slice.stop - 1:x_slice.stop]
			spline_filter(coefficients, order=3, output=coefficients, mode="nearest")
			if model.residual_component is not None:
				allocate("residual")[...] = read_map(model.residual_component, "beam_residual")
		except BaseException:
			self.finish(state)
			raise

	def finish(self, state: PipelineState) -> None:
		prepared = state.artifacts.pop(_PREPARED, None)
		if prepared is not None and state.storage is not None:
			for key, buffer in prepared.buffers.items():
				if isinstance(buffer, ManagedBuffer):
					state.storage.retire(buffer, role=f"beam_band_shift_{key}")

	def run(self, state: PipelineState) -> PipelineState:
		from lftomo.pipeline.execution import _prepare_bounded_work, _release_bounded_work

		owned_preparation = _PREPARED not in state.artifacts
		if owned_preparation:
			self.prepare(state)
		requests = ()
		try:
			if _WARP not in state.artifacts:
				requests = _prepare_bounded_work(state, (self,), "Theta")
			output, data = _normalization_output(state)
			prepared = state.artifacts[_PREPARED]
			model = prepared.bundle.component(self.params.schedule_component).band_shift
			source = state.source_window.with_pad(["Y"])
			selection = source.index[source.shape.axis_of("Y")], source.index[source.shape.axis_of("X")]
			gain, dark, texture = (buffer_view(prepared.buffers[key])[selection] for key in ("gain", "dark", "texture"))
			coefficients = buffer_view(prepared.buffers["spline"])
			residual = None if "residual" not in prepared.buffers else buffer_view(prepared.buffers["residual"])[selection]
			work = buffer_view(state.artifacts[_WARP])
			lines = buffer_view(state.artifacts[_LINE])
			for index in state.work_indices("Theta"):
				source_index = state.source_window.source_index("Theta", index)
				projection = int(state.dataset.acquisition_indices[source_index])
				bracket = prepared.bundle.read_time_bracket(self.params.schedule_component, projection)
				if bracket.before_time != projection and bracket.after_time != projection:
					raise ValueError(f"beam schedule has no measurement for projection {projection}")
				dy, magnification = bracket.before if bracket.before_time == projection else bracket.after
				if not np.isfinite((dy, magnification)).all() or magnification <= 0:
					raise ValueError(f"invalid beam shift/stretch for projection {projection}")
				warped = work[index]
				affine_transform(
					coefficients, ((1 / magnification, 0), (0, 1)),
					offset=(model.yc + (source.pos.Y - model.yc - dy) / magnification - prepared.spline_window.pos.Y,
						source.pos.X - prepared.spline_window.pos.X),
					output=warped, order=3, mode="nearest", prefilter=False,
				)
				np.subtract(warped, texture, out=warped)
				np.exp(warped, out=warped)
				if residual is not None:
					scale = float(np.clip(dy / model.dy_end, 0, model.s_max))
					for row in range(warped.shape[0]):
						np.multiply(residual[row], scale, out=lines[index])
						np.exp(lines[index], out=lines[index])
						np.multiply(warped[row], lines[index], out=warped[row])
				np.multiply(warped, gain, out=warped)
				image = data[state.layout.point_index(Theta=index)]
				np.subtract(image, dark, out=image)
				np.divide(image, warped, out=image)
		finally:
			_release_bounded_work(state, requests)
			if owned_preparation:
				self.finish(state)
		return _finish_normalization(state, output)
