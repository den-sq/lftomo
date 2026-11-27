"""Measured end/start beam residual and object-leak centre scan."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from lftomo.storage import PipelineRunContext, StorageBackend
from lftomo.util.shapes import GenericOrder, Proj
from lftomo.util.window import TomoDataSet, TomoWindow
from .band_shift import _validate_band_inputs
from .build import CalibrationBuildSession
from .models import BeamBandShiftModel, CalibrationComponent, ValidationResult

if TYPE_CHECKING:
	from lftomo.pipeline.stages.center import ManualCenterStage


@dataclass(frozen=True)
class BeamResidualBuild:
	"""Completed residual descriptor and the model to use for its schedule."""

	component: CalibrationComponent
	model: BeamBandShiftModel


def _fit_endpoint(change, texture, rows, fit_rows, model, shifts, stretches, sigma):
	"""Fit band profiles by high-pass RMS, independently of schedule ZNCC."""

	observed = change - gaussian_filter1d(change, sigma, axis=-1, mode="nearest")
	best = (np.inf, 0.0, 1.0)
	for stretch in stretches:
		for shift in shifts:
			source_rows = model.source_y(rows, shift, stretch)
			predicted = np.stack([np.interp(source_rows, rows, band) - band for band in texture])
			predicted -= gaussian_filter1d(predicted, sigma, axis=-1, mode="nearest")
			residual = (observed - predicted)[:, fit_rows]
			rms = float(np.sqrt(np.mean(residual ** 2)))
			if rms < best[0]:
				best = rms, float(shift), float(stretch)
	return best


def build_beam_band_shift_residual(
	build: CalibrationBuildSession,
	source: Any,
	dataset: TomoDataSet,
	*,
	model: BeamBandShiftModel,
	center: ManualCenterStage.Params,
	fit_rows: tuple[int, int],
	fit_columns: tuple[int, int],
	pairs: int,
	pair_start: int = 0,
	scan_center: tuple[float, float] = (4.0, 0.5),
	highpass_sigma: float = 30,
	bands: int = 8,
	y_chunk: int = 64,
	storage: StorageBackend | None = None,
	name: str = "beam_residual",
) -> BeamResidualBuild:
	"""Cancel mirrored object structure, fit endpoint motion, and persist R.

	This implements the ``endstart_xy.py`` method: average log raw counts at
	opposing scan ends, subtract the white's same mirror difference, choose the
	centre by fine-X leak, then subtract a fitted shift/stretch of the smoothed
	log white. High-pass Y filtering is used only for fitting, never on R.
	The returned model owns the measured endpoint dy and residual reference;
	pass it to ``build_beam_band_shift_schedule``.

	Inputs and component output use bounded Y reads/writes. Full detector work
	planes use the supplied storage backend (borrowed), or a scoped run context.
	SciPy owns its internal cubic prefilter/edge-extension and filter-line work;
	map outputs and coordinates use managed buffers. No workers are dispatched.
	"""

	_validate_band_inputs(build, source, dataset, model)
	physical = dataset.physical_shape
	if pairs < 1 or pair_start < 0 or 2 * (pair_start + pairs) > physical.Theta:
		raise ValueError("pair selection must contain positive, non-overlapping opposing endpoint pairs")
	if y_chunk < 1 or bands < 1 or not np.isfinite(highpass_sigma) or highpass_sigma <= 0:
		raise ValueError("y_chunk, bands and highpass_sigma must be positive")
	half_range, step = scan_center
	if not np.isfinite((half_range, step)).all() or half_range < 0 or step < 0:
		raise ValueError("scan_center requires a finite nonnegative half-range and step")
	if not np.isfinite((center.value_px, center.tilt, center.reference_y)).all():
		raise ValueError("mirror centre and tilt must be finite")
	# A missing frame must not silently change which angle is mirrored. The
	# original method assumes theta_i + theta_j = pi, not just two end blocks.
	first = np.arange(pair_start, pair_start + pairs)
	last = physical.Theta - 1 - first
	if not np.allclose(dataset.angles[first] + dataset.angles[last], np.pi,
			rtol=0, atol=np.deg2rad(1e-3)):
		raise ValueError("opposing endpoint pair angles must sum to 180 degrees")
	plane = physical.extract(Proj)
	window = TomoWindow(plane, plane.make_with(0), plane.make_with(0), plane.make_with(1), np.float64)
	roi = window.with_crop(
		Proj(fit_rows[0], fit_columns[0]),
		Proj(plane.Y - fit_rows[1], plane.X - fit_columns[1]),
	)
	if bands > roi.shape.X:
		raise ValueError("each fitting column band must contain at least one column")
	edges = np.linspace(roi.pos.X, roi.interval("X").stop.X, bands + 1).astype(int)
	band_windows = [window.rescope(pos=Proj(0, int(left)), shape=Proj(plane.Y, int(right - left)))
		for left, right in zip(edges[:-1], edges[1:])]
	chunks = [window.rescope(pos=Proj(start, 0), shape=Proj(min(y_chunk, plane.Y - start), plane.X))
		for start in range(window.pos.Y, window.interval("Y").stop.Y, y_chunk)]
	rows = np.fromiter(window.int_iter("Y"), dtype=np.float64)
	columns = np.fromiter(window.int_iter("X"), dtype=np.float64)

	with ExitStack() as lifetime:
		if storage is None:
			storage = lifetime.enter_context(PipelineRunContext()).storage
		buffers = []
		def allocate(role, shape=window.shape, dtype=np.float64):
			buffer = storage.allocate(shape, np.dtype(dtype), role=f"beam_residual_{role}")
			buffers.append((role, buffer))
			return buffer.array()

		try:
			start_log, end_log, white_log = (allocate(role) for role in ("start", "end", "white"))
			# The first dimension here is a numerical batch/coordinate slot, not
			# another acquisition axis. Detector extents come from the window.
			logs = allocate("log_batch", GenericOrder(16, chunks[0].shape.Y, plane.X), np.float32)
			acc = allocate("pair_sums", GenericOrder(4, chunks[0].shape.Y, plane.X))
			for chunk in chunks:
				selection = chunk.axis_slice("Y")
				read = {"y_start": selection.start, "y_stop": selection.stop}
				dark = np.asarray(build.read_component(model.dark_component, **read), dtype=np.float32)
				white = np.asarray(build.read_component(model.white_component, **read), dtype=np.float32) - dark
				if not np.isfinite(dark).all() or not np.isfinite(white).all():
					raise ValueError("white and dark must contain finite counts")
				np.maximum(white, 1, out=white)
				np.log(white, out=white)
				white_log[chunk.index] = white
				sums = acc[:, :chunk.shape.Y]
				sums.fill(0)
				for offset in range(0, pairs, 16):
					count = min(16, pairs - offset)
					for end, indices in enumerate((first, last)):
						block = logs[:count, :chunk.shape.Y]
						for local, index in enumerate(indices[offset:offset + count]):
							block[local] = source.read(int(index), **read)
						if not np.isfinite(block).all():
							raise ValueError("endpoint projections must contain finite counts")
						np.subtract(block, dark, out=block)
						np.maximum(block, 1, out=block)
						np.log(block, out=block)
						# Preserve the producer's float32, 16-frame parity sums,
						# then accumulate in float64. Parity starts at pair_start.
						for parity in (0, 1):
							sums[2 * end + parity] += block[parity::2].sum(axis=0)
				start_log[chunk.index] = (sums[0] + sums[1]) / pairs
				end_log[chunk.index] = (sums[2] + sums[3]) / pairs

			texture, change, work, smoothed = (allocate(role) for role in ("texture", "change", "work", "smoothed"))
			coordinates = allocate("coordinates", GenericOrder(2, plane.Y, plane.X))
			coordinates[0] = rows[:, None]
			gaussian_filter1d(white_log, model.sigma_x, axis=1, output=texture, mode="nearest")

			def measure_change(value):
				candidate = center.model_copy(update={"value_px": value})
				# The coordinate arrays are SciPy boundary adapters derived from
				# the canonical centre model and detector window.
				np.subtract(2 * candidate.values_at_rows(rows[:, None]), columns[None, :], out=coordinates[1])
				map_coordinates(start_log, coordinates, output=work, order=1, mode="nearest")
				np.subtract(end_log, work, out=change)
				map_coordinates(white_log, coordinates, output=work, order=1, mode="nearest")
				np.subtract(white_log, work, out=work)
				np.subtract(change, work, out=change)

			def leak(value):
				measure_change(value)
				gaussian_filter1d(change, 8, axis=1, output=work, mode="nearest")
				np.subtract(change, work, out=work)
				np.square(work, out=work)
				return float(np.sqrt(np.mean(work[roi.index])))

			center_scan = {}
			selected = center.value_px
			if step > 0:
				for value in np.arange(selected - half_range, selected + half_range + 1e-9, step):
					center_scan[round(float(value), 3)] = leak(value)
				coarse_center = min(center_scan, key=center_scan.get)
				for value in np.arange(coarse_center - step, coarse_center + step + 1e-9, step / 5):
					center_scan[round(float(value), 3)] = leak(value)
				selected = min(center_scan, key=center_scan.get)
			edge = bool(step > 0 and abs(selected - center.value_px) >= half_range - 1e-6)
			initial_leak = leak(center.value_px)
			selected_leak = leak(selected)
			# leak leaves change intact. Only X smoothing is applied to the full
			# measured change; the later Y high-pass belongs solely to the fit.
			gaussian_filter1d(change, model.sigma_x, axis=1, output=smoothed, mode="nearest")
			observed, reference = (np.stack([field[band.index].mean(axis=1) for band in band_windows])
				for field in (smoothed, texture))
			if np.ptp(reference - gaussian_filter1d(reference, highpass_sigma, axis=-1, mode="nearest")) <= np.finfo(float).eps:
				raise ValueError("endpoint fitting requires beam texture in the white field")
			coarse = _fit_endpoint(observed, reference, rows, roi.axis_slice("Y"), model,
				np.arange(-5, 45.1, 1.0), np.arange(0.975, 1.0051, 0.0025), highpass_sigma)
			fine = _fit_endpoint(observed, reference, rows, roi.axis_slice("Y"), model,
				np.arange(coarse[1] - 1.5, coarse[1] + 1.51, 0.25),
				np.arange(coarse[2] - 0.003, coarse[2] + 0.0031, 0.0005), highpass_sigma)
			if fine[1] == 0:
				raise ValueError("a zero endpoint dy cannot scale a beam residual")
			fitted = BeamBandShiftModel.model_validate({**model.model_dump(), "dy_end": fine[1], "residual_component": name})
			coordinates[0] = fitted.source_y(rows[:, None], fitted.dy_end, fine[2])
			coordinates[1] = columns[None, :]
			map_coordinates(texture, coordinates, output=work, order=3, mode="nearest")
			np.subtract(work, texture, out=work)
			np.subtract(smoothed, work, out=change)
			qc = [
				ValidationResult(
					name="band_shift_residual_center_scan", status="warning" if edge else "pass",
					message="centre scan reaches its search edge; inspect or widen scan_center" if edge
						else "fine-X object-leak centre measurement",
					metrics={"center_used": selected, "center_scan": {str(key): value for key, value in center_scan.items()},
						"at_search_edge": edge, "initial_leak_rms": initial_leak, "selected_leak_rms": selected_leak},
				),
				ValidationResult(
					name="band_shift_residual_fit", status="pass", message="endpoint band-profile high-pass RMS fit",
					metrics={"dy": fitted.dy_end, "m": fine[2], "resid_rms_bands": fine[0]},
				),
			]
			component = build.write_array_component(
				name, (change[chunk.index] for chunk in chunks), shape=window.shape.as_tuple(), dtype="float32",
				axes=("Y", "X"), kind="beam_residual", coordinate_frame="beam", validation_results=qc,
				metadata={
					"method": "mirrored_end_start_shift_stretch_residual",
					"center": center.model_dump(), "scan_center": list(scan_center),
					"fit_rows": [roi.pos.Y, roi.interval("Y").stop.Y],
					"band_edges": edges.tolist(), "highpass_sigma": highpass_sigma,
					"sigma_x": model.sigma_x, "yc": model.yc, "smoothing_mode": "nearest",
					"pairs": pairs, "pair_start": pair_start,
					"pair_acquisition_indices": np.column_stack((dataset.acquisition_indices[first], dataset.acquisition_indices[last])).tolist(),
					"pair_angles_radians": np.column_stack((dataset.angles[first], dataset.angles[last])).tolist(),
					"y_chunk": y_chunk, "log_accumulation": "float32_parity_sums_in_blocks_of_16_to_float64",
				},
			)
			return BeamResidualBuild(component, fitted)
		finally:
			for role, buffer in reversed(buffers):
				storage.retire(buffer, role=f"beam_residual_{role}")
