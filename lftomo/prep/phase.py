"""
Module for phase retrieval.  Rewritten from tomopy.
"""

# ADD COPYRIGHT NOTICE - SEE TOMOPY

from multiprocessing import cpu_count, Pool
from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from scipy import fft as sfft

from lftomo.util.shapes import ImgSlice, Proj
from lftomo.util.window import TomoWindow

if TYPE_CHECKING:
	from lftomo.util.mem import SharedNPArray

BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]

prime_set = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}


def wavelength_m(energy_kev: float) -> float:
	"""Wavelength in metres used by the calibrated material-line model."""

	return 1.23984193e-9 / energy_kev


def phase_fft_window(shape: tuple[int, int], pad: bool = True) -> TomoWindow:
	"""Own the numerical FFT canvas and its crop in the canonical window model.

	This edge-filled numerical canvas is separate from source-context padding.
	Odd extensions retain their asymmetric remainder in ``pad_after``.
	"""

	interior = Proj(*shape)
	canvas = Proj(*(
		sfft.next_fast_len(int(size * 1.5)) if pad else size for size in shape
	))
	before = Proj(*((outer - inner) // 2 for outer, inner in zip(
		canvas.as_tuple(), interior.as_tuple(), strict=True,
	)))
	return TomoWindow(
		interior, before, interior.make_with(0), interior.make_with(1),
		pad_after=canvas - interior - before,
	)


def phase_pad(array: np.ndarray, window: TomoWindow, *, out: np.ndarray | None = None) -> np.ndarray:
	"""Edge-fill the canvas described by ``window`` around trailing Y/X axes."""

	return window.pad_array(array, out=out)


def phase_unpad(array: np.ndarray, window: TomoWindow) -> np.ndarray:
	"""Return the canonical interior view of a trailing-axis FFT canvas."""

	return array[(Ellipsis,) + window.internal.index]


def phase_frequency_squared(
	window: TomoWindow, pixel_m: float, *, real: bool = True, angular: bool = False,
) -> np.ndarray:
	"""Squared DFT-bin frequencies for the window's one authoritative canvas.

	New material-line callers use metres. The full-grid legacy adapter retains
	centimetres throughout its existing expression for bitwise compatibility.
	"""

	canvas = window.with_pad().shape
	factor = 2 * np.pi if angular else 1
	fy = factor * np.fft.fftfreq(canvas.Y, d=pixel_m)
	fx = factor * (np.fft.rfftfreq if real else np.fft.fftfreq)(canvas.X, d=pixel_m)
	return np.add.outer(np.square(fy), np.square(fx))


def slope_denominator(window: TomoWindow, pixel_m: float, distance_m: float, slope: float) -> np.ndarray:
	"""Slope-form Paganin denominator on the shared real-frequency grid."""

	return 1 + 4 * np.pi**2 * distance_m * slope * phase_frequency_squared(window, pixel_m)


def single_material_filter(
	projections: np.ndarray, pixel_m: float, distance_m: float, slope: float, *,
	pad: bool = True, material_alpha: float = 0, envelope=None,
	dtype=np.float32, workers: int = 1, denominator: np.ndarray | None = None,
	window: TomoWindow | None = None, padding_work: np.ndarray | None = None,
	envelope_work: np.ndarray | None = None,
) -> np.ndarray:
	"""Filter transmission with the material line and one affine defect step.

	Nonzero ``material_alpha`` needs the spatial thickness envelope in metres.
	A supplied ``window`` owns the canvas instead of the ``pad`` convenience
	argument. Optional work arrays have its padded shape, including leading
	batch axes, and the requested ``dtype``; their contents are overwritten.
	The affine step is approximate. SciPy FFT and gradient outputs are unavoidable
	numerical-library temporaries; persistent filters remain caller-owned.
	"""

	image = np.asarray(projections, dtype=dtype)
	window = phase_fft_window(image.shape[-2:], pad) if window is None else window
	if denominator is None:
		denominator = slope_denominator(window, pixel_m, distance_m, slope)
	values = phase_pad(image, window, out=padding_work)
	if material_alpha != 0:
		if envelope is None:
			raise ValueError("nonzero material_alpha requires envelope_m for one-step defect correction")
		thickness = phase_pad(
			np.broadcast_to(np.asarray(envelope, dtype=dtype), image.shape), window, out=envelope_work,
		)
		initial = np.clip(sfft.irfft2(
			sfft.rfft2(values, workers=workers) / denominator,
			s=window.with_pad().shape.as_tuple(), workers=workers,
		), 1e-6, None)
		gy, gx = np.gradient(thickness, pixel_m, axis=(-2, -1))
		correction = (
			np.gradient(initial * gx, pixel_m, axis=-1)
			+ np.gradient(initial * gy, pixel_m, axis=-2)
		)
		values = values - distance_m * material_alpha * correction
	filtered = sfft.irfft2(
		sfft.rfft2(values, workers=workers) / denominator,
		s=window.with_pad().shape.as_tuple(), workers=workers,
	)
	return np.asarray(phase_unpad(filtered, window), dtype=dtype)


def negative_log(
	image: np.ndarray,
	*,
	epsilon: float | None,
	maximum: float | None,
	invert: bool = False,
) -> np.ndarray:
	"""Apply the one terminal transmission-to-attenuation conversion."""

	if epsilon is None:
		if not np.isfinite(image).all() or np.any(image <= 0):
			raise ValueError(
				"phase.negative_log requires finite positive transmission when epsilon is null"
			)
		transmission = np.minimum(image, maximum) if maximum is not None else image
	else:
		transmission = np.clip(image, epsilon, maximum)
	result = -np.log(transmission)
	return -result if invert else result


def retrieve_single(
	projections: np.ndarray, pixel_m: float, distance_m: float, slope: float, *,
	pad: bool = True, material_alpha: float = 0, envelope=None,
	dtype=np.float32, workers: int = 1, denominator: np.ndarray | None = None,
	epsilon: float = 1e-6,
	window: TomoWindow | None = None, padding_work: np.ndarray | None = None,
	envelope_work: np.ndarray | None = None,
) -> np.ndarray:
	"""Retrieve attenuation, forwarding optional canonical canvas and owned work."""

	transmission = single_material_filter(
		projections, pixel_m, distance_m, slope, pad=pad,
		material_alpha=material_alpha, envelope=envelope, dtype=dtype,
		workers=workers, denominator=denominator,
		window=window, padding_work=padding_work, envelope_work=envelope_work,
	)
	return negative_log(transmission, epsilon=epsilon, maximum=None)


def _paganin_filter_factor(energy, dist, alpha, w2, tp=False):
	if tp:
		return 1 / (_wavelength(energy) * dist * w2 / (4 * np.pi) + alpha)
	else:
		return 1 / (1 + (dist * alpha * _wavelength(energy) * w2 / (4 * np.pi)))


def _reciprocal_coord(pixel_size, num_grid, tp=False):
	"""
	Calculate reciprocal grid coordinates for a given pixel size
	and discretization.

	Parameters
	----------
	pixel_size : float
		Detector pixel size in cm.
	num_grid : int
		Size of the reciprocal grid.

	Returns
	-------
	ndarray
		Grid coordinates.
	"""
	n = num_grid - 1
	if tp:
		return np.arange(-n, num_grid, 2, dtype=np.float32) * 0.5 / (n * pixel_size)
	else:
		return np.arange(-n, num_grid, 2, dtype=np.float32) * 2 * np.pi / (n * pixel_size)


def _wavelength(energy):
	return 2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def _prime_factor_pad(base_size, wide_size):
	prime_factors = [[2]]
	for _i in range(1, max(len(bin(wide_size)) // 4, 2)):
		prime_factors = [x + [y] for x in prime_factors for y in prime_set if y not in x]

	prime_factors = sorted(prime_factors, key=lambda item: sum(item))
	for factor_set in prime_factors:
		if np.prod(factor_set) >= wide_size:
			return (np.prod(factor_set) - base_size) // 2


def _phase_pad(dim, pixel_size, dist, energy, force_pow2=False):
	wavelength = _wavelength(energy)
	pad_pix = np.ceil(np.pi * wavelength * dist / pixel_size ** 2)
	# Next power of 2.
	next_pow2 = int((pow(2, np.ceil(np.log2(dim + pad_pix))) - dim) * 0.5)
	# Next smallest prime factors.
	next_factors = _prime_factor_pad(dim, int(dim + pad_pix))
	return min(next_pow2, next_factors) if not force_pow2 else next_pow2


def legacy_phase_window(base_shape, pixel_size, dist, energy, height_pad=False) -> TomoWindow:
	"""Own the compatibility algorithm's symmetric height-only FFT canvas."""

	if height_pad:
		from lftomo.util.log import log, LOG

		pad_height = _phase_pad(base_shape.Y, pixel_size, dist, energy)
		log.write("Phase Prep", f"Pad Height {pad_height} on {base_shape.Y}, {pixel_size}, {dist}, {energy}",
							log_level=LOG.INFO)
	else:
		pad_height = 0

	shape = Proj(base_shape.Y, base_shape.X)
	return TomoWindow(shape, Proj(pad_height, 0), Proj(0, 0), Proj(1, 1))


def filter_shape(base_shape, pixel_size, dist, energy, height_pad=False):
	return legacy_phase_window(base_shape, pixel_size, dist, energy, height_pad).with_pad().shape.to(ImgSlice)


def filter_create(mem_name, mem_dtype, base_shape, alpha, pixel_size, dist, energy, tp=False, height_pad=False,
					*, window: TomoWindow | None = None):
	from lftomo.util.mem import SharedNPArray

	window = legacy_phase_window(base_shape, pixel_size, dist, energy, height_pad) if window is None else window
	phase_mem = SharedNPArray(mem_name, window.with_pad().shape.to(ImgSlice),
						mem_dtype, allocate=True)

	_set_phase_filter(alpha, pixel_size, dist, energy, phase_mem, tp)

	return phase_mem


def _set_phase_filter(alpha, pixel_size, dist, energy, phase_mem, tp=False):
	from lftomo.util.log import log, LOG

	with phase_mem as phase_filter:
		log.write("Phase Prep", log_level=LOG.INFO,
			statement=f'Filter: {phase_mem.shape} | {phase_mem} | {pixel_size} | {dist} | {energy} | {np.mean(phase_filter)}')

		w2 = np.add.outer(np.square(_reciprocal_coord(pixel_size, phase_mem.shape.Y, tp=tp)),
						np.square(_reciprocal_coord(pixel_size, phase_mem.shape.X, tp=tp)))
		unnormalized_filter = np.fft.fftshift(_paganin_filter_factor(energy, dist, alpha, w2, tp=tp))

		phase_filter[:, :] = np.divide(unnormalized_filter[:, :], unnormalized_filter.max())

		log.write("Phase Prep", log_level=LOG.INFO,
			statement=f'Filter: {phase_mem.shape} | {phase_mem} | {pixel_size} | {dist} | {energy} | {np.mean(phase_filter)}')


def phase_retrieval(dw: TomoWindow, source_mem: "SharedNPArray", phase_alpha, pixel_size, energy, dist, sino_order=True,
					tp=False, height_pad=False, proc_count=cpu_count()):

	def get_slice(sino_order, i):
		return np.s_[:, i, :] if sino_order else np.s_[i, :, :]

	from lftomo.util.mem import SharedNPArray

	window = legacy_phase_window(source_mem.shape, pixel_size, dist, energy, height_pad)
	phase_mem = filter_create("phase_filter", source_mem.dtype, source_mem.shape, phase_alpha,
								pixel_size, dist, energy, tp, height_pad, window=window)
	padding_work = None
	try:
		if window.pad_before.Y:
			padding_work = SharedNPArray(
				f"phase-pad-{uuid4().hex}", replace(source_mem.shape, Y=window.with_pad().shape.Y),
				source_mem.dtype, allocate=True,
			)
		with Pool(proc_count) as pool:
			pool.starmap(_phase_retrieval, [(get_slice(sino_order, i), source_mem, phase_mem,
												pixel_size, energy, dist, tp, height_pad, cpu_count(),
												window, None if padding_work is None else padding_work[get_slice(sino_order, i)])
												for i in range(dw.shape.Theta)])
	finally:
		if padding_work is not None:
			padding_work.shutdown()
		phase_mem.shutdown()


def _phase_retrieval(y_slice, work_mem: "SharedNPArray", phase_mem: "SharedNPArray",
					pixel_size, energy, dist, tp=False, height_pad=False,
					thread_fallback=cpu_count(), window: TomoWindow | None = None, padding_work=None):
	import numexpr as ne

	with work_mem as tomo, phase_mem as phase_filter:
		window = legacy_phase_window(work_mem.shape, pixel_size, dist, energy, height_pad) if window is None else window

		if window.pad_before.Y > 0:
			tomo_complex = np.fft.fft2(phase_pad(
				tomo[y_slice], window, out=None if padding_work is None else np.asarray(padding_work),
			), axes=(-2, -1))
			# log.write("Phase Type Check", f"{type(tomo_complex)}:{type(phase_filter)}", LOG.INFO)
			tomo_complex *= phase_filter
			tomo[y_slice] = np.real(np.fft.ifft2(phase_unpad(tomo_complex, window), axes=(-2, -1)))
		else:
			tomo_complex = np.fft.fft2(tomo[y_slice], axes=(-2, -1))
			# log.write("Phase Type Check", f"{type(tomo_complex)}:{type(phase_filter)}", LOG.INFO)
			tomo_complex *= phase_filter
			tomo[y_slice] = np.real(np.fft.ifft2(tomo_complex, axes=(-2, -1)))

		if not tp:
			# Post-Paganin filter minus_log
			low = np.where(tomo[y_slice] <= 0)
			high = np.where(tomo[y_slice] >= 1)
			tomo[y_slice][low] = 0.00000001
			tomo[y_slice][high] = 0.99999999

			ne.set_num_threads(1)
			tomo[y_slice] = ne.evaluate('-log(p)', local_dict={'p': tomo[y_slice]})
			ne.set_num_threads(thread_fallback)

			#  The final result will be the value of beta
			np.divide(tomo[y_slice], (4 * np.pi / _wavelength(energy)), out=tomo[y_slice])
