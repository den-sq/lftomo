"""
Module for phase retrieval.  Rewritten from tomopy.
"""

# ADD COPYRIGHT NOTICE - SEE TOMOPY

from multiprocessing import cpu_count, Pool

import numpy as np
import numexpr as ne

from lftomo.util.log import log, LOG
from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import ImgSlice
from lftomo.util.window import TomoWindow

BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]

prime_set = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}


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


def filter_shape(base_shape, pixel_size, dist, energy, height_pad=False):
	if height_pad:
		pad_height = _phase_pad(base_shape.Y, pixel_size, dist, energy)
		log.write("Phase Prep", f"Pad Height {pad_height} on {base_shape.Y}, {pixel_size}, {dist}, {energy}",
							log_level=LOG.INFO)
	else:
		pad_height = 0

	return ImgSlice(base_shape.Y + 2 * pad_height, base_shape.X)


def filter_create(mem_name, mem_dtype, base_shape, alpha, pixel_size, dist, energy, tp=False, height_pad=False):
	phase_mem = SharedNPArray(mem_name, filter_shape(base_shape, pixel_size, dist, energy, height_pad),
						mem_dtype, allocate=True)

	_set_phase_filter(alpha, pixel_size, dist, energy, phase_mem, tp)

	return phase_mem


def _set_phase_filter(alpha, pixel_size, dist, energy, phase_mem, tp=False):
	with phase_mem as phase_filter:
		log.write("Phase Prep", log_level=LOG.INFO,
			statement=f'Filter: {phase_mem.shape} | {phase_mem} | {pixel_size} | {dist} | {energy} | {np.mean(phase_filter)}')

		w2 = np.add.outer(np.square(_reciprocal_coord(pixel_size, phase_mem.shape.Y, tp=tp)),
						np.square(_reciprocal_coord(pixel_size, phase_mem.shape.X, tp=tp)))
		unnormalized_filter = np.fft.fftshift(_paganin_filter_factor(energy, dist, alpha, w2, tp=tp))

		phase_filter[:, :] = np.divide(unnormalized_filter[:, :], unnormalized_filter.max())

		log.write("Phase Prep", log_level=LOG.INFO,
			statement=f'Filter: {phase_mem.shape} | {phase_mem} | {pixel_size} | {dist} | {energy} | {np.mean(phase_filter)}')


def phase_retrieval(dw: TomoWindow, source_mem: SharedNPArray, phase_alpha, pixel_size, energy, dist, sino_order=True,
					tp=False, height_pad=False, proc_count=cpu_count()):

	def get_slice(sino_order, i):
		return np.s_[:, i, :] if sino_order else np.s_[i, :, :]

	phase_mem = filter_create("phase_filter", source_mem.dtype, source_mem.shape, phase_alpha,
												pixel_size, dist, energy, tp, height_pad)
	try:
		with Pool(proc_count) as pool:
			pool.starmap(_phase_retrieval, [(get_slice(sino_order, i), source_mem, phase_mem,
												pixel_size, energy, dist, tp, height_pad)
												for i in range(dw.shape.Theta)])
	finally:
		phase_mem.shutdown()


def _phase_retrieval(y_slice, work_mem: SharedNPArray, phase_mem: SharedNPArray,
					pixel_size, energy, dist, tp=False, height_pad=False,
					thread_fallback=cpu_count()):

	with work_mem as tomo, phase_mem as phase_filter:
		if height_pad:
			pad_height = _phase_pad(work_mem.shape.Y, pixel_size, energy, dist)
		else:
			pad_height = 0

		if pad_height > 0:
			tomo_complex = np.fft.fft2(np.pad(tomo[y_slice], ((pad_height, pad_height), (0, 0)), mode="edge"), axes=(-2, -1))
			# log.write("Phase Type Check", f"{type(tomo_complex)}:{type(phase_filter)}", LOG.INFO)
			tomo_complex *= phase_filter
			tomo[y_slice] = np.real(np.fft.ifft2(tomo_complex[pad_height:-pad_height, :], axes=(-2, -1)))
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
