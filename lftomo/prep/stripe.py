#!/usr/bin/env python
from collections import namedtuple
from multiprocessing import Pool

import numpy as np
import pywt
from scipy.signal import medfilt2d
from scipy.ndimage import median_filter
from scipy.ndimage import binary_dilation
from scipy.ndimage import uniform_filter1d
from scipy import interpolate
from tomopy.util import mproc


from lftomo.util.mem import SharedNP
from lftomo.util.window import DataWindow

DetailCoeff = namedtuple("DetailCoeff", ["cH", "cV", "cD"])

# Add Copyright Notice


def mem_size(mem, swap=None):
	if swap is None:
		shape = mem.shape
	else:
		shape = [swap[x] if swap[x] is not None else mem.shape[x] for x in range(len(swap))]
	return int(np.prod(shape, dtype=np.float64) * np.dtype(mem.dtype).itemsize)


def remove_sort_vo(dw: DataWindow, sino_mem: SharedNP, size: int = 21, dim: int = 1, proc_count: int = 12):
	with Pool(proc_count) as pool:
		pool.starmap(_remove_sort_vo, [(sino_mem, i, size, dim) for i in dw.internal.y_range])


def _remove_sort_vo(sino_mem: SharedNP, i: int, size: int = 21, dim: int = 1):
	with (sino_mem[i] as sino):

		f_shape = (size, 1 if dim == 1 else size)

		sino_mod = np.transpose(sino[:, :])
		sort_idx = sino_mod.argsort(axis=1) 			# Track index separately to sort back.
		sino_mod[:, :] = np.take_along_axis(sino_mod, sort_idx, axis=1)
		sort_idx[:, :] = sort_idx.argsort(axis=1) 	# Now transition it to sort back index.

		# Apply filter to sorted memory.
		sino_mod[:, :] = medfilt2d(sino_mod, f_shape)

		# sort back and assign.
		sino_mod[:, :] = np.take_along_axis(sino_mod, sort_idx, axis=1)
		sino[:, :] = np.transpose(sino_mod)


def _detect_stripe(listdata, snr):
	"""
	Algorithm 4 in :cite:`Vo:18`. Used to locate stripes.
	"""
	numdata = len(listdata)
	listsorted = np.sort(listdata)[::-1]
	xlist = np.arange(0, numdata, 1.0)
	ndrop = np.int16(0.25 * numdata)
	(_slope, _intercept) = np.polyfit(
		xlist[ndrop:-ndrop - 1], listsorted[ndrop:-ndrop - 1], 1)
	numt1 = _intercept + _slope * xlist[-1]
	noiselevel = np.abs(numt1 - _intercept)
	noiselevel = np.clip(noiselevel, 1e-6, None)
	val1 = np.abs(listsorted[0] - _intercept) / noiselevel
	val2 = np.abs(listsorted[-1] - numt1) / noiselevel
	listmask = np.zeros_like(listdata)
	if (val1 >= snr):
		upper_thresh = _intercept + noiselevel * snr * 0.5
		listmask[listdata > upper_thresh] = 1.0
	if (val2 >= snr):
		lower_thresh = numt1 - noiselevel * snr * 0.5
		listmask[listdata <= lower_thresh] = 1.0
	return listmask


def remove_large_vo(sino_mem: SharedNP, i: int, snr: int = 3, size: int = 51, drop_ratio: float = 0.1,
					norm: bool = True):
	with sino_mem[i] as sino_base:

		sino = sino_base.copy()

		# Calculate # of outliers to drop.
		drop_ratio = np.clip(drop_ratio, 0.0, 0.8)
		nrow = sino_mem.shape.Theta
		ndrop = int(0.5 * drop_ratio * nrow)

		# Get Sort Index (so sortback is easy)
		sort_idx = sino.argsort(axis=0)

		# Get mean of original and smoothed values after throwing out outliers.
		sino_smooth = np.take_along_axis(sino, sort_idx, axis=0)
		sort_mean = np.mean(sino_smooth[ndrop:-ndrop, :], axis=0)
		sino_smooth[:, :] = medfilt2d(sino_smooth, (1, size))
		smooth_mean = np.mean(sino_smooth[ndrop:-ndrop, :], axis=0)

		# Sort back sino_smooth.
		sort_idx[:, :] = sort_idx.argsort(axis=0)
		sino_smooth[:, :] = np.take_along_axis(sino_smooth[:, :], sort_idx, axis=0)

		# Locate stripes
		factor = np.divide(sort_mean, smooth_mean, out=np.ones_like(sort_mean), where=smooth_mean != 0)
		sino_mask = _detect_stripe(factor, snr)
		sino_mask = binary_dilation(sino_mask, iterations=1)

		# Alt Version - Test Speed - meaningful speed impact unlikely
		# sino[v_slice] = np.divide(sino[v_slice], np.tile(factor, (nrow, 1)), out=smooth_sino[v_slice],
		# 							where=~np.tile(sino_mask, (nrow, 1))))

		# Normalize  - Why separate to mask application (?)
		if norm:
			np.divide(sino, np.tile(factor, (nrow, 1)), out=sino_base[:, :])

		# Apply mask.  I DO NOT KNOW WHY TRANSPOSE IS NECESSARY, NP ADVANCED INDEXING IS WEIRD
		sino_base[:, sino_mask] = sino_smooth[:, sino_mask]


def remove_dead_vo(dw: DataWindow, sino_mem: SharedNP, snr: int = 3, size: int = 51, norm: float = True,
					proc_count: int = 12):
	with Pool(proc_count) as pool:
		pool.starmap(_remove_dead_vo, [(sino_mem, i, snr, size, norm) for i in dw.internal.y_range])


def _remove_dead_vo(sino_mem: SharedNP, i: int, snr: int = 3, size: int = 51, norm: float = True):
	with sino_mem[i] as sino_base:
		# Portion used for binary dilation.
		mask_cut = np.s_[2:-2]

		# Copy sinogram into contigous array.  Not strictly necessary but will allow multiple orders in future.
		# So leave it here for now.
		sino = np.ascontiguousarray(sino_base[:, :], dtype=sino_mem.dtype)

		# Find areas that deviate from smoothed setup.
		smooth_diff = np.sum(np.abs(sino - np.apply_along_axis(uniform_filter1d, 0, sino, 10)), axis=0)
		median_diff = median_filter(smooth_diff, size)
		factor = np.divide(smooth_diff, median_diff, out=np.ones_like(smooth_diff), where=median_diff != 0)

		# Dilate deviated areas, as a boolean mask.
		mask = np.full(factor.shape, False)
		mask[mask_cut] = binary_dilation(_detect_stripe(factor, snr), iterations=1)[mask_cut]
		x_change = mask.nonzero()[0]

		y_index = np.arange(sino_mem.shape.Theta)

		# Use correct areas as an interpolated mask to apply to bad areas.
		x_keep = np.invert(mask).nonzero()[0]
		func_inter = interpolate.interp2d(x_keep, y_index, sino[:, x_keep], kind='linear')

		if len(x_change) > 0:
			# I DO NOT KNOW WHY TRANSPOSE IS NECESSARY, NP ADVANCED INDEXING IS WEIRD
			sino_base[:, mask] = func_inter(x_change, y_index)

	# Remove residual stripes
	if norm is True:
		remove_large_vo(sino_mem, i, snr, size, norm=norm)


def remove_stripe_fw(dw: DataWindow, sino_mem: SharedNP, level: int = None, wname: str = 'db5', sigma: float = 2.0,
						zero_pad: bool = False, proc_count: int = 12):
	"""
	Remove horizontal stripes from sinogram using the Fourier-Wavelet (FW)
	based method :cite:`Munch:09`.

	:param dw: Datawindow specifying range to operate on..
	:param sino_mem: Sinogram memory of data.
	:param level: Number of levels of wavelet decomposition.
				Defaults to log2 of the largest dimension of the input.
	:param uname: Wavelet decomposition filter.
	:param sigma: Damping parameter in Fourier space.
	:param zero_pad: Whether to pad with zeroes instead of symmetrically during
				wavelet transformation. :cite:`Munch:09` specifices symmetric but
				tomopy offers additional optional zero-padding before wavelet
				transformation; so here's just zero-padding during transformation
				as an option.
	:param proc_count: int, optional
	"""
	if level is None:
		level = int(np.ceil(np.log2(np.max(sino_mem.shape))))
	with Pool(proc_count) as pool:
		pool.starmap(_remove_stripe_fw, [(sino_mem, i, level, wname, sigma, zero_pad) for i in dw.internal.y_range])


def _compounding_dwt2(sli, wname, mode, level):
	sli, coeff = pywt.dwt2(sli, wname, mode=mode)
	return _compounding_dwt2(sli, wname, mode, level - 1) + [DetailCoeff(*coeff)] if level else [sli]


def _remove_stripe_fw(sino_mem, i, level, wname, sigma, zero_pad=False):
	extension_mode = "zero" if zero_pad else "symmetric"
	theta_pad = sino_mem.shape.Theta // 16

	with sino_mem[i] as sino:
		sino_indexing = np.s_[theta_pad:theta_pad + sino_mem.shape.Theta, 0:sino_mem.shape.X]
		sli = np.pad(sino[:, :], ((theta_pad, ), (0, )), 'constant', constant_values=(0.0, 0.0))
		cWavelet = []

		# Wavelet decomposition
		for _j in range(level):
			sli, coeff = pywt.dwt2(sli, wname, mode=extension_mode)
			cWavelet = [DetailCoeff(*coeff)] + cWavelet

		# Ring artifact removal in FFT space.
		def fft_space_removal(j):
			fcV = np.fft.fftshift(np.fft.fft(cWavelet[j].cV, axis=0))
			my, mx = fcV.shape

			# Damping of ring artifact information.
			y_hat = (np.arange(-my, my, 2, dtype='float32') + 1) / 2
			damp = -np.expm1(-np.square(y_hat) / (2 * np.square(sigma)))
			fcV *= np.transpose(np.tile(damp, (mx, 1)))

			# Inverse FFT.
			cWavelet[j] = cWavelet[j]._replace(cV=np.real(np.fft.ifft(np.fft.ifftshift(fcV), axis=0)))

		# map doesn't execute by default, so let's just do this
		[fft_space_removal(x) for x in range(0, len(cWavelet))]

		# Multilevel Wavelet Reconstruction
		sino[:, :] = pywt.waverec2([sli] + cWavelet, wname, mode=extension_mode)[sino_indexing]
