#!/usr/bin/env python
from collections import namedtuple
from dataclasses import astuple
from multiprocessing import Pool
from numbers import Integral

import numpy as np
import pywt
from scipy.signal import medfilt2d
from scipy.ndimage import median_filter
from scipy.ndimage import binary_dilation
from scipy.ndimage import uniform_filter1d


from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import DataShape, Theta, YSlice
from lftomo.util.window import TomoWindow

DetailCoeff = namedtuple("DetailCoeff", ["cH", "cV", "cD"])

# Add Copyright Notice


def remove_sort_vo(dw: TomoWindow, sino_mem: SharedNPArray, size: int = 21, dim: int = 1, proc_count: int = 12):
	with Pool(proc_count) as pool:
		pool.starmap(_remove_sort_vo, [(sino_mem, i, size, dim) for i in dw.internal.int_iter()])


def _remove_sort_vo(sino_mem: SharedNPArray, i: int, size: int = 21, dim: int = 1):
	with (sino_mem[i] as sino):
		_remove_sort_array(sino, size=size, dim=dim)


def _work_plane(work: np.ndarray | None, sino: np.ndarray, name: str) -> np.ndarray:
	"""Return a reusable plane with the sinogram's shape and numeric dtype."""

	if work is None:
		return np.empty_like(sino)
	if work.shape != sino.shape:
		raise ValueError(f"{name} shape {work.shape} does not match sinogram {sino.shape}")
	if work.dtype != sino.dtype:
		raise TypeError(f"{name} dtype {work.dtype} does not match sinogram {sino.dtype}")
	return work


def _index_plane(index_work: np.ndarray | None, sino: np.ndarray) -> np.ndarray:
	"""Return a reusable uint32 sorting-index plane."""

	if index_work is None:
		return np.empty(sino.shape, dtype=np.uint32)
	if index_work.shape != sino.shape:
		raise ValueError(
			f"Vo sorting-index shape {index_work.shape} does not match sinogram {sino.shape}"
		)
	if index_work.dtype != np.dtype(np.uint32):
		raise TypeError("Vo sorting-index work must use uint32 storage")
	return index_work


def _bottleneck_line_half_width(size: int) -> int:
	"""Return reusable border width for a supported odd line median."""

	if not isinstance(size, Integral) or size < 1 or size % 2 == 0:
		return 0
	return int(size) // 2


def _move_median(values: np.ndarray, *, window: int, min_count: int, axis: int) -> np.ndarray:
	"""Import Bottleneck only in workers that execute a Vo line median."""

	from bottleneck import move_median

	return move_median(values, window=window, min_count=min_count, axis=axis)


def _line_median(
	values: np.ndarray,
	size: int,
	axis: int,
	*,
	pad_work: np.ndarray | None = None,
	use_bottleneck: bool = True,
) -> np.ndarray:
	"""Match a centered, zero-padded SciPy line median with Bottleneck.

	Only native float32/float64 finite inputs use Bottleneck. Other dtypes and
	non-finite values retain the established SciPy behavior.
	"""

	if axis not in (0, 1):
		raise ValueError("Vo line median axis must be 0 or 1")
	kernel = (size, 1) if axis == 0 else (1, size)
	half_width = _bottleneck_line_half_width(size)
	if (
		not use_bottleneck
		or not isinstance(size, Integral) or size < 1 or size % 2 == 0
		or values.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
		or not values.dtype.isnative
		or not np.isfinite(values).all()
	):
		return medfilt2d(values, kernel)

	if pad_work is None:
		if half_width:
			padding = [(0, 0), (0, 0)]
			padding[axis] = (half_width, half_width)
			padded = np.pad(values, padding, mode="constant")
		else:
			padded = values
	else:
		if pad_work.dtype != values.dtype:
			raise TypeError("Vo median-pad work dtype does not match sinogram")
		other_axis = 1 - axis
		if pad_work.shape[other_axis] != values.shape[other_axis]:
			raise ValueError("Vo median-pad work does not match the unfiltered axis")
		extra = pad_work.shape[axis] - values.shape[axis]
		if extra < 2 * half_width or extra % 2:
			raise ValueError("Vo median-pad work cannot provide centered line padding")
		managed_half_width = extra // 2
		selection = [slice(None), slice(None)]
		selection[axis] = slice(
			managed_half_width - half_width,
			managed_half_width + values.shape[axis] + half_width,
		)
		padded = pad_work[tuple(selection)]
		interior = [slice(None), slice(None)]
		interior[axis] = slice(half_width, half_width + values.shape[axis])
		padded[tuple(interior)] = values

	filtered = _move_median(
		padded, window=int(size), min_count=int(size), axis=axis,
	)
	crop = [slice(None), slice(None)]
	crop[axis] = slice(int(size) - 1, int(size) - 1 + values.shape[axis])
	return filtered[tuple(crop)]


def _remove_sort_array(
	sino: np.ndarray,
	size: int = 21,
	dim: int = 1,
	*,
	work: np.ndarray | None = None,
	index_work: np.ndarray | None = None,
	median_pad_work: np.ndarray | None = None,
	use_bottleneck: bool = True,
) -> None:
	"""Apply the local Vo sorting algorithm to one ``(Theta, X)`` sinogram."""
	f_shape = (size, 1 if dim == 1 else size)

	# The source kernel reinterprets (rather than stride-transposes) each plane as
	# (X, Theta), keeping the sort axis contiguous.  The same physical plane is
	# viewed as (Theta, X) by the dead/large kernels later in a fused chain.
	sino_mod = _work_plane(work, sino, "Vo floating-point work").reshape(
		sino.shape[1], sino.shape[0],
	)
	sort_idx = _index_plane(index_work, sino).reshape(sino.shape[1], sino.shape[0])
	sino_mod[:, :] = sino.T
	sort_idx[:, :] = sino_mod.argsort(axis=1) 	# Track index separately to sort back.
	sino_mod[:, :] = np.take_along_axis(sino_mod, sort_idx, axis=1)
	# Invert each permutation directly instead of running a second argsort.  The
	# advanced-index arrays are resolved before assignment, so this safely
	# converts the plane in place to the sort-back index.
	sort_idx[np.arange(sort_idx.shape[0])[:, None], sort_idx] = np.arange(
		sort_idx.shape[1], dtype=np.uint32,
	)

	# Apply filter to sorted memory.
	if dim == 1:
		pad = None if median_pad_work is None else median_pad_work.T
		sino_mod[:, :] = _line_median(
			sino_mod, size, axis=0, pad_work=pad,
			use_bottleneck=use_bottleneck,
		)
	else:
		# The square dim=2 kernel is intentionally retained on SciPy.
		sino_mod[:, :] = medfilt2d(sino_mod, f_shape)

	# sort back and assign.
	sino_mod[:, :] = np.take_along_axis(sino_mod, sort_idx, axis=1)
	sino[:, :] = sino_mod.T


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


def remove_large_vo(sino_mem: SharedNPArray, i: int, snr: int = 3, size: int = 51, drop_ratio: float = 0.1,
					norm: bool = True):
	with sino_mem[i] as sino_base:
		_remove_large_array(sino_base, snr=snr, size=size, drop_ratio=drop_ratio, norm=norm)


def _remove_large_array(
	sino_base: np.ndarray,
	snr: int = 3,
	size: int = 51,
	drop_ratio: float = 0.1,
	norm: bool = True,
	*,
	work: np.ndarray | None = None,
	index_work: np.ndarray | None = None,
	median_pad_work: np.ndarray | None = None,
	use_bottleneck: bool = True,
) -> None:
	"""Apply the local Vo large-stripe algorithm to one sinogram."""
	sino = sino_base
	sino_smooth = _work_plane(work, sino, "Vo floating-point work")
	sort_idx = _index_plane(index_work, sino)

	# Calculate # of outliers to drop.
	drop_ratio = np.clip(drop_ratio, 0.0, 0.8)
	nrow = sino.shape[0]
	ndrop = int(0.5 * drop_ratio * nrow)
	kept = slice(ndrop, -ndrop if ndrop else None)

	# Get Sort Index (so sortback is easy)
	sort_idx[:, :] = sino.argsort(axis=0)

	# Get mean of original and smoothed values after throwing out outliers.
	sino_smooth[:, :] = np.take_along_axis(sino, sort_idx, axis=0)
	sort_mean = np.mean(sino_smooth[kept, :], axis=0)
	sino_smooth[:, :] = _line_median(
		sino_smooth, size, axis=1, pad_work=median_pad_work,
		use_bottleneck=use_bottleneck,
	)
	smooth_mean = np.mean(sino_smooth[kept, :], axis=0)

	# Sort back sino_smooth.
	sort_idx[:, :] = sort_idx.argsort(axis=0)
	sino_smooth[:, :] = np.take_along_axis(sino_smooth[:, :], sort_idx, axis=0)

	# Locate stripes
	factor = np.divide(sort_mean, smooth_mean, out=np.ones_like(sort_mean), where=smooth_mean != 0)
	sino_mask = binary_dilation(_detect_stripe(factor, snr), iterations=1)

	if norm:
		np.divide(sino, factor[None, :], out=sino_base[:, :])
	sino_base[:, sino_mask] = sino_smooth[:, sino_mask]


def remove_dead_vo(dw: TomoWindow, sino_mem: SharedNPArray, snr: int = 3, size: int = 51, norm: float = True,
					proc_count: int = 12):
	with Pool(proc_count) as pool:
		pool.starmap(_remove_dead_vo, [(sino_mem, i, snr, size, norm) for i in dw.internal.int_iter()])


def _remove_dead_vo(sino_mem: SharedNPArray, i: int, snr: int = 3, size: int = 51, norm: float = True):
	with sino_mem[i] as sino_base:
		_remove_dead_array(sino_base, snr=snr, size=size)

	# Remove residual stripes
	if norm is True:
		remove_large_vo(sino_mem, i, snr, size, norm=norm)


def _remove_dead_array(
	sino_base: np.ndarray,
	snr: int = 3,
	size: int = 51,
	*,
	work: np.ndarray | None = None,
) -> None:
	"""Apply the local Vo dead-line interpolation to one sinogram."""
	mask_cut = np.s_[2:-2]
	sino = sino_base
	smoothed = _work_plane(work, sino, "Vo floating-point work")

	uniform_filter1d(sino, 10, axis=0, output=smoothed)
	np.subtract(sino, smoothed, out=smoothed)
	np.abs(smoothed, out=smoothed)
	smooth_diff = np.sum(smoothed, axis=0)
	median_diff = median_filter(smooth_diff, size)
	factor = np.divide(smooth_diff, median_diff, out=np.ones_like(smooth_diff), where=median_diff != 0)

	mask = np.full(factor.shape, False)
	mask[mask_cut] = binary_dilation(_detect_stripe(factor, snr), iterations=1)[mask_cut]
	x_change = mask.nonzero()[0]
	x_keep = np.invert(mask).nonzero()[0]

	if len(x_change) > 0:
		for theta in range(sino.shape[0]):
			sino_base[theta, x_change] = np.interp(x_change, x_keep, sino[theta, x_keep])


def remove_stripe_fw(dw: TomoWindow, sino_mem: SharedNPArray, level: int = None, wname: str = 'db5', sigma: float = 2.0,
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
		level = int(np.ceil(np.log2(np.max(astuple(sino_mem.shape)))))
	with Pool(proc_count) as pool:
		pool.starmap(_remove_stripe_fw, [(sino_mem, i, level, wname, sigma, zero_pad)
											for i in dw.internal.int_iter()])


def _compounding_dwt2(sli, wname, mode, level):
	sli, coeff = pywt.dwt2(sli, wname, mode=mode)
	return _compounding_dwt2(sli, wname, mode, level - 1) + [DetailCoeff(*coeff)] if level else [sli]


def _remove_stripe_fw(sino_mem, i, level, wname, sigma, zero_pad=False):
	with sino_mem[i] as sino:
		_remove_stripe_fw_array(sino, level=level, wname=wname, sigma=sigma, zero_pad=zero_pad)


def _fw_theta_mirror_window(shape: DataShape) -> TomoWindow:
	"""Describe the input followed by its Theta-reversed numerical extension."""
	zero = shape.make_with(0)
	return TomoWindow(
		shape, zero, zero, shape.make_with(1),
		pad_after=zero + Theta(shape.Theta),
	)


def _remove_stripe_fw_array(sino: np.ndarray, level: int, wname: str = "db5",
							sigma: float = 2.0, zero_pad: bool = False,
							theta_mirror: bool = False, *, mirror_work: np.ndarray | None = None) -> None:
	"""Apply Fourier-wavelet stripe suppression to one ``(Theta, X)`` sinogram.

	Mirror extension makes drifting columns continuous at the FFT seam. Sigma
	is measured in FFT bins, so it scales with the extended Theta length. The
	caller may supply managed mirror work; direct array calls allocate locally.
	PyWavelets coefficients/reconstruction and NumPy FFT results remain
	unavoidable external-library temporaries in both modes.
	"""
	if theta_mirror:
		window = _fw_theta_mirror_window(YSlice(*sino.shape))
		canvas = window.with_pad()
		if mirror_work is None:
			mirror_work = np.empty(canvas.shape.as_tuple(), dtype=sino.dtype)
		elif mirror_work.shape != canvas.shape.as_tuple() or mirror_work.dtype != sino.dtype:
			raise ValueError("FW mirror work must match the extended sinogram shape and dtype")
		mirror_work[window.internal.index] = sino
		reflection = canvas.with_crop(Theta(window.shape.Theta), Theta(0))
		mirror_work[reflection.index] = sino[::-1]
		_remove_stripe_fw_array(
			mirror_work, level, wname,
			sigma * canvas.shape.Theta / window.shape.Theta, zero_pad,
		)
		sino[:, :] = mirror_work[window.internal.index]
		return
	extension_mode = "zero" if zero_pad else "symmetric"
	sino_indexing = np.s_[0:sino.shape[0], 0:sino.shape[1]]
	sli = sino[:, :]
	wavelets = []

	for _ in range(level):
		sli, coeff = pywt.dwt2(sli, wname, mode=extension_mode)
		wavelets = [DetailCoeff(*coeff)] + wavelets

	for index, wavelet in enumerate(wavelets):
		vertical = np.fft.fftshift(np.fft.fft(wavelet.cV, axis=0))
		my, mx = vertical.shape
		y_hat = (np.arange(-my, my, 2, dtype="float32") + 1) / 2
		damp = -np.expm1(-np.square(y_hat) / (2 * np.square(sigma)))
		vertical *= np.transpose(np.tile(damp, (mx, 1)))
		wavelets[index] = wavelet._replace(
			cV=np.real(np.fft.ifft(np.fft.ifftshift(vertical), axis=0))
		)

	sino[:, :] = pywt.waverec2([sli] + wavelets, wname, mode=extension_mode)[sino_indexing]
