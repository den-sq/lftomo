"""Per-slice material primitives for propagation phase correction.

Ported from mctutil.phase.phase_tomo. Lengths are metres, attenuation is
inverse metres, and projection angles are degrees. The working Ram-Lak FBP
shares the forward projector's geometry; it is not the pipeline recon stage.
These pure functions return NumPy arrays and use NumPy/SciPy temporaries.
"""

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

from lftomo.prep import phase
from lftomo.util.shapes import ImgSlice
from lftomo.util.window import TomoWindow


def fit_material_line(mu, delta, weights=None, through_origin=False):
	"""Fit delta = alpha + slope*mu and return ordered residuals and weighted RMS."""
	mu = np.asarray(mu, float)
	delta = np.asarray(delta, float)
	w = np.ones_like(mu) if weights is None else np.asarray(weights, float)
	if through_origin:
		slope = np.sum(w * mu * delta) / np.sum(w * mu**2)
		alpha = 0.0
	else:
		weight = np.sum(w)
		mu_mean = np.sum(w * mu) / weight
		delta_mean = np.sum(w * delta) / weight
		slope = np.sum(w * (mu - mu_mean) * (delta - delta_mean)) / np.sum(w * (mu - mu_mean)**2)
		alpha = delta_mean - slope * mu_mean
	eps = delta - (alpha + slope * mu)
	return dict(alpha=alpha, slope=slope, eps=eps, rms_eps=float(np.sqrt(np.sum(w * eps**2) / np.sum(w))))


def forward_project(image, angles_deg, pixel_m):
	"""Project a square slice on the working FBP grid, discarding off-grid mass."""
	image = np.asarray(image, float)
	if image.ndim != 2 or image.shape[0] != image.shape[1]:
		raise ValueError("forward_project expects a square 2D slice")
	n = image.shape[0]
	xs = (np.arange(n) - n / 2 + 0.5) * pixel_m
	x_grid, y_grid = np.meshgrid(xs, xs)
	flat = image.ravel()
	sinogram = np.zeros((len(angles_deg), n))
	for a, theta in enumerate(np.deg2rad(angles_deg)):
		q = ((x_grid * np.cos(theta) + y_grid * np.sin(theta)).ravel() - xs[0]) / pixel_m
		index = np.floor(q).astype(int)
		weight = q - index
		acc = np.zeros(n)
		left = (index >= 0) & (index < n)
		right = (index >= -1) & (index < n - 1)
		np.add.at(acc, index[left], (flat * (1 - weight))[left])
		np.add.at(acc, index[right] + 1, (flat * weight)[right])
		sinogram[a] = acc * pixel_m
	return sinogram


def fbp(sinogram, angles_deg, pixel_m):
	"""Working parallel-beam Ram-Lak FBP with the reference's equal angle weights.

	Input is (n_angles, nx); output is a square (nx, nx) integrand map. The
	pi/n_angles quadrature assumes uniform coverage of a half turn, as in the
	reference correction loop. It does not choose the pipeline recon engine.
	"""
	sinogram = np.asarray(sinogram, float)
	if sinogram.ndim != 2 or sinogram.shape[0] != len(angles_deg):
		raise ValueError("sinogram must be (n_ang, nx) with n_ang matching angles_deg")
	window = TomoWindow(
		ImgSlice(*sinogram.shape), ImgSlice(0, 0), ImgSlice(0, 0), ImgSlice(1, 1),
		dtype=sinogram.dtype,
		pad_after=ImgSlice(0, sfft.next_fast_len(2 * sinogram.shape[1]) - sinogram.shape[1]),
	)
	frequency = np.abs(sfft.fftfreq(window.with_pad().shape.X, pixel_m))
	padded = np.pad(sinogram, tuple(zip(window.pad_before.as_tuple(), window.pad_after.as_tuple())))
	filtered = sfft.ifft(sfft.fft(padded, axis=1) * frequency, axis=1).real[window.internal.index]
	xs = (np.arange(window.shape.X) - window.shape.X / 2 + 0.5) * pixel_m
	x_grid, y_grid = np.meshgrid(xs, xs)
	reconstruction = np.zeros((window.shape.X, window.shape.X))
	for a, theta in enumerate(np.deg2rad(angles_deg)):
		u = x_grid * np.cos(theta) + y_grid * np.sin(theta)
		reconstruction += np.interp(u, xs, filtered[a], left=0, right=0)
	return reconstruction * np.pi / window.shape.Y


def classify_materials(mu_map, mu_table):
	"""Nearest-material labels: 0 is air and j corresponds to mu_table[j-1]."""
	centroids = np.concatenate([[0.0], np.asarray(mu_table, float)])
	return np.argmin(np.abs(mu_map[..., None] - centroids), axis=-1)


def segment_materials(mu_map, mu_table, sigma=1.5, core_frac=0.3, core_erode=2):
	"""Propagate material identity from confident cores by gradient watershed.

	Core erosion relaxes until each class has a seed. A missing class, including
	air, raises an error naming its index rather than silently deleting it.
	"""
	mu_table = np.asarray(mu_table, float)
	if not np.all(np.isfinite(mu_table)) or np.any(mu_table <= 0):
		raise ValueError("mu_table must be finite and positive")
	if len(np.unique(mu_table)) != len(mu_table):
		raise ValueError("mu_table entries must be distinct")
	mu_lowpass = ndimage.gaussian_filter(mu_map, sigma)
	centroids = np.concatenate([[0.0], mu_table])
	band = core_frac * np.diff(np.sort(centroids)).min()
	markers = np.zeros(mu_map.shape, np.int32)
	for j, centroid in enumerate(centroids):
		core_uneroded = np.abs(mu_lowpass - centroid) < band
		core = core_uneroded
		for erosion in range(core_erode, -1, -1):
			core = ndimage.binary_erosion(core_uneroded, iterations=erosion) if erosion else core_uneroded
			if core.any():
				break
		if not core.any():
			raise ValueError(
				f"segment_materials: no core pixels for class {j} (centroid {centroid:.4g}); "
				"check mu_table, sigma, or region sizes"
			)
		markers[core] = j + 1
	gradient = ndimage.gaussian_gradient_magnitude(mu_lowpass, sigma)
	gradient_u8 = (255 * gradient / max(gradient.max(), 1e-30)).astype(np.uint8)
	labels = ndimage.watershed_ift(gradient_u8, markers) - 1
	return np.clip(labels, 0, len(mu_table))


def predict_filter_error(
	psi, I0, energy_kev, pixel_m, distance_m, slope, pad=True, workers=-1, *, denominator=None,
	window=None, psi_work=None, intensity_work=None,
):
	"""Return the backward-error dB to subtract from slope-form retrieval.

	dB = (lambda*z/2pi) / I0 * L_s^-1[div(I0*grad(psi))]. Inputs use
	(..., ny, nx); a two-dimensional psi is promoted to (1, ny, nx).
	Padding and FFT coordinates share the Paganin geometry implementation.
	Composite callers can supply their existing shared ``denominator``; plain
	library calls resolve it from the same canonical slope operator.
	An optional canonical ``window`` takes precedence over ``pad``. Work arrays
	match the input rank and padded extents; ``psi_work`` uses float64 and
	``intensity_work`` preserves the intensity dtype. Their contents are replaced.
	"""
	psi = np.asarray(psi, float)
	if psi.ndim == 2:
		psi = psi[None]
		psi_work = None if psi_work is None else psi_work[None]
		intensity_work = None if intensity_work is None else intensity_work[None]
	I0 = np.broadcast_to(I0, psi.shape)
	window = phase.phase_fft_window(psi.shape[-2:], pad) if window is None else window
	padded_psi = phase.phase_pad(psi, window, out=psi_work)
	padded_intensity = phase.phase_pad(I0, window, out=intensity_work)
	gradient_x = np.gradient(padded_psi, pixel_m, axis=-1)
	divergence = np.gradient(padded_intensity * gradient_x, pixel_m, axis=-1)
	if window.with_pad().shape.Y >= 2:
		gradient_y = np.gradient(padded_psi, pixel_m, axis=-2)
		divergence += np.gradient(padded_intensity * gradient_y, pixel_m, axis=-2)
	if denominator is None:
		denominator = phase.slope_denominator(window, pixel_m, distance_m, slope)
	filtered = sfft.irfft2(
		sfft.rfft2(divergence, workers=workers) / denominator,
		s=window.with_pad().shape.as_tuple(), workers=workers,
	)
	return (phase.wavelength_m(energy_kev) * distance_m / (2 * np.pi)) * phase.phase_unpad(filtered, window) / I0
