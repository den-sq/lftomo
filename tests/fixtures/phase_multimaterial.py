"""Exact Fresnel phantoms from the pinned mctutil phase reference.

Source commit: fbe3251c70de45180462ad24e556b7e1a86483ce.
The original case preserves validate_multimat_single.py; the compact baseline
preserves test_phase_tomo_unit.py's single-row multimaterial case. Optional
non-collinearity and row displacement are explicitly simulated variants.
"""

import numpy as np


def original_multimaterial_case():
	"""Return the original 181-angle, 64-row, 512-column three-material case."""
	energy_kev, pixel_m, distance_m = 20.0, 0.8e-6, 0.05
	wavelength = 1.23984193e-9 / energy_kev
	nx, ny, n_angles = 512, 64, 181
	angles = np.linspace(0, 180, n_angles, endpoint=False)
	mu = np.array([60.8, 100.0, 140.0])
	delta = np.array([9.12e-8, 1.50e-7, 1.20e-7])
	xs = (np.arange(nx) - nx / 2 + 0.5) * pixel_m
	fy = np.fft.fftfreq(ny, pixel_m)[:, None]
	fx = np.fft.fftfreq(nx, pixel_m)[None, :]
	propagator = np.exp(-1j * np.pi * wavelength * distance_m * (fx**2 + fy**2))
	projections = np.empty((n_angles, ny, nx), np.float32)
	thicknesses = np.empty((3, n_angles, nx))

	def chord(position, radius):
		return 2 * np.sqrt(np.clip(radius**2 - position**2, 0, None))

	for a, theta in enumerate(np.deg2rad(angles)):
		t2 = chord(xs - (-32e-6 * np.cos(theta) + 8e-6 * np.sin(theta)), 30e-6)
		t3 = chord(xs - (30e-6 * np.cos(theta) - 14e-6 * np.sin(theta)), 26e-6)
		t1 = chord(xs, 80e-6) - t2 - t3
		thicknesses[:, a] = (t1, t2, t3)
		absorption = mu @ thicknesses[:, a]
		phase = -(2 * np.pi / wavelength) * (delta @ thicknesses[:, a])
		wave = np.broadcast_to(np.exp(-absorption / 2 + 1j * phase), (ny, nx))
		projections[a] = np.abs(np.fft.ifft2(np.fft.fft2(wave) * propagator))**2
	return {
		"projections": projections,
		"energy_kev": energy_kev,
		"pixel_m": pixel_m,
		"distance_m": distance_m,
		"angles_deg": angles,
		"mu_table": mu,
		"delta_table": delta,
		"rows": [ny // 2],
	}


def compact_multimaterial_case(n_rows=1, row_varying=False, noncollinear=False):
	"""Return the 24-angle, 96-column unit phantom or a declared Fresnel variant.

	The baseline has delta = 2e-9*mu and one row. ``noncollinear`` reduces the
	second material's delta from 2.4e-7 to 1.8e-7, producing nonzero correction.
	``row_varying`` shifts its inclusion X center by -2 to +2 micrometres across
	rows before the full two-dimensional Fresnel propagation.
	"""
	energy_kev, pixel_m, distance_m = 20.0, 1e-6, 0.01
	wavelength = 1.23984193e-9 / energy_kev
	nx, n_angles = 96, 24
	angles = np.linspace(0, 180, n_angles, endpoint=False)
	mu = np.array([50.0, 120.0])
	delta = 2e-9 * mu
	if noncollinear:
		delta[1] = 1.8e-7
	xs = (np.arange(nx) - nx / 2 + 0.5) * pixel_m
	fy = np.fft.fftfreq(n_rows, pixel_m)[:, None]
	fx = np.fft.fftfreq(nx, pixel_m)[None, :]
	propagator = np.exp(-1j * np.pi * wavelength * distance_m * (fx**2 + fy**2))
	projections = np.empty((n_angles, n_rows, nx), np.float32)
	centers = 8e-6 + np.linspace(-2e-6, 2e-6, n_rows) if row_varying else 8e-6

	def chord(position, radius):
		return 2 * np.sqrt(np.clip(radius**2 - position**2, 0, None))

	for a, theta in enumerate(np.deg2rad(angles)):
		if row_varying:
			position = xs[None, :] - (centers[:, None] * np.cos(theta) - 5e-6 * np.sin(theta))
		else:
			position = xs - (centers * np.cos(theta) - 5e-6 * np.sin(theta))
		t2 = chord(position, 10e-6)
		t1 = chord(xs, 30e-6) - t2
		absorption = mu[0] * t1 + mu[1] * t2
		phase = -(2 * np.pi / wavelength) * (delta[0] * t1 + delta[1] * t2)
		wave = np.broadcast_to(np.exp(-absorption / 2 + 1j * phase), (n_rows, nx))
		projections[a] = np.abs(np.fft.ifft2(np.fft.fft2(wave) * propagator))**2
	return {
		"projections": projections,
		"energy_kev": energy_kev,
		"pixel_m": pixel_m,
		"distance_m": distance_m,
		"angles_deg": angles,
		"mu_table": mu,
		"delta_table": delta,
		"rows": list(range(n_rows)),
	}
