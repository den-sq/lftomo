"""Numerical regressions for fixed libtomo gridrec (#383 and #384)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


pytestmark = pytest.mark.gridrec_fixed


@pytest.fixture(scope="module")
def tomopy():
	# Collection's gridrec_fixed gate checks the canonical runtime metadata
	# before this fixture imports the numerical library or any test calls it.
	return pytest.importorskip("tomopy")


def _disk_projections(width):
	"""Analytic line integrals through two off-axis absorbing disks."""
	theta = np.linspace(0, np.pi, 13, endpoint=False, dtype=np.float32)
	detector = np.arange(width, dtype=np.float32) - width / 2
	sino = np.zeros((len(theta), width), dtype=np.float32)
	for x, y, radius, attenuation in ((-1, .5, 4, .03), (2, -2, 1.8, .08)):
		distance = detector - (x * np.cos(theta) + y * np.sin(theta))[:, None]
		chord = 2 * np.sqrt(np.maximum(radius**2 - distance**2, 0))
		sino += attenuation * chord
	scale = 1 + np.arange(16, dtype=np.float32)[:, None, None] / 16
	return scale * sino, theta


def _reconstruct(tomopy, data, theta):
	return tomopy.recon(
		data, theta, algorithm="gridrec", sinogram_order=True,
		ncore=1, filter_name="shepp", num_gridx=16, num_gridy=16,
	)


def _edges(images):
	return np.stack((
		images[:, 0, :], images[:, -1, :],
		images[:, :, 0], images[:, :, -1],
	), axis=1)


@pytest.fixture(scope="module", params=(16, 15), ids=("width16", "width15"))
def full_grid(request, tomopy):
	width = request.param
	data, theta = _disk_projections(width)
	full = _reconstruct(tomopy, data, theta)
	return width, data, theta, full


def test_full_grid_edges_match_reference(full_grid):
	width, _, _, full = full_grid
	assert full.shape == (16, 16, 16)
	assert np.isfinite(full).all()
	edges = _edges(full)

	# Preserve den-sq/tomopy's validated fixed-gridrec references and tolerance.
	# These nonzero samples cover all four outer edges and corners, catching a
	# consistently incorrect border that batching comparisons alone accept.
	expected = {
		16: [[-0.014514812, -0.017745465, -0.011935343, -0.020524167],
			 [-0.090851687, -0.013721272, -0.006832186, -0.053797100],
			 [-0.014514812, -0.024034632, -0.011040418, -0.090851687],
			 [-0.020524167, -0.006192388, -0.011042548, -0.053797100]],
		15: [[-0.019327067, -0.012183571, -0.018951282, -0.018941890],
			 [-0.102695331, -0.029816074, -0.016673394, -0.033196878],
			 [-0.019327067, -0.039638739, -0.008431993, -0.102695331],
			 [-0.018941890, -0.000608895, -0.018098855, -0.033196878]],
	}
	scale = 1 + np.arange(16)[:, None, None] / 16
	assert_allclose(
		edges[:, :, ::5], scale * np.array(expected[width]),
		rtol=1e-2, atol=1e-7,
	)


@pytest.mark.parametrize("batch", (5, 6, 16))
def test_full_grid_batching(tomopy, full_grid, batch):
	_, data, theta, full = full_grid
	pieces = [
		_reconstruct(tomopy, data[start:start + batch].copy(), theta)
		for start in range(0, len(data), batch)
	]
	# Odd batches change the native FFT partners. Retain the existing gridrec
	# tolerance for the resulting float roundoff, including every border pixel.
	assert_allclose(np.concatenate(pieces), full, rtol=1e-2, atol=1e-7)


def _blob_integral(image, x, y):
	half_width = 8
	# Recenter on the local peak to allow for the subpixel placement difference
	# between gridrec and FBP, using the validated upstream integration window.
	row = image.shape[0] // 2 - 1 - y
	column = image.shape[1] // 2 + x
	window = image[
		row - half_width:row + half_width + 1,
		column - half_width:column + half_width + 1,
	]
	peak_row, peak_column = np.unravel_index(np.argmax(window), window.shape)
	row += peak_row - half_width
	column += peak_column - half_width
	return image[
		row - half_width:row + half_width + 1,
		column - half_width:column + half_width + 1,
	].sum()


@pytest.mark.parametrize("width", (256, 255))
def test_spatial_gain_matches_fbp(tomopy, width):
	# Both widths use a 256-pixel FFT grid. Analytic Gaussian projections avoid
	# adding an interpolation-dependent projector to the scientific reference.
	theta = np.linspace(0, np.pi, 450, endpoint=False, dtype=np.float32)
	detector = np.arange(width)
	sigma = 1.5
	positions = [(0, 0), (0, 0)]
	for fraction in (0.5, 0.85):
		offset = round(fraction * 128)
		for position in ((offset, 0), (-offset, 0), (0, offset), (0, -offset)):
			# Each position occupies both native real and imaginary FFT channels.
			positions.extend((position, position))

	projections = []
	for x, y in positions:
		projected_center = x * np.cos(theta) + y * np.sin(theta) + width / 2
		projections.append(
			np.exp(-0.5 * ((detector[None, :] -
				projected_center[:, None]) / sigma)**2) /
			(np.sqrt(2 * np.pi) * sigma),
		)
	projections = np.stack(projections, axis=1).astype(np.float32)

	normalized = {}
	for algorithm in ("gridrec", "fbp"):
		# A single native call preserves each pair of real/imaginary slices.
		result = tomopy.recon(
			projections, theta, center=width / 2, algorithm=algorithm,
			filter_name="shepp", ncore=1,
		)
		integrals = np.array([
			_blob_integral(image, x, y)
			for image, (x, y) in zip(result, positions)
		]).reshape(-1, 2)
		# Normalize each FFT channel to its own central reference; the tested
		# quantity is spatial gain, independent of absolute normalization.
		normalized[algorithm] = (integrals / integrals[0]).ravel()

	for index, (x, y) in enumerate(positions[2:], start=2):
		assert_allclose(
			normalized["gridrec"][index], normalized["fbp"][index],
			rtol=0.01, atol=0,
			err_msg=f"width={width}, position=({x}, {y}), channel={index % 2}",
		)
