import json
from pathlib import Path

import numpy as np
import pytest

from lftomo.prep import materials, phase


@pytest.fixture(scope="module")
def reference():
	path = Path(__file__).parents[1] / "fixtures" / "phase_materials_reference.npz"
	with np.load(path) as arrays:
		return dict(arrays), json.loads(path.with_suffix(".json").read_text())


@pytest.mark.parametrize("through_origin", [False, True])
def test_material_line_preserves_weights_and_order(through_origin):
	# The non-collinear material set from the reference multimaterial phantom.
	mu = np.array([60.8, 100.0, 140.0])
	delta = np.array([9.12e-8, 1.50e-7, 1.20e-7])
	weights = np.array([1.0, 2.0, 3.0])
	design = mu[:, None] if through_origin else np.column_stack((np.ones(3), mu))
	coefficients = np.linalg.lstsq(
		design * np.sqrt(weights[:, None]), delta * np.sqrt(weights), rcond=None,
	)[0]

	fit = materials.fit_material_line(mu, delta, weights, through_origin=through_origin)

	assert fit["alpha"] == 0 if through_origin else fit["alpha"] != 0
	np.testing.assert_allclose(fit["slope"], coefficients[-1], rtol=1e-12)
	if not through_origin:
		np.testing.assert_allclose(fit["alpha"], coefficients[0], rtol=1e-12)
	expected_residuals = delta - design @ coefficients
	np.testing.assert_allclose(fit["eps"], expected_residuals, rtol=1e-12, atol=1e-22)
	np.testing.assert_allclose(fit["rms_eps"], np.sqrt(np.average(expected_residuals**2, weights=weights)), rtol=1e-12)
	assert fit["slope"] != materials.fit_material_line(mu, delta, through_origin=through_origin)["slope"]


def test_projector_discards_off_detector_mass_and_conserves_inside_mass():
	corner = np.zeros((8, 8))
	corner[7, 7] = 1
	assert materials.forward_project(corner, [45.0], 1e-6).sum() == 0
	disk = np.hypot(*np.meshgrid(np.arange(24) - 11.5, np.arange(24) - 11.5)) < 8
	sinogram = materials.forward_project(disk.astype(float), [0.0, 30.0, 77.0], 1e-6)
	np.testing.assert_allclose(sinogram.sum(1), disk.sum() * 1e-6, rtol=1e-12)
	with pytest.raises(ValueError, match="square 2D slice"):
		materials.forward_project(np.ones((8, 7)), [0.0], 1e-6)


def test_projector_matches_its_linear_interpolation_adjoint():
	rng = np.random.default_rng(0)
	image = rng.normal(size=(16, 16))
	data = rng.normal(size=(5, 16))
	angles = [0.0, 22.0, 45.0, 90.0, 133.0]
	pixel_m = 1e-6
	projected = materials.forward_project(image, angles, pixel_m)
	xs = (np.arange(16) - 8 + 0.5) * pixel_m
	x_grid, y_grid = np.meshgrid(xs, xs)
	adjoint = np.zeros(16 * 16)
	for a, theta in enumerate(np.deg2rad(angles)):
		q = ((x_grid * np.cos(theta) + y_grid * np.sin(theta)).ravel() - xs[0]) / pixel_m
		index = np.floor(q).astype(int)
		weight = q - index
		left = (index >= 0) & (index < 16)
		right = (index >= -1) & (index < 15)
		adjoint[left] += data[a][index[left]] * (1 - weight[left])
		adjoint[right] += data[a][index[right] + 1] * weight[right]
	lhs = np.sum(projected * data)
	rhs = np.sum(image * adjoint.reshape(16, 16)) * pixel_m
	assert abs(lhs - rhs) < 1e-9 * abs(lhs)


def test_working_projection_and_fbp_match_reference_with_canonical_padding(reference, monkeypatch):
	arrays, metadata = reference
	windows = []
	window_type = materials.TomoWindow

	def record_window(*args, **kwargs):
		window = window_type(*args, **kwargs)
		windows.append(window)
		return window

	monkeypatch.setattr(materials, "TomoWindow", record_window)
	sinogram = materials.forward_project(arrays["image"], arrays["angles_deg"], metadata["pixel_m"])
	reconstruction = materials.fbp(sinogram, arrays["angles_deg"], metadata["pixel_m"])

	np.testing.assert_array_equal(sinogram, arrays["sinogram"])
	np.testing.assert_array_equal(reconstruction, arrays["reconstruction"])
	assert len(windows) == 1
	window = windows[0]
	assert window.shape.as_tuple() == sinogram.shape
	assert window.pad_before.as_tuple() == (0, 0)
	assert window.pad_after.as_tuple() == (0, 48)
	assert window.with_pad().shape.as_tuple() == (45, 96)
	assert window.internal.index == (slice(0, 45, 1), slice(0, 48, 1))
	with pytest.raises(ValueError, match="n_ang matching angles_deg"):
		materials.fbp(np.ones((5, 32)), [0.0, 90.0], metadata["pixel_m"])


def test_nearest_labels_keep_material_order_and_air_ties():
	labels = materials.classify_materials(np.array([[0.0, 25.0, 50.0, 85.0, 120.0]]), [120.0, 50.0])
	np.testing.assert_array_equal(labels, [[0, 0, 2, 1, 1]])


def test_core_identity_watershed_matches_reference(reference):
	arrays, _ = reference
	labels = materials.segment_materials(arrays["image"], arrays["mu_table"])
	np.testing.assert_array_equal(labels, arrays["labels"])
	assert set(np.unique(labels)) == {0, 1, 2}


def test_core_erosion_relaxes_without_deleting_small_material():
	image = np.zeros((24, 24))
	image[8:11, 8:11] = 50
	labels = materials.segment_materials(image, [50.0], sigma=0)
	assert labels[9, 9] == 1
	assert labels[0, 0] == 0


@pytest.mark.parametrize("table, message", [
	([50.0, 120.0], r"no core pixels for class 2 \(centroid 120\)"),
	([50.0, 50.0], "entries must be distinct"),
	([0.0, 50.0], "finite and positive"),
	([np.nan, 50.0], "finite and positive"),
])
def test_segmentation_reports_unusable_material_classes(table, message):
	image = np.zeros((64, 64))
	image[10:30, 10:30] = 50
	with pytest.raises(ValueError, match=message):
		materials.segment_materials(image, table)


@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("single_row", [False, True])
def test_backward_error_matches_physical_reference(reference, pad, single_row):
	arrays, metadata = reference
	selection = np.s_[:, 1:2] if single_row else np.s_[:]
	result = materials.predict_filter_error(
		arrays["psi"][selection], arrays["intensity"][selection],
		metadata["energy_kev"], metadata["pixel_m"], metadata["distance_m"], metadata["slope"],
		pad=pad, workers=1,
	)
	key = ("single_row_" if single_row else "") + "error_" + ("padded" if pad else "unpad")
	np.testing.assert_allclose(result, arrays[key], rtol=1e-12, atol=1e-16)
	assert result.shape == arrays["psi"][selection].shape


def test_backward_error_uses_the_shared_window_and_paganin_kernel(reference, monkeypatch):
	arrays, metadata = reference
	windows = []
	calls = []
	make_window = phase.phase_fft_window

	def record_window(*args, **kwargs):
		window = make_window(*args, **kwargs)
		windows.append(window)
		return window

	def record_consumer(name, function):
		def record(*args, **kwargs):
			calls.append((name, args))
			return function(*args, **kwargs)
		return record

	monkeypatch.setattr(phase, "phase_fft_window", record_window)
	for name in ("phase_pad", "slope_denominator", "phase_unpad"):
		monkeypatch.setattr(phase, name, record_consumer(name, getattr(phase, name)))
	result = materials.predict_filter_error(
		arrays["psi"][0], arrays["intensity"][0], metadata["energy_kev"],
		metadata["pixel_m"], metadata["distance_m"], metadata["slope"], workers=1,
	)

	assert len(windows) == 1
	assert [name for name, _ in calls] == ["phase_pad", "phase_pad", "slope_denominator", "phase_unpad"]
	assert all(args[0] is windows[0] if name == "slope_denominator" else args[1] is windows[0] for name, args in calls)
	assert windows[0].shape.as_tuple() == (3, 48)
	assert windows[0].with_pad().shape.as_tuple() == (4, 72)
	assert result.shape == (1, 3, 48)
	np.testing.assert_allclose(result, arrays["error_padded"][:1], rtol=1e-12, atol=1e-16)
