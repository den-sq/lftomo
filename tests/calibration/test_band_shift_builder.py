"""Recover a measured schedule from physically displaced beam texture."""

import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from lftomo.calibration import (
	BeamBandShiftModel, CalibrationBuildSession, CalibrationBundle,
	HDF5RandomFrameSource, build_beam_band_shift_schedule,
)
from lftomo.util.shapes import ProjOrder, Y
from lftomo.util.window import CTFov, TomoDataSet


def _beam(y, x):
	return 7 + 0.12 * np.sin(y / 5.3) + 0.09 * np.cos(y / 9.7) + 0.04 * np.sin(y / 2.8) + 0.03 * np.cos(x / 11)


@pytest.fixture
def measured_scan(tmp_path):
	count, rows, columns = 41, 384, 32
	y, x = np.indices((rows, columns), dtype=np.float64)
	yc = 192.0
	dy = 0.8 + 0.9 * np.sin(np.arange(count) * 1.7) + np.linspace(0, 1, count)
	stretch = 1.003 + 0.001 * np.cos(np.arange(count) / 13)
	dark = (10 + 0.02 * y + 0.03 * x).astype(np.float32)
	white = (dark + np.exp(_beam(y, x))).astype(np.float32)
	path = tmp_path / "measured.h5"
	with h5py.File(path, "w") as handle:
		data = handle.create_dataset("exchange/data", shape=(count, rows, columns), dtype="float32")
		for index in range(count):
			source_y = yc + (y - yc - dy[index]) / stretch[index]
			# A slowly varying object attenuates the raw beam. High-pass profile
			# registration separates its broad Y scale from narrow beam bands.
			object_log = -0.15 * np.exp(-((y - 180) / 180) ** 2) * np.exp(-((x - 15) / 12) ** 2)
			data[index] = dark + np.exp(_beam(source_y, x) + object_log)
	source = HDF5RandomFrameSource(path, "exchange/data")
	indices = np.arange(count) + np.arange(count) // 7
	dataset = TomoDataSet.from_range(
		CTFov(count, 180, 1), ProjOrder(count, rows, columns), Y(rows),
		dtype=np.float32, angles=indices * (np.pi / (indices[-1] + 1)),
		acquisition_indices=indices,
	)
	build = CalibrationBuildSession(
		tmp_path / "bundle", calibration_id="measured-bands",
		producer={"package": "lftomo", "version": "test"},
		compatibility={"detector_id": "camera", "region": {"height": rows, "width": columns}},
	)
	build.add_source(path, metadata={"dataset": "exchange/data"})
	for name, values, kind in (("pre_gain", white, "detector_static"), ("pre_dark", dark, "dark")):
		build.write_array_component(
			name, (values,), shape=values.shape, dtype=values.dtype,
			axes=("Y", "X"), kind=kind, coordinate_frame="detector",
		)
	params = {
		"model": BeamBandShiftModel(yc=yc),
		"row_windows": [(30, 70), (85, 125), (140, 180), (195, 235), (250, 290), (305, 345)],
		"highpass_sigma": 12, "search_shifts": np.linspace(-4, 5, 46), "y_chunk": 31,
	}
	return build, source, dataset, params, dy, stretch


def test_schedule_recovers_jitter_and_smoothed_stretch_from_raw_scan(measured_scan):
	build, source, dataset, params, dy, stretch = measured_scan
	component = build_beam_band_shift_schedule(build, source, dataset, **params)
	ref = build.finalize()
	bundle = CalibrationBundle.open(ref.manifest_path, verify_hashes=True)
	actual = np.array([
		bundle.read_component("beam_schedule", projection_index=int(index))
		for index in dataset.acquisition_indices
	])
	# Resolution follows the requested 0.2-pixel search spacing and 275-pixel
	# separation between the first/last fitting-window centres.
	np.testing.assert_allclose(actual[:, 0], dy, atol=0.2, rtol=0)
	np.testing.assert_allclose(actual[:, 1], gaussian_filter1d(stretch, 25, mode="reflect"), atol=0.2 / 275, rtol=0)
	assert np.std(np.diff(actual[:, 0])) > 0.9 * np.std(np.diff(dy))
	assert component.band_shift == params["model"]
	assert component.time_axis.values == dataset.acquisition_indices.tolist()
	qc = component.validation_results[0].metrics
	assert qc["window_residual_rms_px"] < 0.2
	assert qc["frame_to_frame_dy_rms_px"] == pytest.approx(float(np.sqrt(np.mean(np.diff(dy) ** 2))), abs=0.2)
	assert bundle.component("beam_schedule").metadata["angles_radians"] == dataset.angles.tolist()


@pytest.mark.system
def test_schedule_is_independent_of_source_read_chunking(measured_scan):
	build, source, dataset, params, _dy, _stretch = measured_scan
	first = build_beam_band_shift_schedule(build, source, dataset, **params)
	params["y_chunk"] = 57
	second = build_beam_band_shift_schedule(build, source, dataset, name="second", **params)
	assert first.validation_results == second.validation_results
	for index in dataset.acquisition_indices:
		np.testing.assert_array_equal(
			build.read_component("beam_schedule", projection_index=int(index)),
			build.read_component("second", projection_index=int(index)),
		)
	build.finalize()


def test_inadequate_shift_search_does_not_publish_a_saturated_schedule(measured_scan):
	build, source, dataset, params, _dy, _stretch = measured_scan
	params["search_shifts"] = np.linspace(-0.1, 0.1, 5)
	with pytest.raises(ValueError, match="shift-search boundary"):
		build_beam_band_shift_schedule(build, source, dataset, **params)
	assert "beam_schedule" not in build.components


@pytest.mark.parametrize("bundle_x,component_x", [(17, None), (0, 17), (17, 0)])
def test_schedule_requires_effective_full_detector_column_coverage(measured_scan, bundle_x, component_x):
	build, source, dataset, params, _dy, _stretch = measured_scan
	build.compatibility.region.x_start = bundle_x
	if component_x is not None:
		for name in ("pre_gain", "pre_dark"):
			build.components[name].compatibility.region = build.compatibility.region.model_copy(
				update={"x_start": component_x},
			)
	if component_x == 0:
		# An explicit full-detector component overrides the bundle's default ROI.
		component = build_beam_band_shift_schedule(build, source, dataset, **params)
		assert component.payload.shape == (source.frame_count, 2)
		build.finalize()
	else:
		with pytest.raises(ValueError, match="must cover detector columns"):
			build_beam_band_shift_schedule(build, source, dataset, **params)
		assert "beam_schedule" not in build.components
