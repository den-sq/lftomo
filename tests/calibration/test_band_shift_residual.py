"""Recover endpoint beam change independently of mirrored object attenuation."""

from dataclasses import replace

import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from lftomo.calibration import (
	BeamBandShiftModel, CalibrationBuildSession, CalibrationBundle,
	HDF5RandomFrameSource, build_beam_band_shift_residual,
)
from lftomo.pipeline.stages.center import ManualCenterStage
from lftomo.util.shapes import ProjOrder, Y
from lftomo.util.window import CTFov, TomoDataSet


def _beam(y):
	return 7 + 0.12 * np.sin(y / 5.3) + 0.09 * np.cos(y / 9.7) + 0.04 * np.sin(y / 2.8)


def _object(y, x):
	return -0.4 + (0.16 * np.sin(x / 3.7) + 0.07 * np.cos(x / 1.5)) * (0.8 + y / 1920)


@pytest.fixture
def endpoint_scan(tmp_path, request):
	count, rows, columns = 12, 384, 64
	y, x = np.indices((rows, columns), dtype=np.float64)
	yc, center, dy, stretch = 192.0, 31.5, 6.75, 0.9915
	tilt = getattr(request, "param", 0.0)
	dark = (10 + 0.02 * y + 0.03 * x).astype(np.float32)
	white = (dark + np.exp(_beam(y))).astype(np.float32)
	# This broad change is deliberately outside the high-pass fit's signal.
	# It must remain in the published residual, which contains all Y scales.
	residual = 0.025 + 0.00003 * y + 0.006 * np.cos(x / 23)
	start = dark + np.exp(_beam(y) + _object(y, x))
	end = dark + np.exp(
		_beam(yc + (y - yc - dy) / stretch) + residual
		+ _object(y, 2 * (center + tilt * (y - yc)) - x),
	)
	path = tmp_path / "endpoints.h5"
	with h5py.File(path, "w") as handle:
		data = handle.create_dataset("exchange/data", shape=(count, rows, columns), dtype="float32")
		for index in range(count):
			data[index] = start if index < count // 2 else end
	source = HDF5RandomFrameSource(path, "exchange/data")
	dataset = TomoDataSet.from_range(
		CTFov(count, 180, 1), ProjOrder(count, rows, columns), Y(rows),
		dtype=np.float32, angles=np.linspace(0, np.pi, count),
		acquisition_indices=3 + 2 * np.arange(count),
	)
	build = CalibrationBuildSession(
		tmp_path / "bundle", calibration_id="measured-endpoint-residual",
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
		"model": BeamBandShiftModel(yc=yc, sigma_x=3),
		"center": ManualCenterStage.Params(value_px=32.0, tilt=tilt, reference_y=yc),
		"fit_rows": (40, 344), "fit_columns": (8, 56), "pairs": 4,
		"scan_center": (1.0, 0.5), "highpass_sigma": 12, "bands": 4, "y_chunk": 31,
	}
	return build, source, dataset, params, residual, center, dy, stretch


def _qc(component, name):
	return next(result for result in component.validation_results if result.name == name)


@pytest.mark.parametrize("endpoint_scan", [0.0, 0.004], indirect=True)
def test_residual_recovers_independent_beam_change_and_preserves_broad_signal(endpoint_scan):
	build, source, dataset, params, residual, center, dy, stretch = endpoint_scan
	result = build_beam_band_shift_residual(build, source, dataset, **params)
	ref = build.finalize()
	bundle = CalibrationBundle.open(ref.manifest_path, verify_hashes=True)
	actual = bundle.read_component("beam_residual")
	assert actual.dtype == np.float32
	assert actual.shape == residual.shape
	fit = _qc(result.component, "band_shift_residual_fit").metrics
	# The estimator searches at 0.25-pixel and 0.0005-stretch resolution.
	assert fit["dy"] == pytest.approx(dy, abs=0.25)
	assert fit["m"] == pytest.approx(stretch, abs=0.0005)
	assert result.model.dy_end == fit["dy"]
	assert result.model.residual_component == "beam_residual"
	assert params["model"].residual_component is None
	rows = slice(*params["fit_rows"])
	expected = gaussian_filter1d(residual, params["model"].sigma_x, axis=1, mode="nearest")
	# Fit quantization and interpolation contribute at most a small fraction
	# of the broad residual, so losing that residual cannot pass this check.
	np.testing.assert_allclose(actual[rows].mean(axis=1), expected[rows].mean(axis=1), atol=0.004, rtol=0)
	assert actual[rows].mean() > 0.02
	center_qc = _qc(result.component, "band_shift_residual_center_scan")
	metrics = center_qc.metrics
	assert center_qc.status == "pass"
	assert metrics["center_used"] == pytest.approx(center, abs=0.1)
	scores = {float(candidate): score for candidate, score in metrics["center_scan"].items()}
	assert scores[metrics["center_used"]] == min(scores.values())
	assert scores[metrics["center_used"]] < scores[params["center"].value_px] / 10
	assert not metrics["at_search_edge"]


def test_residual_is_independent_of_detector_row_read_chunking(endpoint_scan):
	build, source, dataset, params, *_ = endpoint_scan
	first = build_beam_band_shift_residual(build, source, dataset, **params)
	params["y_chunk"] = 57
	second = build_beam_band_shift_residual(build, source, dataset, name="second", **params)
	np.testing.assert_array_equal(build.read_component("beam_residual"), build.read_component("second"))
	for name in ("band_shift_residual_fit", "band_shift_residual_center_scan"):
		assert _qc(first.component, name).metrics == _qc(second.component, name).metrics
	build.finalize()


def test_center_search_edge_is_visible_in_component_qc(endpoint_scan):
	build, source, dataset, params, *_ = endpoint_scan
	params.update(center=ManualCenterStage.Params(value_px=33.0), scan_center=(0.5, 0.5))
	result = build_beam_band_shift_residual(build, source, dataset, **params)
	qc = _qc(result.component, "band_shift_residual_center_scan")
	assert qc.status == "warning"
	assert qc.metrics["at_search_edge"]
	assert qc.metrics["center_used"] < params["center"].value_px
	build.finalize()


def test_known_center_can_skip_search_and_still_measure_residual(endpoint_scan):
	build, source, dataset, params, _residual, center, dy, stretch = endpoint_scan
	params.update(center=params["center"].model_copy(update={"value_px": center}), scan_center=(0.0, 0.0))
	result = build_beam_band_shift_residual(build, source, dataset, **params)
	qc = _qc(result.component, "band_shift_residual_center_scan")
	assert qc.status == "pass"
	assert qc.metrics["center_used"] == center
	assert qc.metrics["center_scan"] == {}
	assert not qc.metrics["at_search_edge"]
	fit = _qc(result.component, "band_shift_residual_fit").metrics
	assert fit["dy"] == pytest.approx(dy, abs=0.25)
	assert fit["m"] == pytest.approx(stretch, abs=0.0005)
	assert build.read_component("beam_residual")[slice(*params["fit_rows"])].mean() > 0.02
	build.finalize()


@pytest.mark.parametrize("selection", [{"pairs": 0}, {"pairs": 7}, {"pair_start": -1}, {"pair_start": 3}])
def test_invalid_endpoint_selection_does_not_publish_residual(endpoint_scan, selection):
	build, source, dataset, params, *_ = endpoint_scan
	params.update(selection)
	with pytest.raises(ValueError, match="(?i)(pair|endpoint)"):
		build_beam_band_shift_residual(build, source, dataset, **params)
	assert "beam_residual" not in build.components


def test_unpaired_acquisition_angles_do_not_publish_residual(endpoint_scan):
	build, source, dataset, params, *_ = endpoint_scan
	angles = dataset.angles.copy()
	angles[-2] -= 0.05
	dataset = replace(dataset, angles=angles)
	with pytest.raises(ValueError, match="(?i)(angle|oppos|pair)"):
		build_beam_band_shift_residual(build, source, dataset, **params)
	assert "beam_residual" not in build.components


@pytest.mark.parametrize("selection", [{"fit_rows": (-1, 344)}, {"fit_columns": (8, 65)}, {"bands": 49}])
def test_fit_region_must_retain_real_detector_samples(endpoint_scan, selection):
	build, source, dataset, params, *_ = endpoint_scan
	params.update(selection)
	with pytest.raises(ValueError, match="(?i)(crop|column)"):
		build_beam_band_shift_residual(build, source, dataset, **params)
	assert "beam_residual" not in build.components


def test_zero_endpoint_shift_cannot_publish_an_unscalable_residual(endpoint_scan):
	build, source, dataset, params, residual, center, *_ = endpoint_scan
	y, x = np.indices(residual.shape, dtype=np.float64)
	dark = build.read_component("pre_dark")
	# A beam whose shape changes without moving has no dy_end with which the
	# runtime's dy/dy_end residual scaling can represent that change.
	with h5py.File(source.path, "r+") as handle:
		handle["exchange/data"][source.frame_count // 2:] = (
			dark + np.exp(_beam(y) + residual + _object(y, 2 * center - x))
		)
	with pytest.raises(ValueError, match="zero endpoint dy"):
		build_beam_band_shift_residual(build, source, dataset, **params)
	assert "beam_residual" not in build.components
