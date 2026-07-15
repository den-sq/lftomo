from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import shift
import tifffile as tf

from lftomo.calibration import (
	CalibrationBuildSession,
	CalibrationBundle,
	CompatibilityMetadata,
	DetectorRegion,
	ProducerInfo,
	SegmentedTimePoint,
	TimeAxis,
	build_static_dynamic_components,
	calibrate_motor_axis,
	decompose_static_dynamic,
	register_beam_offset,
	segment_time_bracket,
	write_camera_registration_component,
)
from lftomo.prep.flat_median import TIFFFrameSeries


def _synthetic_flats(shape=(96, 128), count=5):
	y, x = np.indices(shape)
	detector = np.ones(shape)
	detector[22:25, 37:40] = 0.65
	detector[66:68, 91:96] = 1.35
	frames = []
	for index in range(count):
		beam = 1000 * (1 + 0.12 * np.sin((y - index * 3) / 6))
		beam *= 1 + 0.04 * np.cos((x - index * 2) / 29)
		frames.append(detector * beam)
	return detector, np.asarray(frames, dtype=np.float32)


def test_decomposition_keeps_fixed_compact_defects_static_and_reconstructs_exactly():
	true_detector, flats = _synthetic_flats()
	detector, beam = decompose_static_dynamic(
		flats, beam_sigma=4, band_orientation="horizontal", band_length=41,
	)
	np.testing.assert_allclose(detector * beam, flats, rtol=2e-7, atol=2e-4)
	assert detector[23, 38] < 0.82
	assert detector[67, 93] > 1.15
	# Horizontally elongated moving beam bands belong in B, not D.
	assert np.std(detector.mean(axis=1)) < np.std(flats[0].mean(axis=1)) / 100


def test_windowed_decomposition_writes_distinct_validity_scopes(tmp_path):
	_, flats = _synthetic_flats(shape=(48, 64), count=4)
	stack = tmp_path / "flats.tif"
	tf.imwrite(stack, flats, photometric="minisblack")
	series = TIFFFrameSeries(stack)
	build = CalibrationBuildSession(
		tmp_path / "bundle", calibration_id="segment-a",
		producer=ProducerInfo(package="lftomo", version="test"),
		compatibility=CompatibilityMetadata(
			detector_id="camera", beamline="ALS 8.3.2", energy_keV=20,
			region=DetectorRegion(y_start=0, x_start=0, height=48, width=64),
			static_artifact_epoch="epoch-2", camera_segment="segment-a",
		),
	)
	static, dynamic = build_static_dynamic_components(
		build, series,
		time_axis=TimeAxis(kind="timestamp_seconds", values=[0, 1, 2, 3]),
		camera_segment="segment-a", static_artifact_epoch="epoch-2",
		y_chunk=11, beam_sigma=3, band_length=31,
	)
	ref = build.finalize()
	assert static.kind == "detector_static"
	assert static.compatibility.static_artifact_epoch == "epoch-2"
	assert dynamic.kind == "dynamic_beam"
	assert dynamic.compatibility.camera_segment == "segment-a"
	assert dynamic.metadata["camera_segment_boundary"] == "hard"
	bundle = CalibrationBundle.open(ref.manifest_path, verify_hashes=True)
	assert bundle.read_component("detector_static", y_start=7, y_stop=12).shape == (5, 64)
	assert bundle.read_component("dynamic_beam", time_value=2, y_start=7, y_stop=12).shape == (5, 64)


def test_camera_registration_reports_ambiguity_and_uses_encoder_prior():
	y, x = np.indices((128, 160))
	old = 1000 * (1 + 0.18 * np.sin(y / 5) + 0.025 * np.cos(x / 33))
	old *= 1 + 0.03 * np.exp(-((y - 70) ** 2 + (x - 55) ** 2) / 300)
	new = shift(old, shift=(5.0, -7.0), order=1, mode="nearest")
	result = register_beam_offset(old, new, max_shift=15)
	assert result.dy.measured_shift == pytest.approx(5, abs=0.8)
	assert result.dx.peak_ambiguity >= result.dy.peak_ambiguity

	prior = register_beam_offset(
		old, new, max_shift=15, expected=(5, -7), refine=2,
		max_ambiguity=0.2,
	)
	assert prior.resolved_shift == pytest.approx((5, -7), abs=0.8)
	assert prior.dx.zncc >= 0

	pure_stripes = 1000 * (1 + 0.18 * np.sin(y / 5))
	stripe_move = register_beam_offset(
		pure_stripes, shift(pure_stripes, shift=(5, -7), order=1, mode="nearest"),
		max_shift=15,
	)
	assert stripe_move.dy.zncc > 0.9
	assert stripe_move.dx.zncc < 1e-10
	assert stripe_move.dx.peak_ambiguity == 1


def test_motor_axis_calibration_returns_typed_signed_metadata(tmp_path):
	y, x = np.indices((96, 128))
	old = 1000 * (1 + 0.2 * np.sin(y / 7) + 0.04 * np.cos(x / 21))
	new = shift(old, shift=(6, 0), order=1, mode="nearest")
	result = calibrate_motor_axis(
		old, new, motor_delta=-2, unit="mm", axis="vertical", max_shift=12,
	)
	assert result.calibration.unit == "mm"
	assert result.calibration.pixels_per_unit == pytest.approx(3, abs=0.4)
	assert result.calibration.sign == -1
	assert result.expected_shift(-2) == pytest.approx(6, abs=0.8)

	build = CalibrationBuildSession(
		tmp_path / "registration", calibration_id="registration",
		producer={"package": "lftomo", "version": "test"},
		compatibility={
			"detector_id": "camera",
			"region": {"height": 96, "width": 128},
			"motor_axes": {"vertical": result.calibration},
		},
	)
	registration = register_beam_offset(old, new, max_shift=12)
	component = write_camera_registration_component(
		build, {"move-1": registration}, motor_axes={"vertical": result.calibration},
	)
	ref = build.finalize()
	assert component.kind == "camera_registration"
	assert component.metadata["columns"][3] == "dy_peak_ambiguity"
	assert CalibrationBundle.open(ref.manifest_path).component("camera_registration")


def test_camera_segments_are_hard_interpolation_boundaries():
	points = [
		SegmentedTimePoint("before", 0), SegmentedTimePoint("before", 10),
		SegmentedTimePoint("after", 11), SegmentedTimePoint("after", 20),
	]
	assert segment_time_bracket(points, segment="before", time=5) == (points[0], points[1])
	with pytest.raises(ValueError, match="across a camera move is forbidden"):
		segment_time_bracket(points, segment="before", time=11)


@pytest.mark.integration
def test_ra00808_camera_move_matches_encoder_and_is_segmented(request):
	if not request.config.getoption("--run-integration"):
		pytest.skip("requires --run-integration and local B11 data")
	root = Path(
		"/mnt/wsl/PHYSICALDRIVE2p1/B11/b11_median_flats/gains/(2160, 2560)"
	)
	before = root / "20260611_143029_cyrothin1__white_median.tif"
	after = root / "20260611_145551_cyrothin2__white_median.tif"
	if not before.is_file() or not after.is_file():
		pytest.skip("local RA-00808 median pair is unavailable")
	# Bounded 2x view: the manifest records i0_y -25 -> -40, delta -15.
	old = tf.memmap(before)[256:1904:2, ::2]
	new = tf.memmap(after)[256:1904:2, ::2]
	registration = register_beam_offset(old, new, max_shift=40)
	full_resolution_dy = registration.dy.measured_shift * 2
	assert full_resolution_dy == pytest.approx(-30, abs=2)
	assert full_resolution_dy / -15 == pytest.approx(2, abs=0.15)
	assert registration.dx.zncc < 0.25 or registration.dx.peak_ambiguity > 0.9
	points = [SegmentedTimePoint("pre-move", 0), SegmentedTimePoint("post-move", 1)]
	with pytest.raises(ValueError, match="across a camera move"):
		segment_time_bracket(points, segment="pre-move", time=1)
