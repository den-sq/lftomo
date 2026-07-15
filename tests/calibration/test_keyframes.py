from pathlib import Path

import numpy as np
import pytest
import tifffile as tf

from lftomo.calibration import (
	CalibrationBuildSession, CalibrationBundle, CompatibilityMetadata, DetectorRegion,
	ProducerInfo, TIFFRandomFrameSource, TimeAxis, build_flat_series_digest,
)
from lftomo.config import Recon, Scan, StageConfig
from lftomo.pipeline import NumpyBackend, PipelineState, validate_pipeline
from lftomo.pipeline.domains import DataDomain
from lftomo.pipeline.stages import BeamKeyframesNormalizeStage, create_stage
from lftomo.util.shapes import ProjOrder


def _build_bundle(tmp_path, keyframes, times):
	build = CalibrationBuildSession(
		tmp_path / "bundle", calibration_id="event",
		producer=ProducerInfo(package="lftomo", version="test"),
		compatibility=CompatibilityMetadata(
			detector_id="camera", region=DetectorRegion(height=keyframes.shape[1], width=keyframes.shape[2]),
			camera_segment="segment-a",
		),
	)
	build.write_array_component(
		"beam_keyframes", [keyframes], shape=keyframes.shape, dtype="float32",
		axes=("T", "Y", "X"), kind="beam_keyframes", coordinate_frame="beam",
		time_axis=TimeAxis(kind="timestamp_seconds", unit="s", values=times),
		compatibility=CompatibilityMetadata(camera_segment="segment-a"), interpolation="linear",
	)
	return build.finalize()


def _scan(rows, columns, count):
	return Scan.model_validate({
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": count, "degrees_per_fov": 180,
		"detector": {"name": "camera", "pixel_size_um": 1, "rows": rows, "columns": columns},
		"acquisition": {"n_projections": count, "rotation_end_deg": 180},
		"projections": {"format": "tiff_stack", "path": ".", "pattern": "*.tif"},
	})


def test_event_keyframes_beat_endpoint_model_and_keep_detector_defect_fixed(tmp_path):
	y, x = np.indices((20, 28))
	detector = np.ones((20, 28)); detector[6:8, 9:11] = 0.65
	keyframes = np.stack([
		detector * 1000 * (1 + 0.08 * np.sin((y - shift) / 4))
		for shift in (0, 5, 0)
	]).astype(np.float32)
	ref = _build_bundle(tmp_path, keyframes, [0, 5, 10])
	projections = keyframes.copy()
	state = PipelineState(
		projections, _scan(20, 28, 3), Recon.model_validate({"pipeline": []}),
		artifacts={"calibration_bundle": ref, "projection_times": np.array([0, 5, 10])},
		backend=NumpyBackend(), layout=ProjOrder, domain=DataDomain.RAW,
	)
	BeamKeyframesNormalizeStage(transport_sigma=2).run(state)
	assert np.std(projections - 1) < 0.003
	endpoint_mid = np.sqrt(keyframes[0] * keyframes[-1])
	assert np.std(keyframes[1] / endpoint_mid - 1) > 0.03
	assert state.artifacts["beam_keyframe_qc"]["maximum_gap_seen"] == 5


def test_heldout_midpoints_beat_a_scan_endpoint_model(tmp_path):
	y, x = np.indices((20, 28))
	detector = np.ones((20, 28)); detector[6:8, 9:11] = 0.65
	keyframes = np.stack([
		detector * 1000 * (1 + 0.08 * np.sin((y - shift) / 4) + 0.02 * np.cos(x / 9))
		for shift in (0, 5, 0)
	]).astype(np.float32)
	ref = _build_bundle(tmp_path, keyframes, [0, 5, 10])
	heldout = np.stack([
		np.sqrt(keyframes[0] * keyframes[1]),
		np.sqrt(keyframes[1] * keyframes[2]),
	])
	endpoint = np.sqrt(keyframes[0] * keyframes[-1])
	endpoint_error = np.std(heldout / endpoint - 1)
	state = PipelineState(
		heldout.copy(), _scan(20, 28, 2), Recon.model_validate({"pipeline": []}),
		artifacts={"calibration_bundle": ref, "projection_times": np.array([2.5, 7.5])},
		backend=NumpyBackend(), layout=ProjOrder, domain=DataDomain.RAW,
	)
	BeamKeyframesNormalizeStage(transport_sigma=2).run(state)
	assert np.std(np.asarray(state.working_array) - 1) < endpoint_error / 20


def test_digest_samples_short_centered_medians_and_writes_trajectory(tmp_path):
	source_dir = tmp_path / "frames"; source_dir.mkdir()
	y, x = np.indices((12, 16))
	for index in range(21):
		frame = 1000 * (1 + 0.05 * np.sin((y - 4 * np.exp(-((index - 10) / 3) ** 2)) / 3))
		if index in (3, 10, 17):
			frame = frame.copy(); frame[0, 0] = 60000
		tf.imwrite(source_dir / f"frame_{index:03d}.tif", frame.astype(np.uint16))
	source = TIFFRandomFrameSource(source_dir)
	build = CalibrationBuildSession(
		tmp_path / "digest", calibration_id="digest",
		producer={"package": "lftomo", "version": "test"},
		compatibility={"detector_id": "camera", "region": {"height": 12, "width": 16},
			"camera_segment": "segment-a"},
	)
	digest = build_flat_series_digest(
		build, source, keep=3, median_window=3, y_chunk=5,
		camera_segment="segment-a", frame_times=np.arange(21) * 0.5,
		time_kind="timestamp_seconds", time_unit="s",
	)
	ref = build.finalize()
	assert digest.sample_centres == (3, 10, 17)
	assert digest.times == (1.5, 5.0, 8.5)
	assert len(digest.trajectory) == 3
	bundle = CalibrationBundle.open(ref.manifest_path, verify_hashes=True)
	source_metadata = bundle.component("beam_keyframes").metadata["source"]
	assert source_metadata == {
		"path": str(source_dir.resolve()), "pattern": "*.tif*", "frame_count": 21,
		"first_frame": str((source_dir / "frame_000.tif").resolve()),
		"last_frame": str((source_dir / "frame_020.tif").resolve()),
	}
	values = bundle.read_component("beam_keyframes", time_value=5, y_start=0, y_stop=2)
	assert values[0, 0] < 2000  # the short median rejected the centre-frame zinger


def test_keyframe_stage_rejects_missing_times_and_large_gaps(tmp_path):
	keyframes = np.ones((2, 4, 5), dtype=np.float32)
	ref = _build_bundle(tmp_path, keyframes, [0, 10])
	state = PipelineState(
		np.ones((1, 4, 5), np.float32), _scan(4, 5, 1), Recon.model_validate({"pipeline": []}),
		artifacts={"calibration_bundle": ref}, backend=NumpyBackend(), layout=ProjOrder,
		domain=DataDomain.RAW,
	)
	with pytest.raises(ValueError, match="projection_times"):
		BeamKeyframesNormalizeStage().run(state)
	state.artifacts["projection_times"] = [5]
	with pytest.raises(ValueError, match="keyframe gap"):
		BeamKeyframesNormalizeStage(maximum_gap=4).run(state)


def test_beam_keyframes_is_the_sole_normalization_in_a_complete_pipeline():
	configs = [
		StageConfig(kind="read", algorithm="tiff_stack", params={"order": "projection"}),
		StageConfig(kind="normalize", algorithm="beam_keyframes"),
		StageConfig(kind="phase", algorithm="paganin"),
		StageConfig(kind="recon", algorithm="gridrec", params={"center": 14}),
	]
	plan = validate_pipeline(
		[create_stage(config) for config in configs], _scan(20, 28, 3), mode="complete",
	)
	normalize = next(step for step in plan.steps if step.stage_name == "normalize.beam_keyframes")
	assert normalize.input_domain is DataDomain.RAW
	assert normalize.output_domain is DataDomain.TRANSMISSION


@pytest.mark.integration
def test_real_flat_series_keyframes_reduce_heldout_error(tmp_path, request):
	if not request.config.getoption("--run-integration"):
		pytest.skip("requires --run-integration and the copied pure-flat series")
	path = Path(
		"/mnt/wsl/PHYSICALDRIVE2p1/flatfield_advanced/10min_white_field_20keV/"
		"1687581222_1200projs_500000us_SW_Trigger_tiff"
	)
	if not path.is_dir():
		pytest.skip("pure-flat copy is unavailable")
	source = TIFFRandomFrameSource(path)
	build = CalibrationBuildSession(
		tmp_path / "real-digest", calibration_id="real-digest",
		producer={"package": "lftomo", "version": "test"},
		compatibility={"detector_id": "als", "region": {"height": 128, "width": 512},
			"camera_segment": "10min"},
	)
	digest = build_flat_series_digest(
		build, source, keep=20, median_window=5, y_chunk=32,
		crop=(0, 128, 0, 512), camera_segment="10min",
	)
	ref = build.finalize()
	bundle = CalibrationBundle.open(ref.manifest_path)
	centres = digest.sample_centres
	first = bundle.read_component("beam_keyframes", projection_index=centres[0])
	last = bundle.read_component("beam_keyframes", projection_index=centres[-1])
	endpoint_errors, keyframe_errors = [], []
	for left, right in zip(centres[:-1], centres[1:], strict=True):
		index = (left + right) // 2
		actual = source.read(index, y_start=0, y_stop=128, x_start=0, x_stop=512)
		fraction = (index - centres[0]) / (centres[-1] - centres[0])
		endpoint = np.exp((1 - fraction) * np.log(first) + fraction * np.log(last))
		predicted = bundle.read_component("beam_keyframes", projection_index=index)
		scale = np.mean(actual)
		endpoint_errors.append(float(np.sqrt(np.mean((actual - endpoint) ** 2)) / scale))
		keyframe_errors.append(float(np.sqrt(np.mean((actual - predicted) ** 2)) / scale))
	assert np.median(keyframe_errors) < np.median(endpoint_errors)
	assert digest.maximum_gap <= 61
