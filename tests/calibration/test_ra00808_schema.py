from pathlib import Path

import h5py
import pytest

from lftomo.calibration import (
	ArrayPayload,
	CalibrationBundle,
	CalibrationBundleRef,
	CalibrationComponent,
	CalibrationManifest,
	CalibrationTarget,
	CompatibilityMetadata,
	DetectorRegion,
	ProducerInfo,
	check_compatibility,
)


RA_ROOT = Path("/mnt/z/gpfs/phenome/Berkeley/RA-00808_ang")


def _h5_facts(path):
	with h5py.File(path, "r") as handle:
		white = handle["exchange/data_white"]
		dark = handle["exchange/data_dark"]
		return {
			"white_shape": tuple(white.shape),
			"dark_shape": tuple(dark.shape),
			"i0_x": float(handle["process/acquisition/flat_fields/i0_move_x"][0]),
			"i0_y": float(handle["process/acquisition/flat_fields/i0_move_y"][0]),
			"camera_distance": float(
				handle["measurement/instrument/camera_motor_stack/setup/camera_distance"][0]
			),
			"camera_elevation": float(
				handle["measurement/instrument/camera_motor_stack/setup/camera_elevation"][0]
			),
		}


@pytest.mark.integration
def test_ra00808_headers_express_camera_segment_and_artifact_epoch_boundary(tmp_path):
	before_path = RA_ROOT / "20260610_101717_toothpick.h5"
	boundary_path = RA_ROOT / "20260611_143029_cyrothin1.h5"
	if not before_path.is_file() or not boundary_path.is_file():
		pytest.skip(f"RA-00808_ang is unavailable at {RA_ROOT}")
	before = _h5_facts(before_path)
	boundary = _h5_facts(boundary_path)

	assert before["white_shape"] == (15, 688, 2560)
	assert boundary["white_shape"] == (15, 2160, 2560)
	assert before["dark_shape"][1:] == before["white_shape"][1:]
	assert boundary["dark_shape"][1:] == boundary["white_shape"][1:]
	assert before["camera_distance"] != boundary["camera_distance"]
	assert before["camera_elevation"] != boundary["camera_elevation"]
	assert (before["i0_x"], before["i0_y"]) == (0.0, -25.0)
	assert (boundary["i0_x"], boundary["i0_y"]) == (0.0, -25.0)

	region = DetectorRegion(height=before["white_shape"][1], width=before["white_shape"][2])
	manifest = CalibrationManifest(
		calibration_id="ra00808-before-static-artifact",
		producer=ProducerInfo(package="lftomo", version="0.1.0"),
		compatibility=CompatibilityMetadata(
			detector_id="ra00808-camera", beamline="ALS 8.3.2", region=region,
			camera_segment="distance-110.387658", static_artifact_epoch="before-20260611-143029",
		),
		components={
			"detector": CalibrationComponent(
				kind="detector_static", coordinate_frame="detector",
				payload=ArrayPayload(
					path="detector.npy", format="npy", sha256="0" * 64,
					shape=(region.height, region.width),
					dtype="float32", axes=("Y", "X"),
				),
			),
		},
	)
	manifest_path = tmp_path / "calibration.yaml"
	manifest_path.write_text(manifest.to_yaml(), encoding="utf-8")
	bundle = CalibrationBundle(
		manifest, CalibrationBundleRef(
			manifest_path=manifest_path, calibration_id=manifest.calibration_id,
		),
	)
	report = check_compatibility(
		bundle,
		CalibrationTarget(
			detector_id="ra00808-camera", beamline="ALS 8.3.2",
			region=DetectorRegion(
				height=boundary["white_shape"][1], width=boundary["white_shape"][2],
			),
			camera_segment="distance-150.079114",
			static_artifact_epoch="from-20260611-143029",
		),
		components=["detector"], policy="report",
	)
	assert not report.compatible
	assert {item.field for item in report.diagnostics} == {"region", "static_artifact_epoch"}
