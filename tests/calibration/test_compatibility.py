from datetime import datetime, timezone

import numpy as np
import pytest

from lftomo.calibration import (
	ArrayPayload,
	CalibrationBundle,
	CalibrationBundleRef,
	CalibrationCompatibilityError,
	CalibrationComponent,
	CalibrationManifest,
	CalibrationTarget,
	CompatibilityMetadata,
	DetectorRegion,
	ProducerInfo,
	check_compatibility,
)


def _bundle(tmp_path):
	region = DetectorRegion(y_start=0, x_start=0, height=8, width=10)
	components = {}
	for name, kind in (
		("dark", "dark"),
		("detector", "detector_static"),
		("beam", "dynamic_beam"),
		("basis", "beam_basis"),
	):
		path = tmp_path / f"{name}.npy"
		shape = (1, 8, 10) if kind == "dynamic_beam" else (8, 10)
		np.save(path, np.ones(shape, dtype=np.float32))
		from lftomo.calibration import stream_sha256
		components[name] = CalibrationComponent(
			kind=kind,
			coordinate_frame="detector" if kind in ("dark", "detector_static") else "beam",
			payload=ArrayPayload(
				path=path.name, format="npy", sha256=stream_sha256(path),
				shape=shape,
				dtype="float32",
				axes=("Theta", "Y", "X") if kind == "dynamic_beam" else ("Y", "X"),
			),
			time_axis={"kind": "projection_index"} if kind == "dynamic_beam" else None,
			compatibility=CompatibilityMetadata(
				exposure_ms=175 if kind == "dark" else None,
				training_coverage=["quiet", "refill"] if kind == "beam_basis" else [],
			),
		)
	manifest = CalibrationManifest(
		calibration_id="als832-epoch-1",
		producer=ProducerInfo(package="lftomo", version="0.1.0"),
		compatibility=CompatibilityMetadata(
			detector_id="pco-edge", beamline="ALS 8.3.2", optics_id="20keV",
			energy_keV=20, region=region, static_artifact_epoch="epoch-1",
			camera_segment="camera-a",
		),
		components=components,
	)
	manifest_path = tmp_path / "calibration.yaml"
	manifest_path.write_text(manifest.to_yaml(), encoding="utf-8")
	return CalibrationBundle(
		manifest,
		CalibrationBundleRef(
			manifest_path=manifest_path,
			calibration_id=manifest.calibration_id,
		),
	)


def _target(**updates):
	values = {
		"detector_id": "pco-edge",
		"beamline": "ALS 8.3.2",
		"optics_id": "20keV",
		"energy_keV": 20,
		"exposure_ms": 175,
		"region": DetectorRegion(y_start=2, x_start=0, height=4, width=10),
		"static_artifact_epoch": "epoch-1",
		"camera_segment": "camera-a",
		"beam_state": "quiet",
		"acquisition_time": datetime(2026, 6, 11, tzinfo=timezone.utc),
	}
	values.update(updates)
	return CalibrationTarget(**values)


def test_component_specific_compatibility_accepts_valid_subregion(tmp_path):
	report = check_compatibility(_bundle(tmp_path), _target())
	assert report.compatible
	assert not report.diagnostics


@pytest.mark.parametrize(
	("components", "updates", "field"),
	[
		(["dark"], {"exposure_ms": 50}, "exposure_ms"),
		(["detector"], {"static_artifact_epoch": "epoch-2"}, "static_artifact_epoch"),
		(["beam"], {"camera_segment": "camera-b"}, "camera_segment"),
		(["basis"], {"beam_state": "unseen-event"}, "training_coverage"),
	],
)
def test_component_mismatch_reports_exact_field(tmp_path, components, updates, field):
	with pytest.raises(CalibrationCompatibilityError) as caught:
		check_compatibility(_bundle(tmp_path), _target(**updates), components=components)
	assert caught.value.report.diagnostics[0].field == field


def test_region_mismatch_and_override_are_provenance_ready(tmp_path):
	target = _target(region=DetectorRegion(y_start=7, x_start=0, height=2, width=10))
	report = check_compatibility(
		_bundle(tmp_path), target, components=["detector"],
		policy="override", override_reason="commissioning comparison only",
	)
	assert report.compatible
	assert report.override.reason == "commissioning comparison only"
	assert report.override.diagnostics[0].field == "region"
	assert "commissioning comparison" in report.to_yaml()


def test_report_policy_does_not_silently_accept_mismatch(tmp_path):
	report = check_compatibility(
		_bundle(tmp_path), _target(detector_id="other"), policy="report",
	)
	assert not report.compatible
	assert {item.field for item in report.diagnostics} == {"detector_id"}
