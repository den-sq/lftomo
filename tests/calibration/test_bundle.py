from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile as tf

from lftomo.calibration import (
	ArrayPayload,
	CalibrationBundle,
	CalibrationBundleError,
	CalibrationComponent,
	CalibrationManifest,
	CompatibilityMetadata,
	DetectorRegion,
	ProducerInfo,
	TimeAxis,
	stream_sha256,
	validate_bundle,
)


def _payload(path: Path, axes, *, format="npy", dataset=None):
	if format == "npy":
		array = np.load(path, mmap_mode="r")
		shape, dtype = array.shape, array.dtype
	else:
		with h5py.File(path, "r") as handle:
			shape, dtype = handle[dataset].shape, handle[dataset].dtype
	return ArrayPayload(
		path=path.name,
		format=format,
		dataset=dataset,
		sha256=stream_sha256(path),
		shape=shape,
		dtype=np.dtype(dtype).name,
		axes=axes,
	)


def _write_manifest(tmp_path: Path, components):
	manifest = CalibrationManifest(
		calibration_id="als832-epoch-1",
		created_at=datetime(2026, 6, 11, tzinfo=timezone.utc),
		producer=ProducerInfo(package="lftomo", version="0.1.0", git_commit="abc123"),
		compatibility=CompatibilityMetadata(
			detector_id="als832-pco-edge",
			beamline="ALS 8.3.2",
			optics_id="20keV-110mm",
			energy_keV=20,
			region=DetectorRegion(y_start=100, x_start=0, height=4, width=5),
			static_artifact_epoch="epoch-1",
			camera_segment="camera-a",
		),
		components=components,
	)
	path = tmp_path / "calibration.yaml"
	path.write_text(manifest.to_yaml(), encoding="utf-8")
	return path, manifest


def test_manifest_round_trip_and_payload_validation(tmp_path):
	static = np.arange(20, dtype=np.float32).reshape(4, 5)
	np.save(tmp_path / "static.npy", static)
	path, manifest = _write_manifest(tmp_path, {
		"detector": CalibrationComponent(
			kind="detector_static",
			coordinate_frame="detector",
			payload=_payload(tmp_path / "static.npy", ("Y", "X")),
		),
	})

	loaded = CalibrationManifest.from_yaml(path.read_text(encoding="utf-8"))
	assert loaded == manifest
	report = validate_bundle(path)
	assert report.valid
	assert report.results[0].status == "pass"


def test_validation_reports_missing_corrupt_shape_and_hash(tmp_path):
	values = np.ones((4, 5), dtype=np.float32)
	np.save(tmp_path / "field.npy", values)
	payload = _payload(tmp_path / "field.npy", ("Y", "X"))
	payload.shape = (3, 5)
	path, _ = _write_manifest(tmp_path, {
		"bad": CalibrationComponent(
			kind="detector_static", coordinate_frame="detector", payload=payload,
		),
	})
	report = validate_bundle(path)
	assert not report.valid
	assert {result.name for result in report.results if result.status == "fail"} == {
		"payload_shape", "payload_region",
	}

	payload.shape = (4, 5)
	payload.sha256 = "0" * 64
	path, _ = _write_manifest(tmp_path, {
		"bad": CalibrationComponent(
			kind="detector_static", coordinate_frame="detector", payload=payload,
		),
	})
	report = validate_bundle(path)
	assert [result.name for result in report.results if result.status == "fail"] == [
		"payload_sha256",
	]

	(tmp_path / "field.npy").unlink()
	report = validate_bundle(path)
	assert [result.name for result in report.results if result.status == "fail"] == [
		"payload_exists",
	]


def test_static_and_dynamic_reads_are_y_and_time_bounded(tmp_path):
	static = np.arange(20, dtype=np.float32).reshape(4, 5)
	dynamic = np.stack([static + offset for offset in (0, 10, 30)])
	np.save(tmp_path / "static.npy", static)
	np.save(tmp_path / "dynamic.npy", dynamic)
	path, _ = _write_manifest(tmp_path, {
		"detector": CalibrationComponent(
			kind="detector_static",
			coordinate_frame="detector",
			payload=_payload(tmp_path / "static.npy", ("Y", "X")),
		),
		"beam": CalibrationComponent(
			kind="dynamic_beam",
			coordinate_frame="beam",
			payload=_payload(tmp_path / "dynamic.npy", ("T", "Y", "X")),
			time_axis=TimeAxis(
				kind="timestamp_seconds", unit="s", values=[10.0, 20.0, 40.0],
			),
			interpolation="linear",
		),
	})
	bundle = CalibrationBundle.open(path, verify_hashes=True)

	window = bundle.read_component("detector", y_start=101, y_stop=103)
	np.testing.assert_array_equal(window, static[1:3])
	interpolated = bundle.read_component(
		"beam", y_start=101, y_stop=103, time_value=15.0,
	)
	np.testing.assert_allclose(interpolated, dynamic[0, 1:3] * 0.5 + dynamic[1, 1:3] * 0.5)
	bracket = bundle.read_time_bracket("beam", 35.0, y_start=102, y_stop=104)
	assert bracket.before.shape == bracket.after.shape == (2, 5)
	assert bracket.weight == pytest.approx(0.75)


def test_projection_index_selects_one_theta_frame(tmp_path):
	values = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
	np.save(tmp_path / "beam.npy", values)
	path, _ = _write_manifest(tmp_path, {
		"beam": CalibrationComponent(
			kind="dynamic_beam",
			coordinate_frame="beam",
			payload=_payload(tmp_path / "beam.npy", ("Theta", "Y", "X")),
			time_axis=TimeAxis(kind="projection_index"),
		),
	})
	bundle = CalibrationBundle.open(path)
	np.testing.assert_array_equal(
		bundle.read_component("beam", projection_index=2, y_start=100, y_stop=102),
		values[2, :2],
	)
	with pytest.raises(CalibrationBundleError, match="requires projection_index"):
		bundle.read_component("beam")


def test_hdf5_component_reads_only_selected_window(tmp_path):
	path = tmp_path / "components.h5"
	values = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
	with h5py.File(path, "w") as handle:
		handle.create_dataset("beam/keyframes", data=values, chunks=(1, 2, 5))
	manifest_path, _ = _write_manifest(tmp_path, {
		"beam": CalibrationComponent(
			kind="beam_keyframes",
			coordinate_frame="beam",
			payload=_payload(path, ("T", "Y", "X"), format="hdf5", dataset="beam/keyframes"),
			time_axis=TimeAxis(kind="timestamp_seconds", unit="s", values=[0, 1, 2]),
		),
	})
	bundle = CalibrationBundle.open(manifest_path)
	result = bundle.read_component("beam", time_value=1, y_start=102, y_stop=104)
	np.testing.assert_array_equal(result, values[1, 2:4])


def test_memory_mappable_tiff_component_reads_y_window(tmp_path):
	values = np.arange(20, dtype=np.float32).reshape(4, 5)
	path = tmp_path / "dark.tif"
	tf.imwrite(path, values)
	manifest_path, _ = _write_manifest(tmp_path, {
		"dark": CalibrationComponent(
			kind="dark", coordinate_frame="detector",
			payload=ArrayPayload(
				path=path.name, format="tiff", sha256=stream_sha256(path),
				shape=values.shape, dtype="float32", axes=("Y", "X"),
			),
		),
	})
	bundle = CalibrationBundle.open(manifest_path)
	np.testing.assert_array_equal(
		bundle.read_component("dark", y_start=101, y_stop=103), values[1:3],
	)


def test_manifest_rejects_unsafe_paths_and_bad_time_axes(tmp_path):
	with pytest.raises(ValueError, match="inside the bundle"):
		ArrayPayload(
			path="../outside.npy", format="npy", sha256="0" * 64,
			shape=(2, 2), dtype="float32", axes=("Y", "X"),
		)
	with pytest.raises(ValueError, match="strictly increasing"):
		TimeAxis(kind="timestamp_seconds", values=[0, 2, 1])
	with pytest.raises(ValueError, match="schema_version"):
		CalibrationManifest.model_validate({
			"schema_version": "2.0",
			"calibration_id": "future",
			"producer": {"package": "lftomo", "version": "0.1.0"},
			"compatibility": {"detector_id": "camera"},
			"components": {"field": {
				"kind": "dark", "coordinate_frame": "detector",
				"payload": {
					"path": "field.npy", "format": "npy", "sha256": "0" * 64,
					"shape": [2, 2], "dtype": "float32", "axes": ["Y", "X"],
				},
			}},
		})


def test_dynamic_reads_reject_wrong_time_axis_and_out_of_range_query(tmp_path):
	values = np.ones((2, 4, 5), dtype=np.float32)
	np.save(tmp_path / "beam.npy", values)
	path, _ = _write_manifest(tmp_path, {
		"beam": CalibrationComponent(
			kind="dynamic_beam", coordinate_frame="beam",
			payload=_payload(tmp_path / "beam.npy", ("T", "Y", "X")),
			time_axis=TimeAxis(kind="timestamp_seconds", unit="s", values=[10, 20]),
		),
	})
	bundle = CalibrationBundle.open(path)
	with pytest.raises(CalibrationBundleError, match="physical timestamps"):
		bundle.read_component("beam", projection_index=1)
	with pytest.raises(CalibrationBundleError, match="outside component range"):
		bundle.read_component("beam", time_value=25)
