from pathlib import Path
import tracemalloc

import h5py
import numpy as np
import pytest

from lftomo.calibration import (
	CalibrationBuildError,
	CalibrationBuildSession,
	CalibrationBundle,
	CompatibilityMetadata,
	DetectorRegion,
	ProducerInfo,
	TimeAxis,
	inspect_build,
	stream_sha256,
)


def _session(tmp_path, name="bundle"):
	return CalibrationBuildSession(
		tmp_path / name,
		calibration_id="calibration-test",
		producer=ProducerInfo(
			package="lftomo", version="0.1.0", git_commit="abc123",
			parameters={"rank": 3},
		),
		compatibility=CompatibilityMetadata(
			detector_id="camera-a",
			region=DetectorRegion(y_start=0, x_start=0, height=32, width=64),
			static_artifact_epoch="epoch-a",
		),
		metadata={"camera_segments": ["trip-1", "trip-2"]},
	)


def test_incremental_multiscan_build_is_chunk_bounded_and_atomic(tmp_path):
	session = _session(tmp_path)
	assert not session.target.exists()
	assert inspect_build(session.staging_path).state == "building"
	for index in range(3):
		source = tmp_path / f"scan-{index}.h5"
		with h5py.File(source, "w") as handle:
			handle.create_dataset(
				"exchange/data_white",
				data=np.full((4, 32, 64), index, dtype=np.float32),
				chunks=(1, 8, 64),
			)
		session.add_source(
			source,
			uuid=f"uuid-{index}",
			hash_source=True,
			metadata={"camera_segment": f"trip-{index}", "static_artifact_epoch": "epoch-a"},
		)

	writer = session.create_array_component(
		"beam_keyframes",
		shape=(12, 32, 64),
		dtype="float32",
		axes=("T", "Y", "X"),
		kind="beam_keyframes",
		coordinate_frame="beam",
		time_axis=TimeAxis(kind="timestamp_seconds", unit="s", values=list(range(12))),
		metadata={"confidence_metric": "median-absolute-deviation"},
	)
	tracemalloc.start()
	for start in range(0, 12, 2):
		chunk = np.full((2, 32, 64), start, dtype=np.float32)
		writer.append(chunk)
	current, peak = tracemalloc.get_traced_memory()
	tracemalloc.stop()
	writer.add_validation_result({
		"name": "keyframe_qc", "status": "pass", "message": "synthetic QC passed",
		"metrics": {"confidence": 0.99},
	})
	writer.close()
	assert writer.max_buffer_bytes == 2 * 32 * 64 * 4
	assert peak < 2_000_000
	assert not session.target.exists()

	ref = session.finalize()
	assert ref.manifest_path == session.target / "calibration.yaml"
	assert not session.staging_path.exists()
	inspection = inspect_build(session.target, verify_hashes=True)
	assert inspection.state == "complete"
	assert inspection.calibration_id == "calibration-test"
	bundle = CalibrationBundle.open(ref.manifest_path, verify_hashes=True)
	assert len(bundle.manifest.sources) == 3
	assert bundle.manifest.sources[0].sha256 == stream_sha256(tmp_path / "scan-0.h5")
	assert bundle.component("beam_keyframes").validation_results[0].metrics["confidence"] == 0.99
	np.testing.assert_array_equal(
		bundle.read_component("beam_keyframes", time_value=7, y_start=3, y_stop=5),
		np.full((2, 64), 6, dtype=np.float32),
	)


def test_peak_memory_scales_with_append_chunk_not_total_component(tmp_path):
	session = CalibrationBuildSession(
		tmp_path / "large-bundle", calibration_id="large-test",
		producer={"package": "lftomo", "version": "test"},
		compatibility={
			"detector_id": "camera-a", "region": {"height": 256, "width": 256},
		},
	)
	frame_count = 128
	writer = session.create_array_component(
		"beam_keyframes", shape=(frame_count, 256, 256), dtype="float32",
		axes=("T", "Y", "X"), kind="beam_keyframes", coordinate_frame="beam",
		time_axis=TimeAxis(kind="projection_index", values=list(range(frame_count))),
	)
	tracemalloc.start()
	for index in range(frame_count):
		writer.append(np.full((1, 256, 256), index, dtype=np.float32))
	_, peak = tracemalloc.get_traced_memory()
	tracemalloc.stop()

	chunk_bytes = 256 * 256 * np.dtype("float32").itemsize
	full_bytes = frame_count * chunk_bytes
	assert writer.max_buffer_bytes == chunk_bytes
	assert full_bytes > 32_000_000
	assert peak < 2_000_000
	writer.close()


def test_table_writer_records_columns_and_uses_row_chunks(tmp_path):
	session = _session(tmp_path)
	component = session.write_table_component(
		"beam_schedule",
		[np.array([[0, 1], [1, 2]]), np.array([[2, 3]])],
		n_rows=3,
		columns=("time_seconds", "keyframe"),
		kind="beam_schedule",
		coordinate_frame="time",
		time_axis=TimeAxis(kind="timestamp_seconds", values=[0, 1, 2]),
	)
	ref = session.finalize()
	assert component.metadata["columns"] == ["time_seconds", "keyframe"]
	bundle = CalibrationBundle.open(ref.manifest_path)
	np.testing.assert_array_equal(
		bundle.read_component("beam_schedule", time_value=1), np.array([1, 2]),
	)


def test_build_reads_only_closed_components_with_detector_coordinates(tmp_path):
	session = _session(tmp_path)
	writer = session.create_array_component(
		"white", shape=(3, 4), dtype="float32", axes=("Y", "X"),
		kind="detector_static", coordinate_frame="detector",
		compatibility={"region": {"y_start": 7, "height": 3, "width": 4}},
	)
	values = np.arange(12, dtype=np.float32).reshape(3, 4)
	writer.append(values)
	with pytest.raises(CalibrationBuildError, match="closed before reading"):
		session.read_component("white", y_start=8, y_stop=9)
	writer.close()
	np.testing.assert_array_equal(session.read_component("white", y_start=8, y_stop=9), values[1:2])
	assert not (session.staging_path / "calibration.yaml").exists()
	session.finalize()
	np.testing.assert_array_equal(session.read_component("white", y_start=8, y_stop=9), values[1:2])


def test_context_exception_preserves_diagnosable_failed_build(tmp_path):
	staging = None
	with pytest.raises(RuntimeError, match="algorithm failed"):
		with _session(tmp_path) as session:
			staging = session.staging_path
			writer = session.create_array_component(
				"detector", shape=(32, 64), dtype="float32", axes=("Y", "X"),
				kind="detector_static", coordinate_frame="detector",
			)
			writer.append(np.ones((4, 64), dtype=np.float32))
			raise RuntimeError("algorithm failed")

	assert not (tmp_path / "bundle").exists()
	inspection = inspect_build(staging)
	assert inspection.state == "failed"
	assert "algorithm failed" in inspection.message
	assert inspection.payloads


def test_incomplete_component_cannot_be_finalized(tmp_path):
	session = _session(tmp_path)
	writer = session.create_array_component(
		"detector", shape=(32, 64), dtype="float32", axes=("Y", "X"),
		kind="detector_static", coordinate_frame="detector",
	)
	writer.append(np.ones((4, 64), dtype=np.float32))
	with pytest.raises(CalibrationBuildError, match="incomplete") as caught:
		session.finalize()
	assert caught.value.staging_path == session.staging_path
	assert inspect_build(session.staging_path).state == "failed"
	assert not session.target.exists()


def test_descriptor_roi_mismatch_prevents_atomic_publication(tmp_path):
	session = _session(tmp_path)
	session.write_array_component(
		"detector",
		[np.ones((16, 64), dtype=np.float32)],
		shape=(16, 64),
		dtype="float32",
		axes=("Y", "X"),
		kind="detector_static",
		coordinate_frame="detector",
	)
	with pytest.raises(CalibrationBuildError, match="validation failed"):
		session.finalize()
	assert not session.target.exists()
	inspection = inspect_build(session.staging_path)
	assert inspection.state == "failed"
	assert "validation failed" in inspection.message
	assert (session.staging_path / "calibration.yaml").is_file()


def test_source_hash_mismatch_is_reported_without_unbounded_read(tmp_path):
	session = _session(tmp_path)
	source = tmp_path / "source.h5"
	source.write_bytes(b"source")
	with pytest.raises(CalibrationBuildError, match="source hash"):
		session.add_source(source, sha256="0" * 64, hash_source=True)


def test_writer_rejects_unknown_manifest_schema(tmp_path):
	with pytest.raises(ValueError, match="unsupported calibration build schema"):
		CalibrationBuildSession(
			tmp_path / "future",
			calibration_id="future",
			producer={"package": "lftomo", "version": "test"},
			compatibility={"detector_id": "camera-a"},
			schema_version="2.0",
		)
