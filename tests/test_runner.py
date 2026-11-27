from pathlib import Path
from types import SimpleNamespace
import json

import h5py
import numpy as np
import pytest
import tifffile as tf
import yaml
from click.testing import CliRunner

from lftomo.calibration import (
	ArrayPayload,
	CalibrationBundleRef,
	CalibrationComponent,
	CalibrationCompatibilityError,
	CalibrationManifest,
	CompatibilityMetadata,
	DetectorRegion,
	ProducerInfo,
	load_manifest,
	stream_sha256,
)
from lftomo.config import Recon, Scan, StageConfig
from lftomo.pipeline.stages import create_stage
from lftomo.runner import (
	_center_artifacts,
	_detector_shape,
	load_recon,
	load_scan,
	main,
	resolve_calibration,
	resolve_scan,
	run_reconstruction,
)
from lftomo.util.window import TomoWindow


def write_yaml(path, value):
	path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_center_prepass_retains_durable_diagnostic_outputs():
	state = type("State", (), {"artifacts": {
		"center": 4.0,
		"center_entropy": {"termination": "converged"},
		"output_files": [Path("center-diagnostics/candidate.tif")],
		"unrelated": "discarded",
	}})()

	assert _center_artifacts(state) == {
		"center": 4.0,
		"center_entropy": {"termination": "converged"},
		"output_files": [Path("center-diagnostics/candidate.tif")],
	}


def test_load_scan_resolves_hardware_timestamp_source_relative_to_scan(tmp_path):
	scan_path = tmp_path / "scan.yaml"
	document = scan_document("projections")
	document["n_projections"] = 2
	document["acquisition"] = {
		"n_projections": 3,
		"rotation_end_deg": 180,
		"angle_source": {
			"kind": "hardware_timestamps",
			"path": "scanlog.txt",
		},
	}
	document["projections"] = {
		"format": "tiff_stack",
		"path": "projections",
	}
	write_yaml(scan_path, document)

	scan, _ = load_scan(scan_path)

	assert scan.projections.path == (tmp_path / "projections").resolve()
	assert scan.acquisition.angle_source.path == (tmp_path / "scanlog.txt").resolve()


def scan_document(data_path):
	return {
		"energy_keV": 20,
		"pixel_size_um": 1,
		"ssd_cm": 10,
		"n_projections": 3,
		"degrees_per_fov": 180,
		"detector": {"name": "camera-a", "pixel_size_um": 1, "rows": 2, "columns": 4},
		"acquisition": {"n_projections": 3, "rotation_end_deg": 180},
		"projections": {
			"format": "hdf5_dxchange",
			"path": data_path,
			"data_path": "exchange/data",
			"white_path": "exchange/data_white",
			"dark_path": "exchange/data_dark",
		},
	}


def calibration_manifest(tmp_path, *, detector_id="camera-a", corrupt_hash=False):
	components = {}
	for index, name in enumerate(("pre_gain", "post_gain", "pre_dark", "post_dark")):
		path = tmp_path / f"{name}.npy"
		np.save(path, np.full((4, 4), index + 10, dtype=np.float32))
		payload_hash = "0" * 64 if corrupt_hash and name == "pre_gain" else stream_sha256(path)
		components[name] = CalibrationComponent(
			kind="dark" if "dark" in name else "detector_static",
			coordinate_frame="detector",
			payload=ArrayPayload(
				path=path.name, format="npy", sha256=payload_hash,
				shape=(4, 4), dtype="float32", axes=("Y", "X"),
			),
		)
	manifest = CalibrationManifest(
		calibration_id="test-calibration",
		producer=ProducerInfo(package="lftomo", version="test"),
		compatibility=CompatibilityMetadata(
			detector_id=detector_id,
			region=DetectorRegion(y_start=0, x_start=0, height=4, width=4),
			static_artifact_epoch="epoch-a",
		),
		components=components,
	)
	path = tmp_path / "calibration.yaml"
	path.write_text(manifest.to_yaml(), encoding="utf-8")
	return path


def normalization_recon(**params):
	return Recon.model_validate({
		"pipeline": [
			{"kind": "read", "algorithm": "hdf5_dxchange", "params": {
				"order": "projection", "y_start": 1, "y_stop": 3,
				"calibration_policy": "required",
			}},
			{"kind": "normalize", "algorithm": "weighted", "params": params},
		],
	})


def resolve_test_calibration(scan, recon, **kwargs):
	read = next((stage for stage in recon.pipeline if stage.kind == "read"), None)
	rows = scan.projection_shape().Y
	window = TomoWindow.from_bounds(
		scan.projection_shape(),
		y=(
			0 if read is None else int(read.params.get("y_start", 0)),
			rows if read is None or read.params.get("y_stop") is None
			else int(read.params["y_stop"]),
		),
	)
	return resolve_calibration(scan, recon, source_window=window, **kwargs)


def hdf5_scan(tmp_path, *, detector_id="camera-a"):
	data_path = tmp_path / "scan.h5"
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=np.ones((3, 4, 4), dtype=np.float32))
		handle.create_dataset("exchange/data_white", data=np.full((5, 4, 4), 100, dtype=np.float32))
		handle.create_dataset("exchange/data_dark", data=np.full((5, 4, 4), 2, dtype=np.float32))
	document = scan_document(data_path)
	document["detector"]["rows"] = 4
	document["metadata"] = {
		"detector_id": detector_id,
		"static_artifact_epoch": "epoch-a",
	}
	return Scan.model_validate(document)


def test_hdf5_probe_resolves_detector_before_canonical_shape_bridge(tmp_path):
	data_path = tmp_path / "shape-only.h5"
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", shape=(3, 5, 7), dtype=np.uint16)
	document = scan_document(data_path)
	document["detector"].pop("rows")
	document["detector"].pop("columns")

	assert _detector_shape(Scan.model_validate(document)) == (5, 7)


def test_all_bundled_canonical_templates_instantiate():
	root = Path(__file__).parents[1] / "lftomo" / "pipelines"
	for path in root.glob("*.yaml"):
		recon, _ = load_recon(path)
		stages = [create_stage(config) for config in recon.pipeline]
		assert stages, path.name


def test_empty_pipeline_reports_complete_pipeline_contract(tmp_path):
	from lftomo.pipeline import CompletePipelineError

	scan_path = tmp_path / "scan.yaml"
	write_yaml(scan_path, scan_document("scan.h5"))
	pipeline_path = tmp_path / "pipeline.yaml"
	write_yaml(pipeline_path, {"pipeline": []})
	with pytest.raises(CompletePipelineError, match="must begin"):
		run_reconstruction(pipeline_path, scan_path)


def test_runner_executes_hdf5_to_tiff_and_resolves_relative_scan_path(
	tmp_path, monkeypatch,
):
	import lftomo.runner as runner_module

	plans, checked_peaks, reservations, loaded_scans, planned_scans = [], [], [], [], []
	validate = runner_module.validate_pipeline
	check_capacity = runner_module.SharedNPManager.check_capacity
	reserve = runner_module.SharedNPManager.reserve
	def record_scan(path):
		loaded = load_scan(path)
		loaded_scans.append(loaded[0])
		return loaded
	def record_plan(*args, **kwargs):
		plan = validate(*args, **kwargs)
		plans.append(plan)
		planned_scans.append(args[1])
		return plan
	def record_capacity(peak):
		checked_peaks.append(peak)
		check_capacity(peak)
	def record_reservation(manager, role, size):
		reservations.append((role, size))
		reserve(manager, role, size)
	monkeypatch.setattr(runner_module, "validate_pipeline", record_plan)
	monkeypatch.setattr(runner_module, "load_scan", record_scan)
	monkeypatch.setattr(runner_module.SharedNPManager, "check_capacity", staticmethod(record_capacity))
	monkeypatch.setattr(runner_module.SharedNPManager, "reserve", record_reservation)
	data_path = tmp_path / "scan.h5"
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=np.arange(24, dtype=np.float32).reshape(3, 2, 4))
		handle.create_dataset("exchange/data_white", data=np.ones((2, 2, 4), dtype=np.float32))
		handle.create_dataset("exchange/data_dark", data=np.zeros((2, 2, 4), dtype=np.float32))
	scan_path = tmp_path / "scan.yaml"
	write_yaml(scan_path, scan_document("scan.h5"))
	pipeline_path = tmp_path / "pipeline.yaml"
	write_yaml(pipeline_path, {
		"pipeline": [
			{"kind": "read", "algorithm": "hdf5_dxchange", "params": {"order": "sinogram", "calibration_policy": "required"}},
			{"kind": "normalize", "algorithm": "weighted"},
			{"kind": "phase", "algorithm": "paganin"},
			{"kind": "center", "algorithm": "manual", "params": {"value_px": 2}},
			{"kind": "recon", "algorithm": "gridrec", "params": {"circle_mask": False}},
			{"kind": "write", "algorithm": "tiff_slices", "params": {"naming_template": "slice_{slice:03d}.tif"}},
		],
	})

	import tomopy
	monkeypatch.setattr(tomopy, "recon", lambda data, *args, **kwargs:
		np.repeat(np.asarray(data).mean(axis=1)[:, None, :], data.shape[2], axis=1))
	output = tmp_path / "output"
	scan_input = resolve_scan(scan_path)
	assert resolve_scan(scan_input) is scan_input
	scan_path.unlink()
	results = run_reconstruction(pipeline_path, scan_input, output)

	assert len(loaded_scans) == 1
	assert planned_scans[0].detector is loaded_scans[0].detector
	assert planned_scans[0].metadata is not loaded_scans[0].metadata
	assert "flat_fields" not in loaded_scans[0].metadata
	assert len(results) == 1
	assert results[0].output_path == output
	assert [path.name for path in results[0].artifacts["output_files"]] == ["slice_000.tif", "slice_001.tif"]
	assert tf.imread(output / "slice_000.tif").shape == (4, 4)
	assert results[0].artifacts["calibration_resolution"]["source"] == "embedded_hdf5"
	assert "white_fields" not in results[0].artifacts
	assert "dark_fields" not in results[0].artifacts
	assert checked_peaks == [plans[0].shared_memory_peak_bytes]
	assert reservations == [(event.role, event.bytes) for event in plans[0].allocation_schedule]


def test_intermediate_capacity_is_advisory_and_report_matches_coexisting_files(
	tmp_path, monkeypatch,
):
	import lftomo.runner as runner_module
	from tests.fixtures.tier1 import flat_dark_frames, projection_stack

	pytest.importorskip("tomopy")
	size, theta = 8, 7
	flats, darks = flat_dark_frames(size=size, n_flats=2, gain=10000, dark_level=20)
	transmission = np.exp(-0.7 * projection_stack(size=size, n_projections=theta))
	counts = np.rint(darks[0] + transmission * (flats[0] - darks[0])).astype(np.uint16)
	# Avoid the upstream gridrec correction-table overread at power-of-two widths.
	columns = size - 1
	counts, flats, darks = (values[..., :columns] for values in (counts, flats, darks))
	data_path = tmp_path / "scan.h5"
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=counts)
		handle.create_dataset("exchange/data_white", data=flats)
		handle.create_dataset("exchange/data_dark", data=darks)
	document = scan_document(str(data_path))
	document["n_projections"] = document["acquisition"]["n_projections"] = theta
	document["detector"].update(rows=size, columns=columns)
	scan_path = tmp_path / "scan.yaml"
	write_yaml(scan_path, document)
	pipeline_path = tmp_path / "pipeline.yaml"
	write_yaml(pipeline_path, {"process_workers": 1, "pipeline": [
		{"kind": "read", "algorithm": "hdf5_dxchange"},
		{"kind": "normalize", "algorithm": "weighted"},
		{"kind": "phase", "algorithm": "paganin", "sweep": {
			"kind": "list", "params": {"slope": [1e-10, 2e-10]},
		}},
		{"kind": "recon", "algorithm": "gridrec", "params": {"center": columns / 2, "ncore": 1}},
		{"kind": "write", "algorithm": "tiff_slices"},
	]})
	checked = []
	def low_space(path):
		checked.append((path, tuple(root.rglob("*.npy")) if root.exists() else ()))
		return SimpleNamespace(free=0)
	monkeypatch.setattr(runner_module.shutil, "disk_usage", low_space)
	read = h5py.Dataset.__getitem__
	def check_before_read(dataset, *args, **kwargs):
		if dataset.name == "/exchange/data":
			assert checked, "intermediate storage was not reported before reading projections"
		return read(dataset, *args, **kwargs)
	monkeypatch.setattr(h5py.Dataset, "__getitem__", check_before_read)
	root = tmp_path / "intermediates" / "nested"
	output = tmp_path / "output"
	cli_result = CliRunner().invoke(main, [
		str(pipeline_path), str(scan_path), "--output", str(output),
		"--window-rows", "4", "--window-theta", "3", "--intermediate-root", str(root),
		"--retain-intermediates", "--json-output",
	])
	assert cli_result.exit_code == 0, cli_result.output
	results = json.loads(cli_result.stdout)
	assert len(results) == len(checked) == 2
	assert checked[0] == (tmp_path, ())
	assert checked[1][0] == root
	assert set(checked[1][1]) == {
		Path(item["path"]) for item in results[0]["artifacts"]["intermediates"]
	}
	for result in results:
		report = result["artifacts"]["intermediate_storage"]
		assert report["root"] == str(root.resolve())
		assert report["available_bytes"] == 0
		files = [Path(item["path"]) for item in result["artifacts"]["intermediates"]]
		assert len(files) == len(report["boundaries"]) == 2
		for boundary, path in zip(report["boundaries"], files, strict=True):
			assert boundary["estimated_file_bytes"] == path.stat().st_size
			assert boundary["payload_bytes"] == np.load(path, mmap_mode="r").nbytes
		assert report["estimated_peak_bytes"] == sum(path.stat().st_size for path in files)
		manifest = yaml.safe_load((Path(result["output_path"]) / "recon_run.yaml").read_text())
		assert manifest["intermediate_storage"] == report
		assert manifest["resolved_recon"]["metadata"]["intermediate_storage"] == report
		assert len(result["artifacts"]["output_files"]) == size
		assert np.isfinite(tf.memmap(result["artifacts"]["output_files"][0], mode="r")).all()
	message = cli_result.stderr
	assert "advisory only, proceeding" in message
	assert str(root.resolve()) in message
	assert f"peak coexisting bytes={report['estimated_peak_bytes']}" in message


@pytest.mark.parametrize("window_rows, vo_chain, available_mib, fits", [
	(None, False, 64, False),
	(1, False, 32, False),
	(1, False, 64, True),
	(2, True, 64, False),
])
def test_shared_memory_preflight_uses_window_plan_before_read_or_manager(
	tmp_path, monkeypatch, window_rows, vo_chain, available_mib, fits,
):
	import lftomo.runner as runner_module
	import lftomo.util.mem as mem_module
	from lftomo.pipeline import WindowConfig

	# Reproduce the reported source dimensions without materializing its 22 GB
	# projection payload. Preflight needs the real HDF5 header, not pixel values.
	data_path = tmp_path / "b10.h5"
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", shape=(1969, 2160, 2560), dtype=np.uint16)
		handle.create_dataset("exchange/data_white", shape=(2, 2160, 2560), dtype=np.uint16, fillvalue=100)
		handle.create_dataset("exchange/data_dark", shape=(2, 2160, 2560), dtype=np.uint16)
	document = scan_document(str(data_path))
	document["n_projections"] = document["acquisition"]["n_projections"] = 1969
	document["detector"].update(rows=2160, columns=2560)
	scan_path = tmp_path / "scan.yaml"
	write_yaml(scan_path, document)
	recon, _ = load_recon("b10-recon-canonical")
	recon.pipeline[0].params.update(y_start=1076, y_stop=1086)
	if vo_chain:
		recon.pipeline[0].params.update(y_stop=1084, theta_stride=5)
		recon.pipeline[3:4] = [
			StageConfig(kind="stripe", algorithm="vo_sort", params={"size": 21}),
			StageConfig(kind="stripe", algorithm="vo_large", params={"snr": 3, "size": 51}),
		]
		recon.process_workers = 4
	pipeline_path = tmp_path / "pipeline.yaml"
	write_yaml(pipeline_path, recon.model_dump(mode="json"))

	available = available_mib * 1024**2
	monkeypatch.setattr(mem_module, "platform", "linux")
	statvfs = mem_module.os.statvfs
	def space(path):
		if path != "/dev/shm":
			return statvfs(path)
		return SimpleNamespace(f_bavail=available // 4096, f_frsize=4096)
	monkeypatch.setattr(mem_module.os, "statvfs", space)
	checked_peaks = []
	check_capacity = runner_module.SharedNPManager.check_capacity
	def record_capacity(peak):
		checked_peaks.append(peak)
		check_capacity(peak)
	monkeypatch.setattr(runner_module.SharedNPManager, "check_capacity", staticmethod(record_capacity))
	plans, window_counts = [], []
	validate = runner_module.validate_pipeline
	def record_plan(*args, **kwargs):
		plan = validate(*args, **kwargs)
		plans.append(plan)
		window_counts.append(len(kwargs["dataset"].frange))
		return plan
	monkeypatch.setattr(runner_module, "validate_pipeline", record_plan)

	read = h5py.Dataset.__getitem__
	def no_projection_read(dataset, *args, **kwargs):
		assert dataset.name != "/exchange/data", "projection data read before capacity check"
		return read(dataset, *args, **kwargs)
	monkeypatch.setattr(h5py.Dataset, "__getitem__", no_projection_read)
	class ReachedManager(Exception):
		pass
	def stop_before_manager(_context):
		raise ReachedManager
	monkeypatch.setattr(runner_module.PipelineRunContext, "__enter__", stop_before_manager)

	window_config = WindowConfig(rows=window_rows) if window_rows is not None else None
	if not fits:
		with pytest.raises(MemoryError) as error:
			run_reconstruction(pipeline_path, scan_path, window_config=window_config)
		peak, = checked_peaks
		assert peak > available
		message = str(error.value)
		assert f"planned peak {peak} bytes" in message
		assert f"available /dev/shm space {available} bytes" in message
		assert "shared memory" in message
		assert "--window-rows" in message and "--shm-size" in message
	else:
		with pytest.raises(ReachedManager):
			run_reconstruction(pipeline_path, scan_path, window_config=window_config)
		peak, = checked_peaks
		assert 0 < peak < available
	if window_rows is None:
		assert len(plans) == 1
		assert peak == plans[0].shared_memory_peak_bytes
	else:
		# BAC now uses the full selected detector extent in a Theta sweep.
		# Preflight covers all three sweep reservations, after complete validation.
		assert len(plans) == 4
		assert peak == runner_module.reserve_streaming_buffers(
			plans[1:], window_counts=window_counts[1:],
		)


def test_shared_memory_preflight_does_not_assume_linux_mount_on_other_platforms(monkeypatch):
	import lftomo.util.mem as mem_module

	monkeypatch.setattr(mem_module, "platform", "darwin")
	def unexpected_probe(_path):
		raise AssertionError("/dev/shm is Linux-specific")
	monkeypatch.setattr(mem_module.os, "statvfs", unexpected_probe)
	mem_module.SharedNPManager.check_capacity(1)


def test_external_bundle_overrides_hdf5_and_reads_only_y_window(tmp_path):
	manifest_path = calibration_manifest(tmp_path)
	scan = hdf5_scan(tmp_path)
	recon = normalization_recon(calibration_manifest=manifest_path.name)

	resolution = resolve_test_calibration(
		scan, recon, pipeline_root=tmp_path, scan_root=tmp_path,
	)

	assert resolution.source == "external_bundle"
	assert resolution.bundle_ref.calibration_id == "test-calibration"
	assert resolution.compatibility.compatible
	assert scan.metadata["flat_fields"].shape == (4, 2, 4)
	np.testing.assert_array_equal(scan.metadata["flat_fields"][:, 0, 0], [10, 11, 12, 13])
	read = next(stage for stage in recon.pipeline if stage.kind == "read")
	assert read.params["calibration_policy"] == "ignore"
	normalize = next(stage for stage in recon.pipeline if stage.kind == "normalize")
	assert "calibration_manifest" not in normalize.params


def test_direct_bundle_ref_and_yaml_must_identify_same_calibration(tmp_path):
	manifest_path = calibration_manifest(tmp_path)
	_, ref = load_manifest(manifest_path)
	scan = hdf5_scan(tmp_path)
	recon = normalization_recon(calibration_manifest=manifest_path)
	resolution = resolve_test_calibration(
		scan, recon, pipeline_root=tmp_path, scan_root=tmp_path, seeded_bundle=ref,
	)
	assert resolution.bundle_ref == ref

	other = CalibrationBundleRef(
		manifest_path=tmp_path / "other.yaml", calibration_id=ref.calibration_id,
	)
	with pytest.raises(ValueError, match="conflicts"):
		resolve_test_calibration(
			hdf5_scan(tmp_path), normalization_recon(calibration_manifest=manifest_path),
			pipeline_root=tmp_path, scan_root=tmp_path, seeded_bundle=other,
		)


def test_compatibility_mismatch_defaults_to_error_and_override_is_provenance(tmp_path):
	manifest_path = calibration_manifest(tmp_path)
	with pytest.raises(CalibrationCompatibilityError, match="detector_id"):
		resolve_test_calibration(
			hdf5_scan(tmp_path, detector_id="camera-b"),
			normalization_recon(calibration_manifest=manifest_path),
			pipeline_root=tmp_path, scan_root=tmp_path,
		)

	resolution = resolve_test_calibration(
		hdf5_scan(tmp_path, detector_id="camera-b"),
		normalization_recon(
			calibration_manifest=manifest_path,
			compatibility_policy="override",
			compatibility_override_reason="validated camera replacement",
		),
		pipeline_root=tmp_path, scan_root=tmp_path,
	)
	provenance = resolution.provenance()
	assert provenance["compatibility"]["override"]["reason"] == "validated camera replacement"
	assert provenance["compatibility"]["diagnostics"][0]["field"] == "detector_id"


def test_runner_reports_region_and_artifact_epoch_mismatches(tmp_path):
	manifest_path = calibration_manifest(tmp_path)
	scan = hdf5_scan(tmp_path)
	scan.detector.columns = 5
	scan.metadata["static_artifact_epoch"] = "epoch-b"

	with pytest.raises(CalibrationCompatibilityError) as caught:
		resolve_test_calibration(
			scan, normalization_recon(calibration_manifest=manifest_path),
			pipeline_root=tmp_path, scan_root=tmp_path,
		)

	assert {item.field for item in caught.value.report.diagnostics} == {
		"region", "static_artifact_epoch",
	}


def test_dynamic_normalizer_keeps_explicit_bundle_for_direct_stage_consumption(tmp_path):
	manifest_path = calibration_manifest(tmp_path)
	recon = Recon.model_validate({
		"pipeline": [{"kind": "normalize", "algorithm": "beam_tracking", "params": {
			"calibration_manifest": str(manifest_path),
		}}],
	})
	scan = hdf5_scan(tmp_path)

	resolution = resolve_test_calibration(
		scan, recon, pipeline_root=tmp_path, scan_root=tmp_path,
	)

	assert resolution.source == "external_bundle"
	assert resolution.bundle_ref is not None
	assert "flat_fields" not in scan.metadata
	assert "calibration_manifest" not in recon.pipeline[0].params


def test_explicit_legacy_maps_precede_embedded_hdf5(tmp_path):
	scan = hdf5_scan(tmp_path)
	scan.metadata["flat_fields"] = np.full((4, 2, 4), 7, dtype=np.float32)
	recon = normalization_recon()
	resolution = resolve_test_calibration(
		scan, recon, pipeline_root=tmp_path, scan_root=tmp_path,
	)
	assert resolution.source == "legacy_flat_fields"
	assert np.all(scan.metadata["flat_fields"] == 7)
	assert recon.pipeline[0].params["calibration_policy"] == "ignore"


def test_selected_bundle_is_integrity_checked_before_stages(tmp_path):
	manifest_path = calibration_manifest(tmp_path, corrupt_hash=True)
	with pytest.raises(ValueError, match="validation failed"):
		resolve_test_calibration(
			hdf5_scan(tmp_path), normalization_recon(calibration_manifest=manifest_path),
			pipeline_root=tmp_path, scan_root=tmp_path,
		)


def test_runner_seeds_typed_bundle_and_serializable_resolution(tmp_path, monkeypatch):
	manifest_path = calibration_manifest(tmp_path)
	_, bundle_ref = load_manifest(manifest_path)
	scan_path = tmp_path / "scan.yaml"
	scan_path.write_text(hdf5_scan(tmp_path).to_yaml(), encoding="utf-8")
	pipeline_path = tmp_path / "pipeline.yaml"
	write_yaml(pipeline_path, {
		"pipeline": [
			{"kind": "read", "algorithm": "hdf5_dxchange", "params": {
				"order": "projection", "y_start": 1, "y_stop": 3,
				"calibration_policy": "required",
			}},
			{"kind": "normalize", "algorithm": "weighted"},
		],
	})
	monkeypatch.setattr("lftomo.runner.Pipeline.run", lambda self, state: state)

	result = run_reconstruction(
		pipeline_path, scan_path, calibration_bundle=bundle_ref,
	)[0]

	assert result.artifacts["calibration_bundle"] == bundle_ref
	assert result.artifacts["calibration_compatibility"].compatible
	assert result.artifacts["calibration_resolution"]["source"] == "external_bundle"


def test_normalization_without_any_calibration_source_fails(tmp_path):
	projection = tmp_path / "projection.tif"
	tf.imwrite(projection, np.ones((4, 4), dtype=np.float32))
	scan = Scan.model_validate({
		**scan_document(projection),
		"projections": {
			"format": "tiff_stack", "path": projection, "pattern": "*.tif",
		},
	})
	with pytest.raises(ValueError, match="normalization requires"):
		resolve_test_calibration(
			scan,
			Recon.model_validate({
				"pipeline": [{"kind": "normalize", "algorithm": "weighted"}],
			}),
			pipeline_root=tmp_path,
			scan_root=tmp_path,
		)


def test_cli_emits_json_summary(tmp_path, monkeypatch):
	monkeypatch.setattr("lftomo.runner.run_reconstruction", lambda *args, **kwargs: [])
	scan_path = tmp_path / "scan.yaml"
	scan_path.write_text("placeholder: true\n", encoding="utf-8")
	pipeline_path = tmp_path / "pipeline.yaml"
	pipeline_path.write_text("pipeline: []\n", encoding="utf-8")

	result = CliRunner().invoke(main, [str(pipeline_path), str(scan_path), "--json-output"])

	assert result.exit_code == 0
	assert result.output.strip() == "[]"


def test_plain_runner_executes_algorithm_candidates_with_stable_outputs(tmp_path, monkeypatch):
	data_path = tmp_path / "scan.h5"
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=np.ones((3, 2, 4), dtype=np.float32))
		handle.create_dataset("exchange/data_white", data=np.full((2, 2, 4), 2, dtype=np.float32))
		handle.create_dataset("exchange/data_dark", data=np.zeros((2, 2, 4), dtype=np.float32))
	scan_path = tmp_path / "scan.yaml"
	write_yaml(scan_path, scan_document(str(data_path)))
	pipeline_path = tmp_path / "pipeline.yaml"
	write_yaml(pipeline_path, {
		"output_path": str(tmp_path / "output"),
		"pipeline": [
			{
				"id": "read",
				"kind": "read",
				"algorithm": "hdf5_dxchange",
				"params": {"order": "projection", "calibration_policy": "required"},
			},
			{"id": "normalize", "kind": "normalize", "algorithm": "weighted"},
			{
				"id": "stripe",
				"kind": "stripe",
				"sweep": {
					"kind": "cases",
					"cases": [
						{"name": "sort", "algorithm": "vo_sort", "params": {"size": 3}},
						{"name": "dead", "algorithm": "vo_dead", "params": {"size": 5, "snr": 2}},
					],
				},
			},
		],
	})
	monkeypatch.setattr("lftomo.runner.Pipeline.run", lambda self, state: state)

	results = run_reconstruction(pipeline_path, scan_path)

	assert len(results) == 2
	assert results[0].parameters["stage[stripe].algorithm"] == "vo_sort"
	assert results[1].parameters["stage[stripe].algorithm"] == "vo_dead"
	assert results[0].candidate_key != results[1].candidate_key
	assert results[0].output_path.name == f"candidate-{results[0].candidate_key[:16]}"
	assert results[1].output_path.name == f"candidate-{results[1].candidate_key[:16]}"
	assert all(
			stage.get("sweep") is None
			for result in results
			for stage in result.resolved_recon["pipeline"]
		)
	assert Recon.model_validate(results[0].resolved_recon)
