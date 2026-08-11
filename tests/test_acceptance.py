from pathlib import Path

import h5py
import numpy as np
import tifffile as tf
import yaml

from lftomo.acceptance import (
	AcceptanceLog,
	_experiment_config,
	_prepare_scan,
	_run_interrupted_cleanup,
	audit_pin_events,
	memory_proof,
	run_experiment_acceptance,
)
from lftomo.runner import load_scan
from lftomo.util.log import log


def test_memory_proof_keeps_the_three_components_independent():
	proof = memory_proof(
		{
			"shared_memory_bytes": 100,
			"per_worker_private_scratch_bytes": 20,
			"external_engine_temporary_bytes": 30,
		},
		{
			"shared_memory_bytes": 90,
			"per_worker_private_scratch_bytes": 21,
			"external_engine_temporary_bytes": 10,
		},
	)

	assert not proof["passed"]
	assert proof["components"]["shared_memory_bytes"]["within_bound"]
	assert not proof["components"]["per_worker_private_scratch_bytes"]["within_bound"]
	assert proof["components"]["external_engine_temporary_bytes"]["within_bound"]


def test_pin_audit_requires_balanced_pins_and_rejects_alias_reuse():
	valid = audit_pin_events([
		{"action": "pin", "name": "one", "alias": "cache"},
		{"action": "allocate", "name": "two", "alias": None},
		{"action": "unpin", "name": "one", "alias": "cache"},
	])
	invalid = audit_pin_events([
		{"action": "pin", "name": "one", "alias": "cache"},
		{"action": "alias", "name": "one", "alias": "retired:raw_staging"},
	])

	assert valid["passed"]
	assert not invalid["passed"]
	assert invalid["violations"][0]["reason"] == "alias reused pinned storage"
	assert invalid["unreleased"] == {"one": ["cache"]}


def test_acceptance_experiment_is_concurrent_whole_volume_and_preserves_d7():
	config = _experiment_config(y_start=2, y_stop=6, workers=2)

	assert config.resources.max_concurrency == 2
	assert config.windows[0].y_start == 2
	assert config.windows[0].y_stop == 6
	assert len(config.fidelities) == 1
	assert config.metadata["windowed_prefix_caching"] == "deferred to issue #184"
	assert config.candidate_space.pipeline[2].sweep.params["gamma"] == [1.1, 1.2]


def test_acceptance_scan_materializes_the_named_oracles_presubtracted_flats(tmp_path):
	data_path = tmp_path / "scan.h5"
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=np.ones((2, 3, 3), dtype=np.uint16))
	flat_root = tmp_path / "gains"
	flat_root.mkdir()
	for name, value in (
		("pregain", 12), ("postgain", 20), ("predark", 2), ("postdark", 5),
	):
		tf.imwrite(flat_root / f"{name}_median.tif", np.full((3, 3), value, dtype=np.uint16))
	(tmp_path / "scan.yaml").write_text(yaml.safe_dump({
		"energy_keV": 20,
		"pixel_size_um": 1,
		"ssd_cm": 10,
		"n_projections": 2,
		"degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": 3, "columns": 3},
		"acquisition": {"n_projections": 2, "rotation_end_deg": 180},
		"projections": {"format": "hdf5_dxchange", "path": str(data_path)},
	}, sort_keys=False), encoding="utf-8")

	scan_path, source, digest = _prepare_scan(
		tmp_path,
		tmp_path / "invocation",
		flat_root,
		"2" * 64,
		AcceptanceLog(tmp_path / "progress.jsonl"),
	)
	scan, _ = load_scan(scan_path)
	fields = [tf.imread(path) for path in scan.metadata["flat_fields"]]

	assert source == data_path
	assert digest == "2" * 64
	assert [float(field[0, 0]) for field in fields] == [10, 15, 2, 5]


def test_interrupted_owner_process_group_leaves_no_shared_segment(tmp_path):
	data_path = tmp_path / "scan.h5"
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=np.ones((3, 4, 4), dtype=np.uint16))
		handle.create_dataset("exchange/data_white", data=np.full((2, 4, 4), 2, dtype=np.uint16))
		handle.create_dataset("exchange/data_dark", data=np.zeros((2, 4, 4), dtype=np.uint16))
	scan_path = tmp_path / "scan.yaml"
	scan_path.write_text(yaml.safe_dump({
		"energy_keV": 20,
		"pixel_size_um": 1,
		"ssd_cm": 10,
		"n_projections": 3,
		"degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": 4, "columns": 4},
		"acquisition": {"n_projections": 3, "rotation_end_deg": 180},
		"projections": {
			"format": "hdf5_dxchange", "path": str(data_path),
			"data_path": "exchange/data", "white_path": "exchange/data_white",
			"dark_path": "exchange/data_dark",
		},
	}, sort_keys=False), encoding="utf-8")
	pipeline_path = tmp_path / "pipeline.yaml"
	pipeline_path.write_text(yaml.safe_dump({"pipeline": [
		{"kind": "read", "algorithm": "hdf5_dxchange", "params": {
			"order": "projection", "calibration_policy": "required",
		}},
		{"kind": "normalize", "algorithm": "weighted"},
		{"kind": "phase", "algorithm": "negative_log"},
		{"kind": "center", "algorithm": "manual", "params": {"value_px": 2}},
		{"kind": "recon", "algorithm": "gridrec", "params": {"circle_mask": False}},
		{"kind": "write", "algorithm": "tiff_slices"},
	]}, sort_keys=False), encoding="utf-8")

	proof = _run_interrupted_cleanup(
		pipeline_path,
		scan_path,
		tmp_path / "output",
		tmp_path / "proof",
		2,
		startup_timeout=30,
		cleanup_timeout=10,
	)

	assert proof["passed"], proof
	assert proof["inspected"]
	assert proof["child_returncode"] == -9


def test_whole_volume_acceptance_proves_cache_isolation_end_to_end(tmp_path, monkeypatch):
	log.set_logdir(tmp_path / "logs")
	data_path = tmp_path / "scan.h5"
	y, theta, x = np.indices((8, 4, 8))
	projections = np.transpose(0.5 + 0.01 * y + 0.005 * theta + 0.002 * x, (1, 0, 2))
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=projections.astype(np.float32))
		handle.create_dataset("exchange/data_white", data=np.full((2, 8, 8), 2, dtype=np.float32))
		handle.create_dataset("exchange/data_dark", data=np.zeros((2, 8, 8), dtype=np.float32))
	flat_paths = []
	for index, value in enumerate((2.0, 2.0, 0.0, 0.0)):
		path = tmp_path / f"flat-{index}.tif"
		tf.imwrite(path, np.full((8, 8), value, dtype=np.float32))
		flat_paths.append(str(path))
	scan_path = tmp_path / "scan.yaml"
	scan_path.write_text(yaml.safe_dump({
		"energy_keV": 20,
		"pixel_size_um": 1,
		"ssd_cm": 10,
		"n_projections": 4,
		"degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": 8, "columns": 8},
		"acquisition": {"n_projections": 4, "rotation_end_deg": 180},
		"projections": {
			"format": "hdf5_dxchange", "path": str(data_path),
			"data_path": "exchange/data", "white_path": "exchange/data_white",
			"dark_path": "exchange/data_dark",
		},
		"metadata": {
			"dataset_content_hash": "1" * 64,
			"dataset_id": "synthetic",
			"flat_fields": flat_paths,
		},
	}, sort_keys=False), encoding="utf-8")

	import tomopy

	def reconstruct(data, *_args, **_kwargs):
		return np.repeat(np.asarray(data).mean(axis=0)[:, None, :], data.shape[2], axis=1)

	monkeypatch.setattr(tomopy, "recon", reconstruct)
	monkeypatch.setattr(tomopy, "circ_mask", lambda data, **_kwargs: data)
	record = run_experiment_acceptance(
		tmp_path / "acceptance",
		scan_path=scan_path,
		source=data_path,
		workers=2,
		log=AcceptanceLog(tmp_path / "acceptance" / "progress.jsonl"),
		y_start=2,
		y_stop=6,
	)

	assert record["passed"], record
	assert set(record["proofs"]["shared_prefix_cache"]["statuses"]) == {"miss", "hit"}
	assert record["proofs"]["pins_honored"]["pin_events_present"]
	assert record["proofs"]["stable_fingerprints_and_no_cross_trial_contamination"]["passed"]
