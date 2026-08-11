from pathlib import Path
import json

import h5py
import numpy as np
import tifffile as tf
import yaml

from lftomo.acceptance import (
	AcceptanceProgressRecorder,
	ProcessTreeMemorySampler,
	_experiment_config,
	_memory_record_logger,
	_prepare_scan,
	_progress_observer,
	_run_interrupted_cleanup,
	_write_json,
	audit_pin_events,
	run_experiment_acceptance,
)
from lftomo.runner import load_scan
from lftomo.util.log import LOG, log


class RecordingLogger:
	def __init__(self):
		self.calls = []

	def write(self, step, statement, log_level=LOG.TIME):
		self.calls.append((step, statement, log_level))


def test_acceptance_progress_uses_shared_logger_and_persists_structured_event(tmp_path):
	logger = RecordingLogger()
	progress = AcceptanceProgressRecorder(tmp_path / "progress.jsonl", logger=logger)

	progress.emit("acceptance.started", "acceptance started", mode="windowed")

	record = json.loads(progress.path.read_text(encoding="utf-8"))
	assert record["event"] == "acceptance.started"
	assert record["details"] == {"mode": "windowed"}
	assert logger.calls == [("Acceptance", "acceptance started", LOG.TIME)]


def test_acceptance_stage_progress_names_the_independent_run_and_window(tmp_path):
	progress = AcceptanceProgressRecorder(
		tmp_path / "progress.jsonl", logger=RecordingLogger(),
	)
	observer = _progress_observer(
		progress,
		"windowed.current_oracle_reproduction",
		"current-oracle reproduction",
		y_start=1068,
		y_stop=1092,
		window_rows=12,
	)
	stage = type("Stage", (), {"name": "read.hdf5_dxchange"})()

	observer("started", 0, stage, None)
	observer("completed", 0, stage, None)
	observer("started", 0, stage, None)

	records = [json.loads(line) for line in progress.path.read_text().splitlines()]
	assert records[0]["message"].startswith(
		"current-oracle reproduction: window 1/2 [1068, 1080)"
	)
	assert records[0]["details"]["core_interval"] == [1068, 1080]
	assert records[2]["message"].startswith(
		"current-oracle reproduction: window 2/2 [1080, 1092)"
	)


def test_memory_sampler_records_each_stage_without_turning_it_into_a_proof():
	recorded = []
	sampler = ProcessTreeMemorySampler(
		y_start=1068, y_stop=1092, window_rows=12, on_record=recorded.append,
	)
	snapshots = iter((
		{1: 100},
		{1: 140, 2: 25},
		{1: 120, 2: 35},
	))
	sampler._snapshot = lambda: next(snapshots)
	stage = type("Stage", (), {"name": "read.test"})()

	sampler.observer("started", 0, stage, None)
	sampler._sample()
	sampler.observer("completed", 0, stage, None)

	assert sampler.records == recorded
	assert len(recorded) == 1
	observation = recorded[0]
	assert observation["stage"] == "read.test"
	assert observation["window_index"] == 0
	assert observation["core_interval"] == [1068, 1080]
	assert observation["dispatch_kind"] == "exempt"
	assert observation["outcome"] == "completed"
	assert observation["peak_process_private_delta_bytes"] == 40
	assert observation["peak_process_tree_private_delta_bytes"] == 65
	assert observation["peak_private_delta_bytes_by_pid"] == {"1": 40, "2": 35}
	assert "passed" not in observation


def test_stage_memory_observation_uses_shared_logger(tmp_path):
	logger = RecordingLogger()
	progress = AcceptanceProgressRecorder(tmp_path / "progress.jsonl", logger=logger)
	emit = _memory_record_logger(progress, "windowed.frozen", "frozen reproduction")

	emit({
		"window_index": 1,
		"core_interval": [1080, 1092],
		"stage": "recon.gridrec",
		"peak_process_tree_private_delta_bytes": 3 * 1024**2,
	})

	record = json.loads(progress.path.read_text(encoding="utf-8"))
	assert record["event"] == "windowed.frozen.window.stage.memory_observed"
	assert record["details"]["stage"] == "recon.gridrec"
	assert "3.0 MiB" in logger.calls[0][1]


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
		AcceptanceProgressRecorder(tmp_path / "progress.jsonl", logger=RecordingLogger()),
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
		log=AcceptanceProgressRecorder(tmp_path / "acceptance" / "progress.jsonl"),
		y_start=2,
		y_stop=6,
	)

	assert record["passed"], record
	assert set(record["proofs"]["shared_prefix_cache"]["statuses"]) == {"miss", "hit"}
	assert record["proofs"]["pins_honored"]["pin_events_present"]
	assert record["proofs"]["stable_fingerprints_and_no_cross_trial_contamination"]["passed"]
	record_path = tmp_path / "acceptance-record.json"
	_write_json(record_path, record)
	persisted = json.loads(record_path.read_text(encoding="utf-8"))
	assert persisted["proofs"]["concurrent_trials"]["attempt_intervals"][0][
		"finished_at"
	].endswith("+00:00")
