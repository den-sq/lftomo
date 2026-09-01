from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile as tf
import yaml

from lftomo.config import Acquisition, Detector, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	PadRequirement,
	Stage,
	StageContract,
	WindowConfig,
	WindowPlanError,
	resolve_streaming_plan,
)
from lftomo.pipeline.stages import (
	BilateralFilterStage,
	HelicalIndexRemapStage,
	MovementCorrectStage,
	NegativeLogStage,
	PaganinFilterStage,
	create_stage,
)
from lftomo.prep.phase import _phase_pad
from lftomo.runner import run_reconstruction
from lftomo.util.shapes import ProjOrder, Y
from lftomo.util.window import TomoDataSet, TomoWindow
from tests.fixtures.tier1 import projection_stack


def _scan(rows=12, columns=8, theta=4):
	return Scan(
		energy_keV=20, pixel_size_um=1, ssd_cm=10,
		n_projections=theta, degrees_per_fov=180,
		detector=Detector(pixel_size_um=1, rows=rows, columns=columns),
		acquisition=Acquisition(n_projections=theta, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


class _FullYStage(Stage):
	name = "test.full_y"
	contract = StageContract(requires_full_axes=("Y",))

	def run(self, state):
		return state


class _AsymmetricStage(Stage):
	name = "test.asymmetric"
	contract = StageContract(
		required_pad=PadRequirement(fixed=Y(1), fixed_after=Y(3)),
	)

	def run(self, state):
		return state


def test_stage_and_operator_padding_compose_into_the_canonical_dataset():
	stages = [
		BilateralFilterStage(diameter=5),
		BilateralFilterStage(diameter=5),
		MovementCorrectStage(y=3),
	]
	dataset = resolve_streaming_plan(
		stages, _scan(), WindowConfig(rows=3, extra_pad=Y(5)),
	)
	assert dataset.step == Y(3)
	assert dataset.pad_before == ProjOrder(Theta=0, Y=7, X=4)
	assert dataset.pad_after == ProjOrder(Theta=0, Y=7, X=4)

	user_dataset = resolve_streaming_plan(
		stages, _scan(), WindowConfig(rows=3, extra_pad=Y(10)),
	)
	assert user_dataset.pad_before.Y == 10
	assert user_dataset.pad_after.Y == 10


def test_asymmetric_stage_and_operator_requirements_compose_by_side():
	dataset = resolve_streaming_plan(
		[_AsymmetricStage()], _scan(),
		WindowConfig(rows=4, extra_pad=Y(2), extra_pad_after=Y(1)),
	)

	assert dataset.pad_before == ProjOrder(Theta=0, Y=2, X=0)
	assert dataset.pad_after == ProjOrder(Theta=0, Y=3, X=0)
	windows = tuple(dataset)
	assert windows[0].pad_before == ProjOrder(Theta=0, Y=0, X=0)
	assert windows[0].pad_after == ProjOrder(Theta=0, Y=3, X=0)
	assert windows[1].pad_before == ProjOrder(Theta=0, Y=2, X=0)
	assert windows[1].pad_after == ProjOrder(Theta=0, Y=3, X=0)
	record = dataset.to_document()
	assert TomoDataSet.from_document(record).to_document() == record


def test_phase_pad_uses_the_canonical_phase_rule_and_warns_about_extent(caplog):
	scan = _scan(rows=32)
	dataset = resolve_streaming_plan(
		[PaganinFilterStage(), NegativeLogStage()], scan, WindowConfig(rows=8),
	)
	assert dataset.pad_before.Y == _phase_pad(8, 1e-4, 10, 20)
	assert dataset.pad_after.Y == _phase_pad(8, 1e-4, 10, 20)
	assert "phase.paganin_filter" in caplog.text
	assert "whole-volume equivalence" in caplog.text


def test_full_axis_contract_rejects_silent_window_degradation():
	with pytest.raises(WindowPlanError, match="test.full_y.*full Y axis"):
		resolve_streaming_plan([_FullYStage()], _scan(), WindowConfig(rows=4))
	with pytest.raises(WindowPlanError, match="helical_correct.index_remap"):
		resolve_streaming_plan([
			HelicalIndexRemapStage(pitch_px_per_projection=0.1),
		], _scan(), WindowConfig(rows=4))


def test_canonical_dataset_reiterates_without_a_window_cache():
	dataset = resolve_streaming_plan(
		[BilateralFilterStage(diameter=5)], _scan(rows=10), WindowConfig(rows=3),
	)
	first_pass = tuple(dataset)
	second_pass = tuple(dataset)
	assert len(dataset.frange) == 4
	assert dataset.interval("Y") == Y.interval(0, 10)
	assert all(isinstance(window, TomoWindow) for window in first_pass)
	assert [window.index for window in first_pass] == [
		window.index for window in second_pass
	]


def test_streaming_windows_read_physical_neighbors_outside_selection(tmp_path):
	projection_dir = tmp_path / "projections"
	projection_dir.mkdir()
	for index in range(4):
		tf.imwrite(projection_dir / f"projection_{index}.tif", np.zeros((12, 8), dtype=np.uint16))
	scan = _scan(rows=12)
	scan.projections.path = projection_dir
	read = StageConfig(
		kind="read", algorithm="tiff_stack",
		params={"y_start": 2, "y_stop": 10},
	)
	stages = [create_stage(read), BilateralFilterStage(diameter=5)]
	dataset = resolve_streaming_plan(stages, scan, WindowConfig(rows=4))
	windows = tuple(dataset)

	assert dataset.physical_shape == ProjOrder(Theta=4, Y=12, X=8)
	assert dataset.interval("Y") == Y.interval(2, 10)
	assert dataset.step == Y(4)
	assert [window.index[1] for window in windows] == [
		slice(2, 6, 1), slice(6, 10, 1),
	]
	assert [window.with_pad(["Y"]).index[1] for window in windows] == [
		slice(0, 8, 1), slice(4, 12, 1),
	]


def _write_yaml(path: Path, value):
	path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_windowed_runner_matches_whole_non_fft_and_replays_center_prefix(
	tmp_path, monkeypatch,
):
	import lftomo.pipeline.stages.read as read_module
	import lftomo.runner as runner_module

	bound_datasets = []
	read_stage_ids = []
	original_bind = read_module._bind_source_state
	original_read = read_module.HDF5DxchangeReadStage.run
	original_planned_pipeline = runner_module._planned_pipeline
	planned_scans = []

	def record_bind(state, working, read_window):
		result = original_bind(state, working, read_window)
		bound_datasets.append((state.dataset, result.dataset))
		return result

	def record_read(stage, state):
		read_stage_ids.append(id(stage))
		return original_read(stage, state)

	def record_planned_pipeline(stages, scan, *args, **kwargs):
		planned_scans.append(scan)
		return original_planned_pipeline(stages, scan, *args, **kwargs)

	monkeypatch.setattr(read_module, "_bind_source_state", record_bind)
	monkeypatch.setattr(read_module.HDF5DxchangeReadStage, "run", record_read)
	monkeypatch.setattr(runner_module, "_planned_pipeline", record_planned_pipeline)
	data_path = tmp_path / "scan.h5"
	rows, theta, columns = 6, 3, 6
	projections = 20 + 20 * projection_stack(size=rows, n_projections=theta)
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=projections)
		handle.create_dataset("exchange/data_white", data=np.full((2, rows, columns), 100))
		handle.create_dataset("exchange/data_dark", data=np.zeros((2, rows, columns)))
	scan_path = tmp_path / "scan.yaml"
	_write_yaml(scan_path, {
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": theta, "degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": rows, "columns": columns},
		"acquisition": {"n_projections": theta, "rotation_end_deg": 180},
		"projections": {
			"format": "hdf5_dxchange", "path": str(data_path),
			"data_path": "exchange/data", "white_path": "exchange/data_white",
			"dark_path": "exchange/data_dark",
		},
	})
	pipeline_path = tmp_path / "pipeline.yaml"
	_write_yaml(pipeline_path, {"pipeline": [
		{"kind": "read", "algorithm": "hdf5_dxchange", "params": {
			"order": "projection", "calibration_policy": "required",
		}},
		{"kind": "normalize", "algorithm": "weighted"},
		{"kind": "filter", "algorithm": "bilateral", "params": {"diameter": 3}},
		{"kind": "phase", "algorithm": "negative_log"},
		{"kind": "center", "algorithm": "pc"},
		{"kind": "recon", "algorithm": "gridrec", "params": {"circle_mask": False}},
		{"kind": "write", "algorithm": "tiff_slices", "params": {
			"naming_template": "slice_{slice:03d}.tif",
		}},
	]})

	import tomopy

	monkeypatch.setattr(tomopy, "find_center_pc", lambda *args, **kwargs: 2.0)
	reconstruction_inputs = []

	def reconstruct(data, *args, **kwargs):
		reconstruction_inputs.append(tuple(data.shape))
		kwargs["init_recon"][...] = np.repeat(
			np.asarray(data).mean(axis=0)[:, None, :], data.shape[2], axis=1,
		)
		return kwargs["init_recon"]

	monkeypatch.setattr(tomopy, "recon", reconstruct)
	whole_output = tmp_path / "whole"
	window_output = tmp_path / "windowed"
	whole = run_reconstruction(pipeline_path, scan_path, whole_output)[0]
	planned_scans.clear()
	bound_datasets.clear()
	read_stage_ids.clear()
	windowed = run_reconstruction(
		pipeline_path, scan_path, window_output,
		window_config=WindowConfig(rows=2),
	)[0]
	assert len(bound_datasets) == 6
	assert len(set(read_stage_ids)) == 1
	assert len({id(before) for before, _after in bound_datasets}) == 1
	assert all(before is after for before, after in bound_datasets)
	assert len({id(scan) for scan in planned_scans}) == 1
	assert reconstruction_inputs == [(theta, rows, columns)] + [
		(theta, 2, columns), (theta, 2, columns), (theta, 2, columns),
	]

	for row in range(rows):
		np.testing.assert_allclose(
			tf.imread(window_output / f"slice_{row:03d}.tif"),
			tf.imread(whole_output / f"slice_{row:03d}.tif"),
			rtol=1e-6, atol=1e-6,
		)
	assert len(windowed.artifacts["window_results"]) == 3
	assert all(
		item["center_artifacts"]["center"] == 2.0
		for item in windowed.artifacts["window_results"]
	)
	restored_dataset = TomoDataSet.from_document(windowed.artifacts["streaming_dataset"])
	assert restored_dataset.interval("Y") == Y.interval(0, rows)
	assert restored_dataset.step == Y(2)
	assert windowed.artifacts["measured_shared_memory_bytes"] > 0
	manifest = yaml.safe_load((window_output / "recon_run.yaml").read_text())
	assert TomoDataSet.from_document(manifest["streaming_dataset"]).step == Y(2)
	assert "streaming_dataset" in windowed.resolved_recon["metadata"]
	assert "source_window" not in windowed.resolved_recon["metadata"]
