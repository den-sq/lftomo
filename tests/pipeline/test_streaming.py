from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile as tf
import yaml

from lftomo.config import Acquisition, Detector, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	AxisPad,
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
from lftomo.util.window import TomoWindow
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


def test_pad_quantities_are_separate_and_neighborhood_stages_compose():
	stages = [
		BilateralFilterStage(diameter=5),
		BilateralFilterStage(diameter=5),
		MovementCorrectStage(y=3),
	]
	plan = resolve_streaming_plan(
		stages, _scan(), WindowConfig(rows=3, extra_pad=AxisPad(y=5)),
	)
	assert plan.window_rows == 3
	assert plan.required_pad == AxisPad(y=7, x=4)
	assert plan.extra_pad == AxisPad(y=5)
	assert plan.effective_pad == AxisPad(y=7, x=4)

	user_dominates = resolve_streaming_plan(
		stages, _scan(), WindowConfig(rows=3, extra_pad=AxisPad(y=10)),
	)
	assert user_dominates.required_pad.y == 7
	assert user_dominates.effective_pad.y == 10


def test_phase_pad_uses_the_canonical_phase_rule_and_warns_about_extent():
	scan = _scan(rows=32)
	plan = resolve_streaming_plan(
		[PaganinFilterStage(), NegativeLogStage()], scan, WindowConfig(rows=8),
	)
	assert plan.required_pad.y == _phase_pad(8, 1e-4, 10, 20)
	assert "phase.paganin_filter" in plan.warnings[0]
	assert "whole-volume equivalence" in plan.warnings[0]


def test_full_axis_contract_rejects_silent_window_degradation():
	with pytest.raises(WindowPlanError, match="test.full_y.*full Y axis"):
		resolve_streaming_plan([_FullYStage()], _scan(), WindowConfig(rows=4))
	with pytest.raises(WindowPlanError, match="helical_correct.index_remap"):
		resolve_streaming_plan([
			HelicalIndexRemapStage(pitch_px_per_projection=0.1),
		], _scan(), WindowConfig(rows=4))


def test_retained_fresh_transition_is_explicit_but_prefix_is_not_retained():
	plan = resolve_streaming_plan(
		[BilateralFilterStage(diameter=5)], _scan(rows=10), WindowConfig(rows=3),
	)
	assert all(isinstance(window, TomoWindow) for window in plan.windows)
	first, second = plan.transitions[:2]
	assert first.retained is None
	assert first.fresh == ((0, 5),)
	assert second.adjacent
	assert second.retained == (1, 5)
	assert second.fresh == ((5, 8),)
	assert second.prefix_valid is False


def test_streaming_windows_read_physical_neighbors_outside_selection():
	read = StageConfig(
		kind="read", algorithm="hdf5_dxchange",
		params={"y_start": 2, "y_stop": 10},
	)
	stages = [create_stage(read), BilateralFilterStage(diameter=5)]
	plan = resolve_streaming_plan(stages, _scan(rows=12), WindowConfig(rows=4))

	assert plan.dataset.physical_shape == ProjOrder(Theta=4, Y=12, X=8)
	assert plan.dataset.frange == Y.drange(2, 10, 4)
	assert [window.index[1] for window in plan.windows] == [
		slice(2, 6, 1), slice(6, 10, 1),
	]
	assert [window.with_pad(["Y"]).index[1] for window in plan.windows] == [
		slice(0, 8, 1), slice(4, 12, 1),
	]
	assert plan.record()["windows"] == [
		{
			"index": 0, "core": [2, 6], "source": [0, 8], "interior": [2, 6],
			"transition": {
				"adjacent": False, "retained": None, "fresh": ((0, 8),),
				"prefix_valid": False,
			},
		},
		{
			"index": 1, "core": [6, 10], "source": [4, 12], "interior": [2, 6],
			"transition": {
				"adjacent": True, "retained": (4, 8), "fresh": ((8, 12),),
				"prefix_valid": False,
			},
		},
	]


def _write_yaml(path: Path, value):
	path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_windowed_runner_matches_whole_non_fft_and_replays_center_prefix(
	tmp_path, monkeypatch,
):
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
	monkeypatch.setattr(tomopy, "recon", lambda data, *args, **kwargs:
		np.repeat(np.asarray(data).mean(axis=0)[:, None, :], data.shape[2], axis=1))
	whole_output = tmp_path / "whole"
	window_output = tmp_path / "windowed"
	whole = run_reconstruction(pipeline_path, scan_path, whole_output)[0]
	windowed = run_reconstruction(
		pipeline_path, scan_path, window_output,
		window_config=WindowConfig(rows=2),
	)[0]

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
	assert windowed.artifacts["window_plan"]["center_prepass"] == (
		"replay_processed_prefix_per_window"
	)
	assert windowed.artifacts["measured_shared_memory_bytes"] > 0
	manifest = yaml.safe_load((window_output / "recon_run.yaml").read_text())
	assert manifest["window_plan"]["window_rows"] == 2
	assert "resolved_window" in windowed.resolved_recon["metadata"]
