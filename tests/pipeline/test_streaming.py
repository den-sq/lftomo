from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile as tf
import yaml

from lftomo.config import Acquisition, Detector, Scan, TIFFStackProjections
from lftomo.pipeline import (
	Stage,
	StageContract,
	WindowConfig,
	WindowPlanError,
	Y,
	resolve_streaming_plan,
)
from lftomo.pipeline.stages import (
	BilateralFilterStage,
	HelicalIndexRemapStage,
	MovementCorrectStage,
	NegativeLogStage,
	PaganinFilterStage,
	TIFFStackReadStage,
)
from lftomo.prep.phase import _phase_pad
from lftomo.runner import run_reconstruction
from lftomo.util.shapes import ProjOrder, Y as CanonicalY
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


def test_pipeline_exports_the_canonical_y_geometry_type():
	assert Y is CanonicalY


def test_pad_quantities_are_separate_and_neighborhood_stages_compose():
	stages = [
		BilateralFilterStage(diameter=5),
		BilateralFilterStage(diameter=5),
		MovementCorrectStage(y=3),
	]
	plan = resolve_streaming_plan(
		stages, _scan(), WindowConfig(rows=3, extra_pad=Y(5)),
	)
	assert plan.window_rows == 3
	assert plan.required_pad == ProjOrder(Theta=0, Y=7, X=4)
	assert plan.extra_pad == ProjOrder(Theta=0, Y=5, X=0)
	assert plan.effective_pad == ProjOrder(Theta=0, Y=7, X=4)

	user_dominates = resolve_streaming_plan(
		stages, _scan(), WindowConfig(rows=3, extra_pad=Y(10)),
	)
	assert user_dominates.required_pad.Y == 7
	assert user_dominates.effective_pad.Y == 10


def test_phase_pad_uses_the_canonical_phase_rule_and_warns_about_extent():
	scan = _scan(rows=32)
	plan = resolve_streaming_plan(
		[PaganinFilterStage(), NegativeLogStage()], scan, WindowConfig(rows=8),
	)
	assert plan.required_pad.Y == _phase_pad(8, 1e-4, 10, 20)
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
	first, second = plan.windows[:2]
	first_transition, second_transition = plan.transitions[:2]
	assert first.dataset is plan.dataset
	assert first.pos == ProjOrder(Theta=0, Y=0, X=0)
	assert first.shape == ProjOrder(Theta=4, Y=3, X=8)
	assert first.source_pos == ProjOrder(Theta=0, Y=0, X=0)
	assert first.source_shape == ProjOrder(Theta=4, Y=5, X=8)
	assert first.internal.pos == ProjOrder(Theta=0, Y=0, X=0)
	assert second.internal.pos.Y == 2
	assert first_transition.retained is None
	assert first_transition.fresh == ((0, 5),)
	assert second_transition.adjacent
	assert second_transition.retained == (1, 5)
	assert second_transition.fresh == ((5, 8),)
	assert second_transition.prefix_valid is False


def test_canonical_windows_clip_to_nonzero_read_selection_on_both_edges():
	plan = resolve_streaming_plan(
		[
			TIFFStackReadStage(y_start=2, y_stop=9),
			BilateralFilterStage(diameter=5),
		],
		_scan(rows=10),
		WindowConfig(rows=3),
	)
	first, middle, last = plan.windows

	assert plan.selection == (2, 9)
	assert first.pos.Y == 2
	assert first.source_pos.Y == 2
	assert first.source_shape.Y == 5
	assert first.realized_pad_before.Y == 0
	assert first.realized_pad_after.Y == 2
	assert middle.pos.Y == 5
	assert middle.source_pos.Y == 3
	assert middle.source_shape.Y == 6
	assert middle.realized_pad_before.Y == 2
	assert middle.realized_pad_after.Y == 1
	assert last.pos.Y == 8
	assert last.source_pos.Y == 6
	assert last.source_shape.Y == 3
	assert last.realized_pad_before.Y == 2
	assert last.realized_pad_after.Y == 0


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
		{"kind": "geometry", "algorithm": "affine_skew_movement", "params": {
			"transition_to": "sinogram", "pad_x": 1, "skew_degrees": 0,
		}},
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
		np.repeat(np.asarray(data).mean(axis=1)[:, None, :], data.shape[2], axis=1))
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
