"""Numerical and lifecycle contracts for changes of streaming direction."""

import h5py
import numpy as np
import pytest
import tifffile as tf

from lftomo.pipeline import WindowConfig, WindowPlanError
from lftomo.runner import run_reconstruction
from tests.fixtures.tier1 import flat_dark_frames, projection_stack
from tests.pipeline.test_streaming import _write_yaml


@pytest.fixture
def directional_scan(tmp_path):
	pytest.importorskip("tomopy")
	size, theta = 16, 13
	flats, darks = flat_dark_frames(size=size, n_flats=2, gain=10000, dark_level=20)
	# Form detector counts from positive transmission and measured-field units.
	# The existing phantom varies in Y, X and angle, exposing misplaced chunks.
	transmission = np.exp(-0.7 * projection_stack(size=size, n_projections=theta))
	projections = darks[0] + transmission * (flats[0] - darks[0])
	data_path = tmp_path / "scan.h5"
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=np.rint(projections).astype(np.uint16))
		handle.create_dataset("exchange/data_white", data=flats)
		handle.create_dataset("exchange/data_dark", data=darks)
	scan_path = tmp_path / "scan.yaml"
	_write_yaml(scan_path, {
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": theta, "degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": size, "columns": size},
		"acquisition": {"n_projections": theta, "rotation_end_deg": 180},
		"projections": {
			"format": "hdf5_dxchange", "path": str(data_path),
			"data_path": "exchange/data", "white_path": "exchange/data_white",
			"dark_path": "exchange/data_dark",
		},
	})
	return scan_path


def _pipeline(tmp_path, phase="slope", *, read_params=None, after_phase=()):
	if phase == "slope":
		phase_stage = {
			"kind": "phase", "algorithm": "paganin",
			"params": {"slope": 1e-10, "fft_pad": True},
		}
	elif phase == "bac":
		phase_stage = {
			"kind": "phase", "algorithm": "bac",
			"params": {"alpha": 2.0, "gamma": 1.25, "device": "cpu"},
		}
	else:
		phase_stage = {"kind": "phase", "algorithm": "negative_log"}
	path = tmp_path / "pipeline.yaml"
	_write_yaml(path, {"process_workers": 1, "pipeline": [
		{"kind": "read", "algorithm": "hdf5_dxchange", "params": {
			"order": "projection", "calibration_policy": "required", **(read_params or {}),
		}},
		{"kind": "normalize", "algorithm": "weighted"},
		phase_stage,
		*after_phase,
		{"kind": "stripe", "algorithm": "vo_sort", "params": {"size": 3}},
		{"kind": "recon", "algorithm": "gridrec", "params": {
			"center": 8.0, "ncore": 1,
		}},
		{"kind": "write", "algorithm": "tiff_slices", "params": {
			"naming_template": "slice_{slice:03d}.tif",
		}},
	]})
	return path


def _run(pipeline, scan, output, config):
	run_reconstruction(pipeline, scan, output, window_config=config)
	paths = sorted(output.glob("slice_*.tif"))
	return [path.name for path in paths], np.stack([tf.memmap(path, mode="r") for path in paths])


def _assert_same_reconstruction(actual, expected, rows):
	assert actual[0] == expected[0] == [f"slice_{row:03d}.tif" for row in rows]
	assert np.isfinite(actual[1]).all()
	assert np.ptp(expected[1]) > 0
	np.testing.assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("phase", ["slope", "bac"])
def test_phase_direction_changes_match_whole_reconstruction(tmp_path, directional_scan, phase):
	if phase == "bac":
		pytest.importorskip("hotopy.holo")
	pipeline = _pipeline(tmp_path, phase)
	whole = _run(pipeline, directional_scan, tmp_path / "whole", WindowConfig())
	windowed = _run(
		pipeline, directional_scan, tmp_path / "windowed", WindowConfig(rows=6, theta=4),
	)

	_assert_same_reconstruction(windowed, whole, range(16))


def test_x_windows_match_whole_pointwise_phase_reconstruction(tmp_path, directional_scan):
	pipeline = _pipeline(tmp_path, "negative_log")
	whole = _run(pipeline, directional_scan, tmp_path / "whole", WindowConfig())
	windowed = _run(
		pipeline, directional_scan, tmp_path / "windowed",
		WindowConfig(rows=6, columns=6, axis="X"),
	)

	_assert_same_reconstruction(windowed, whole, range(16))


def test_direction_changes_preserve_strided_angles_and_selected_row_offsets(tmp_path, directional_scan):
	pipeline = _pipeline(tmp_path, read_params={
		"y_start": 2, "y_stop": 11, "theta_start": 1, "theta_stop": 12, "theta_stride": 2,
	})
	whole = _run(pipeline, directional_scan, tmp_path / "whole", WindowConfig())
	windowed = _run(
		pipeline, directional_scan, tmp_path / "windowed", WindowConfig(rows=4, theta=4),
	)

	_assert_same_reconstruction(windowed, whole, range(2, 11))


def test_downstream_bilateral_reads_processed_neighbors_across_direction_changes(tmp_path, directional_scan):
	pipeline = _pipeline(
		tmp_path,
		after_phase=[{"kind": "filter", "algorithm": "bilateral", "params": {"diameter": 5}}],
	)
	whole = _run(pipeline, directional_scan, tmp_path / "whole", WindowConfig())
	windowed = _run(
		pipeline, directional_scan, tmp_path / "windowed", WindowConfig(rows=4, theta=4),
	)

	_assert_same_reconstruction(windowed, whole, range(16))


def test_automatic_center_replay_reloads_processed_phase_input(tmp_path, directional_scan, monkeypatch):
	monkeypatch.setattr("tomopy.find_center_pc", lambda *_args, **_kwargs: 8.0)
	pipeline = _pipeline(
		tmp_path, after_phase=[{"kind": "center", "algorithm": "pc"}],
	)
	whole = _run(pipeline, directional_scan, tmp_path / "whole", WindowConfig())
	windowed = _run(
		pipeline, directional_scan, tmp_path / "windowed", WindowConfig(rows=6, theta=4),
	)

	_assert_same_reconstruction(windowed, whole, range(16))


def test_direction_boundary_rejects_geometry_that_changes_after_first_theta_window(
	tmp_path, directional_scan, monkeypatch,
):
	import lftomo.runner as runner_module

	pipeline = _pipeline(tmp_path, "negative_log", after_phase=[{
		"kind": "helical_correct", "algorithm": "index_remap",
		"params": {"pitch_px_per_projection": 0.1, "crop_valid": True},
	}])
	def unexpected_manager(_context):
		raise AssertionError("invalid geometry reached storage allocation")
	monkeypatch.setattr(runner_module.PipelineRunContext, "__enter__", unexpected_manager)
	with pytest.raises(WindowPlanError, match="changes the materialized shape"):
		run_reconstruction(
			pipeline, directional_scan, window_config=WindowConfig(rows=4, theta=1),
		)


def test_intermediates_clean_up_by_default_and_retained_runs_start_fresh(tmp_path, directional_scan):
	pipeline = _pipeline(tmp_path)
	intermediate_root = tmp_path / "intermediates"
	clean = _run(
		pipeline, directional_scan, tmp_path / "clean",
		WindowConfig(rows=6, theta=4, intermediate_root=intermediate_root),
	)
	assert not list(intermediate_root.iterdir())
	retained_config = WindowConfig(
		rows=6, theta=4, intermediate_root=intermediate_root, retain_intermediates=True,
	)
	retained = _run(pipeline, directional_scan, tmp_path / "retained", retained_config)
	first_directories = set(intermediate_root.iterdir())
	assert first_directories
	retained_arrays = list(intermediate_root.rglob("*.npy"))
	assert retained_arrays
	_assert_same_reconstruction(retained, clean, range(16))
	# Retention is inspectable output, not an implicit restart input. Poison the
	# previous run's arrays so any stale reuse corrupts the next reconstruction.
	for path in retained_arrays:
		values = np.load(path, mmap_mode="r+")
		values[...] = np.nan
		values.flush()
		del values
	rerun = _run(pipeline, directional_scan, tmp_path / "rerun", retained_config)
	assert first_directories < set(intermediate_root.iterdir())
	_assert_same_reconstruction(rerun, clean, range(16))
