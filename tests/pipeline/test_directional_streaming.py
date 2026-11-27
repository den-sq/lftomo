"""Numerical and lifecycle contracts for changes of streaming direction."""

import h5py
import json
import numpy as np
import pytest
import tifffile as tf

from lftomo.pipeline import WindowConfig, WindowPlanError
from lftomo.runner import run_reconstruction
from tests.fixtures.tier1 import flat_dark_frames, projection_stack, shepp_logan
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
	# TomoPy 1.15.3 gridrec reads past its correction table at power-of-two
	# output widths. Keep the same phantom and physical center while excluding
	# its last detector column, so these checks isolate streaming behavior.
	columns = size - 1
	projections, flats, darks = (
		values[..., :columns] for values in (projections, flats, darks)
	)
	data_path = tmp_path / "scan.h5"
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=np.rint(projections).astype(np.uint16))
		handle.create_dataset("exchange/data_white", data=flats)
		handle.create_dataset("exchange/data_dark", data=darks)
	scan_path = tmp_path / "scan.yaml"
	_write_yaml(scan_path, {
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": theta, "degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": size, "columns": columns},
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


def test_bac_row_roi_matches_full_detector_and_persists_window_qc(tmp_path, directional_scan):
	pytest.importorskip("hotopy.holo")
	pipeline = _pipeline(tmp_path, "bac")
	full = _run(pipeline, directional_scan, tmp_path / "full", WindowConfig())
	pipeline = _pipeline(tmp_path, "bac", read_params={"y_start": 5, "y_stop": 11})
	for label, config in [("whole", WindowConfig()), ("theta", WindowConfig(rows=3, theta=4))]:
		output = tmp_path / label
		roi = _run(pipeline, directional_scan, output, config)
		assert roi[0] == full[0][5:11]
		np.testing.assert_allclose(roi[1], full[1][5:11], rtol=1e-6, atol=1e-6)
		qc = json.loads((output / "bac_qc.json").read_text())
		assert qc["complete"] is True
		counts = [item for window in qc["windows"] for item in window["projections"]]
		assert [item["projection_index"] for item in counts] == list(range(13))
		assert all(item["nonfinite_count"] == 0 for item in counts)


def test_bac_nonfinite_runner_failure_persists_qc_before_reconstruction(tmp_path, directional_scan):
	import yaml
	from lftomo.pipeline.stages.phase import BACNonFiniteError

	scan = yaml.safe_load(directional_scan.read_text())
	with h5py.File(scan["projections"]["path"], "r+") as handle:
		values = handle["exchange/data"][...].astype(np.float32)
		values[2, 4, 5] = np.nan
		del handle["exchange/data"]
		handle.create_dataset("exchange/data", data=values)
	pipeline = _pipeline(tmp_path, "bac")
	document = yaml.safe_load(pipeline.read_text())
	document["pipeline"][2] = {
		"kind": "phase", "algorithm": "bac_correction", "params": {"alpha": 1.6, "gamma": 1.2},
	}
	document["pipeline"].insert(3, {"kind": "phase", "algorithm": "negative_log"})
	_write_yaml(pipeline, document)
	output = tmp_path / "failed"
	with pytest.raises(BACNonFiniteError, match="projection 2"):
		_run(pipeline, directional_scan, output, WindowConfig(rows=3, theta=4))
	qc = json.loads((output / "bac_qc.json").read_text())
	assert qc["complete"] is False
	counts = qc["windows"][0]["projections"]
	assert [item["projection_index"] for item in counts] == [0, 1, 2, 3]
	assert [item["nonfinite_count"] for item in counts] == [0, 0, 16 * 15, 0]
	assert list(output.glob("slice_*.tif")) == []


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


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("rows, slope", [(48, 1e-9), (96, 1e-12)])
def test_phase_row_roi_uses_detector_context_in_whole_and_theta_runs(
	tmp_path, monkeypatch, affine, rows, slope,
):
	"""A requested row interval is an output ROI, not a new detector edge."""
	tomopy = pytest.importorskip("tomopy")
	columns, theta_count = 31, 7
	y_start, y_stop = rows // 2 - 4, rows // 2 + 4
	y = np.linspace(-1, 1, rows, dtype=np.float32)[:, None, None]
	volume = (0.4 + 0.5 * np.exp(-((y - 0.35) / 0.2) ** 2)) * shepp_logan(columns)
	angles = np.linspace(0, np.pi, theta_count, dtype=np.float32)
	line_integrals = tomopy.project(volume, angles, pad=False, emission=True, ncore=1)
	transmission = np.exp(-0.04 * line_integrals)
	flat = 10000 * np.linspace(0.9, 1.1, rows, dtype=np.float32)[:, None] * np.ones((1, columns))
	dark = np.full((rows, columns), 20, dtype=np.float32)
	counts = np.rint(dark + transmission * (flat - dark)).astype(np.uint16)
	data_path = tmp_path / "roi.h5"
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=counts)
		handle.create_dataset("exchange/data_white", data=np.stack([flat, flat]))
		handle.create_dataset("exchange/data_dark", data=np.stack([dark, dark]))
	scan_path = tmp_path / "roi-scan.yaml"
	_write_yaml(scan_path, {
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": theta_count, "degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": rows, "columns": columns},
		"acquisition": {"n_projections": theta_count, "rotation_end_deg": 180},
		"projections": {"format": "hdf5_dxchange", "path": str(data_path)},
	})
	phase_params = {"slope": slope, "fft_pad": True}
	if affine:
		envelope = 40e-6 * (
			1 + 0.15 * np.cos(np.linspace(-np.pi, np.pi, rows))[:, None]
			+ 0.1 * np.sin(np.linspace(-np.pi, np.pi, columns))[None, :]
		)
		phase_params.update(material_alpha=5e-8, envelope_m=envelope.tolist())
	captured = []
	original_recon = tomopy.recon
	def capture_phase_input(data, angles, **kwargs):
		values = data if kwargs["sinogram_order"] else data.transpose(1, 0, 2)
		captured.append(values.copy())
		return original_recon(data, angles, **kwargs)
	monkeypatch.setattr(tomopy, "recon", capture_phase_input)
	results = []
	for name, selection, config in (
		("full", {}, WindowConfig()),
		("roi-whole", {"y_start": y_start, "y_stop": y_stop}, WindowConfig()),
		("roi-theta", {"y_start": y_start, "y_stop": y_stop}, WindowConfig(
			rows=4, theta=3, intermediate_root=tmp_path / "intermediates",
		)),
	):
		pipeline = tmp_path / f"{name}.yaml"
		_write_yaml(pipeline, {"process_workers": 1, "pipeline": [
			{"kind": "read", "algorithm": "hdf5_dxchange", "params": selection},
			{"kind": "normalize", "algorithm": "weighted"},
			{"kind": "phase", "algorithm": "paganin", "params": phase_params},
			{"kind": "recon", "algorithm": "gridrec", "params": {"center": columns / 2, "ncore": 1}},
			{"kind": "write", "algorithm": "tiff_slices", "params": {
				"naming_template": "slice_{slice:03d}.tif",
			}},
		]})
		captured.clear()
		_run(pipeline, scan_path, tmp_path / name, config)
		results.append(np.concatenate(captured, axis=0))
	assert np.ptp(results[0][y_start:y_stop]) > 0.01
	for result in results[1:]:
		# The weak-slope case retains sampled-FFT tails beyond the continuum
		# decay length. Both cases must share measured rows and detector edges.
		np.testing.assert_allclose(result, results[0][y_start:y_stop], rtol=1e-6, atol=1e-6)


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
