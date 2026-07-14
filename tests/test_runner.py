from pathlib import Path

import h5py
import numpy as np
import tifffile as tf
import yaml
from click.testing import CliRunner

from lftomo.pipeline.stages import create_stage
from lftomo.runner import _make_executable, load_recon, main, run_reconstruction


def write_yaml(path, value):
	path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def scan_document(data_path):
	return {
		"energy_keV": 20,
		"pixel_size_um": 1,
		"ssd_cm": 10,
		"n_projections": 3,
		"degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": 2, "columns": 4},
		"acquisition": {"n_projections": 3, "rotation_end_deg": 180},
		"projections": {
			"format": "hdf5_dxchange",
			"path": data_path,
			"data_path": "exchange/data",
			"white_path": "exchange/data_white",
			"dark_path": "exchange/data_dark",
		},
	}


def test_all_bundled_canonical_templates_instantiate():
	root = Path(__file__).parents[1] / "lftomo" / "pipelines"
	for path in root.glob("*.yaml"):
		recon, _ = load_recon(path)
		stages = [create_stage(config) for config in _make_executable(recon).pipeline]
		assert stages, path.name


def test_runner_executes_hdf5_to_tiff_and_resolves_relative_scan_path(tmp_path, monkeypatch):
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
			{"kind": "read", "algorithm": "hdf5_dxchange", "params": {"order": "sinogram", "calibration_policy": "ignore"}},
			{"kind": "center", "algorithm": "manual", "params": {"value_px": 2}},
			{"kind": "recon", "algorithm": "gridrec", "params": {"sinogram_order": True, "circle_mask": False}},
			{"kind": "write", "algorithm": "tiff_slices", "params": {"naming_template": "slice_{slice:03d}.tif"}},
		],
	})

	import tomopy
	monkeypatch.setattr(tomopy, "recon", lambda data, *args, **kwargs:
		np.repeat(np.asarray(data).mean(axis=1)[:, None, :], data.shape[2], axis=1))
	output = tmp_path / "output"
	results = run_reconstruction(pipeline_path, scan_path, output)

	assert len(results) == 1
	assert results[0].output_path == output
	assert [path.name for path in results[0].artifacts["output_files"]] == ["slice_000.tif", "slice_001.tif"]
	assert tf.imread(output / "slice_000.tif").shape == (4, 4)


def test_cli_emits_json_summary(tmp_path, monkeypatch):
	monkeypatch.setattr("lftomo.runner.run_reconstruction", lambda *args, **kwargs: [])
	scan_path = tmp_path / "scan.yaml"
	scan_path.write_text("placeholder: true\n", encoding="utf-8")
	pipeline_path = tmp_path / "pipeline.yaml"
	pipeline_path.write_text("pipeline: []\n", encoding="utf-8")

	result = CliRunner().invoke(main, [str(pipeline_path), str(scan_path), "--json-output"])

	assert result.exit_code == 0
	assert result.output.strip() == "[]"
