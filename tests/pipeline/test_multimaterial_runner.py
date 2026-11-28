"""Required phase diagnostics survive the real runner and its JSON boundary."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from lftomo.config import Recon, Scan
from lftomo.pipeline import WindowConfig
from lftomo.runner import _json_result, run_reconstruction
from lftomo.util.shapes import Y
from tests.fixtures.phase_multimaterial import compact_multimaterial_case


@pytest.mark.system
def test_runner_retains_reference_diagnostics_for_every_window_and_json(tmp_path, caplog):
	case = compact_multimaterial_case(n_rows=4, row_varying=True, noncollinear=True)
	projections = case["projections"]
	theta, rows, columns = projections.shape
	source = tmp_path / "scan.h5"
	with h5py.File(source, "w") as handle:
		handle.create_dataset("exchange/data", data=projections)
		# Identity calibration preserves the simulated transmission exactly.
		handle.create_dataset("exchange/data_white", data=np.ones((2, rows, columns), np.float32))
		handle.create_dataset("exchange/data_dark", data=np.zeros((2, rows, columns), np.float32))
	scan = Scan.model_validate({
		"energy_keV": case["energy_kev"], "pixel_size_um": case["pixel_m"] * 1e6,
		"ssd_cm": case["distance_m"] * 100, "n_projections": theta, "degrees_per_fov": 180,
		"detector": {"pixel_size_um": case["pixel_m"] * 1e6, "rows": rows, "columns": columns},
		"acquisition": {"n_projections": theta, "rotation_end_deg": 180},
		"projections": {
			"format": "hdf5_dxchange", "path": source.name,
			"data_path": "exchange/data", "white_path": "exchange/data_white",
			"dark_path": "exchange/data_dark",
		},
	})
	scan_path = tmp_path / "scan.yaml"
	scan_path.write_text(scan.to_yaml(), encoding="utf-8")
	recon = Recon.model_validate({"pipeline": [
		{"kind": "read", "algorithm": "hdf5_dxchange", "params": {
			"order": "projection", "calibration_policy": "required",
		}},
		{"kind": "normalize", "algorithm": "weighted"},
		{"kind": "phase", "algorithm": "multimaterial", "params": {
			"mu_table": case["mu_table"].tolist(), "delta_table": case["delta_table"].tolist(),
			"n_iter": 2,
		}},
	]})
	pipeline_path = tmp_path / "pipeline.yaml"
	pipeline_path.write_text(recon.to_yaml(), encoding="utf-8")
	result, = run_reconstruction(
		pipeline_path, scan_path, window_config=WindowConfig(rows=2, extra_pad=Y(rows)),
	)
	assert "Extent-sensitive" in caplog.text
	assert "phase.multimaterial" in caplog.text
	assert "whole-volume equivalence" in caplog.text
	assert result.artifacts["calibration_resolution"]["source"] == "embedded_hdf5"
	window_results = result.artifacts["window_results"]
	assert len(window_results) == 2

	reference_path = Path(__file__).parents[1] / "fixtures" / "phase_multimaterial_reference.npz"
	with np.load(reference_path) as reference:
		for key, length in (("correction_norms", 2), ("label_change", 1)):
			expected = reference[f"compact_rows4_noncollinear_n2_{key}"]
			for record in window_results:
				history = record["artifacts"][key]
				assert len(history) == length
				np.testing.assert_allclose(history, expected, rtol=1e-6, atol=1e-12)
			assert result.artifacts[key] == window_results[-1]["artifacts"][key]
			assert window_results[0]["artifacts"][key] is not window_results[1]["artifacts"][key]
		assert np.any(reference["compact_rows4_noncollinear_n2_correction_norms"] > 0)

	# Exercise the same summary adapter used by the runner's --json-output.
	document = json.loads(json.dumps(_json_result(result), allow_nan=False))
	for actual, retained in zip(document["artifacts"]["window_results"], window_results, strict=True):
		for key in ("correction_norms", "label_change"):
			assert actual["artifacts"][key] == retained["artifacts"][key]
