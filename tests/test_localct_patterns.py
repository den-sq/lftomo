from pathlib import Path

import pytest

tf = pytest.importorskip("tifffile")
np = pytest.importorskip("numpy")

from lftomo.localct_patterns import (  # noqa: E402
	ReconstructionRequest,
	ReconstructionResult,
	RunSummary,
	check_tomopy_runtime,
	inspect_scan_folder,
	read_run_summary,
	write_run_summary,
)


def write_tiff(path: Path):
	tf.imwrite(path, np.arange(4, dtype=np.uint16).reshape(2, 2))


def test_scan_folder_inspection_uses_tiny_generated_tiffs(tmp_path):
	(root := tmp_path / "scan").mkdir()
	(root / "projections").mkdir()
	(root / "flats").mkdir()
	(root / "scan.yaml").write_text("sample: demo\n")
	write_tiff(root / "projections" / "projection_0001.tif")
	write_tiff(root / "projections" / "projection_0002.tif")
	write_tiff(root / "flats" / "pregain_flat.tif")
	write_tiff(root / "flats" / "dark_0001.tif")

	inspection = inspect_scan_folder(root)

	assert len(inspection.projections) == 2
	assert len(inspection.flats) == 1
	assert len(inspection.darks) == 1
	assert inspection.metadata_files == [root / "scan.yaml"]


def test_run_summary_json_roundtrip(tmp_path):
	request = ReconstructionRequest(
		session_path=tmp_path / "session",
		recon_config=tmp_path / "recon.yaml",
		output_path=tmp_path / "output",
		scan_config=tmp_path / "scan.yaml",
		metadata={"lineage": "localCT-pattern"},
	)
	result = ReconstructionResult(
		output_path=tmp_path / "output",
		artifacts={"slice": tmp_path / "output" / "slice_0001.tif"},
		metrics={"seconds": 1.25},
	)
	summary = RunSummary(request=request, result=result, metadata={"status": "demo"})

	path = tmp_path / "run-summary.json"
	write_run_summary(summary, path)

	assert read_run_summary(path) == summary


def test_tomopy_runtime_check_is_structured():
	runtime = check_tomopy_runtime()

	assert runtime.module == "tomopy"
	assert isinstance(runtime.available, bool)
	assert runtime.message
