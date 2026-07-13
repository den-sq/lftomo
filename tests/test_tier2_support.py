"""Unit tests for Tier-2 path resolution and layout validation."""

from pathlib import Path

import pytest

from lftomo.config import Scan
from tests.tier2 import DATASETS, tier2_path, validate_canonical_layout


SCAN_YAML = """\
energy_keV: 20.0
pixel_size_um: 0.637
ssd_cm: 2.0
n_projections: 1969
degrees_per_fov: 180.0
n_fov: 1
helical:
  enabled: false
detector:
  pixel_size_um: 0.637
  rows: 2160
  columns: 2560
  name: PCO Edge
acquisition:
  n_projections: 1969
  rotation_start_deg: 0.0
  rotation_end_deg: 180.0
  exposure_ms: 200.0
projections:
  format: hdf5_dxchange
  path: projections/data.h5
metadata:
  tier: 2
  dataset: phase_contrast_hdf5
"""


def _canonical_dataset(tmp_path: Path) -> Path:
	(tmp_path / "projections").mkdir()
	(tmp_path / "projections" / "data.h5").touch()
	(tmp_path / "flats").mkdir()
	(tmp_path / "reference" / "phase_contrast").mkdir(parents=True)
	(tmp_path / "scan.yaml").write_text(SCAN_YAML, encoding="utf-8")
	return tmp_path


def test_tier2_path_honors_mount_override(tmp_path):
	dataset = DATASETS["striping_tiff"]
	assert tier2_path(dataset, {dataset.environment_variable: str(tmp_path)}) == tmp_path


@pytest.mark.parametrize("dataset_name", ["striping_tiff", "helical_tiff"])
def test_network_dataset_defaults_use_z_drive_mount(dataset_name):
	assert DATASETS[dataset_name].default_path.is_relative_to("/mnt/z/gpfs")


@pytest.mark.parametrize("dataset_name", ["striping_tiff", "helical_tiff"])
def test_argonne_tiff_pixel_size(dataset_name):
	assert DATASETS[dataset_name].pixel_size_um == pytest.approx(0.7046)


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_installed_scan_template_validates(dataset_name):
	template = Path(__file__).with_name("tier2_metadata") / f"{dataset_name}.yaml"
	scan = Scan.from_yaml(template.read_text(encoding="utf-8"))
	assert scan.metadata["tier"] == 2
	assert scan.metadata["dataset"] == dataset_name
	assert scan.pixel_size_um == pytest.approx(DATASETS[dataset_name].pixel_size_um)


def test_validate_canonical_layout_accepts_empty_reference_slot(tmp_path):
	root = _canonical_dataset(tmp_path)
	assert validate_canonical_layout(root).metadata["tier"] == 2


def test_validate_canonical_layout_rejects_partial_reference(tmp_path):
	root = _canonical_dataset(tmp_path)
	(root / "reference" / "phase_contrast" / "metadata.yaml").touch()
	with pytest.raises(FileNotFoundError, match="incomplete reference archive"):
		validate_canonical_layout(root)
