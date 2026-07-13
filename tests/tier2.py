"""Registry and layout checks for lab-internal Tier-2 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from lftomo.config import Scan


@dataclass(frozen=True)
class Tier2Dataset:
	"""A stable Tier-2 dataset location and its applicable producers."""

	name: str
	environment_variable: str
	default_path: Path
	producers: tuple[str, ...]


DATASETS = {
	"striping_tiff": Tier2Dataset(
		name="striping_tiff",
		environment_variable="LFTOMO_DATASET_STRIPING_TIFF",
		default_path=Path(
			"/gpfs/Influenza_APS_2022/Mouse_Lungs/AAA588_25.45kV/Scan/"
			"1651106162_77000projs_90000us_Mouse_Lung_PTA_Normal_Real_Scan"
		),
		producers=("r254", "recon_minimal"),
	),
	"phase_contrast_hdf5": Tier2Dataset(
		name="phase_contrast_hdf5",
		environment_variable="LFTOMO_DATASET_PHASE_CONTRAST_HDF5",
		default_path=Path("/mnt/wsl/PHYSICALDRIVE2p1/B10/PCRBox_Octo0/20kev/voi2.1"),
		producers=("phase_contrast", "r254", "b10_recon"),
	),
	"helical_tiff": Tier2Dataset(
		name="helical_tiff",
		environment_variable="LFTOMO_DATASET_HELICAL_TIFF",
		default_path=Path(
			"/gpfs/phenome/Argonne/2022_April/initial_setup/AAAPhantom_25.45kV/scan/"
			"1650859932_32000projs_60000us_Helical_Whole_Phantom_Scan"
		),
		producers=("r254",),
	),
}


def tier2_path(dataset: Tier2Dataset, environment: Mapping[str, str]) -> Path:
	"""Resolve a dataset path, allowing each runner to override its mount point."""

	return Path(environment.get(dataset.environment_variable, dataset.default_path)).expanduser()


def validate_canonical_layout(root: Path) -> Scan:
	"""Validate the canonical dataset and any populated reference archives."""

	if not root.is_dir():
		raise FileNotFoundError(f"Tier-2 dataset root does not exist: {root}")

	scan_path = root / "scan.yaml"
	if not scan_path.is_file():
		raise FileNotFoundError(f"missing canonical scan metadata: {scan_path}")
	scan = Scan.from_yaml(scan_path.read_text(encoding="utf-8"))

	projection_path = scan.projections.path
	if not projection_path.is_absolute():
		projection_path = root / projection_path
	if not projection_path.exists():
		raise FileNotFoundError(f"scan.yaml projection path does not exist: {projection_path}")

	for directory in (root / "flats", root / "reference"):
		if not directory.is_dir():
			raise FileNotFoundError(f"missing canonical directory: {directory}")

	for producer in (root / "reference").iterdir():
		if not producer.is_dir():
			continue
		if not any(producer.iterdir()):
			continue
		for relative in ("recon", "metadata.yaml", "pipeline.yaml", "recon.sha256"):
			if not (producer / relative).exists():
				raise FileNotFoundError(f"incomplete reference archive: {producer / relative}")

	return scan
