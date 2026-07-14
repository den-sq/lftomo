"""Canonical-layout checks executed on the Yarrkha Tier-2 runner."""

from pathlib import Path

import pytest

from tests.tier2 import DATASETS, validate_canonical_layout


pytestmark = pytest.mark.tier2


@pytest.mark.parametrize(
	"fixture_name,dataset_name",
	[
		("dataset_striping_tiff", "striping_tiff"),
		("dataset_phase_contrast_hdf5", "phase_contrast_hdf5"),
		("dataset_helical_tiff", "helical_tiff"),
	],
)
def test_tier2_dataset_layout(request, fixture_name: str, dataset_name: str):
	root: Path = request.getfixturevalue(fixture_name)
	scan = validate_canonical_layout(root)
	dataset = DATASETS[dataset_name]
	assert scan.metadata.get("tier") == 2
	assert scan.metadata.get("dataset") == dataset_name
	assert scan.pixel_size_um == pytest.approx(dataset.pixel_size_um)
	assert scan.detector.pixel_size_um == pytest.approx(dataset.pixel_size_um)


@pytest.mark.parametrize("dataset", DATASETS.values(), ids=lambda dataset: dataset.name)
def test_tier2_producer_directories(request, dataset):
	root: Path = request.getfixturevalue(f"dataset_{dataset.name}")
	validate_canonical_layout(root)
	for producer in dataset.producers:
		assert (root / "reference" / producer).is_dir(), f"missing {producer} reference slot"
