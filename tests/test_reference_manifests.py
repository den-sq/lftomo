from pathlib import Path

import yaml


MANIFEST_ROOT = Path(__file__).parent / "tier2_metadata" / "references"


def test_versioned_reference_manifests_are_complete():
	producer_dirs = sorted(path.parent for path in MANIFEST_ROOT.glob("*/*/metadata.yaml"))
	assert producer_dirs

	for producer_dir in producer_dirs:
		metadata = yaml.safe_load(producer_dir.joinpath("metadata.yaml").read_text())
		pipeline = yaml.safe_load(producer_dir.joinpath("pipeline.yaml").read_text())
		checksums = producer_dir.joinpath("recon.sha256").read_text().splitlines()

		assert metadata["immutable"] is True
		assert metadata["producer"] == pipeline["metadata"]["producer"]
		assert metadata["dataset"] == pipeline["metadata"]["dataset"]
		assert len(metadata["reference_selection"]["slices"]) == 10
		assert len(checksums) == 10
		assert all("  recon/" in line for line in checksums)
