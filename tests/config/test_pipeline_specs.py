from pathlib import Path

import pytest

from lftomo.config import Recon


PIPELINES = Path(__file__).parents[2] / "lftomo" / "pipelines"

EXPECTED_ORDERS = {
	"r254-canonical.yaml": [
		"read", "normalize", "phase", "geometry", "stripe", "stripe", "center", "recon", "write",
	],
	"recon-minimal-canonical.yaml": ["read", "normalize", "stripe", "phase", "center", "recon", "write"],
	"phase-contrast-canonical.yaml": [
		"read", "normalize", "geometry", "phase", "filter", "geometry", "geometry", "stripe", "center", "recon", "write",
	],
	"b10-recon-canonical.yaml": ["read", "normalize", "phase", "stripe", "center", "recon", "write"],
}


@pytest.mark.parametrize("filename, expected_order", EXPECTED_ORDERS.items())
def test_canonical_pipeline_spec_validates_and_pins_literal_order(filename, expected_order):
	text = (PIPELINES / filename).read_text(encoding="utf-8")
	recon = Recon.from_yaml(text)

	assert [stage.kind for stage in recon.pipeline] == expected_order
	assert all(stage.algorithm for stage in recon.pipeline)
	assert all(stage.params for stage in recon.pipeline)
	assert recon.pipeline_template == filename.removesuffix(".yaml")
	assert "# Producer:" in text
	assert "# Survey source:" in text


def test_snapshot_b10_spec_remains_distinct_from_current_api_driver():
	recon = Recon.from_yaml((PIPELINES / "b10-recon-canonical.yaml").read_text(encoding="utf-8"))

	assert recon.metadata["lftomo_snapshot"].startswith("ddfcdf6")
	assert all("b10_recon_current" not in str(value) for stage in recon.pipeline for value in stage.params.values())
