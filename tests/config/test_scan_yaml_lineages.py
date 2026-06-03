from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")
yaml = pytest.importorskip("yaml")

from lftomo.config import Scan  # noqa: E402


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "scan_yaml_lineages"


@pytest.mark.parametrize("fixture_path", sorted(FIXTURE_DIR.glob("*.scan.yaml")), ids=lambda p: p.stem)
def test_lineage_scan_yaml_roundtrips_without_dropped_fields(fixture_path):
	source = yaml.safe_load(fixture_path.read_text())

	model = Scan.model_validate(source)
	roundtripped = yaml.safe_load(model.to_yaml())

	assert roundtripped == source
