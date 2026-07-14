"""Numerically reproduce populated frozen references on the recon-data runner."""

import json
import os
from pathlib import Path

import pytest

from lftomo.reference import ReferenceNotReproducible, run_reference_archive
from tests.tier2 import DATASETS


pytestmark = pytest.mark.tier2


CASES = [
	(dataset, producer)
	for dataset in DATASETS.values()
	for producer in dataset.producers
]


@pytest.mark.parametrize("dataset,producer", CASES, ids=lambda value: getattr(value, "name", value))
def test_tier2_numerical_reference(request, dataset, producer):
	root: Path = request.getfixturevalue(f"dataset_{dataset.name}")
	archive = root / "reference" / producer
	if not any((archive / "recon").glob("*.tif*")):
		pytest.skip(f"reference slot is not populated: {archive}")
	limit = os.environ.get("LFTOMO_TIER2_SLICES")
	slices = None
	if limit:
		metadata = __import__("yaml").safe_load((archive / "metadata.yaml").read_text(encoding="utf-8"))
		slices = metadata["reference_selection"]["slices"][:int(limit)]
	try:
		report = run_reference_archive(root, producer, slices=slices)
	except ReferenceNotReproducible as error:
		pytest.skip(f"exact producer input is unavailable: {error}")

	report_dir = os.environ.get("LFTOMO_TIER2_REPORT_DIR")
	if report_dir:
		path = Path(report_dir)
		path.mkdir(parents=True, exist_ok=True)
		(path / f"{dataset.name}-{producer}.json").write_text(
			json.dumps(report.as_dict(), indent=2) + "\n",
			encoding="utf-8",
		)
	failed = [comparison for comparison in report.comparisons if not comparison.passed]
	assert not failed, failed
