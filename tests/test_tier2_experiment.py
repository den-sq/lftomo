"""Audit a lab-run experiment bundle without making it part of lightweight CI."""

import os
from pathlib import Path

import pytest

from lftomo.experiment import (
	FidelityRunSummary,
	PreviewBundle,
	Tier2ExperimentPlan,
	audit_tier2_results,
)


pytestmark = pytest.mark.tier2


def test_tier2_experiment_bundle_matches_immutable_protocol():
	root_value = os.environ.get("LFTOMO_TIER2_EXPERIMENT_ROOT")
	plan_value = os.environ.get("LFTOMO_TIER2_EXPERIMENT_PLAN")
	if not root_value or not plan_value:
		pytest.skip(
			"set LFTOMO_TIER2_EXPERIMENT_ROOT and LFTOMO_TIER2_EXPERIMENT_PLAN "
			"to audit a real experiment bundle"
		)
	root = Path(root_value)
	plan = Tier2ExperimentPlan.model_validate_json(Path(plan_value).read_text(encoding="utf-8"))
	summary = FidelityRunSummary.model_validate_json(
		(root / "fidelity-results.json").read_text(encoding="utf-8")
	)
	previews = PreviewBundle.model_validate_json(
		(root / "previews" / "mapping.json").read_text(encoding="utf-8")
	)

	audit = audit_tier2_results(plan, summary, previews)
	assert audit.passed, audit.errors
