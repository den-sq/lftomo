import os

import pytest

from tests.tier2 import DATASETS, tier2_path


def _integration_host_enabled():
	"""Return whether this host should run lab-local integration tests."""

	return (
		os.environ.get("LFTOMO_RUNNER_CAPABILITY") == "recon-data"
		or os.environ.get("LFTOMO_RUN_INTEGRATION") == "1"
	)


def pytest_addoption(parser):
	parser.addoption(
		"--run-integration",
		action="store_true",
		default=False,
		help="run integration tests that require local lab data or hardware",
	)
	parser.addoption(
		"--run-system",
		action="store_true",
		default=False,
		help="run system tests: end-to-end runs, reference parity, and real optional engines",
	)
	parser.addoption(
		"--run-tier2",
		action="store_true",
		default=False,
		help="run tests that require the lab-internal Tier-2 datasets",
	)


def pytest_configure(config):
	config.addinivalue_line(
		"markers",
		"integration: requires local lab data or hardware; skipped unless --run-integration is set",
	)
	config.addinivalue_line(
		"markers",
		"system: end-to-end, reference-parity, or real-engine test outside the unit suite; "
		"skipped unless --run-system is set",
	)
	config.addinivalue_line(
		"markers",
		"tier2: requires lab-internal Tier-2 reference data; skipped unless --run-tier2 is set",
	)
	config.addinivalue_line(
		"markers", "legacy_gridrec: compares stored gridrec output from stock libtomo",
	)
	config.addinivalue_line(
		"markers", "gridrec_fixed: requires the corrected densq1 native gridrec build",
	)


def pytest_collection_modifyitems(config, items):
	from lftomo.pipeline.runtime import collect_libtomo_identity

	libtomo = collect_libtomo_identity()
	run_integration = config.getoption("--run-integration")
	integration_enabled = run_integration and _integration_host_enabled()
	if run_integration:
		reason = "integration tests require the recon-data capability or LFTOMO_RUN_INTEGRATION=1"
	else:
		reason = "need --run-integration to run"
	skip_integration = pytest.mark.skip(reason=reason)
	skip_system = pytest.mark.skip(reason="need --run-system to run")
	skip_tier2 = pytest.mark.skip(reason="need --run-tier2 to run")
	for item in items:
		if "integration" in item.keywords and not integration_enabled:
			item.add_marker(skip_integration)
		if "system" in item.keywords and not config.getoption("--run-system"):
			item.add_marker(skip_system)
		if "tier2" in item.keywords and not config.getoption("--run-tier2"):
			item.add_marker(skip_tier2)
		if "legacy_gridrec" in item.keywords and (libtomo is None or libtomo.gridrec_fixed):
			item.add_marker(pytest.mark.skip(reason="legacy gridrec values require stock libtomo"))
		if "gridrec_fixed" in item.keywords and (libtomo is None or not libtomo.gridrec_fixed):
			item.add_marker(pytest.mark.skip(reason="gridrec regressions require patched densq1 libtomo"))


def _dataset_fixture(dataset_name):
	@pytest.fixture(name=f"dataset_{dataset_name}")
	def fixture():
		return tier2_path(DATASETS[dataset_name], os.environ)

	return fixture


dataset_striping_tiff = _dataset_fixture("striping_tiff")
dataset_phase_contrast_hdf5 = _dataset_fixture("phase_contrast_hdf5")
dataset_phase_contrast_tiff = _dataset_fixture("phase_contrast_tiff")
dataset_helical_tiff = _dataset_fixture("helical_tiff")
dataset_dropped_frame_tiff = _dataset_fixture("dropped_frame_tiff")
