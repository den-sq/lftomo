import pytest


def pytest_addoption(parser):
	parser.addoption(
		"--run-integration",
		action="store_true",
		default=False,
		help="run integration tests that require local lab data or hardware",
	)


def pytest_configure(config):
	config.addinivalue_line(
		"markers",
		"integration: requires local lab data or hardware; skipped unless --run-integration is set",
	)


def pytest_collection_modifyitems(config, items):
	if config.getoption("--run-integration"):
		return

	skip_integration = pytest.mark.skip(reason="need --run-integration to run")
	for item in items:
		if "integration" in item.keywords:
			item.add_marker(skip_integration)
