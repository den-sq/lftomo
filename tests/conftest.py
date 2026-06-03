import os
import socket

import pytest


def _integration_host_enabled():
	"""Return whether this host should run lab-local integration tests."""

	if os.environ.get("LFTOMO_RUN_INTEGRATION") == "1":
		return True
	return "yarrkha" in socket.gethostname().lower()


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
	if config.getoption("--run-integration") and _integration_host_enabled():
		return

	if config.getoption("--run-integration"):
		reason = "integration tests run only on Yarrkha unless LFTOMO_RUN_INTEGRATION=1"
	else:
		reason = "need --run-integration to run"
	skip_integration = pytest.mark.skip(reason=reason)
	for item in items:
		if "integration" in item.keywords:
			item.add_marker(skip_integration)
