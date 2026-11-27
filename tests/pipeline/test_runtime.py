"""Native build provenance and environment-specific numerical test selection."""

import json
from types import SimpleNamespace

import pytest

from lftomo.pipeline import runtime
from tests.conftest import pytest_collection_modifyitems


@pytest.mark.parametrize("build, fixed", [
	("cuda12.9h1234567_densq1_103", True),
	("cuda12.9h1234567_102", False),
])
def test_libtomo_identity_uses_interpreter_prefix_and_build(tmp_path, monkeypatch, build, fixed):
	metadata = tmp_path / "conda-meta"
	metadata.mkdir()
	(metadata / f"libtomo-1.15.3-{build}.json").write_text(json.dumps({
		"name": "libtomo", "version": "1.15.3", "build": build,
	}))
	monkeypatch.setattr(runtime.sys, "prefix", str(tmp_path))
	monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "another-environment"))

	identity = runtime.collect_libtomo_identity()

	assert identity.model_dump() == {"version": "1.15.3", "build": build}
	assert identity.gridrec_fixed is fixed


def test_missing_native_record_is_unknown(tmp_path):
	assert runtime.collect_libtomo_identity(tmp_path) is None


@pytest.mark.parametrize("build, skipped", [
	("cuda12.9h1234567_densq1_103", {"legacy_gridrec"}),
	("cuda12.9h1234567_102", {"gridrec_fixed"}),
	(None, {"legacy_gridrec", "gridrec_fixed"}),
])
def test_collection_gates_only_environment_specific_tests(monkeypatch, build, skipped):
	identity = None if build is None else runtime.LibTomoIdentity(version="1.15.3", build=build)
	monkeypatch.setattr(runtime, "collect_libtomo_identity", lambda: identity)
	marks = {name: [] for name in ("legacy_gridrec", "gridrec_fixed", "ordinary")}
	items = [
		SimpleNamespace(keywords={name}, add_marker=markers.append)
		for name, markers in marks.items()
	]
	config = SimpleNamespace(getoption=lambda name: False)

	pytest_collection_modifyitems(config, items)

	assert {name for name, markers in marks.items() if markers} == skipped
	assert all(marker.name == "skip" for markers in marks.values() for marker in markers)
