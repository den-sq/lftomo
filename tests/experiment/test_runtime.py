import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lftomo.experiment import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	ExperimentConfig,
	RuntimeCompatibilityDiagnostic,
	RuntimeCompatibilityError,
	RuntimeCompatibilityReport,
	check_runtime_compatibility,
)
from lftomo.experiment.runner import (
	ExperimentInputs,
	IndependentExperimentRunner,
	TrialExecution,
)
from lftomo.experiment.store import PersistedExperimentRunner
from lftomo.pipeline import RuntimeRequirement, Stage, StageContract
from lftomo.pipeline.stages import STAGE_REGISTRY


HASHES = [f"{index:064x}" for index in range(1, 5)]


def config(*, kind="phase", algorithm="paganin", sweep=None, pipeline=None):
	stages = pipeline or [{
		"id": "algorithm", "kind": kind, "algorithm": algorithm,
	}]
	document = {
		"name": "runtime-preflight",
		"candidate_space": {"pipeline": stages},
		"sampler": {"name": "exhaustive", "seed": 1},
		"metrics": [{
			"id": "quality", "metric": "quality", "protocol_version": "1",
			"direction": "minimize",
		}],
		"objectives": [{"metric_id": "quality"}],
		"windows": [{"id": "screen", "y_start": 0, "y_stop": 1}],
		"fidelities": [{"id": "screen", "windows": ["screen"]}],
		"aggregation": {"metric_ids": ["quality"]},
	}
	if sweep is not None:
		document["candidate_space"]["pipeline"][0]["sweep"] = sweep
	return ExperimentConfig.model_validate(document)


def tomopy_module(version="1.14.0"):
	return SimpleNamespace(
		__version__=version,
		recon=lambda *_args, **_kwargs: None,
		circ_mask=lambda *_args, **_kwargs: None,
		find_center_pc=lambda *_args, **_kwargs: None,
		find_center_vo=lambda *_args, **_kwargs: None,
		remove_all_stripe=lambda *_args, **_kwargs: None,
		astra=object(),
	)


def loader(modules):
	def load(name):
		try:
			return modules[name]
		except KeyError as error:
			raise ImportError(f"missing {name}") from error
	return load


def test_numpy_2_tomopy_1_is_rejected_with_actionable_structured_diagnostic():
	report = check_runtime_compatibility(
		config(kind="recon", algorithm="gridrec"),
		package_versions={
			"numpy": "2.4.6", "scipy": "1.16.0", "pydantic": "2.11.0",
			"tomopy": "1.14.0",
		},
		module_loader=loader({"tomopy": tomopy_module()}),
	)

	assert report.compatible is False
	diagnostic = next(
		item for item in report.diagnostics
		if item.package == "numpy" and item.requirement == "<2"
	)
	assert diagnostic.stage_ids == ("algorithm",)
	assert "NumPy 2.4.6" in diagnostic.message
	assert "activate/rebuild" in diagnostic.corrective_action
	with pytest.raises(RuntimeCompatibilityError, match="numpy<2"):
		raise RuntimeCompatibilityError(report)


def test_compatible_tomopy_runtime_runs_bounded_import_and_api_smoke_check():
	report = check_runtime_compatibility(
		config(kind="recon", algorithm="gridrec"),
		package_versions={
			"numpy": "1.26.4", "scipy": "1.13.0", "pydantic": "2.11.0",
			"tomopy": "1.14.0",
		},
		module_loader=loader({"tomopy": tomopy_module()}),
	)

	assert report.compatible is True
	assert report.enabled_features == ("tomopy",)
	assert report.smoke_checks[0].status == "passed"
	assert report.smoke_checks[0].checked_attributes == ("circ_mask", "recon")
	assert report.environment.dependencies["tomopy"] == "1.14.0"


def test_numpy_2_with_future_tomopy_2_is_not_blocked_by_a_static_version_ceiling():
	report = check_runtime_compatibility(
		config(kind="recon", algorithm="gridrec"),
		package_versions={
			"numpy": "2.4.6", "scipy": "1.16.0", "pydantic": "2.11.0",
			"tomopy": "2.0.0",
		},
		module_loader=loader({"tomopy": tomopy_module("2.0.0")}),
	)

	assert report.compatible is True
	assert report.diagnostics == ()


def test_stage_owned_requirements_are_discovered_without_a_runtime_key_table(monkeypatch):
	class ExternalStage(Stage):
		contract = StageContract(runtime_requirements=(RuntimeRequirement(
			capability="external",
			package="external-package",
			module="external_module",
			corrective_action="install the external Stage dependency",
		),))

		def run(self, state):
			return state

	monkeypatch.setitem(STAGE_REGISTRY, "phase.external", ExternalStage)
	report = check_runtime_compatibility(
		config(kind="phase", algorithm="external"),
		package_versions={
			"numpy": "1.26.4", "scipy": "1.13.0", "pydantic": "2.11.0",
			"external-package": None,
		},
		module_loader=loader({}),
	)

	assert report.compatible is False
	assert report.enabled_features == ("external",)
	assert report.diagnostics[0].stage_ids == ("algorithm",)
	assert report.diagnostics[0].corrective_action == "install the external Stage dependency"


def test_stage_registry_can_be_inspected_without_importing_optional_tomopy():
	script = """
import importlib.abc
import sys

class BlockTomopy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tomopy" or fullname.startswith("tomopy."):
            raise ModuleNotFoundError("blocked optional TomoPy")

sys.meta_path.insert(0, BlockTomopy())
from lftomo.pipeline.stages import STAGE_REGISTRY
assert STAGE_REGISTRY
assert "tomopy" not in sys.modules
"""
	result = subprocess.run(
		[sys.executable, "-c", script],
		check=False,
		capture_output=True,
		text=True,
	)
	assert result.returncode == 0, result.stderr


def test_shared_stage_requirement_is_imported_once_with_all_stage_api_probes():
	loaded = []

	def recording_loader(name):
		loaded.append(name)
		return tomopy_module()

	report = check_runtime_compatibility(
		config(pipeline=[
			{"id": "center", "kind": "center", "algorithm": "pc"},
			{"id": "reconstruction", "kind": "recon", "algorithm": "gridrec"},
		]),
		package_versions={
			"numpy": "1.26.4", "scipy": "1.13.0", "pydantic": "2.11.0",
			"tomopy": "1.14.0",
		},
		module_loader=recording_loader,
	)

	assert report.compatible is True
	assert loaded == ["tomopy"]
	assert report.smoke_checks[0].checked_attributes == (
		"circ_mask", "find_center_pc", "recon",
	)


def test_missing_enabled_optional_feature_is_rejected():
	report = check_runtime_compatibility(
		config(kind="phase", algorithm="bac"),
		package_versions={
			"numpy": "1.26.4", "scipy": "1.13.0", "pydantic": "2.11.0",
			"hotopy": None,
		},
		module_loader=loader({}),
	)

	assert report.compatible is False
	assert report.smoke_checks[0].status == "failed"
	assert report.diagnostics[0].capability == "bac"
	assert "bac extra" in report.diagnostics[0].corrective_action


def test_source_import_version_is_used_when_distribution_metadata_is_absent():
	report = check_runtime_compatibility(
		config(kind="recon", algorithm="gridrec"),
		package_versions={
			"numpy": "1.26.4", "scipy": "1.13.0", "pydantic": "2.11.0",
			"tomopy": None,
		},
		module_loader=loader({"tomopy": tomopy_module("1.15.0")}),
	)

	assert report.compatible is True
	assert report.environment.dependencies["tomopy"] == "1.15.0"


def test_case_sweeps_preflight_every_selectable_optional_algorithm():
	report = check_runtime_compatibility(
		config(
			algorithm="paganin",
			sweep={
				"kind": "cases",
				"cases": [
					{"name": "native", "algorithm": "paganin"},
					{"name": "hotopy", "algorithm": "bac", "params": {"alpha": 1e-3}},
				],
			},
		),
		package_versions={
			"numpy": "1.26.4", "scipy": "1.13.0", "pydantic": "2.11.0",
			"hotopy": None,
		},
		module_loader=loader({}),
	)

	assert report.compatible is False
	assert report.diagnostics[0].stage_ids == ("algorithm",)


def inputs(environment):
	return ExperimentInputs(
		dataset=DatasetIdentity(dataset_id="fixture", content_hash=HASHES[0]),
		configuration=ConfigurationIdentity(scan_config_hash=HASHES[1]),
		code=CodeIdentity(version="test", git_commit="abcdef123456"),
		environment=environment,
	)


def incompatible_report():
	environment = check_runtime_compatibility(
		config(),
		package_versions={
			"numpy": "2.4.6", "scipy": "1.16.0", "pydantic": "2.11.0",
		},
		module_loader=loader({}),
	).environment
	return RuntimeCompatibilityReport(
		compatible=False,
		environment=environment,
		enabled_features=("tomopy",),
		diagnostics=(RuntimeCompatibilityDiagnostic(
			capability="tomopy",
			severity="error",
			message="injected incompatibility",
			evidence="test",
			corrective_action="use a compatible runtime",
		),),
	)


def test_runner_rejects_runtime_before_validation_cache_store_claim_or_executor(tmp_path):
	calls = []

	class Executor:
		def plan_prefix_cache(self, _trials):
			calls.append("cache")

		def __call__(self, *_args):
			calls.append("execute")
			return TrialExecution(metrics={"quality": 1})

	report = incompatible_report()
	runner = IndependentExperimentRunner(
		config(), inputs(report.environment), tmp_path, Executor(),
		validator=lambda _recon: calls.append("validate"),
		runtime_report=report,
	)
	with pytest.raises(RuntimeCompatibilityError, match="injected incompatibility"):
		PersistedExperimentRunner(runner).run()

	assert calls == []
	assert list(tmp_path.iterdir()) == []


def test_accepted_runtime_report_is_in_preflight_and_manifest(tmp_path):
	report = check_runtime_compatibility(
		config(),
		package_versions={
			"numpy": "1.26.4", "scipy": "1.13.0", "pydantic": "2.11.0",
		},
		module_loader=loader({}),
	)
	runner = IndependentExperimentRunner(
		config(),
		inputs(report.environment),
		tmp_path,
		lambda *_args: TrialExecution(metrics={"quality": 1}),
		runtime_report=report,
	)

	summary = PersistedExperimentRunner(runner).run()
	manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

	assert summary.preflight.runtime == report
	assert manifest["metadata"]["runtime_compatibility"] == report.model_dump(mode="json")
	assert summary.attempts[0].identity.environment == report.environment
