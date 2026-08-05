import json
from types import SimpleNamespace

import pytest

from lftomo.experiment import (
	EnvironmentIdentity,
	ExperimentConfig,
	PathContextRoots,
	PathResourceError,
	PathResourceRequest,
	compare_path_resource_reports,
	preflight_path_resources,
)
from lftomo.experiment.runner import (
	ExperimentInputs,
	IndependentExperimentRunner,
	TrialExecution,
)
from lftomo.experiment.store import PersistedExperimentRunner


HASHES = [f"{index:064x}" for index in range(1, 5)]


def config():
	return ExperimentConfig.model_validate({
		"name": "path-preflight",
		"candidate_space": {"pipeline": [{
			"id": "phase", "kind": "phase", "algorithm": "paganin",
			"params": {"alpha": 0.01},
		}]},
		"sampler": {"name": "exhaustive", "seed": 1},
		"metrics": [{
			"id": "quality", "metric": "quality", "protocol_version": "1",
			"direction": "minimize",
		}],
		"objectives": [{"metric_id": "quality"}],
		"windows": [{"id": "screen", "y_start": 0, "y_stop": 1}],
		"fidelities": [{"id": "screen", "windows": ["screen"]}],
		"aggregation": {"metric_ids": ["quality"]},
	})


def inputs():
	return ExperimentInputs(
		dataset={"dataset_id": "fixture", "content_hash": HASHES[0]},
		configuration={"scan_config_hash": HASHES[1]},
		code={"version": "test", "git_commit": "abcdef123456"},
		environment=EnvironmentIdentity(
			python="3.11", platform="test", backend="numpy",
			dependencies={"numpy": "1.26.4"},
		),
	)


def test_safe_external_roots_report_context_without_enforcing_caller_policy(tmp_path):
	repository = tmp_path / "repo"
	result = repository / "artifacts"
	scratch = tmp_path / "external-scratch"
	repository.mkdir()
	report = preflight_path_resources(PathResourceRequest(
		result_root=result,
		scratch_root=scratch,
		estimated_result_bytes=1024,
		estimated_scratch_bytes=2048,
		contexts=PathContextRoots(
			cwd=repository,
			repository=repository,
			config_directory=repository / "config",
		),
	))

	assert report.accepted is True
	assert report.root("result").will_create is True
	assert report.root("result").context_flags == ("inside_cwd", "inside_repository")
	assert report.root("scratch").context_flags == ()
	assert report.relationships[0].relationship == "distinct"


@pytest.mark.parametrize("scratch_relative", ["result", "result/scratch", "."])
def test_result_and_disposable_roots_cannot_be_identical_or_nested(
	tmp_path, scratch_relative,
):
	result = tmp_path / "result"
	scratch = tmp_path / scratch_relative
	report = preflight_path_resources(PathResourceRequest(
		result_root=result,
		scratch_root=scratch,
	))

	assert report.accepted is False
	assert any(item.code == "ambiguous_root_overlap" for item in report.diagnostics)
	with pytest.raises(PathResourceError, match="cleanup boundaries"):
		raise PathResourceError(report)


def test_nested_disposable_scratch_and_cache_are_reported_but_accepted(tmp_path):
	report = preflight_path_resources(PathResourceRequest(
		result_root=tmp_path / "result",
		scratch_root=tmp_path / "scratch",
		cache_root=tmp_path / "scratch" / "prefix-cache",
	))

	assert report.accepted is True
	assert any(item.code == "nested_disposable_roots" for item in report.diagnostics)


def test_unwritable_root_and_non_directory_parent_are_structured_errors(tmp_path):
	unwritable = preflight_path_resources(
		PathResourceRequest(result_root=tmp_path / "result"),
		access=lambda _path, _mode: False,
	)
	assert unwritable.accepted is False
	assert unwritable.diagnostics[0].code == "not_writable"

	blocking_file = tmp_path / "blocking"
	blocking_file.write_text("not a directory", encoding="utf-8")
	blocked = preflight_path_resources(PathResourceRequest(
		result_root=blocking_file / "result",
	))
	assert blocked.accepted is False
	assert any(item.code == "not_a_directory" for item in blocked.diagnostics)


def test_guards_and_shared_filesystem_capacity_fail_before_allocation(tmp_path):
	guarded = preflight_path_resources(PathResourceRequest(
		result_root=tmp_path / "result",
		estimated_result_bytes=101,
		result_guard_bytes=100,
	))
	assert guarded.accepted is False
	assert any(item.code == "guard_exceeded" for item in guarded.diagnostics)

	usage = SimpleNamespace(total=1000, free=100)
	capacity = preflight_path_resources(
		PathResourceRequest(
			result_root=tmp_path / "result",
			scratch_root=tmp_path / "scratch",
			estimated_result_bytes=70,
			estimated_scratch_bytes=40,
		),
		disk_usage=lambda _path: usage,
	)
	assert capacity.accepted is False
	assert any(item.code == "insufficient_capacity" for item in capacity.diagnostics)


def test_relocation_fixes_result_identity_and_invalidates_moved_cache(tmp_path):
	prior = preflight_path_resources(PathResourceRequest(
		result_root=tmp_path / "result",
		scratch_root=tmp_path / "scratch-a",
		cache_root=tmp_path / "cache-a",
	))
	relocated = preflight_path_resources(PathResourceRequest(
		result_root=tmp_path / "result",
		scratch_root=tmp_path / "scratch-b",
		cache_root=tmp_path / "cache-b",
	))
	comparison = compare_path_resource_reports(prior, relocated)
	assert comparison.compatible is True
	assert comparison.cache_reusable is False
	assert {item.code for item in comparison.diagnostics} == {
		"scratch_relocated", "cache_relocated",
	}

	moved_result = preflight_path_resources(PathResourceRequest(
		result_root=tmp_path / "other-result",
	))
	incompatible = compare_path_resource_reports(prior, moved_result)
	assert incompatible.compatible is False
	assert incompatible.diagnostics[0].code == "result_identity_changed"


def test_runner_rejects_unsafe_roots_before_store_claim_or_execution(tmp_path):
	result = tmp_path / "result"
	calls = []
	runner = IndependentExperimentRunner(
		config(),
		inputs(),
		result,
		lambda *_args: calls.append("execute") or TrialExecution(metrics={"quality": 1}),
		path_resources=PathResourceRequest(
			result_root=result,
			scratch_root=result,
		),
	)

	with pytest.raises(PathResourceError, match="cleanup boundaries"):
		PersistedExperimentRunner(runner).run()

	assert calls == []
	assert not result.exists()


def test_manifest_records_accepted_report_and_compatible_scratch_relocation(tmp_path):
	result = tmp_path / "result"
	calls = []

	def execute(*_args):
		calls.append("execute")
		return TrialExecution(metrics={"quality": 1})

	first_runner = IndependentExperimentRunner(
		config(), inputs(), result, execute,
		path_resources=PathResourceRequest(
			result_root=result,
			scratch_root=tmp_path / "scratch-a",
		),
	)
	first = PersistedExperimentRunner(first_runner).run()
	second_runner = IndependentExperimentRunner(
		config(), inputs(), result, execute,
		path_resources=PathResourceRequest(
			result_root=result,
			scratch_root=tmp_path / "scratch-b",
		),
	)
	second = PersistedExperimentRunner(second_runner).run()
	manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))

	assert calls == ["execute"]
	assert second.attempts == first.attempts
	assert second.preflight.path_resources.root("scratch").resolved_path == (
		tmp_path / "scratch-b"
	).resolve()
	assert manifest["metadata"]["path_resource_preflight"] == (
		second.preflight.path_resources.model_dump(mode="json")
	)
	assert manifest["metadata"]["path_resource_relocation"]["compatible"] is True
