import pytest

from pydantic import ValidationError

from lftomo.config import Recon, StageConfig, SweepConfig
from lftomo.pipeline import (
	CandidateLimitError,
	ContinuousDomainError,
	candidate_key,
	count_recon_candidates,
	expand_recon_sweeps,
	expand_sweep,
)


def sweep(kind, params):
	return SweepConfig(kind=kind, params=params)


def test_list_sweep_accepts_values_and_explicit_cases():
	assert expand_sweep(sweep("list", {"center": [10, 11]})) == [
		{"center": 10},
		{"center": 11},
	]
	assert expand_sweep(sweep("list", {"cases": [{"a": 1, "b": 2}, {"a": 3}]})) == [
		{"a": 1, "b": 2},
		{"a": 3},
	]


def test_grid_sweep_is_cartesian():
	assert expand_sweep(sweep("grid", {"a": [1, 2], "b": ["x", "y"]})) == [
		{"a": 1, "b": "x"},
		{"a": 1, "b": "y"},
		{"a": 2, "b": "x"},
		{"a": 2, "b": "y"},
	]


def test_linspace_sweep_supports_endpoint_control():
	assert expand_sweep(
		sweep("linspace", {"alpha": {"start": 0, "stop": 1, "num": 4, "endpoint": False}})
	) == [{"alpha": 0.0}, {"alpha": 0.25}, {"alpha": 0.5}, {"alpha": 0.75}]


def test_zip_sweep_pairs_values_and_checks_lengths():
	assert expand_sweep(sweep("zip", {"a": [1, 2], "b": [3, 4]})) == [
		{"a": 1, "b": 3},
		{"a": 2, "b": 4},
	]
	with pytest.raises(ValueError, match="equal lengths"):
		expand_sweep(sweep("zip", {"a": [1], "b": [2, 3]}))


def test_pipeline_expansion_combines_stages_without_mutating_source():
	recon = Recon(
		pipeline=[
			StageConfig(
				kind="phase",
				algorithm="paganin",
				params={"alpha": 0.001},
				sweep=sweep("list", {"alpha": [0.01, 0.02]}),
			),
			StageConfig(
				kind="recon",
				algorithm="gridrec",
				params={"filter_name": "parzen"},
				sweep=sweep("grid", {"center": [10, 11], "filter_name": ["hann", "shepp"]}),
			),
		]
	)

	runs = expand_recon_sweeps(recon)

	assert len(runs) == 8
	assert runs[0].index == 0
	assert runs[0].recon.pipeline[0].params["alpha"] == 0.01
	assert runs[0].recon.pipeline[1].params == {"filter_name": "hann", "center": 10}
	assert runs[-1].parameters == {
		"stage[0].alpha": 0.02,
		"stage[1].center": 11,
		"stage[1].filter_name": "shepp",
	}
	assert recon.pipeline[0].params == {"alpha": 0.001}
	assert recon.pipeline[0].sweep is not None
	assert all(stage.sweep is None for run in runs for stage in run.recon.pipeline)


@pytest.mark.parametrize(
	"spec, message",
	[
		(sweep("grid", {}), "must not be empty"),
		(sweep("list", {"a": [1], "b": [2]}), "one parameter"),
		(sweep("linspace", {"a": {"start": 0, "stop": 1}}), "requires start"),
	],
)
def test_invalid_sweeps_fail_early(spec, message):
	with pytest.raises(ValueError, match=message):
		expand_sweep(spec)


def test_stable_stage_ids_are_validated_and_qualify_resolved_values():
	with pytest.raises(ValidationError, match="duplicate pipeline Stage IDs"):
		Recon(pipeline=[
			StageConfig(id="stripe", kind="stripe", algorithm="vo_sort"),
			StageConfig(id="stripe", kind="stripe", algorithm="vo_dead"),
		])
	with pytest.raises(ValidationError, match="string_pattern_mismatch"):
		StageConfig(id="not valid", kind="stripe")

	recon = Recon(pipeline=[
		StageConfig(
			id="stripe-second",
			kind="stripe",
			algorithm="vo_sort",
			sweep=sweep("list", {"size": [3, 5]}),
		),
	])
	runs = expand_recon_sweeps(recon)
	assert runs[1].parameters == {"stage[stripe-second].size": 5}


def test_id_targeting_survives_unrelated_insertion_and_reordering():
	target = StageConfig(
		id="target", kind="stripe", algorithm="vo_sort",
		sweep=sweep("list", {"size": [3, 5]}),
	)
	first = Recon(pipeline=[StageConfig(id="read", kind="read"), target])
	second = Recon(pipeline=[
		StageConfig(id="other", kind="filter"),
		target,
		StageConfig(id="read", kind="read"),
	])

	assert [run.parameters for run in expand_recon_sweeps(first)] == [
		{"stage[target].size": 3}, {"stage[target].size": 5},
	]
	assert [run.parameters for run in expand_recon_sweeps(second)] == [
		{"stage[target].size": 3}, {"stage[target].size": 5},
	]


def test_algorithm_cases_keep_conditional_parameters_disjoint():
	recon = Recon.model_validate({
		"pipeline": [{
			"id": "stripe",
			"kind": "stripe",
			"sweep": {
				"kind": "cases",
				"cases": [
					{
						"name": "fourier",
						"algorithm": "fw",
						"params": {"wname": "db2"},
						"domains": {
							"sigma": {"kind": "linear", "start": 1, "stop": 2, "count": 2},
						},
					},
					{
						"name": "sorting",
						"algorithm": "vo_sort",
						"params": {"dim": 1},
						"domains": {
							"size": {"kind": "discrete", "values": [3, 5]},
						},
					},
				],
			},
		}],
	})

	runs = expand_recon_sweeps(recon)

	assert [run.recon.pipeline[0].algorithm for run in runs] == [
		"fw", "fw", "vo_sort", "vo_sort",
	]
	assert all("size" not in run.recon.pipeline[0].params for run in runs[:2])
	assert all("sigma" not in run.recon.pipeline[0].params for run in runs[2:])
	assert all(run.recon.pipeline[0].sweep is None for run in runs)


@pytest.mark.parametrize("sampled", [False, True], ids=["enumerated", "sampled"])
def test_algorithm_case_validates_after_applying_parameter_overrides(sampled):
	recon = Recon.model_validate({
		"pipeline": [{
			"kind": "recon",
			"algorithm": "sirt_cuda",
			"params": {"iterations": 10, "center": "4"},
			"sweep": {
				"kind": "cases",
				"cases": [{
					"name": "gridrec",
					"algorithm": "gridrec",
					"params": {"iterations": 1},
				}],
			},
		}],
	})
	original = recon.model_dump(mode="json")
	if sampled:
		import numpy as np
		from lftomo.experiment.selection import _resolve_unit_candidate

		resolved, _ = _resolve_unit_candidate(recon, np.array([0.5]))
	else:
		resolved = expand_recon_sweeps(recon)[0].recon
	expected = Recon(pipeline=[StageConfig(
		kind="recon", algorithm="gridrec", params={"iterations": 1, "center": "4"},
	)])
	assert resolved.pipeline[0].model_dump() == expected.pipeline[0].model_dump()
	assert candidate_key(resolved) == candidate_key(expected)
	assert recon.model_dump(mode="json") == original


def test_typed_domains_share_one_finite_and_continuous_vocabulary():
	finite = SweepConfig.model_validate({
		"kind": "domain",
		"params": {
			"choice": {"kind": "categorical", "values": ["a", "b"]},
			"alpha": {"kind": "log", "start": 0.01, "stop": 1, "count": 3},
		},
	})
	assert expand_sweep(finite) == [
		{"choice": "a", "alpha": pytest.approx(0.01)},
		{"choice": "a", "alpha": pytest.approx(0.1)},
		{"choice": "a", "alpha": pytest.approx(1.0)},
		{"choice": "b", "alpha": pytest.approx(0.01)},
		{"choice": "b", "alpha": pytest.approx(0.1)},
		{"choice": "b", "alpha": pytest.approx(1.0)},
	]

	continuous = SweepConfig.model_validate({
		"kind": "domain",
		"params": {"alpha": {"kind": "linear", "start": 0, "stop": 1}},
	})
	with pytest.raises(ContinuousDomainError, match="experiment sampler"):
		expand_sweep(continuous)


def test_named_structural_variants_are_explicit_and_deterministic():
	recon = Recon.model_validate({
		"pipeline": [
			{"id": "read", "kind": "read", "algorithm": "tiff_stack"},
			{"id": "stripe-a", "kind": "stripe", "algorithm": "fw"},
			{"id": "stripe-b", "kind": "stripe", "algorithm": "vo_sort"},
			{"id": "write", "kind": "write", "algorithm": "tiff_slices"},
		],
		"variants": [
			{"name": "both", "stages": ["read", "stripe-a", "stripe-b", "write"]},
			{"name": "sorting-only", "stages": ["read", "stripe-b", "write"]},
		],
	})

	runs = expand_recon_sweeps(recon)

	assert [run.variant for run in runs] == ["both", "sorting-only"]
	assert [[stage.id for stage in run.recon.pipeline] for run in runs] == [
		["read", "stripe-a", "stripe-b", "write"],
		["read", "stripe-b", "write"],
	]
	assert all(not run.recon.variants for run in runs)
	with pytest.raises(ValidationError, match="unknown Stage IDs"):
		Recon.model_validate({
			"pipeline": [{"id": "read", "kind": "read"}],
			"variants": [{"name": "bad", "stages": ["missing"]}],
		})
	with pytest.raises(ValidationError, match="require stable IDs"):
		Recon.model_validate({
			"pipeline": [{"kind": "read"}],
			"variants": [{"name": "bad", "stages": ["read"]}],
		})


def test_candidate_keys_ignore_enumeration_but_track_resolved_content():
	a = Recon(pipeline=[
		StageConfig(
			id="phase", kind="phase", algorithm="paganin",
			sweep=sweep("list", {"alpha": [0.1, 0.2]}),
		),
	])
	b = Recon(pipeline=[
		StageConfig(
			id="phase", kind="phase", algorithm="paganin",
			sweep=sweep("list", {"alpha": [0.2, 0.1]}),
		),
	])
	keys_a = {run.recon.pipeline[0].params["alpha"]: run.candidate_key for run in expand_recon_sweeps(a)}
	keys_b = {run.recon.pipeline[0].params["alpha"]: run.candidate_key for run in expand_recon_sweeps(b)}

	assert keys_a == keys_b
	assert keys_a[0.1] != keys_a[0.2]
	resolved = expand_recon_sweeps(a)[0].recon
	assert candidate_key(resolved) == keys_a[0.1]
	assert Recon.model_validate(resolved.model_dump(mode="json")) == resolved


def test_candidate_count_guard_fails_before_cartesian_materialization():
	recon = Recon.model_validate({
		"pipeline": [{
			"kind": "phase",
			"algorithm": "paganin",
			"sweep": {"kind": "grid", "params": {"a": list(range(20)), "b": list(range(20))}},
		}],
		"candidate_limits": {"max_candidates": 100},
	})

	assert count_recon_candidates(recon) == 400
	with pytest.raises(CandidateLimitError, match="400 runs"):
		expand_recon_sweeps(recon)
