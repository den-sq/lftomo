import pytest

from lftomo.config import Recon, StageConfig, SweepConfig
from lftomo.pipeline import expand_recon_sweeps, expand_sweep


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
