from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import pytest
import yaml

from lftomo.experiment.cli import main
from lftomo.review import CandidateResolutionError, ExperimentCandidateResolver

from .conftest import BLINDING_KEY


@pytest.mark.system
def test_candidate_resolution_joins_preview_trial_pipeline_and_result(experiment_root):
	resolver = ExperimentCandidateResolver(experiment_root)
	inventory = resolver.inventory()
	assert len(inventory) == 3
	winner = next(item for item in inventory if item.promoted)
	report = resolver.resolve(winner.blind_id)
	assert report.matched_namespace == "blind_id"
	assert report.candidate_key == winner.candidate_key
	assert len(report.trials) == 2
	assert {trial.fidelity_id for trial in report.trials} == {"screen", "final"}
	assert report.algorithm_chain[0].algorithm == "paganin"
	assert any(
		key.endswith(".alpha") and value == pytest.approx(0.01)
		for variation in report.variations
		for key, value in variation.parameters.items()
	)
	assert len(report.preview_paths) == 3
	assert all(path.is_file() for path in report.preview_paths)
	assert all(
		trial.latest_attempt.metrics["quality"] == pytest.approx(0.01)
		for trial in report.trials
	)
	assert all(
		trial.latest_attempt.outputs[0].path.is_file()
		for trial in report.trials
	)
	assert any(
		trial.latest_attempt.promotion is not None for trial in report.trials
	)

	candidate_prefix = winner.candidate_key[:12]
	assert resolver.resolve(candidate_prefix).candidate_key == winner.candidate_key
	trial_id = report.trials[0].trial_id
	assert resolver.resolve(trial_id[:12]).candidate_key == winner.candidate_key


def test_ambiguous_prefix_reports_matches_across_namespaces(experiment_root):
	resolver = ExperimentCandidateResolver(experiment_root)
	with pytest.raises(CandidateResolutionError) as caught:
		resolver.resolve("")
	message = str(caught.value)
	assert "ambiguous identifier" in message
	assert "blind_id:" in message
	assert "candidate_key:" in message
	assert "trial_id:" in message


def test_candidate_and_candidates_cli_have_human_json_and_yaml_contracts(experiment_root):
	resolver = ExperimentCandidateResolver(experiment_root)
	entry = resolver.inventory()[0]
	runner = CliRunner()

	json_result = runner.invoke(main, [
		"candidate", str(experiment_root), entry.blind_id, "--json",
	])
	assert json_result.exit_code == 0, json_result.output
	payload = json.loads(json_result.output)
	assert payload["candidate_key"] == entry.candidate_key
	assert payload["algorithm_chain"][0]["algorithm"] == "paganin"
	assert payload["trials"][0]["latest_attempt"]["metrics"]

	yaml_result = runner.invoke(main, [
		"candidate", str(experiment_root), entry.blind_id, "--yaml",
	])
	assert yaml_result.exit_code == 0, yaml_result.output
	assert yaml.safe_load(yaml_result.output)["candidate_key"] == entry.candidate_key

	human_result = runner.invoke(main, [
		"candidate", str(experiment_root), entry.blind_id,
	])
	assert human_result.exit_code == 0, human_result.output
	assert "algorithm chain: phase.paganin" in human_result.output
	assert "resolved pipeline:" in human_result.output
	assert "peak memory bytes: 4096" in human_result.output

	inventory_result = runner.invoke(main, [
		"candidates", str(experiment_root), "--json",
	])
	assert inventory_result.exit_code == 0, inventory_result.output
	inventory_payload = json.loads(inventory_result.output)
	assert len(inventory_payload) == 3
	assert {item["candidate_key"] for item in inventory_payload} == {
		item.candidate_key for item in resolver.inventory()
	}
	assert all(item["trial_ids"] for item in inventory_payload)


def test_candidate_cli_rejects_conflicting_structured_formats(experiment_root):
	entry = ExperimentCandidateResolver(experiment_root).inventory()[0]
	result = CliRunner().invoke(main, [
		"candidate", str(experiment_root), entry.blind_id, "--json", "--yaml",
	])
	assert result.exit_code == 2
	assert "choose only one" in result.output


def test_private_mapping_fallback_recomputes_blind_ids_without_serializing_key(
	experiment_root, tmp_path, monkeypatch,
):
	fallback = tmp_path / "mapping-inaccessible"
	fallback.mkdir()
	for name in ("store.json", "manifest.json"):
		(fallback / name).symlink_to(experiment_root / name)
	for name in ("trials", "previews"):
		(fallback / name).symlink_to(experiment_root / name, target_is_directory=True)
	monkeypatch.setenv("LFTOMO_PREVIEW_BLINDING_KEY", BLINDING_KEY)
	resolver = ExperimentCandidateResolver(fallback)
	assert resolver.preview_bundle is None
	assert all(item.blind_id for item in resolver.inventory())
	assert all(
		Path(path).is_file()
		for item in resolver.inventory()
		for path in resolver.resolve(item.blind_id).preview_paths
	)
