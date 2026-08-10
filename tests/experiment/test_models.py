from datetime import datetime, timedelta, timezone

from pydantic import ValidationError
import pytest

from lftomo.config import Recon
from lftomo.experiment import (
	CandidateKey,
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	ExperimentManifest,
	FailureRecord,
	FidelityIdentity,
	FinalResult,
	MetricProtocolIdentity,
	OutputRecord,
	SamplerIdentity,
	TrialAttemptRecord,
	TrialDescription,
	TrialIdentity,
	WindowIdentity,
	canonical_json,
	compare_trial_identities,
	select_candidates,
)


HASHES = [f"{index:064x}" for index in range(1, 16)]
NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


def resolved_recon(*, alpha=0.01, output_path=None):
	return Recon.model_validate({
		"pipeline": [{
			"id": "phase",
			"kind": "phase",
			"algorithm": "paganin",
			"params": {"alpha": alpha},
		}],
		"output_path": output_path,
	})


def experiment_document(candidate=None, *, sampler=None):
	return {
		"schema_version": 1,
		"name": "stripe-study",
		"candidate_space": (candidate or resolved_recon()).model_dump(mode="json"),
		"sampler": sampler or {"name": "exhaustive", "seed": 17},
		"resources": {"max_concurrency": 2, "memory_gb": 8},
		"metrics": [{
			"id": "nrmse",
			"metric": "reference.nrmse",
			"protocol_version": "1",
			"direction": "minimize",
		}],
		"objectives": [{"metric_id": "nrmse"}],
		"windows": [
			{"id": "top", "y_start": 0, "y_stop": 8},
			{"id": "bottom", "y_start": 24, "y_stop": 32},
		],
		"fidelities": [{
			"id": "screen",
			"windows": ["top", "bottom"],
			"projection_stride": 2,
		}],
		"aggregation": {"name": "mean", "metric_ids": ["nrmse"]},
		"output": {"screening": "metrics_only"},
	}


@pytest.mark.parametrize("sampler_name", ["sobol", "random"])
def test_sampled_structural_variants_are_atomically_materialized(sampler_name):
	structural = Recon.model_validate({
		"pipeline": [
			{"id": "read", "kind": "read", "algorithm": "hdf5_dxchange"},
			{"id": "stripe-a", "kind": "stripe", "algorithm": "fw"},
			{"id": "stripe-b", "kind": "stripe", "algorithm": "vo_sort"},
			{"id": "write", "kind": "write", "algorithm": "tiff_slices"},
		],
		"variants": [
			{"name": "fourier", "stages": ["read", "stripe-a", "write"]},
			{"name": "sorting", "stages": ["read", "stripe-b", "write"]},
		],
	})
	original = structural.model_dump(mode="json")
	config = ExperimentConfig.model_validate(experiment_document(
		structural,
		sampler={"name": sampler_name, "seed": 41, "budget": 8},
	))

	first = select_candidates(config)
	second = select_candidates(config)

	expected_stages = {
		"fourier": ("read", "stripe-a", "write"),
		"sorting": ("read", "stripe-b", "write"),
	}
	assert {item.parameters["pipeline.variant"] for item in first} == set(expected_stages)
	assert [item.candidate_key for item in first] == [item.candidate_key for item in second]
	assert [item.sampler.coordinate for item in first] == [
		item.sampler.coordinate for item in second
	]
	for item in first:
		assert item.recon.variants == []
		assert all(stage.sweep is None for stage in item.recon.pipeline)
		variant = item.parameters["pipeline.variant"]
		assert item.sampler.explicit_case == variant
		assert tuple(stage.id for stage in item.recon.pipeline) == expected_stages[variant]
	assert structural.model_dump(mode="json") == original


def trial_identity(*, recon=None):
	recon = recon or resolved_recon()
	return TrialIdentity(
		candidate=CandidateKey.from_recon(recon),
		resolved_recon=recon,
		dataset=DatasetIdentity(
			dataset_id="scan-a", content_hash=HASHES[0], acquisition_id="acq-a",
		),
		configuration=ConfigurationIdentity(
			scan_config_hash=HASHES[1], calibration_hash=HASHES[2],
		),
		code=CodeIdentity(version="0.1.0", git_commit="abcdef123456", dirty=False),
		environment=EnvironmentIdentity(
			python="3.11.9",
			platform="linux-x86_64",
			backend="numpy",
			dependencies={"numpy": "2.1", "scipy": "1.15"},
		),
		sampler=SamplerIdentity(
			name="sobol", seed=17, coordinate={"alpha": 0.25},
		),
		fidelity=FidelityIdentity(
			id="screen",
			windows=(
				WindowIdentity(id="top", y_start=0, y_stop=8),
				WindowIdentity(id="bottom", y_start=24, y_stop=32),
			),
			projection_stride=2,
		),
		metric_protocols=(
			MetricProtocolIdentity(
				metric_id="nrmse", protocol_version="1", config_hash=HASHES[3],
			),
			MetricProtocolIdentity(
				metric_id="validity", protocol_version="1", config_hash=HASHES[4],
			),
		),
	)


def replace(model, **updates):
	document = model.model_dump(mode="json")
	document.update(updates)
	return type(model).model_validate(document)


def test_experiment_config_composes_recon_and_validates_references():
	config = ExperimentConfig.model_validate(experiment_document())

	assert isinstance(config.candidate_space, Recon)
	assert config.candidate_space.pipeline[0].id == "phase"
	assert len(config.experiment_id) == 64
	with pytest.raises(ValidationError, match="unknown metrics"):
		ExperimentConfig.model_validate({
			**experiment_document(),
			"objectives": [{"metric_id": "missing"}],
		})
	with pytest.raises(ValidationError, match="unknown windows"):
		ExperimentConfig.model_validate({
			**experiment_document(),
			"fidelities": [{"id": "screen", "windows": ["missing"]}],
		})
	with pytest.raises(ValidationError, match="extra_forbidden"):
		ExperimentConfig.model_validate({**experiment_document(), "parameter_space": {}})
	with pytest.raises(ValidationError, match="literal_error"):
		ExperimentConfig.model_validate({**experiment_document(), "schema_version": 2})


def test_exhaustive_rejects_continuous_space_while_budgeted_sampler_accepts_it():
	continuous = Recon.model_validate({
		"pipeline": [{
			"id": "phase",
			"kind": "phase",
			"algorithm": "paganin",
			"sweep": {
				"kind": "domain",
				"params": {"alpha": {"kind": "log", "start": 0.001, "stop": 0.1}},
			},
		}],
	})
	with pytest.raises(ValidationError, match="cannot consume continuous"):
		ExperimentConfig.model_validate(experiment_document(continuous))
	config = ExperimentConfig.model_validate(experiment_document(
		continuous, sampler={"name": "sobol", "seed": 4, "budget": 32},
	))
	assert config.sampler.budget == 32
	with pytest.raises(ValidationError, match="explicit budget"):
		ExperimentConfig.model_validate(experiment_document(
			continuous, sampler={"name": "sobol", "seed": 4},
		))


def test_candidate_key_and_trial_id_have_distinct_stable_identity_domains():
	identity = trial_identity()
	assert identity.candidate.value != identity.trial_id.value

	permuted = replace(
		identity,
		metric_protocols=list(reversed(identity.model_dump(mode="json")["metric_protocols"])),
		fidelity={
			**identity.fidelity.model_dump(mode="json"),
			"windows": list(reversed(identity.fidelity.model_dump(mode="json")["windows"])),
		},
	)
	assert permuted.trial_id == identity.trial_id

	with_other_path = trial_identity(recon=resolved_recon(output_path="/another/location"))
	assert with_other_path.candidate == identity.candidate
	assert with_other_path.trial_id == identity.trial_id

	resampled = replace(identity, sampler={
		"name": "random", "seed": 99, "coordinate": {"alpha": 0.75},
	})
	assert resampled.sampler != identity.sampler
	assert resampled.trial_id == identity.trial_id
	assert compare_trial_identities(identity, resampled).compatible
	assert "sampler" not in identity.identity_payload()

	changed = trial_identity(recon=resolved_recon(alpha=0.02))
	assert changed.candidate != identity.candidate
	assert changed.trial_id != identity.trial_id


@pytest.mark.parametrize(
	"field, update, expected_diagnostic",
	[
		("dataset", {"dataset_id": "scan-a", "content_hash": HASHES[5]}, "dataset.content_hash"),
		("configuration", {"scan_config_hash": HASHES[6], "calibration_hash": HASHES[2]}, "configuration.scan_config_hash"),
		("code", {"version": "0.1.0", "git_commit": "123456789abc", "dirty": False}, "code.git_commit"),
		("environment", {"python": "3.12", "platform": "linux-x86_64", "backend": "numpy"}, "environment"),
		("fidelity", {"id": "full", "windows": [{"id": "top", "y_start": 0, "y_stop": 8}]}, "fidelity"),
	],
)
def test_trial_identity_sensitivity_and_field_specific_compatibility(
	field, update, expected_diagnostic,
):
	expected = trial_identity()
	actual = replace(expected, **{field: update})
	report = compare_trial_identities(expected, actual)

	assert not report.compatible
	assert any(item.field.startswith(expected_diagnostic) for item in report.diagnostics)
	assert expected.trial_id != actual.trial_id


def test_descriptive_metadata_does_not_participate_in_compatibility():
	identity = trial_identity()
	a = TrialDescription(label="first", dataset_path="/mnt/a", notes="old")
	b = TrialDescription(label="renamed", dataset_path="/mnt/b", notes="new")

	assert a != b
	assert compare_trial_identities(identity, identity).compatible


def test_trial_identity_requires_a_matching_fully_resolved_candidate():
	recon = resolved_recon()
	with pytest.raises(ValidationError, match="candidate key does not match"):
		TrialIdentity.model_validate({
			**trial_identity().model_dump(mode="json"),
			"candidate": {"value": HASHES[8]},
		})
	unresolved = Recon.model_validate({
		"pipeline": [{
			"id": "phase",
			"kind": "phase",
			"algorithm": "paganin",
			"sweep": {"kind": "list", "params": {"alpha": [0.01, 0.02]}},
		}],
	})
	with pytest.raises(ValidationError, match="fully resolved"):
		TrialIdentity.model_validate({
			**trial_identity().model_dump(mode="json"),
			"candidate": CandidateKey.from_recon(unresolved).model_dump(mode="json"),
			"resolved_recon": unresolved.model_dump(mode="json"),
		})


def test_attempt_lifecycle_forbids_partial_output_from_appearing_successful():
	identity = trial_identity()
	base = {
		"attempt_id": "attempt-1",
		"sequence": 0,
		"trial_id": identity.trial_id,
		"identity": identity,
		"created_at": NOW,
	}
	pending = TrialAttemptRecord(**base)
	assert pending.status == "pending"
	with pytest.raises(ValidationError, match="pending attempts"):
		TrialAttemptRecord(**base, result={
			"trial_id": identity.trial_id,
			"attempt_id": "attempt-1",
			"finalized_at": NOW,
			"resolved_recon": identity.resolved_recon,
			"metrics": {},
		})
	with pytest.raises(ValidationError, match="finalized result"):
		TrialAttemptRecord(
			**base, status="succeeded", started_at=NOW,
			finished_at=NOW + timedelta(seconds=1),
		)

	result = FinalResult(
		trial_id=identity.trial_id,
		attempt_id="attempt-1",
		finalized_at=NOW + timedelta(seconds=1),
		resolved_recon=identity.resolved_recon,
		metrics={"nrmse": 0.1},
		outputs=(OutputRecord(role="preview", path="preview.png", bytes=120),),
	)
	succeeded = TrialAttemptRecord(
		**base,
		status="succeeded",
		started_at=NOW,
		finished_at=NOW + timedelta(seconds=1),
		elapsed_seconds=1,
		peak_memory_bytes=1024,
		result=result,
	)
	assert succeeded.result.complete is True
	assert Recon.model_validate(succeeded.result.resolved_recon.model_dump(mode="json"))

	failed = TrialAttemptRecord(
		**{**base, "attempt_id": "attempt-2", "sequence": 1},
		status="failed",
		started_at=NOW,
		finished_at=NOW + timedelta(seconds=1),
		failure=FailureRecord(type="ValueError", message="bad ROI", stage_id="evaluate"),
	)
	assert failed.result is None


def test_manifest_and_yaml_json_serialization_are_deterministic_and_versioned():
	config = ExperimentConfig.model_validate(experiment_document())
	manifest = ExperimentManifest(
		experiment_id=config.experiment_id,
		config=config,
		created_at=NOW,
		attempt_ids=("attempt-1",),
	)
	assert canonical_json(manifest) == canonical_json(
		ExperimentManifest.model_validate_json(manifest.model_dump_json())
	)
	assert ExperimentManifest.from_yaml(manifest.to_yaml()) == manifest
	with pytest.raises(ValidationError, match="experiment_id"):
		ExperimentManifest(
			experiment_id=HASHES[9], config=config, created_at=NOW,
		)
