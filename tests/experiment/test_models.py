from datetime import datetime, timedelta, timezone

import numpy as np
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
	FidelityReference,
	FinalResult,
	MetricProtocolIdentity,
	OutputRecord,
	SamplerIdentity,
	TrialAttemptRecord,
	TrialDescription,
	TrialIdentity,
	WindowReference,
	canonical_json,
	compare_trial_identities,
	select_candidates,
)
from lftomo.util.converter import CONVERTER
from lftomo.util.shapes import ProjOrder
from lftomo.util.window import TomoWindow


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


def trial_identity(*, recon=None, fidelity=None):
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
		fidelity=fidelity or FidelityIdentity(
			id="screen",
			windows={
				"top": TomoWindow.from_bounds(
					ProjOrder(2, 32, 1), y=(0, 8), theta=(0, 2, 2), dtype=np.uint16,
				),
				"bottom": TomoWindow.from_bounds(
					ProjOrder(2, 32, 1), y=(24, 32), theta=(0, 2, 2), dtype=np.uint16,
				),
			},
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
			"windows": dict(reversed(list(identity.fidelity.model_dump(mode="json")["windows"].items()))),
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

	# Changing a window's role changes the evaluation identity.
	held_out = trial_identity(fidelity=FidelityIdentity(
		id="screen", windows=identity.fidelity.windows, held_out_window_ids=("bottom",),
	))
	assert held_out.trial_id != identity.trial_id

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
		("fidelity", {"id": "full", "windows": {
			"top": TomoWindow.from_bounds(ProjOrder(2, 32, 1), y=(0, 8), dtype=np.uint16),
		}}, "fidelity"),
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


def test_native_build_changes_numerical_identity_at_the_same_package_version():
	expected = trial_identity()
	environment = expected.environment.model_dump(mode="python")
	environment["libtomo"] = {"version": "1.15.3", "build": "cuda12.9h1234567_102"}
	stock = replace(expected, environment=environment)
	environment["libtomo"] = {"version": "1.15.3", "build": "cuda12.9h1234567_densq1_103"}
	patched = replace(expected, environment=environment)

	report = compare_trial_identities(stock, patched)

	assert not report.compatible
	assert [item.field for item in report.diagnostics] == ["environment.libtomo.build"]
	assert stock.trial_id != patched.trial_id
	for known in (stock, patched):
		assert known.trial_id != expected.trial_id
		assert not compare_trial_identities(expected, known).compatible


def test_persisted_attempt_without_native_identity_keeps_legacy_trial_hash():
	# Frozen using the pre-libtomo TrialIdentity at 15dd32283e84. Recomputing
	# this expectation with the current serializer would hide migration bugs.
	legacy_hash = "829bfd722bf0c7efd4e4eae2a8385f9d06aa2f77184200699c8ad7c1724e9589"
	legacy_identity = trial_identity().model_dump(mode="json")
	legacy_identity["environment"].pop("libtomo")
	serialized = canonical_json({
		"schema_version": 1,
		"attempt_id": "legacy-attempt",
		"sequence": 0,
		"trial_id": {"algorithm": "sha256", "value": legacy_hash},
		"identity": legacy_identity,
		"created_at": NOW.isoformat(),
	})

	restored = TrialAttemptRecord.model_validate_json(serialized)

	assert restored.identity.environment.libtomo is None
	assert "libtomo" not in restored.identity.identity_payload()["environment"]
	assert restored.identity.trial_id.value == legacy_hash
	assert restored.trial_id.value == legacy_hash
	roundtrip = TrialAttemptRecord.model_validate_json(restored.model_dump_json())
	assert roundtrip.model_dump(mode="json") == restored.model_dump(mode="json")
	assert roundtrip.trial_id.value == legacy_hash


def test_descriptive_metadata_does_not_participate_in_compatibility():
	identity = trial_identity()
	a = TrialDescription(label="first", dataset_path="/mnt/a", notes="old")
	b = TrialDescription(label="renamed", dataset_path="/mnt/b", notes="new")

	assert a != b
	assert compare_trial_identities(identity, identity).compatible


def test_realized_halo_and_sampling_are_the_fidelity_identity_authority():
	reference = FidelityReference(
		id="screen", windows=["edge"], projection_stride=2, projection_limit=2,
	)
	window = WindowReference(
		id="edge", y_start=1, y_stop=3, theta_start=1, theta_stop=8, halo=4,
	)
	clipped = FidelityIdentity.from_reference(
		reference, [window], TomoWindow.from_bounds(ProjOrder(8, 5, 4), dtype=np.uint16),
	)
	unclipped = FidelityIdentity.from_reference(
		reference, [window], TomoWindow.from_bounds(ProjOrder(8, 9, 4), dtype=np.uint16),
	)
	selected = clipped.windows["edge"]
	assert selected.pad_before.Y == 1
	assert selected.pad_after.Y == 2
	assert unclipped.windows["edge"].pad_after.Y == 4
	assert selected.axis_slice("Theta") == slice(1, 5, 2)
	assert selected.frange.length.Theta == 2
	assert trial_identity(fidelity=clipped).trial_id != trial_identity(fidelity=unclipped).trial_id

	# Both requests exhaust the same available source context. The larger
	# requested halo must not survive as a second identity-bearing geometry.
	larger_request = window.model_copy(update={"halo": 8})
	equivalent = FidelityIdentity.from_reference(
		reference, [larger_request],
		TomoWindow.from_bounds(ProjOrder(8, 5, 4), dtype=np.uint16),
	)
	assert trial_identity(fidelity=equivalent).trial_id == trial_identity(fidelity=clipped).trial_id

	# Once the requested context is available, extra source rows do not
	# change the evaluated geometry or its identity.
	more_context = FidelityIdentity.from_reference(
		reference, [window], TomoWindow.from_bounds(ProjOrder(8, 13, 4), dtype=np.uint16),
	)
	assert trial_identity(fidelity=more_context).trial_id == trial_identity(fidelity=unclipped).trial_id


def test_window_reference_reverse_rejects_unrepresentable_asymmetric_geometry():
	physical_shape = ProjOrder(8, 10, 4)
	window = TomoWindow.from_bounds(physical_shape, y=(2, 4), halo=(1, 3))
	with pytest.raises(ValueError, match="cannot be represented"):
		WindowReference.from_window(window, physical_shape, id="asymmetric")


@pytest.mark.parametrize("invalid", [
	{"windows": {"broken": {"type": "TomoWindow"}}},
	{"windows": {"broken": {"type": "unknown"}}},
	{"held_out_window_ids": ["unknown"]},
])
def test_corrupt_fidelity_documents_use_existing_value_error_handling(invalid):
	# Persisted-record readers handle ValueError for corrupt geometry and roles;
	# cattrs ExceptionGroups must not escape that established handling path.
	document = trial_identity().fidelity.model_dump(mode="json")
	document.update(invalid)
	with pytest.raises(ValueError):
		FidelityIdentity.model_validate(document)


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
	config.windows[0].halo = 3
	serialized = config.to_yaml()
	restored = ExperimentConfig.from_yaml(serialized)
	assert restored == config
	assert restored.to_yaml() == serialized

	physical_shape = ProjOrder(8, 5, 4)
	window = TomoWindow.from_bounds(
		physical_shape, y=(1, 3), theta=(1, 7, 2), halo=4,
		dtype=np.uint16, work_dtype=np.float64,
	)
	reference = WindowReference.from_window(
		window, physical_shape, id="edge", role="held_out",
	)
	assert reference.halo == 2  # Smallest request that reproduces the clipped pads.
	assert reference.role == "held_out"
	restored_window = reference.to_window(
		physical_shape, projection_stride=window.step.Theta,
		dtype=window.dtype, work_dtype=window.work_dtype,
	)
	assert CONVERTER.unstructure(restored_window) == CONVERTER.unstructure(window)
	identity = trial_identity(fidelity=FidelityIdentity(
		id="screen", windows={"edge": window}, held_out_window_ids=("edge",),
	))
	assert identity.fidelity.windows["edge"] is window
	assert identity.model_dump(mode="json")["fidelity"]["windows"] == {
		"edge": CONVERTER.unstructure(window),
	}
	restored_identity = TrialIdentity.model_validate_json(identity.model_dump_json())
	assert restored_identity.model_dump(mode="json") == identity.model_dump(mode="json")
	assert restored_identity.trial_id == identity.trial_id

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
