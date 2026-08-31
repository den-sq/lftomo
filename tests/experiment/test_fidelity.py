from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lftomo.config import Recon
from lftomo.evaluation import ImageROI, PreviewContract
from lftomo.experiment import (
	AttemptProgress,
	CancellationToken,
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	FidelityExperimentRunner,
	PathResourceError,
	ProgressEventWriter,
	PromotionCandidate,
	PromotionDecisionStore,
	ResumePolicy,
	aggregate_window_metrics,
	export_blinded_previews,
	read_progress_events,
	successive_halving,
)
from lftomo.experiment.cli import PipelineTrialExecutor
from lftomo.experiment.events import ExecutionContext, bind_execution_context
from lftomo.experiment.runner import (
	ExperimentInputs,
	ExperimentRunPolicy,
	IndependentExperimentRunner,
)
from lftomo.util.shapes import ProjOrder
from tests.experiment.process_helpers import AlphaExecutor, WindowedAlphaExecutor


HASHES = [f"{index:064x}" for index in range(1, 20)]


def experiment(values=(0.01, 0.02, 0.03, 0.04), *, ties="stable_cut"):
	return ExperimentConfig.model_validate({
		"name": "fidelity-test",
		"candidate_space": {
			"pipeline": [{
				"id": "phase", "kind": "phase", "algorithm": "paganin",
				"sweep": {"kind": "list", "params": {"alpha": list(values)}},
			}],
		},
		"sampler": {"name": "exhaustive", "seed": 4},
		"resources": {"max_concurrency": 2},
		"metrics": [{
			"id": "quality", "metric": "quality", "protocol_version": "1",
			"direction": "minimize",
		}],
		"objectives": [{"metric_id": "quality"}],
		"windows": [
			{"id": "tune-a", "role": "tuning", "y_start": 2, "y_stop": 4},
			{"id": "tune-b", "role": "tuning", "y_start": 6, "y_stop": 8},
			{"id": "hold", "role": "held_out", "y_start": 10, "y_stop": 12},
		],
		"fidelities": [
			{
				"id": "screen", "windows": ["tune-a", "tune-b", "hold"],
				"projection_stride": 4, "projection_limit": 32,
				"memory_gb": 2, "max_output_bytes": 100000,
			},
			{
				"id": "final", "windows": ["tune-a", "tune-b", "hold"],
				"projection_stride": 1,
			},
		],
		"aggregation": {"name": "mean", "metric_ids": ["quality"]},
		"output": {
			"screening": "metrics_only", "final_fidelity_id": "final",
			"write_full_finalists": True,
		},
		"promotion": {
			"name": "successive_halving", "reduction_factor": 2,
			"min_survivors": 1, "ties": ties,
		},
	})


def inputs():
	return ExperimentInputs(
		dataset=DatasetIdentity(dataset_id="fixture", content_hash=HASHES[0]),
		configuration=ConfigurationIdentity(scan_config_hash=HASHES[1]),
		code=CodeIdentity(version="test", git_commit="abcdef123456"),
		environment=EnvironmentIdentity(
			python="3.11", platform="test", backend="numpy",
		),
	)


def alpha(identity):
	return identity.resolved_recon.pipeline[0].params["alpha"]


def candidate(index, metric, *, status="succeeded", held_out=None):
	return PromotionCandidate(
		candidate_key=HASHES[index],
		trial_id=HASHES[index + 8],
		status=status,
		tuning_metrics={"quality": metric},
		held_out_metrics={"quality": held_out},
	)


def test_window_metrics_aggregate_tuning_and_held_out_separately():
	result = aggregate_window_metrics({
		"tune-a": {"quality": [1, 3]},
		"tune-b": {"quality": [5]},
		"hold": {"quality": [99]},
	}, experiment())

	assert result.per_window["tune-a"]["quality"] == 2
	assert result.tuning["quality"] == 3.5
	assert result.held_out["quality"] == 99


def test_successive_halving_is_deterministic_ignores_held_out_and_audits_ties():
	config = experiment(ties="include_all")
	candidates = [
		candidate(1, 1, held_out=1000),
		candidate(2, 2, held_out=-1000),
		candidate(3, 2, held_out=-2000),
	]
	first = successive_halving(candidates, config, "screen", "final")
	second = successive_halving(list(reversed(candidates)), config, "screen", "final")

	assert first.decision_id == second.decision_id
	assert set(first.promoted_candidate_keys) == {HASHES[1], HASHES[2], HASHES[3]}
	assert first.cutoff_position == 3
	assert first.rankings[0].held_out_metrics == {"quality": 1000}
	assert first.rankings[1].tie_signature == first.rankings[2].tie_signature


def test_invalid_failures_and_manual_promotions_follow_explicit_policy():
	document = experiment().model_dump(mode="json")
	document["promotion"].update({"invalid": "exclude", "failures": "worst"})
	config = ExperimentConfig.model_validate(document)
	decision = successive_halving([
		candidate(1, 1),
		candidate(2, None),
		candidate(3, 0, status="failed"),
	], config, "screen", "final")
	by_key = {item.candidate_key: item for item in decision.rankings}
	assert not by_key[HASHES[2]].eligible
	assert not by_key[HASHES[3]].eligible
	assert not by_key[HASHES[3]].promoted
	assert by_key[HASHES[3]].reason == (
		"terminal status failed; ranked as worst but ineligible for promotion"
	)
	failed_only = successive_halving(
		[candidate(3, 0, status="failed")], config, "screen", "final",
	)
	assert failed_only.promoted_candidate_keys == ()
	with pytest.raises(ValueError, match="manual promotion requires succeeded"):
		successive_halving(
			[candidate(3, 0, status="failed")], config, "screen", "final",
			manual_promote={HASHES[3]},
		)

	manual = successive_halving(
		[candidate(1, 1), candidate(2, 2)],
		config,
		"screen",
		"final",
		manual_promote={HASHES[2]},
	)
	assert manual.manual_override
	assert manual.promoted_candidate_keys == (HASHES[2],)


def test_promotion_store_reuses_or_explicitly_invalidates_stale_boundary(tmp_path):
	config = experiment()
	first = successive_halving(
		[candidate(1, 1), candidate(2, 2)], config, "screen", "final",
	)
	changed = successive_halving(
		[candidate(1, 3), candidate(2, 2)], config, "screen", "final",
	)
	store = PromotionDecisionStore(tmp_path)
	assert store.publish(first).decision_id == first.decision_id
	assert store.publish(first).decision_id == first.decision_id
	with pytest.raises(ValueError, match="promotion boundary is stale"):
		store.publish(changed)
	assert store.publish(changed, stale="invalidate").decision_id == changed.decision_id
	assert any((tmp_path / "quarantine" / "promotions").rglob("*.json"))


def test_blinded_previews_share_scale_crop_and_have_reversible_private_mapping(tmp_path):
	contract = PreviewContract(
		scale_min=0, scale_max=10,
		crop=ImageROI(y_start=1, y_stop=3, x_start=1, x_stop=4),
		blinded=True,
		format="png",
	)
	bundle = export_blinded_previews({
		HASHES[1]: np.arange(16, dtype=np.float32).reshape(4, 4),
		HASHES[2]: np.full((4, 4), 5, dtype=np.float32),
	}, tmp_path, contract, blinding_key="private-review-key")

	assert bundle.mapping_path.is_file()
	assert bundle.mapping_path == tmp_path / "private" / "preview-mapping" / "mapping.json"
	assert not any((tmp_path / "previews").rglob("*.json"))
	assert len({item.blind_id for item in bundle.entries}) == 2
	assert all(item.candidate_key not in str(item.files[0]) for item in bundle.entries)
	constant_entry = next(item for item in bundle.entries if item.candidate_key == HASHES[2])
	image = cv2.imread(str(constant_entry.files[0]), cv2.IMREAD_UNCHANGED)
	assert image.shape == (2, 3)
	assert np.all(image == np.uint16(round(0.5 * 65535)))
	updated = export_blinded_previews(
		{HASHES[3]: np.full((4, 4), 7, dtype=np.float32)},
		tmp_path,
		contract,
		blinding_key="private-review-key",
	)
	assert {item.candidate_key for item in updated.entries} == {
		HASHES[1], HASHES[2], HASHES[3],
	}
	with pytest.raises(ValueError, match="preview output would exceed"):
		export_blinded_previews(
			{HASHES[3]: np.ones((4, 4))}, tmp_path / "guarded", contract,
			blinding_key="key", max_bytes=1,
		)
	assert not any((tmp_path / "guarded").rglob("*.png"))


def test_full_fidelity_workflow_promotes_only_winners_and_resumes_boundary(tmp_path):
	config = experiment()
	writer = ProgressEventWriter(tmp_path, config.experiment_id)
	call_root = tmp_path / "calls"
	executor = WindowedAlphaExecutor(call_root)

	first = FidelityExperimentRunner(
		config, inputs(), tmp_path, executor, event_writer=writer,
	).run()
	assert [len(level.attempts) for level in first.levels] == [4, 2]
	calls = [
		json.loads(path.read_text(encoding="utf-8"))
		for path in call_root.glob("*.json")
	]
	assert len(calls) == 6
	assert sorted(item["alpha"] for item in calls if item["fidelity"] == "final") == [
		0.01, 0.02,
	]
	assert all(
		attempt.result.promotion.from_fidelity == "screen"
		for attempt in first.levels[1].attempts
	)
	events = read_progress_events(tmp_path / "progress.jsonl")
	assert [event.total_candidates for event in events if event.kind == "fidelity.started"] == [
		4, 2,
	]
	assert [
		event.completed_candidates for event in events if event.kind == "fidelity.completed"
	] == [4, 2]

	prior_sequence = events[-1].sequence
	resumed = FidelityExperimentRunner(
		config, inputs(), tmp_path, executor, event_writer=writer,
	).run()
	assert len(list(call_root.glob("*.json"))) == 6
	assert resumed.promotions[0].decision_id == first.promotions[0].decision_id
	resume_events = [
		event for event in read_progress_events(tmp_path / "progress.jsonl")
		if event.sequence > prior_sequence and event.kind == "fidelity.completed"
	]
	assert [(event.total_candidates, event.reused_candidates) for event in resume_events] == [
		(4, 4), (2, 2),
	]


def test_level_specific_memory_and_output_guards_apply_during_preflight(tmp_path):
	config = experiment()
	runner = IndependentExperimentRunner(
		config,
		inputs(),
		tmp_path,
		AlphaExecutor(),
		fidelity_id="screen",
		policy=ExperimentRunPolicy(
			estimated_memory_bytes_per_trial=1500 * 1024**2,
			estimated_output_bytes_per_trial=30000,
		),
	)
	with pytest.raises(PathResourceError, match="override the configured guard"):
		runner.prepare()
	overridden = IndependentExperimentRunner(
		config,
		inputs(),
		tmp_path,
		AlphaExecutor(),
		fidelity_id="screen",
		policy=ExperimentRunPolicy(
			estimated_memory_bytes_per_trial=1500 * 1024**2,
			estimated_output_bytes_per_trial=30000,
			override_limits=True,
		),
	)
	assert overridden.prepare()[0].max_concurrency == 1


def test_fidelity_window_injection_enforces_projection_and_finalist_writer(monkeypatch):
	config = experiment(values=(0.01,))
	executor = PipelineTrialExecutor(config, "scan.yaml")
	scan = SimpleNamespace(
		detector=SimpleNamespace(rows=20),
		n_projections=100,
		metadata={},
		projection_shape=lambda: ProjOrder(100, 20, 8),
	)
	monkeypatch.setattr("lftomo.experiment.cli.load_scan", lambda _path: (scan, _path))
	recon = Recon.model_validate({
		"pipeline": [
			{"id": "read", "kind": "read", "algorithm": "tiff_stack"},
			{"id": "metrics", "kind": "evaluate", "algorithm": "metrics"},
			{"id": "write", "kind": "write", "algorithm": "tiff_slices"},
		],
	})
	window = SimpleNamespace(
		id="tune-a", y_start=2, y_stop=4, halo=1,
		theta_start=5, theta_stop=90,
	)
	base = dict(
		resolved_recon=recon,
		fidelity=SimpleNamespace(
			id="screen", projection_limit=10, projection_stride=4,
		),
	)
	screen = executor._window_recon(SimpleNamespace(**base), window, SimpleNamespace(
		__truediv__=lambda self, value: self,
	))
	read = screen.pipeline[0]
	assert read.params == {
		"y_start": 1, "y_stop": 5, "theta_start": 5, "theta_stop": 45,
		"theta_stride": 4,
	}
	assert screen.pipeline[1].params["windows"] == [{
		"id": "tune-a", "slice_start": 1, "slice_stop": 3,
	}]
	assert all(stage.kind != "write" for stage in screen.pipeline)

	base["fidelity"] = SimpleNamespace(
		id="final", projection_limit=None, projection_stride=1,
	)
	final = executor._window_recon(SimpleNamespace(**base), window, Path("final-window"))
	assert any(stage.kind == "write" for stage in final.pipeline)


def test_pipeline_span_observer_reports_resolved_variant_positions(tmp_path):
	executor = PipelineTrialExecutor(experiment(values=(0.01,)), "scan.yaml")
	configs = (
		SimpleNamespace(id="stripe", kind="stripe", algorithm="vo_sort"),
		SimpleNamespace(id="reconstruct", kind="recon", algorithm="gridrec"),
	)
	writer = ProgressEventWriter(tmp_path, "experiment", durable=True)
	context = ExecutionContext(
		writer=writer,
		progress_sink=lambda _progress: None,
		cancellation=CancellationToken(),
		progress=AttemptProgress(
			experiment_id="experiment", fidelity_id="screen",
			candidate_key="candidate", trial_id="trial", attempt_id="attempt",
			state="started", updated_at=datetime.now(timezone.utc),
		),
	)
	from lftomo.pipeline import (
		DispatchKind,
		ExecutionKind,
		ExecutionSpan,
		ExecutionSpanEvent,
		ExecutionSpanPhase,
		LogicalStageMember,
	)
	members = (
		LogicalStageMember(0, "stripe.vo_sort", "tests.Stripe", DispatchKind.PROCESS),
		LogicalStageMember(1, "recon.gridrec", "tests.Recon", DispatchKind.ENGINE),
	)
	span = ExecutionSpan(
		"run", "run:0", 0, 1, ExecutionKind.BOUNDED_SINOGRAM, 0, 2, members,
	)
	with bind_execution_context(context):
		observe = executor._execution_observer(configs)
		observe(ExecutionSpanEvent(ExecutionSpanPhase.STARTED, span))
		observe(ExecutionSpanEvent(
			ExecutionSpanPhase.COMPLETED, span, coordinator_wall_seconds=0.25,
		))
	events = read_progress_events(tmp_path / "progress.jsonl")
	assert [
		[(member.occurrence_id, member.position, member.total) for member in event.segment_members]
		for event in events
	] == [
		[("stripe", 1, 2), ("reconstruct", 2, 2)],
		[("stripe", 1, 2), ("reconstruct", 2, 2)],
	]
	assert all(event.stage_occurrence_id is None for event in events)
	assert events[-1].segment_coordinator_wall_seconds == 0.25


def test_pipeline_executor_publishes_blinded_screening_previews(monkeypatch, tmp_path):
	document = experiment(values=(0.01,)).model_dump(mode="json")
	document["output"]["screening"] = "metrics_and_previews"
	config = ExperimentConfig.model_validate(document)
	executor = PipelineTrialExecutor(config, "scan.yaml")
	scan = SimpleNamespace(
		detector=SimpleNamespace(rows=20), n_projections=100, metadata={},
		projection_shape=lambda: ProjOrder(100, 20, 8),
	)
	monkeypatch.setattr("lftomo.experiment.cli.load_scan", lambda _path: (scan, _path))
	monkeypatch.setenv("LFTOMO_PREVIEW_BLINDING_KEY", "test-private-key")
	recon = Recon.model_validate({
		"pipeline": [
			{"id": "read", "kind": "read", "algorithm": "tiff_stack"},
			{"id": "metrics", "kind": "evaluate", "algorithm": "metrics"},
		],
	})
	identity = SimpleNamespace(
		candidate=SimpleNamespace(value="a" * 64),
		resolved_recon=recon,
		fidelity=SimpleNamespace(
			id="screen", projection_limit=10, projection_stride=2,
			windows=(SimpleNamespace(
				id="tune-a", y_start=2, y_stop=4, halo=0,
				theta_start=None, theta_stop=None,
			),),
		),
	)
	artifact = {
		"aggregates": {"quality": {"value": 0.25}},
		"observations": [{
			"metric": "quality", "window_id": "tune-a", "valid": True,
			"value": 0.25,
		}],
		"preview": {
			"scale_min": 0, "scale_max": 1, "blinded": True, "format": "png",
		},
	}
	monkeypatch.setattr(
		"lftomo.experiment.cli.run_reconstruction",
		lambda *_args: [SimpleNamespace(artifacts={
			"metrics": artifact,
			"preview_images": {"tune-a-0": np.full((3, 3), 0.5)},
		})],
	)

	result = executor(identity, tmp_path / "trials" / ("b" * 64))

	assert result.metrics == {"quality": 0.25}
	assert result.outputs[0].role == "blinded_preview"
	assert result.outputs[0].path.is_file()
	assert (tmp_path / "private" / "preview-mapping" / "mapping.json").is_file()
	assert not any((tmp_path / "previews").rglob("*.json"))
	assert "preview_images" not in str(result.artifacts)
