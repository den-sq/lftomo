from pathlib import Path

import numpy as np
import pytest

from lftomo.config import (
	Acquisition,
	Detector,
	Recon,
	Scan,
	TIFFStackProjections,
)
from lftomo.evaluation import PreviewContract
from lftomo.experiment import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	FidelityExperimentRunner,
	Tier2Bounds,
	audit_tier2_results,
	build_tier2_plan,
	export_blinded_previews,
	write_tier2_plan,
)
from lftomo.experiment.protocol import Tier2ExperimentPlan
from lftomo.experiment.runner import ExperimentInputs, TrialExecution
from lftomo.experiment.cli import load_experiment_config
from lftomo.experiment.selection import select_candidates
from lftomo.experiment.models import OutputRecord
from lftomo.pipeline import NumpyBackend, Pipeline, PipelineState
from lftomo.pipeline.domains import DataDomain
from lftomo.pipeline.stages import create_stage
from lftomo.util.shapes import ProjOrder


HASHES = [f"{index:064x}" for index in range(1, 10)]


def experiment_config():
	return ExperimentConfig.model_validate({
		"name": "hermetic-known-winner",
		"candidate_space": {
			"pipeline": [
				{
					"id": "stripe-sort", "kind": "stripe", "algorithm": "vo_sort",
					"params": {"size": 3},
				},
				{
					"id": "stripe-large", "kind": "stripe", "algorithm": "vo_large",
					"params": {"size": 3, "snr": 2.5, "drop_ratio": 0.1},
				},
				{
					"id": "phase-filter", "kind": "phase", "algorithm": "paganin_filter",
					"sweep": {
						"kind": "list", "params": {"alpha": [0.02, 0.05, 0.1]},
					},
				},
				{
					"id": "negative-log", "kind": "phase", "algorithm": "negative_log",
					"params": {"epsilon": 1e-6},
				},
			],
		},
		"sampler": {"name": "exhaustive", "seed": 17},
		"resources": {
			"max_concurrency": 2, "memory_gb": 2, "max_output_bytes": 2000000,
		},
		"metrics": [{
			"id": "nrmse", "metric": "reference.nrmse",
			"protocol_version": "1", "direction": "minimize", "units": "ratio",
		}],
		"objectives": [{"metric_id": "nrmse"}],
		"windows": [
			{"id": "top", "role": "tuning", "y_start": 0, "y_stop": 1},
			{"id": "middle", "role": "tuning", "y_start": 1, "y_stop": 3},
			{"id": "blind", "role": "held_out", "y_start": 3, "y_stop": 4},
		],
		"fidelities": [
			{
				"id": "screen", "windows": ["top", "middle", "blind"],
				"projection_stride": 2, "projection_limit": 8,
				"memory_gb": 1, "max_output_bytes": 100000,
			},
			{
				"id": "final", "windows": ["top", "middle", "blind"],
				"projection_stride": 1, "memory_gb": 2,
			},
		],
		"aggregation": {"name": "mean", "metric_ids": ["nrmse"]},
		"output": {
			"screening": "metrics_and_previews", "final_fidelity_id": "final",
			"write_full_finalists": True, "max_bytes": 2000000,
		},
		"promotion": {
			"reduction_factor": 3, "min_survivors": 1,
			"ties": "stable_cut", "invalid": "fail", "failures": "fail",
		},
	})


def scan():
	return Scan(
		energy_keV=20,
		pixel_size_um=0.7,
		ssd_cm=8,
		n_projections=8,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=0.7, rows=4, columns=8),
		acquisition=Acquisition(n_projections=8, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path=Path("unused")),
	)


def inputs():
	return ExperimentInputs(
		dataset=DatasetIdentity(dataset_id="synthetic", content_hash=HASHES[0]),
		configuration=ConfigurationIdentity(scan_config_hash=HASHES[1]),
		code=CodeIdentity(version="test", git_commit="abcdef123456", dirty=False),
		environment=EnvironmentIdentity(
			python="3.11", platform="hermetic", backend="numpy",
			dependencies={"numpy": np.__version__},
		),
	)


def run_fragment(recon, source):
	stages = [create_stage(stage) for stage in recon.pipeline]
	assert [stage.name for stage in stages] == [
		"stripe.vo_sort", "stripe.vo_large", "phase.paganin_filter", "phase.negative_log",
	]
	state = PipelineState(
		working_array=source.copy(),
		scan=scan(),
		recon=recon,
		backend=NumpyBackend(),
		layout=ProjOrder,
		domain=DataDomain.TRANSMISSION,
	)
	return Pipeline(
		stages,
		backend=NumpyBackend(),
		validation_mode="fragment",
		initial_layout=ProjOrder,
		initial_domain=DataDomain.TRANSMISSION,
		bounded_workers=2,
	).run(state).working_array


def test_hermetic_known_winner_full_workflow_is_deterministic_and_auditable(tmp_path):
	config = experiment_config()
	rng = np.random.default_rng(20260804)
	source = np.clip(
		0.82 + rng.normal(0, 0.025, size=(8, 4, 8))
		+ np.linspace(-0.03, 0.03, 8)[:, None, None],
		0.5,
		1.0,
	).astype(np.float32)
	target_recon = config.candidate_space.model_copy(deep=True)
	target_recon.pipeline[2].params["alpha"] = 0.05
	target_recon.pipeline[2].sweep = None
	reference = run_fragment(target_recon, source)

	calls = []
	preview_images = {}

	def execute(identity, output_root):
		result = run_fragment(identity.resolved_recon, source)
		calls.append((identity.fidelity.id, identity.candidate.value))
		window_metrics = {}
		for window in identity.fidelity.windows:
			actual = result[:, window.y_start:window.y_stop]
			expected = reference[:, window.y_start:window.y_stop]
			dynamic_range = float(np.ptp(expected)) or 1.0
			nrmse = float(np.sqrt(np.mean((actual - expected) ** 2)) / dynamic_range)
			window_metrics[window.id] = {"nrmse": [nrmse]}
		if identity.fidelity.id == "screen":
			preview_images[identity.candidate.value] = result[0]
		outputs = ()
		if identity.fidelity.id == "final":
			path = output_root / "finalist.npy"
			path.parent.mkdir(parents=True, exist_ok=True)
			with path.open("wb") as stream:
				np.save(stream, result)
			outputs = (OutputRecord(
				role="finalist_output", path=path, bytes=path.stat().st_size,
			),)
		return TrialExecution(
			metrics={"nrmse": window_metrics["top"]["nrmse"][0]},
			window_metrics=window_metrics,
			outputs=outputs,
		)

	summary = FidelityExperimentRunner(config, inputs(), tmp_path, execute).run()
	assert [len(level.attempts) for level in summary.levels] == [3, 1]
	winner = summary.levels[-1].attempts[0]
	assert winner.identity.resolved_recon.pipeline[2].params["alpha"] == 0.05
	assert winner.result.metrics["nrmse"] == 0
	assert winner.result.promotion.parent_trial_id is not None
	assert winner.result.outputs[0].path.is_file()

	preview_contract = PreviewContract(
		scale_min=float(reference.min()), scale_max=float(reference.max()),
		blinded=True, format="png",
	)
	previews = export_blinded_previews(
		preview_images,
		tmp_path,
		preview_contract,
		blinding_key="hermetic-private-key",
		max_bytes=100000,
	)
	plan = build_tier2_plan(
		config,
		inputs(),
		environment_lock_hash=HASHES[2],
		projection_manifest_hash=HASHES[3],
		preview=preview_contract,
		blinding_key_id="fixture-key-v1",
		bounds=Tier2Bounds(
			max_candidates=3,
			max_screen_window_rows=2,
			max_screen_projections=8,
			max_output_bytes=2000000,
			max_memory_gb=2,
		),
	)
	plan_path = tmp_path / "tier2-plan.json"
	write_tier2_plan(plan_path, plan)
	assert Tier2ExperimentPlan.model_validate_json(plan_path.read_text()) == plan
	tampered = plan.model_copy(update={"blinding_key_id": "different-key"})
	with pytest.raises(ValueError, match="content-addressed payload"):
		write_tier2_plan(tmp_path / "tampered-plan.json", tampered)
	assert audit_tier2_results(plan, summary, previews).passed

	call_count = len(calls)
	resumed = FidelityExperimentRunner(config, inputs(), tmp_path, execute).run()
	assert len(calls) == call_count
	assert resumed.promotions == summary.promotions
	for resumed_level, original_level in zip(resumed.levels, summary.levels):
		assert resumed_level.attempts == original_level.attempts
		assert resumed_level.leaderboard == original_level.leaderboard
		assert resumed_level.preflight.model_dump(
			mode="json", exclude={"path_resources"},
		) == original_level.preflight.model_dump(
			mode="json", exclude={"path_resources"},
		)
		assert resumed_level.preflight.path_resources.accepted is True


def test_tier2_plan_rejects_dirty_code_and_missing_held_out_protocol():
	config = experiment_config()
	preview = PreviewContract(scale_min=0, scale_max=1, blinded=True)
	bounds = Tier2Bounds(
		max_candidates=3,
		max_screen_window_rows=2,
		max_screen_projections=8,
		max_output_bytes=2000000,
		max_memory_gb=2,
	)
	dirty = inputs()
	dirty.code = dirty.code.model_copy(update={"dirty": True})
	with pytest.raises(ValueError, match="clean, pinned code commit"):
		build_tier2_plan(
			config, dirty,
			environment_lock_hash=HASHES[2],
			projection_manifest_hash=HASHES[3],
			preview=preview,
			blinding_key_id="fixture-key-v1",
			bounds=bounds,
		)
	document = config.model_dump(mode="json")
	for window in document["windows"]:
		window["role"] = "tuning"
	without_holdout = ExperimentConfig.model_validate(document)
	with pytest.raises(ValueError, match="held-out window"):
		build_tier2_plan(
			without_holdout, inputs(),
			environment_lock_hash=HASHES[2],
			projection_manifest_hash=HASHES[3],
			preview=preview,
			blinding_key_id="fixture-key-v1",
			bounds=bounds,
		)


def test_documented_tier2_template_resolves_ordered_complete_candidates():
	root = Path(__file__).resolve().parents[2]
	config = load_experiment_config(root / "examples" / "experiment-tier2-template.yaml")
	candidates = select_candidates(config)

	assert len(candidates) == 32
	for candidate in candidates:
		stage_ids = [stage.id for stage in candidate.recon.pipeline]
		assert stage_ids.index("stripe-primary") < stage_ids.index("stripe-sort")
		assert stage_ids.index("phase-filter") < stage_ids.index("negative-log")
		resolved = candidate.recon.model_copy(deep=True)
		for stage in resolved.pipeline:
			stage.params.pop("calibration_manifest", None)
		[create_stage(stage) for stage in resolved.pipeline]
