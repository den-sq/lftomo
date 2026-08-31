import json

import numpy as np
import pytest
import tifffile as tf

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.evaluation import (
	AggregationConfig,
	EvaluationArtifact,
	EvaluationStage,
	METRIC_REGISTRY,
	PROTOCOL_VERSION,
	ReferenceMetricRequest,
	RingMetricRequest,
	RoiNoiseCnrMetricRequest,
	ValidityMetricRequest,
	validate_experiment_metric_references,
)
from lftomo.experiment import ExperimentConfig
from lftomo.pipeline import BufferSpec, DataDomain, Pipeline, PipelineState, PRESERVE
from lftomo.pipeline.stages import STAGE_REGISTRY, create_stage
from lftomo.reference import NumericalTolerance, compare_slice
from lftomo.util.shapes import ReconOrder


def _scan(slices=3, size=24):
	return Scan(
		energy_keV=20,
		pixel_size_um=1,
		n_projections=8,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1, rows=slices, columns=size),
		acquisition=Acquisition(n_projections=8, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


def _state(array, *, artifacts=None):
	return PipelineState(
		working_array=array,
		scan=_scan(*array.shape[:1], size=array.shape[2]),
		recon=Recon(pipeline=[StageConfig(kind="evaluate", algorithm="metrics")]),
		artifacts=artifacts or {},
		buffer_spec=BufferSpec(ReconOrder(*array.shape), array.dtype),
		domain=DataDomain.RECON,
	)


def _values(request, image, context=None):
	return {
		value.metric: value
		for value in METRIC_REGISTRY.evaluate(image, request, context or {})
	}


def test_registry_declares_complete_typed_first_metric_set():
	definitions = {item.name: item for item in METRIC_REGISTRY.definitions}

	assert set(definitions) == {
		"reference.nrmse",
		"reference.correlation",
		"reference.normalized_bias",
		"validity.nonfinite_count",
		"validity.clipped_fraction",
		"artifact.ring_score",
		"roi.background_noise",
		"roi.cnr",
	}
	for definition in definitions.values():
		assert definition.direction in {"minimize", "maximize", "constraint"}
		assert definition.units
		assert definition.normalization
		assert definition.protocol_version == PROTOCOL_VERSION
		assert definition.undefined in {"record_invalid", "fail_stage"}
		assert definition.resource_scope == "slice"


def test_reference_adapter_reproduces_existing_comparison_behavior(tmp_path):
	reference = np.arange(36, dtype=np.float32).reshape(6, 6)
	generated = reference * np.float32(1.01) + np.float32(0.25)
	path = tmp_path / "reference.tif"
	tf.imwrite(path, reference)
	existing = compare_slice(generated, path, NumericalTolerance())
	values = _values(
		ReferenceMetricRequest(),
		generated,
		{"reference": reference},
	)

	assert values["reference.nrmse"].value == pytest.approx(existing.nrmse)
	assert values["reference.correlation"].value == pytest.approx(existing.correlation)
	assert values["reference.normalized_bias"].value == pytest.approx(existing.normalized_bias)


def test_validity_metrics_respond_monotonically_to_nonfinite_and_clipping():
	clean = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
	bad = clean.copy()
	bad.flat[:5] = np.nan
	bad.flat[5:15] = 1
	request = ValidityMetricRequest(clip_min=0, clip_max=1)
	clean_values = _values(request, clean)
	bad_values = _values(request, bad)

	assert clean_values["validity.nonfinite_count"].value == 0
	assert bad_values["validity.nonfinite_count"].value == 5
	assert (
		bad_values["validity.clipped_fraction"].value
		> clean_values["validity.clipped_fraction"].value
	)


def test_ring_score_has_expected_monotonic_synthetic_response():
	y, x = np.indices((96, 96), dtype=np.float64)
	radius = np.hypot(y - 47.5, x - 47.5)
	base = radius / radius.max()
	request = RingMetricRequest(smoothing_sigma=2, inner_radius=4, outer_radius=44)
	scores = []
	for amplitude in (0.0, 0.1, 0.3):
		image = base + amplitude * np.sin(radius * np.pi)
		scores.append(_values(request, image)["artifact.ring_score"].value)

	assert scores[0] < scores[1] < scores[2]


def test_explicit_roi_noise_and_cnr_respond_to_noise_and_contrast():
	rng = np.random.default_rng(4)
	low_noise = np.zeros((24, 24), dtype=np.float32)
	low_noise[:8, :8] = rng.normal(0, 0.1, (8, 8))
	low_noise[12:20, 12:20] = 2 + rng.normal(0, 0.1, (8, 8))
	high_noise = low_noise.copy()
	high_noise[:8, :8] = rng.normal(0, 0.5, (8, 8))
	request = RoiNoiseCnrMetricRequest.model_validate({
		"background": {"y_start": 0, "y_stop": 8, "x_start": 0, "x_stop": 8},
		"signal": {"y_start": 12, "y_stop": 20, "x_start": 12, "x_stop": 20},
	})
	low = _values(request, low_noise)
	high = _values(request, high_noise)

	assert low["roi.background_noise"].value < high["roi.background_noise"].value
	assert low["roi.cnr"].value > high["roi.cnr"].value
	with pytest.raises(ValueError, match="exceeds image shape"):
		_values(
			RoiNoiseCnrMetricRequest.model_validate({
				"background": {"y_start": 0, "y_stop": 30, "x_start": 0, "x_stop": 8},
				"signal": {"y_start": 8, "y_stop": 16, "x_start": 8, "x_stop": 16},
			}),
			low_noise,
		)


def test_evaluation_stage_preserves_array_contract_and_emits_json_artifact(monkeypatch):
	rng = np.random.default_rng(15)
	working = rng.normal(0, 0.2, (3, 24, 24)).astype(np.float32)
	working[:, 12:20, 12:20] += 2
	reference = working + np.float32(0.01)
	before = working.copy()
	kernel_shapes = []
	original_evaluate = METRIC_REGISTRY.evaluate
	monkeypatch.setattr(
		METRIC_REGISTRY,
		"evaluate",
		lambda image, request, context: (
			kernel_shapes.append(image.shape)
			or original_evaluate(image, request, context)
		),
	)
	stage = EvaluationStage(
		metrics=[
			{"kind": "reference"},
			{"kind": "validity", "clip_min": -2, "clip_max": 3},
			{"kind": "ring", "inner_radius": 2, "outer_radius": 10},
			{
				"kind": "roi_noise_cnr",
				"background": {"y_start": 0, "y_stop": 8, "x_start": 0, "x_stop": 8},
				"signal": {"y_start": 12, "y_stop": 20, "x_start": 12, "x_stop": 20},
			},
		],
		windows=[
			{"id": "top", "slice_start": 0, "slice_stop": 1, "weight": 2},
			{"id": "bottom", "slice_start": 1, "slice_stop": 3},
		],
		aggregation={
			"method": "weighted_mean",
			"window_weights": {"top": 2, "bottom": 1},
			"invalid": "exclude",
		},
		preview={
			"scale_min": -1,
			"scale_max": 3,
			"crop": {"y_start": 2, "y_stop": 22, "x_start": 2, "x_stop": 22},
			"blinded": True,
		},
	)
	state = _state(working, artifacts={"evaluation_reference": reference})
	result = Pipeline(
		[stage],
		scan=state.scan,
		initial_layout=ReconOrder,
		initial_domain=DataDomain.RECON,
	).run(state)
	artifact = EvaluationArtifact.model_validate(result.artifacts["metrics"])

	assert result.working_array is working
	np.testing.assert_array_equal(working, before)
	assert result.layout is ReconOrder
	assert result.domain is DataDomain.RECON
	assert stage.contract.mutates_input is False
	assert stage.contract.produces is PRESERVE
	assert stage.contract.produces_domain is PRESERVE
	assert stage.contract.produces_artifacts == ("metrics",)
	assert len(artifact.observations) == 3 * 8
	assert set(artifact.aggregates) == set(artifact.definitions)
	assert artifact.preview.blinded
	assert set(result.artifacts["preview_images"]) == {"top-0", "bottom-1"}
	np.testing.assert_array_equal(result.artifacts["preview_images"]["top-0"], working[0])
	assert artifact.metadata["bounded_axis"] == "Y"
	assert artifact.metadata["maximum_kernel_input_shape"] == [24, 24]
	assert stage.contract.resources.preferred_cpus == 1
	assert kernel_shapes and set(kernel_shapes) == {(24, 24)}
	json.dumps(result.artifacts["metrics"], allow_nan=False)


def test_evaluation_fails_clearly_for_missing_or_malformed_reference():
	working = np.ones((2, 8, 8), dtype=np.float32)
	stage = EvaluationStage(metrics=[{"kind": "reference"}])
	with pytest.raises(ValueError, match="require artifact"):
		stage.run(_state(working))
	with pytest.raises(ValueError, match="does not match reconstruction"):
		stage.run(_state(
			working, artifacts={"evaluation_reference": np.ones((1, 8, 8))},
		))


def test_stage_registry_builds_typed_evaluation_stage():
	config = StageConfig(
		id="quality",
		kind="evaluate",
		algorithm="metrics",
		params={"metrics": [{"kind": "validity", "clip_min": 0}]},
	)
	stage = create_stage(config)

	assert STAGE_REGISTRY["evaluate.metrics"] is EvaluationStage
	assert isinstance(stage, EvaluationStage)
	assert stage.occurrence_id == "quality"


def test_experiment_metric_references_match_registry_protocols():
	document = {
		"schema_version": 1,
		"name": "evaluation-check",
		"candidate_space": {
			"pipeline": [{"id": "evaluate", "kind": "evaluate", "algorithm": "metrics"}],
		},
		"sampler": {"name": "exhaustive"},
		"metrics": [{
			"id": "nrmse",
			"metric": "reference.nrmse",
			"protocol_version": "1",
			"direction": "minimize",
			"units": "ratio",
		}],
		"objectives": [{"metric_id": "nrmse"}],
		"windows": [{"id": "window", "y_start": 0, "y_stop": 1}],
		"fidelities": [{"id": "screen", "windows": ["window"]}],
		"aggregation": {"metric_ids": ["nrmse"]},
	}
	config = ExperimentConfig.model_validate(document)
	validate_experiment_metric_references(config)

	document["metrics"][0]["protocol_version"] = "2"
	with pytest.raises(ValueError, match="protocol_version"):
		validate_experiment_metric_references(ExperimentConfig.model_validate(document))
