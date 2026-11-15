from pathlib import Path

import numpy as np
import pytest
import yaml

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import LocalArray, Pipeline, PipelineState
from lftomo.pipeline.stages import (
	EntropyLocalCenterStage,
	ManualCenterStage,
	PhaseCorrelationCenterStage,
	STAGE_REGISTRY,
	VoCenterStage,
	create_stage,
)
from lftomo.util.shapes import ProjOrder, SinoOrder
from lftomo.util.window import TomoWindow
from tests.sample_data import attach_source_geometry


def _state(array, configs):
	return attach_source_geometry(PipelineState(
		working_array=LocalArray(array, SinoOrder(*array.shape)),
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1,
			n_projections=array.shape[1],
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(n_projections=array.shape[1], rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=configs),
	))


def test_all_center_algorithms_are_registered():
	assert STAGE_REGISTRY["center.manual"] is ManualCenterStage
	assert STAGE_REGISTRY["center.vo"] is VoCenterStage
	assert STAGE_REGISTRY["center.pc"] is PhaseCorrelationCenterStage
	assert STAGE_REGISTRY["center.entropy_local"] is EntropyLocalCenterStage
	for stage_type in STAGE_REGISTRY.values():
		if stage_type in (ManualCenterStage, VoCenterStage, PhaseCorrelationCenterStage, EntropyLocalCenterStage):
			assert SinoOrder in stage_type.contract.accepts
			assert "center" in stage_type.contract.produces_artifacts


@pytest.mark.parametrize(
	"path",
	[
		Path("lftomo/pipelines/r254-canonical.yaml"),
		Path("lftomo/pipelines/recon-minimal-canonical.yaml"),
		Path("lftomo/pipelines/phase-contrast-canonical.yaml"),
		Path("lftomo/pipelines/b10-recon-canonical.yaml"),
		Path("tests/tier2_metadata/references/striping_tiff/recon_minimal/pipeline.yaml"),
	],
)
def test_pinned_center_configs_create_registered_stages(path):
	pipeline = yaml.safe_load(path.read_text())["pipeline"]
	config = StageConfig.model_validate(next(stage for stage in pipeline if stage["kind"] == "center"))

	assert create_stage(config).name == f"center.{config.algorithm}"


def test_manual_center_stores_scalar_or_tilt_profile():
	working = np.ones((4, 3, 8), dtype=np.float32)
	scalar_config = StageConfig(kind="center", algorithm="manual", params={"value_px": 4.25})
	result = Pipeline([create_stage(scalar_config)]).run(_state(working, [scalar_config]))
	assert result.artifacts["center"] == pytest.approx(4.25)
	assert np.asarray(result.working_array) is working

	tilt_config = StageConfig(
		kind="center",
		algorithm="manual",
		params={"value_px": 4, "tilt": 0.25, "reference_y": 1},
	)
	result = Pipeline([create_stage(tilt_config)]).run(_state(working, [tilt_config]))
	np.testing.assert_allclose(result.artifacts["center"], [3.75, 4, 4.25, 4.5])


def test_vo_center_transposes_data_and_fits_tilt(monkeypatch):
	working = np.zeros((5, 3, 8), dtype=np.float32)
	config = StageConfig(
		kind="center",
		algorithm="vo",
		params={"tilt_fit": "polynomial", "tilt_samples": 3},
	)
	calls = []

	def find_center_vo(data, **kwargs):
		calls.append((data.shape, kwargs, np.shares_memory(data, working)))
		return 4 + 0.2 * (len(calls) - 1)

	monkeypatch.setattr("tomopy.find_center_vo", find_center_vo)
	result = Pipeline([create_stage(config)]).run(_state(working, [config]))

	assert all(call[0] == (3, 8) and call[2] for call in calls)
	np.testing.assert_allclose(result.artifacts["center"], 4 + 0.1 * np.arange(5), rtol=1e-6)
	assert result.artifacts["center_fit"]["coefficients"].shape == (2,)
	assert result.artifacts["center_fit"]["qc_status"] == "accepted"
	assert result.artifacts["center_fit"]["row_span"] == 4


def test_vo_tilt_fit_rejects_center_outlier(monkeypatch):
	working = np.zeros((9, 3, 8), dtype=np.float32)
	config = StageConfig(
		kind="center",
		algorithm="vo",
		params={"tilt_fit": "polynomial", "tilt_samples": 5},
	)
	centers = iter([4.0, 4.2, 9.4, 4.6, 4.8])
	monkeypatch.setattr("tomopy.find_center_vo", lambda *_args, **_kwargs: next(centers))

	result = Pipeline([create_stage(config)]).run(_state(working, [config]))

	np.testing.assert_allclose(result.artifacts["center"], 4 + 0.1 * np.arange(9), atol=1e-6)
	metadata = result.artifacts["center_fit"]
	np.testing.assert_array_equal(metadata["inlier_mask"], [True, True, False, True, True])
	np.testing.assert_array_equal(metadata["outlier_source_rows"], [4])
	assert metadata["residual_rms_px"] == pytest.approx(0, abs=1e-12)


def test_vo_tilt_fit_uses_source_y_coordinates_and_row_span(monkeypatch):
	working = np.zeros((7, 3, 8), dtype=np.float32)
	config = StageConfig(
		kind="center",
		algorithm="vo",
		params={"tilt_fit": "polynomial", "tilt_samples": 3},
	)
	state = _state(working, [config])
	state.source_window = TomoWindow(
		ProjOrder(Theta=3, Y=5, X=8),
		ProjOrder(Theta=0, Y=1, X=0),
		ProjOrder(Theta=0, Y=100, X=0),
		ProjOrder(Theta=1, Y=2, X=1),
		np.float32,
		np.float32,
	)
	centers = iter([13.9, 14.5, 15.1])
	monkeypatch.setattr("tomopy.find_center_vo", lambda *_args, **_kwargs: next(centers))

	result = Pipeline([create_stage(config)]).run(state)

	np.testing.assert_allclose(
		result.artifacts["center"], [13.9, 14.1, 14.3, 14.5, 14.7, 14.9, 15.1],
	)
	metadata = result.artifacts["center_fit"]
	np.testing.assert_array_equal(metadata["sampled_source_rows"], [99, 105, 111])
	assert metadata["row_span"] == 12


def test_vo_tilt_fit_rejects_or_forces_failed_residual_qc(monkeypatch):
	working = np.zeros((5, 3, 8), dtype=np.float32)
	base_params = {
		"tilt_fit": "polynomial",
		"tilt_samples": 5,
		"tilt_outlier_sigma": 100,
		"tilt_residual_cutoff_px": 0.1,
	}
	values = [4.0, 4.3, 3.8, 4.7, 4.1]

	def run(policy):
		centers = iter(values)
		monkeypatch.setattr("tomopy.find_center_vo", lambda *_args, **_kwargs: next(centers))
		config = StageConfig(
			kind="center", algorithm="vo",
			params={**base_params, "tilt_qc_policy": policy},
		)
		state = _state(working, [config])
		if policy == "reject":
			with pytest.raises(ValueError, match="tilt fit rejected"):
				Pipeline([create_stage(config)]).run(state)
			assert state.artifacts["center_fit"]["qc_status"] == "rejected"
			return None
		return Pipeline([create_stage(config)]).run(state)

	run("reject")
	forced = run("force")
	assert forced.artifacts["center_fit"]["qc_status"] == "forced"
	assert forced.artifacts["center_fit"]["qc_passed"] is False


def test_vo_tilt_fit_quantizes_output_centers(monkeypatch):
	working = np.zeros((5, 3, 8), dtype=np.float32)
	config = StageConfig(
		kind="center",
		algorithm="vo",
		params={
			"tilt_fit": "polynomial",
			"tilt_samples": 3,
			"tilt_quantization_px": 0.25,
		},
	)
	centers = iter([4.03, 4.23, 4.43])
	monkeypatch.setattr("tomopy.find_center_vo", lambda *_args, **_kwargs: next(centers))

	result = Pipeline([create_stage(config)]).run(_state(working, [config]))

	np.testing.assert_array_equal(result.artifacts["center"], [4, 4.25, 4.25, 4.25, 4.5])
	assert result.artifacts["center_fit"]["quantization_px"] == 0.25


def test_phase_correlation_uses_first_and_last_projection(monkeypatch):
	working = np.arange(4 * 3 * 6, dtype=np.float32).reshape(4, 3, 6)
	config = StageConfig(kind="center", algorithm="pc", params={"tol": 0.25, "rotc_guess": 3})
	calls = {}

	def find_center_pc(first, last, **kwargs):
		calls["first"] = first
		calls["last"] = last
		calls["kwargs"] = kwargs
		return 3.5

	monkeypatch.setattr("tomopy.find_center_pc", find_center_pc)
	result = Pipeline([create_stage(config)]).run(_state(working, [config]))

	np.testing.assert_array_equal(calls["first"], working[:, 0, :])
	np.testing.assert_array_equal(calls["last"], working[:, -1, :])
	assert calls["kwargs"] == {"tol": 0.25, "rotc_guess": 3}
	assert result.artifacts["center"] == pytest.approx(3.5)


def test_entropy_local_selects_lowest_entropy_candidate(monkeypatch):
	working = np.ones((3, 4, 8), dtype=np.float32)
	config = StageConfig(
		kind="center",
		algorithm="entropy_local",
		params={"start": 4, "range": 1, "step": 1, "slice_index": 1},
	)
	calls = []

	def recon(data, angles, center, **kwargs):
		calls.append((data.shape, angles, center, kwargs, np.shares_memory(data, working)))
		return np.full((1, 8, 8), center, dtype=np.float32)

	monkeypatch.setattr("tomopy.recon", recon)
	monkeypatch.setattr("tomopy.circ_mask", lambda data, **_kwargs: data)
	monkeypatch.setattr(
		"skimage.measure.shannon_entropy",
		lambda data: abs(float(np.mean(data)) - 4),
	)
	result = Pipeline([create_stage(config)]).run(_state(working, [config]))

	assert [call[2] for call in calls] == [3, 4, 5]
	assert all(call[0] == (1, 4, 8) for call in calls)
	assert all(call[4] for call in calls)
	assert result.artifacts["center"] == pytest.approx(4)
	np.testing.assert_array_equal(result.artifacts["center_entropy"]["entropies"], [1, 0, 1])
	assert result.artifacts["center_entropy"]["termination"] == "converged"
	assert result.artifacts["center_entropy"]["expansions"] == 0


def test_entropy_local_expands_from_boundary_and_updates_current_center(monkeypatch):
	working = np.ones((3, 4, 8), dtype=np.float32)
	config = StageConfig(
		kind="center",
		algorithm="entropy_local",
		params={"start": 4, "range": 1, "step": 1, "max_expansions": 3},
	)
	candidates = []

	def recon(_data, _angles, center, **_kwargs):
		candidates.append(center)
		return np.full((1, 8, 8), center, dtype=np.float32)

	monkeypatch.setattr("tomopy.recon", recon)
	monkeypatch.setattr("tomopy.circ_mask", lambda data, **_kwargs: data)
	monkeypatch.setattr(
		"skimage.measure.shannon_entropy",
		lambda data: abs(float(np.mean(data)) - 6),
	)
	result = Pipeline([create_stage(config)]).run(_state(working, [config]))

	assert candidates == [3, 4, 5, 5, 6, 7]
	assert result.artifacts["center"] == pytest.approx(6)
	metadata = result.artifacts["center_entropy"]
	assert metadata["termination"] == "converged"
	assert metadata["expansions"] == 1
	assert [search["start"] for search in metadata["searches"]] == [4, 6]


def test_entropy_local_stops_at_expansion_cap(monkeypatch):
	working = np.ones((3, 4, 8), dtype=np.float32)
	config = StageConfig(
		kind="center",
		algorithm="entropy_local",
		params={"start": 4, "range": 1, "step": 1, "max_expansions": 2},
	)
	monkeypatch.setattr(
		"tomopy.recon",
		lambda _data, _angles, center, **_kwargs:
			np.full((1, 8, 8), center, dtype=np.float32),
	)
	monkeypatch.setattr("tomopy.circ_mask", lambda data, **_kwargs: data)
	monkeypatch.setattr(
		"skimage.measure.shannon_entropy",
		lambda data: -float(np.mean(data)),
	)
	result = Pipeline([create_stage(config)]).run(_state(working, [config]))

	assert result.artifacts["center"] == pytest.approx(9)
	metadata = result.artifacts["center_entropy"]
	assert metadata["termination"] == "cap-reached"
	assert metadata["expansions"] == 2
	assert metadata["converged"] is False


def test_entropy_local_stops_when_the_same_candidate_remains_at_a_boundary(monkeypatch):
	working = np.ones((3, 4, 8), dtype=np.float32)
	config = StageConfig(
		kind="center",
		algorithm="entropy_local",
		params={"start": 4, "range": 1, "step": 1, "max_expansions": 3},
	)
	monkeypatch.setattr(
		"tomopy.recon",
		lambda _data, _angles, center, **_kwargs:
			np.full((1, 8, 8), center, dtype=np.float32),
	)
	monkeypatch.setattr("tomopy.circ_mask", lambda data, **_kwargs: data)
	monkeypatch.setattr(
		"skimage.measure.shannon_entropy",
		lambda data: abs(float(np.mean(data)) - 5),
	)
	result = Pipeline([create_stage(config)]).run(_state(working, [config]))

	assert result.artifacts["center"] == pytest.approx(5)
	metadata = result.artifacts["center_entropy"]
	assert metadata["termination"] == "boundary-persisted"
	assert metadata["expansions"] == 1


def test_entropy_local_diagnostics_are_durable_outputs(tmp_path, monkeypatch):
	working = np.ones((3, 4, 8), dtype=np.float32)
	config = StageConfig(
		kind="center",
		algorithm="entropy_local",
		params={
			"start": 4,
			"range": 1,
			"step": 1,
			"max_expansions": 0,
			"diagnostic_images": True,
		},
	)
	state = _state(working, [config])
	state.recon.output_path = tmp_path
	monkeypatch.setattr(
		"tomopy.recon",
		lambda _data, _angles, center, **_kwargs:
			np.full((1, 8, 8), center, dtype=np.float32),
	)
	monkeypatch.setattr("tomopy.circ_mask", lambda data, **_kwargs: data)
	result = Pipeline([create_stage(config)]).run(state)

	outputs = result.artifacts["output_files"]
	assert len(outputs) == 3
	assert all(path.is_file() for path in outputs)
	assert all(path.parent == tmp_path / "center-diagnostics" for path in outputs)
	assert result.artifacts["center_entropy"]["diagnostic_files"] == outputs
	assert "output_files" in create_stage(config).contract.produces_artifacts


def test_vo_can_fall_back_to_entropy_local(monkeypatch):
	working = np.ones((3, 4, 8), dtype=np.float32)
	config = StageConfig(
		kind="center",
		algorithm="vo",
		params={
			"fallback": "entropy_local",
			"fallback_params": {"start": 4, "range": 0},
		},
	)
	monkeypatch.setattr("tomopy.find_center_vo", lambda *_args, **_kwargs: np.nan)
	monkeypatch.setattr(
		"tomopy.recon",
		lambda *_args, **_kwargs: np.ones((1, 8, 8), dtype=np.float32),
	)
	monkeypatch.setattr("tomopy.circ_mask", lambda data, **_kwargs: data)

	result = Pipeline([create_stage(config)]).run(_state(working, [config]))

	assert result.artifacts["center"] == pytest.approx(4)


def test_vo_fallback_does_not_hide_unexpected_programming_errors(monkeypatch):
	working = np.ones((3, 4, 8), dtype=np.float32)
	config = StageConfig(
		kind="center",
		algorithm="vo",
		params={"fallback": "entropy_local"},
	)
	monkeypatch.setattr("tomopy.find_center_vo", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("bug")))

	with pytest.raises(KeyError, match="bug"):
		Pipeline([create_stage(config)]).run(_state(working, [config]))
