from pathlib import Path

import numpy as np
import pytest
import yaml

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import Pipeline, PipelineState
from lftomo.pipeline.stages import (
	EntropyLocalCenterStage,
	ManualCenterStage,
	PhaseCorrelationCenterStage,
	STAGE_REGISTRY,
	VoCenterStage,
	create_stage,
)
from lftomo.util.shapes import SinoOrder


def _state(array, configs):
	return PipelineState(
		working_array=array,
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
	)


def test_all_center_algorithms_are_registered():
	assert STAGE_REGISTRY["center.manual"] is ManualCenterStage
	assert STAGE_REGISTRY["center.vo"] is VoCenterStage
	assert STAGE_REGISTRY["center.pc"] is PhaseCorrelationCenterStage
	assert STAGE_REGISTRY["center.entropy_local"] is EntropyLocalCenterStage
	for stage_type in STAGE_REGISTRY.values():
		if stage_type in (ManualCenterStage, VoCenterStage, PhaseCorrelationCenterStage, EntropyLocalCenterStage):
			assert stage_type.contract.expected_input is SinoOrder
			assert stage_type.contract.produces_artifacts == ("center",)


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
	assert result.working_array is working

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
		calls.append((data.shape, kwargs))
		return 4 + 0.2 * (len(calls) - 1)

	monkeypatch.setattr("tomopy.find_center_vo", find_center_vo)
	result = Pipeline([create_stage(config)]).run(_state(working, [config]))

	assert all(call[0] == (3, 8) for call in calls)
	np.testing.assert_allclose(result.artifacts["center"], 4 + 0.1 * np.arange(5), rtol=1e-6)
	assert result.artifacts["center_fit"]["coefficients"].shape == (2,)


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
		calls.append((data.shape, angles, center, kwargs))
		return np.full((1, 8, 8), center, dtype=np.float32)

	monkeypatch.setattr("tomopy.recon", recon)
	monkeypatch.setattr("tomopy.circ_mask", lambda data, **_kwargs: data)
	monkeypatch.setattr(
		"lftomo.pipeline.stages.center.shannon_entropy",
		lambda data: abs(float(np.mean(data)) - 4),
	)
	result = Pipeline([create_stage(config)]).run(_state(working, [config]))

	assert [call[2] for call in calls] == [3, 4, 5]
	assert all(call[0] == (1, 4, 8) for call in calls)
	assert result.artifacts["center"] == pytest.approx(4)
	np.testing.assert_array_equal(result.artifacts["center_entropy"]["entropies"], [1, 0, 1])


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
