import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import Pipeline, PipelineState
from lftomo.pipeline.stages import (
	AffineSkewMovementStage,
	MovementCorrectStage,
	STAGE_REGISTRY,
	SkewCorrectStage,
	create_stage,
)


def _state(array, config):
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
		recon=Recon(pipeline=[config]),
	)


def test_skew_stage_rotates_each_projection(monkeypatch):
	config = StageConfig(
		kind="geometry",
		algorithm="affine_skew",
		params={"degrees": 0.25, "center_px": 3, "interpolation": "linear"},
	)
	working = np.ones((4, 3, 6), dtype=np.float32)
	calls = []
	monkeypatch.setattr("cv2.getRotationMatrix2D", lambda center, degrees, scale: (center, degrees, scale))

	def warp(image, matrix, size, flags):
		calls.append((matrix, size, flags))
		return image + 1

	monkeypatch.setattr("cv2.warpAffine", warp)
	result = Pipeline([create_stage(config)]).run(_state(working, config))

	assert STAGE_REGISTRY["geometry.affine_skew"] is SkewCorrectStage
	assert len(calls) == 3
	assert calls[0][0] == ((3, 2), 0.25, 1.0)
	assert calls[0][1] == (6, 4)
	np.testing.assert_array_equal(result.working_array, 2)


def test_movement_stage_interpolates_shift_across_projections(monkeypatch):
	config = StageConfig(
		kind="geometry",
		algorithm="movement",
		params={"enabled": True, "y": 4, "x": -2, "order": 1},
	)
	working = np.ones((4, 4, 6), dtype=np.float32)
	calls = []

	def fake_shift(image, offsets, **kwargs):
		calls.append((offsets, kwargs))
		return image

	monkeypatch.setattr("lftomo.pipeline.stages.geometry.shift", fake_shift)
	Pipeline([create_stage(config)]).run(_state(working, config))

	assert STAGE_REGISTRY["geometry.movement"] is MovementCorrectStage
	assert [call[0] for call in calls] == [(0, 0), (1, -0.5), (2, -1), (3, -1.5)]
	assert all(call[1]["order"] == 1 for call in calls)


def test_combined_r254_alias_runs_movement_then_skew(monkeypatch):
	config = StageConfig(
		kind="geometry",
		algorithm="affine_skew_movement",
		params={"skew_degrees": 0.2, "movement_enabled": True, "movement_y": 1},
	)
	working = np.ones((4, 2, 6), dtype=np.float32)
	order = []
	monkeypatch.setattr(
		"lftomo.pipeline.stages.geometry.shift",
		lambda image, *_args, **_kwargs: order.append("movement") or image,
	)
	monkeypatch.setattr("cv2.getRotationMatrix2D", lambda *_args: np.eye(2, 3))
	monkeypatch.setattr(
		"cv2.warpAffine",
		lambda image, *_args, **_kwargs: order.append("skew") or image,
	)
	stage = create_stage(config)
	Pipeline([stage]).run(_state(working, config))

	assert isinstance(stage, AffineSkewMovementStage)
	assert order == ["movement", "movement", "skew", "skew"]
