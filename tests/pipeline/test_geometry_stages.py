import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import BufferSpec, DataDomain, Pipeline, PipelineState
from lftomo.pipeline.stages import (
	AffineSkewMovementStage,
	MovementCorrectStage,
	STAGE_REGISTRY,
	SkewCorrectStage,
	create_stage,
)
from lftomo.util.shapes import ProjOrder, SinoOrder
from lftomo.util.window import TomoWindow
from tests.sample_data import attach_source_geometry


def _state(array, config, layout=SinoOrder):
	shape = layout(*array.shape)
	return attach_source_geometry(PipelineState(
		working_array=array,
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1,
			n_projections=shape.Theta,
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(n_projections=shape.Theta, rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
		buffer_spec=BufferSpec(shape, array.dtype),
	))


def test_skew_stage_rotates_each_projection(monkeypatch):
	config = StageConfig(
		kind="geometry",
		algorithm="affine_skew",
		params={"degrees": 0.25, "center_px": 3, "interpolation": "linear"},
	)
	working = np.ones((4, 3, 6), dtype=np.float32)
	calls = []
	monkeypatch.setattr("cv2.getRotationMatrix2D", lambda center, degrees, scale: (center, degrees, scale))

	def warp(image, matrix, size, flags, **_kwargs):
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


def test_zero_skew_fused_transition_transposes_and_pads_without_interpolation(monkeypatch):
	config = StageConfig(
		kind="geometry", algorithm="affine_skew_movement",
		params={"transition_to": "projection", "pad_x": 1, "skew_degrees": 0},
	)
	working = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
	state = _state(working, config)
	state.domain = DataDomain.ATTENUATION
	state.source_window = TomoWindow(
		ProjOrder(Theta=3, Y=2, X=4), ProjOrder(0, 0, 0),
		ProjOrder(0, 0, 0), ProjOrder(1, 1, 1), np.float32, np.float32,
	)
	monkeypatch.setattr("cv2.warpAffine", lambda *_args, **_kwargs:
		(_ for _ in ()).throw(AssertionError("zero skew must not interpolate")))

	stage = create_stage(config)
	result = stage.run(state)

	assert stage.contract.fused_transition is True
	assert result.layout is ProjOrder
	assert result.working_array.flags.c_contiguous
	expected = np.pad(working.transpose(1, 0, 2), ((0, 0), (0, 0), (1, 1)), mode="edge")
	np.testing.assert_array_equal(result.working_array, expected)
	assert result.source_window.pos == ProjOrder(Theta=0, Y=0, X=-1)
	assert result.source_window.shape == ProjOrder(Theta=3, Y=2, X=6)
	assert result.buffer_spec.shape == result.source_window.shape
	assert result.buffer_spec.layout is ProjOrder


def test_fused_transition_is_direction_parametric():
	config = StageConfig(
		kind="geometry", algorithm="affine_skew_movement",
		params={"transition_to": "sinogram", "pad_x": 0},
	)
	working = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
	state = _state(working, config, ProjOrder)
	state.domain = DataDomain.ATTENUATION

	result = create_stage(config).run(state)

	assert result.layout is SinoOrder
	np.testing.assert_array_equal(result.working_array, working.transpose(1, 0, 2))


def test_nonzero_fused_transition_resamples_once_per_projection(monkeypatch):
	config = StageConfig(
		kind="geometry", algorithm="affine_skew_movement",
		params={"transition_to": "projection", "skew_degrees": 0.2, "pad_x": 1},
	)
	working = np.ones((2, 3, 4), dtype=np.float32)
	state = _state(working, config)
	calls = []
	monkeypatch.setattr("cv2.getRotationMatrix2D", lambda center, degrees, scale:
		(center, degrees, scale))
	def warp(image, matrix, size, flags, **_kwargs):
		calls.append((image.shape, matrix, size, flags))
		return image
	monkeypatch.setattr("cv2.warpAffine", warp)

	stage = create_stage(config)
	result = stage.run(state)

	assert stage.contract.resamples_detector is True
	assert len(calls) == 3
	assert calls[0][0] == (2, 6)
	assert calls[0][2] == (6, 2)
	assert result.layout is ProjOrder


def test_recon_minimal_tilt_radians_is_converted_to_corrective_degrees():
	stage = AffineSkewMovementStage(
		transition_to="projection", tilt_radians=-0.003134796238244514,
	)
	assert stage.params.skew_degrees == pytest.approx(0.17961059408489755)


def test_nontransition_affine_preserves_aaa523_style_projection_layout():
	config = StageConfig(
		kind="geometry", algorithm="affine_skew_movement",
		params={"skew_degrees": 0, "movement_enabled": False},
	)
	working = np.ones((3, 2, 4), dtype=np.float32)
	state = _state(working, config, ProjOrder)

	stage = create_stage(config)
	result = stage.run(state)

	assert stage.contract.fused_transition is False
	assert result.layout is ProjOrder
	assert result.working_array is working
