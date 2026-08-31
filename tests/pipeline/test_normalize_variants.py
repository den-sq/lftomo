import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import BufferSpec, Pipeline, PipelineState
from lftomo.pipeline.stages import (
	STAGE_REGISTRY,
	TaperBlendedNormalizeStage,
	ThetaSwitchNormalizeStage,
	create_stage,
)
from lftomo.util.shapes import SinoOrder
from tests.sample_data import attach_source_geometry


def _state(array, config, flats):
	return attach_source_geometry(PipelineState(
		working_array=array,
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1,
			n_projections=array.shape[1],
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(n_projections=array.shape[1], rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
			metadata={"flat_fields": flats},
		),
		recon=Recon(pipeline=[config]),
		buffer_spec=BufferSpec(SinoOrder(*array.shape), array.dtype),
	))


def _flats():
	return np.stack([
		np.full((2, 3), 10, dtype=np.float32),
		np.full((2, 3), 20, dtype=np.float32),
		np.full((2, 3), 2, dtype=np.float32),
		np.full((2, 3), 4, dtype=np.float32),
	])


def test_theta_switch_uses_pre_then_post_fields():
	config = StageConfig(kind="normalize", algorithm="theta_switch")
	working = np.full((2, 4, 3), 12, dtype=np.float32)
	result = Pipeline([create_stage(config)]).run(_state(working, config, _flats()))

	assert STAGE_REGISTRY["normalize.theta_switch"] is ThetaSwitchNormalizeStage
	np.testing.assert_allclose(result.working_array[:, :3], 1.25)
	np.testing.assert_allclose(result.working_array[:, 3], 0.5)


def test_theta_switch_can_linearly_blend_fields():
	config = StageConfig(
		kind="normalize",
		algorithm="theta_switch",
		params={"gain_averaging": True},
	)
	working = np.full((2, 4, 3), 12, dtype=np.float32)
	result = Pipeline([create_stage(config)]).run(_state(working, config, _flats()))

	np.testing.assert_allclose(result.working_array[:, 0], 1.25)
	np.testing.assert_allclose(result.working_array[:, 2], 9 / 12)


def test_phase_contrast_alias_accepts_pinned_path_parameters():
	config = StageConfig(
		kind="normalize",
		algorithm="taper_blended",
		params={
			"pre_gain": "gains/gain.tif",
			"post_gain": "gains/gain_post.tif",
			"pre_dark": "gains/dark.tif",
			"post_dark": "gains/dark_post.tif",
			"pre_weight": "linear",
			"post_weight": "linear",
		},
	)
	working = np.full((2, 2, 3), 12, dtype=np.float32)
	stage = create_stage(config)
	Pipeline([stage]).run(_state(working, config, _flats()))

	assert isinstance(stage, TaperBlendedNormalizeStage)
	assert STAGE_REGISTRY["normalize.phase_contrast"] is TaperBlendedNormalizeStage
	assert stage.calibration_paths["pre_gain"] == "gains/gain.tif"


def test_frozen_r254_theta_switch_parameters_construct():
	config = StageConfig(
		kind="normalize",
		algorithm="theta_switch",
		params={
			"gain_averaging": True,
			"pre_gain": "gains/gain.tif",
			"post_gain": "gains/gain_post.tif",
			"pre_dark": "gains/dark.tif",
			"post_dark": "gains/dark_post.tif",
			"pre_subtracted": False,
		},
	)

	stage = create_stage(config)

	assert isinstance(stage, ThetaSwitchNormalizeStage)
	assert stage.calibration_paths == {
		"pre_gain": "gains/gain.tif",
		"post_gain": "gains/gain_post.tif",
		"pre_dark": "gains/dark.tif",
		"post_dark": "gains/dark_post.tif",
	}


def test_frozen_phase_contrast_taper_parameters_construct():
	config = StageConfig(
		kind="normalize",
		algorithm="taper_blended",
		params={
			"pre_gain": "gains/gain.tif",
			"post_gain": "gains/gain_post.tif",
			"pre_dark": "gains/dark.tif",
			"post_dark": "gains/dark_post.tif",
			"gain_averaging": True,
			"gain_movement_enabled": False,
		},
	)

	stage = create_stage(config)

	assert isinstance(stage, TaperBlendedNormalizeStage)
	assert stage.gain_averaging is True
