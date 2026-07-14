import numpy as np

from lftomo.config import Acquisition, Detector, Helical, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import Pipeline, PipelineState
from lftomo.pipeline.stages import HelicalIndexRemapStage, STAGE_REGISTRY, create_stage


def _state(array, config, pitch_um=1):
	return PipelineState(
		working_array=array,
		scan=Scan(
			energy_keV=20,
			pixel_size_um=2,
			n_projections=array.shape[1],
			degrees_per_fov=180,
			helical=Helical(enabled=True, pitch_um_per_projection=pitch_um),
			detector=Detector(pixel_size_um=2),
			acquisition=Acquisition(n_projections=array.shape[1], rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
	)


def test_helical_remap_crops_to_rows_valid_for_every_projection():
	config = StageConfig(
		kind="helical_correct",
		algorithm="index_remap",
		params={"pitch_px_per_projection": 1},
	)
	working = np.arange(6 * 3 * 2).reshape(6, 3, 2)
	result = Pipeline([create_stage(config)]).run(_state(working, config))

	assert STAGE_REGISTRY["helical_correct.index_remap"] is HelicalIndexRemapStage
	assert result.working_array.shape == (4, 3, 2)
	for projection in range(3):
		np.testing.assert_array_equal(
			result.working_array[:, projection],
			working[projection:projection + 4, projection],
		)
	np.testing.assert_array_equal(result.artifacts["helical_row_shifts"], [0, 1, 2])


def test_helical_remap_derives_pitch_and_can_preserve_height():
	config = StageConfig(
		kind="geometry",
		algorithm="helical_pitch",
		params={"crop_valid": False, "fill_value": -1},
	)
	working = np.arange(5 * 3).reshape(5, 3, 1)
	result = Pipeline([create_stage(config)]).run(_state(working, config, pitch_um=2))

	assert result.working_array.shape == working.shape
	np.testing.assert_array_equal(result.working_array[:-2, 2], working[2:, 2])
	np.testing.assert_array_equal(result.working_array[-2:, 2], -1)


def test_disabled_helical_alias_is_noop_without_pitch():
	config = StageConfig(
		kind="geometry",
		algorithm="helical_pitch",
		params={"enabled": False, "vertical_velocity_px": 0, "going_up": False},
	)
	working = np.ones((2, 3, 4), dtype=np.float32)
	result = Pipeline([create_stage(config)]).run(_state(working, config, pitch_um=None))

	assert result.working_array is working


def test_berkeley_source_bottom_row_equation_is_preserved():
	"""Source reads height - round(pitch * projection) - 1 when going up."""
	config = StageConfig(
		kind="helical_correct",
		algorithm="index_remap",
		params={
			"pitch_px_per_projection": 1,
			"going_up": True,
			"crop_valid": False,
			"fill_value": -1,
		},
	)
	working = np.broadcast_to(np.arange(6)[:, None, None], (6, 3, 1)).copy()

	result = Pipeline([create_stage(config)]).run(_state(working, config))

	for projection in range(3):
		expected_row = working.shape[0] - round(projection) - 1
		assert result.working_array[-1, projection, 0] == working[expected_row, projection, 0]
