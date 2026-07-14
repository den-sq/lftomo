import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import Pipeline, PipelineState
from lftomo.pipeline.stages import DRHTStage, LDMStage, STAGE_REGISTRY, create_stage


def _state(array, config, mask):
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
		artifacts={"availability_mask": mask},
	)


@pytest.mark.parametrize("algorithm,stage_type", [("drht", DRHTStage), ("ldm", LDMStage)])
def test_super_resolution_stages_are_registered(algorithm, stage_type):
	params = {"allow_bicubic_fallback": True} if algorithm == "drht" else {"allow_interpolation_fallback": True}
	stage = create_stage(StageConfig(kind="super_resolve", algorithm=algorithm, params=params))

	assert isinstance(stage, stage_type)
	assert STAGE_REGISTRY[f"super_resolve.{algorithm}"] is stage_type


def test_drht_explicit_fallback_interpolates_missing_angles():
	config = StageConfig(
		kind="super_resolve",
		algorithm="drht",
		params={"allow_bicubic_fallback": True},
	)
	working = np.zeros((1, 3, 2), dtype=np.float32)
	working[:, 0] = 1
	working[:, 2] = 3
	result = Pipeline([create_stage(config)]).run(_state(working, config, [True, False, True]))

	np.testing.assert_allclose(result.working_array[:, 1], 2)
	assert result.artifacts["super_resolution"]["fallback"] == "interpolation"


def test_ldm_requires_restored_external_assets_without_fallback():
	config = StageConfig(kind="super_resolve", algorithm="ldm")
	working = np.ones((1, 3, 2), dtype=np.float32)

	with pytest.raises(RuntimeError, match="LDM repo path"):
		Pipeline([create_stage(config)]).run(_state(working, config, [True, True, True]))
