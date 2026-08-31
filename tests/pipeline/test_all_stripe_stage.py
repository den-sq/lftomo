import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import Pipeline, PipelineState
from lftomo.pipeline.stages import AllOfTheStripeStage, STAGE_REGISTRY, create_stage
from lftomo.util.shapes import SinoOrder


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
		layout=SinoOrder,
	)


def test_all_of_the_stripe_is_registered_and_transposes_for_tomopy(monkeypatch):
	config = StageConfig(
		kind="stripe",
		algorithm="all_of_the_stripe",
		params={"snr": 2.5, "la_size": 7, "sm_size": 3, "dim": 2, "ncore": 4},
	)
	working = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
	before = working.copy()
	calls = {}

	def remove_all_stripe(data, **kwargs):
		calls["shape"] = data.shape
		calls["kwargs"] = kwargs
		return data + 10

	monkeypatch.setattr("tomopy.remove_all_stripe", remove_all_stripe)
	stage = create_stage(config)
	result = Pipeline([stage]).run(_state(working, config))

	assert STAGE_REGISTRY["stripe.all_of_the_stripe"] is AllOfTheStripeStage
	assert calls["shape"] == (3, 2, 5)
	assert calls["kwargs"] == {
		"snr": 2.5,
		"la_size": 7,
		"sm_size": 3,
		"dim": 2,
		"ncore": 4,
		"nchunk": None,
	}
	assert result.working_array is working
	np.testing.assert_array_equal(working, before + 10)
