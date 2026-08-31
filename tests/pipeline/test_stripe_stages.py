import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import Pipeline, PipelineState
from lftomo.pipeline.stages import STAGE_REGISTRY, create_stage
from lftomo.util.shapes import ProjOrder, SinoOrder


def _state(array, layout, config):
	return PipelineState(
		working_array=array,
		scan=Scan(
			energy_keV=20, pixel_size_um=1,
			n_projections=array.shape[layout.axis_of("Theta")], degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(
				n_projections=array.shape[layout.axis_of("Theta")], rotation_end_deg=180,
			),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
		layout=layout,
	)


def _striped_fixture():
	theta = np.linspace(0.8, 1.2, 24, dtype=np.float32)
	x = np.linspace(0.9, 1.1, 24, dtype=np.float32)
	data = theta[None, :, None] * x[None, None, :]
	data = np.repeat(data, 3, axis=0)
	data[:, :, 11] *= 4
	data[:, :, 17] = 0
	return data


@pytest.mark.parametrize(("algorithm", "params"), [
	("vo_sort", {"size": 3}),
	("vo_dead", {"snr": 1, "size": 5, "norm": False}),
	("vo_large", {"snr": 1, "size": 5, "drop_ratio": 0.25}),
])
def test_registered_vo_stage_mutates_tier1_striped_sinogram(algorithm, params):
	config = StageConfig(kind="stripe", algorithm=algorithm, params=params)
	working = _striped_fixture()
	before = working.copy()

	result = Pipeline([create_stage(config)]).run(_state(working, SinoOrder, config))

	stage = create_stage(config)
	assert f"stripe.{algorithm}" in STAGE_REGISTRY
	assert stage.contract.mutates_input is True
	assert stage.contract.window_axes == ("Y",)
	assert stage.contract.required_metadata == ()
	assert stage.contract.consumes_artifacts == ()
	assert stage.contract.produces_artifacts == ()
	assert result.working_array is working
	assert np.isfinite(working).all()
	assert not np.array_equal(working, before)


def test_fourier_wavelet_projection_order_matches_sinogram_order():
	sinogram_order = _striped_fixture()
	projection_order = np.transpose(sinogram_order.copy(), (1, 0, 2))
	sino_config = StageConfig(
		kind="stripe", algorithm="fw", params={"level": 2, "wavelet": "db2"},
	)
	proj_config = sino_config

	create_stage(sino_config).run(_state(sinogram_order, SinoOrder, sino_config))
	proj_stage = create_stage(proj_config)
	proj_state = _state(projection_order, ProjOrder, proj_config)
	proj_stage.run(proj_state)

	assert proj_stage.contract.accepted_layouts == (SinoOrder,)
	np.testing.assert_allclose(projection_order, np.transpose(sinogram_order, (1, 0, 2)))
