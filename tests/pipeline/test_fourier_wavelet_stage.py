import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import BufferSpec, Pipeline, PipelineState
from lftomo.pipeline.stages import FourierWaveletStripeStage, STAGE_REGISTRY, create_stage
from lftomo.util.shapes import SinoOrder


def test_registered_fourier_wavelet_stage_filters_tier1_striped_sinogram():
	theta = np.linspace(0.8, 1.2, 32, dtype=np.float32)
	x = np.linspace(0.9, 1.1, 32, dtype=np.float32)
	working = np.repeat((theta[:, None] * x[None, :])[None, :, :], 2, axis=0)
	working[:, :, 12] *= 4
	before = working.copy()
	config = StageConfig(kind="stripe", algorithm="fw", params={
		"level": 2, "wname": "db2", "sigma": 2.0,
	})
	state = PipelineState(
		working_array=working,
		scan=Scan(
			energy_keV=20, pixel_size_um=1, n_projections=32, degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(n_projections=32, rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
		buffer_spec=BufferSpec(SinoOrder(*working.shape), working.dtype),
	)

	stage = create_stage(config)
	result = Pipeline([stage]).run(state)

	assert STAGE_REGISTRY["stripe.fw"] is FourierWaveletStripeStage
	assert stage.contract.mutates_input is True
	assert stage.contract.window_axes == ("Y",)
	assert stage.contract.required_metadata == ()
	assert stage.contract.consumes_artifacts == ()
	assert stage.contract.produces_artifacts == ()
	assert result.working_array is working
	assert np.isfinite(working).all()
	assert not np.array_equal(working, before)
