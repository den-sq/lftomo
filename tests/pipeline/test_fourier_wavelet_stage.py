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


def _drifting_stripe_sinogram():
	theta, x = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 64), indexing="ij")
	truth = (1 + 0.1 * np.sin(6 * x + 3 * theta)).astype(np.float64)
	striped = truth.copy()
	striped[:, 30] += np.linspace(0.0, 0.4, 128)    # stripe drifts during the scan
	striped[:, 45] += 0.2                            # stationary stripe
	return truth, striped


def test_theta_mirror_suppresses_drifting_stripe_seam_at_scan_ends():
	from lftomo.prep.stripe import _remove_stripe_fw_array

	truth, striped = _drifting_stripe_sinogram()
	plain, mirrored = striped.copy(), striped.copy()
	_remove_stripe_fw_array(plain, level=4, wname="db5", sigma=2.0)
	_remove_stripe_fw_array(mirrored, level=4, wname="db5", sigma=2.0, theta_mirror=True)

	def error(result, rows, columns):
		return np.abs(result - truth)[rows, columns].mean()

	ends, middle = np.r_[0:12, -12:0], slice(40, 88)
	assert error(mirrored, ends, slice(28, 33)) < 0.9 * error(plain, ends, slice(28, 33))
	# Away from the scan ends and for stationary stripes the result is unchanged.
	assert error(mirrored, middle, slice(28, 33)) < 1.05 * error(plain, middle, slice(28, 33))
	np.testing.assert_allclose(error(mirrored, slice(None), 45), error(plain, slice(None), 45), rtol=0.01)


def test_fourier_wavelet_stage_forwards_theta_mirror():
	from lftomo.prep.stripe import _remove_stripe_fw_array

	_, striped = _drifting_stripe_sinogram()
	expected = striped.copy()
	_remove_stripe_fw_array(expected, level=4, wname="db5", sigma=2.0, theta_mirror=True)
	working = striped[None].copy()
	stage = FourierWaveletStripeStage(level=4, wname="db5", sigma=2.0, theta_mirror=True)
	stage.run_sinogram(working[0], None)
	np.testing.assert_allclose(working[0], expected)
