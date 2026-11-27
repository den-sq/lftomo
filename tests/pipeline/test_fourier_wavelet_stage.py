import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import LocalArray, Pipeline, PipelineState
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
		working_array=LocalArray(working, SinoOrder(*working.shape)),
		scan=Scan(
			energy_keV=20, pixel_size_um=1, n_projections=32, degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(n_projections=32, rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
	)

	stage = create_stage(config)
	result = Pipeline([stage]).run(state)

	assert STAGE_REGISTRY["stripe.fw"] is FourierWaveletStripeStage
	assert stage.contract.mutates_input is True
	assert stage.contract.window_axes == ("Y",)
	assert stage.contract.required_metadata == ()
	assert stage.contract.consumes_artifacts == ()
	assert stage.contract.produces_artifacts == ()
	assert np.asarray(result.working_array) is working
	assert np.isfinite(working).all()
	assert not np.array_equal(working, before)


def _drifting_stripe_sinogram():
	theta, x = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 64), indexing="ij")
	truth = 1 + 0.1 * np.sin(6 * x + 3 * theta)
	striped = truth.copy()
	striped[:, 30] += np.linspace(0.0, 0.4, 128)
	striped[:, 45] += 0.2
	return truth, striped


def test_theta_mirror_suppresses_drifting_stripe_seam_at_scan_ends():
	from lftomo.prep.stripe import _remove_stripe_fw_array

	truth, striped = _drifting_stripe_sinogram()
	plain, mirrored = striped.copy(), striped.copy()
	_remove_stripe_fw_array(plain, level=4, sigma=2.0)
	_remove_stripe_fw_array(mirrored, level=4, sigma=2.0, theta_mirror=True)

	def error(result, rows, columns):
		return np.abs(result - truth)[rows, columns].mean()

	ends, middle = np.r_[0:12, -12:0], slice(40, 88)
	assert error(mirrored, ends, slice(28, 33)) < 0.9 * error(plain, ends, slice(28, 33))
	assert error(mirrored, middle, slice(28, 33)) < 1.05 * error(plain, middle, slice(28, 33))
	assert error(mirrored, slice(None), 45) < 1.01 * error(plain, slice(None), 45)


def test_theta_mirror_keeps_damping_strength_for_periodic_stripes():
	from lftomo.prep.stripe import _remove_stripe_fw_array

	theta, x = np.meshgrid(np.arange(256), np.linspace(0, 1, 64), indexing="ij")
	truth = 1 + 0.1 * np.sin(6 * x)
	striped = truth.copy()
	striped[:, 30] += 0.2 * np.cos(2 * np.pi * theta[:, 0] / 256)
	plain, mirrored = striped.copy(), striped.copy()
	_remove_stripe_fw_array(plain, level=4, sigma=2.0)
	_remove_stripe_fw_array(mirrored, level=4, sigma=2.0, theta_mirror=True)
	assert np.abs(mirrored - truth)[:, 28:33].mean() < 1.1 * np.abs(plain - truth)[:, 28:33].mean()


@pytest.mark.parametrize("zero_pad", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("level", [None, 4])
def test_theta_mirror_matches_doubled_sigma_and_crops_to_original_extent(zero_pad, dtype, level):
	from lftomo.prep.stripe import _remove_stripe_fw_array

	_, striped = _drifting_stripe_sinogram()
	# Odd dimensions also exercise the wavelet reconstruction's existing crop.
	striped = striped[:-1, :-1].astype(dtype)
	expected = np.concatenate((striped, striped[::-1]), axis=0)
	resolved_level = level or int(np.ceil(np.log2(max(striped.shape))))
	_remove_stripe_fw_array(expected, level=resolved_level, sigma=4.0, zero_pad=zero_pad)
	stage = create_stage(StageConfig(kind="stripe", algorithm="fw", params={
		"level": level, "sigma": 2.0, "zero_pad": zero_pad, "theta_mirror": True,
	}))
	stage.run_sinogram(striped, None)
	assert striped.shape == (127, 63)
	assert striped.dtype == dtype
	np.testing.assert_array_equal(striped, expected[:127])


def test_theta_mirror_is_disabled_by_default():
	_, striped = _drifting_stripe_sinogram()
	default, disabled = striped.copy(), striped.copy()
	stage = FourierWaveletStripeStage(level=4, sigma=2.0)
	assert stage.params.theta_mirror is False
	stage.run_sinogram(default, None)
	FourierWaveletStripeStage(level=4, sigma=2.0, theta_mirror=False).run_sinogram(disabled, None)
	np.testing.assert_array_equal(default, disabled)
