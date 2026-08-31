import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import Pipeline, PipelineState
from lftomo.pipeline.stages import BilateralFilterStage, STAGE_REGISTRY, create_stage
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


def test_bilateral_stage_is_registered_and_filters_each_projection(monkeypatch):
	config = StageConfig(
		kind="filter",
		algorithm="bilateral",
		params={"diameter": 5, "sigma_color": 2, "sigma_space": 3},
	)
	working = np.ones((4, 3, 6), dtype=np.float64)
	calls = []

	def bilateral(image, diameter, sigma_color, sigma_space, borderType):
		calls.append((image.dtype, diameter, sigma_color, sigma_space, borderType))
		return image + 1

	monkeypatch.setattr("cv2.bilateralFilter", bilateral)
	stage = create_stage(config)
	result = Pipeline([stage]).run(_state(working, config))

	assert STAGE_REGISTRY["filter.bilateral"] is BilateralFilterStage
	assert len(calls) == 3
	assert all(call[0] == np.float32 for call in calls)
	assert all(call[1:4] == (5, 2, 3) for call in calls)
	assert result.working_array is working
	assert result.working_array.dtype == np.float64
	np.testing.assert_array_equal(working, 2)


def test_disabled_bilateral_stage_is_noop(monkeypatch):
	config = StageConfig(
		kind="filter", algorithm="bilateral", params={"enabled": False}
	)
	working = np.ones((2, 3, 4), dtype=np.float32)
	monkeypatch.setattr(
		"cv2.bilateralFilter",
		lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
	)

	result = Pipeline([create_stage(config)]).run(_state(working, config))

	assert result.working_array is working
