import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import Pipeline, PipelineState
from lftomo.pipeline.stages import BronnikovAidedCorrectionStage, STAGE_REGISTRY, create_stage


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
	)


def test_bac_aliases_are_registered_and_process_each_projection(monkeypatch):
	config = StageConfig(
		kind="phase",
		algorithm="bronnikov_aided",
		params={"alpha": 2, "gamma": 1.25, "invert": True, "processes": 2},
	)
	working = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
	before = working.copy()
	calls = []

	class FakeBAC:
		def __init__(self, shape, alpha, gamma, device):
			calls.append((shape, alpha, gamma, device))

		def __call__(self, image):
			return image + 2

	monkeypatch.setattr("hotopy.holo.BronnikovAidedCorrection", FakeBAC)
	stage = create_stage(config)
	result = Pipeline([stage]).run(_state(working, config))

	assert STAGE_REGISTRY["phase.bac"] is BronnikovAidedCorrectionStage
	assert STAGE_REGISTRY["phase.bronnikov_aided"] is BronnikovAidedCorrectionStage
	assert len(calls) == working.shape[1]
	assert all(call[:3] == ((3, 5), 2, 1.25) for call in calls)
	assert result.working_array is working
	np.testing.assert_array_equal(working, -(before + 2))


def test_disabled_bac_is_noop(monkeypatch):
	config = StageConfig(
		kind="phase",
		algorithm="bac",
		params={"alpha": 2, "enabled": False},
	)
	working = np.ones((2, 3, 4), dtype=np.float32)
	monkeypatch.setattr(
		"hotopy.holo.BronnikovAidedCorrection",
		lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not construct")),
	)

	result = Pipeline([create_stage(config)]).run(_state(working, config))

	assert result.working_array is working
	np.testing.assert_array_equal(working, 1)
