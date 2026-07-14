import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import Pipeline, PipelineState
from lftomo.pipeline.stages import (
	PaganinPhaseAbsorptionStage,
	PaganinStage,
	STAGE_REGISTRY,
	create_stage,
)


def _state(array, config):
	return PipelineState(
		working_array=array,
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1.0,
			ssd_cm=10.0,
			n_projections=array.shape[1],
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1.0),
			acquisition=Acquisition(n_projections=array.shape[1], rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
	)


def _fixture():
	data = np.full((16, 3, 16), 0.8, dtype=np.float32)
	data[6:10, :, 6:10] = 0.4
	return data


def test_standard_paganin_stage_is_registered_and_filters_tier1_fixture():
	config = StageConfig(kind="phase", algorithm="paganin", params={"alpha": 0.06})
	working = _fixture()

	result = Pipeline([create_stage(config)]).run(_state(working, config))

	assert STAGE_REGISTRY["phase.paganin"] is PaganinStage
	assert result.working_array is working
	assert np.isfinite(working).all()
	assert working[8, 0, 8] > working[0, 0, 0]


def test_coupled_phase_absorption_is_distinct_registered_algorithm():
	config = StageConfig(
		kind="phase", algorithm="paganin_phase_absorption", params={"alpha": 0.06}
	)
	working = _fixture()
	standard = _fixture()
	PaganinStage(alpha=0.06).run(_state(standard, config))

	result = Pipeline([create_stage(config)]).run(_state(working, config))

	assert STAGE_REGISTRY["phase.paganin_phase_absorption"] is PaganinPhaseAbsorptionStage
	assert result.working_array is working
	assert np.isfinite(working).all()
	assert not np.allclose(working, standard, rtol=1e-5, atol=0)
	assert np.all(working >= 0)
