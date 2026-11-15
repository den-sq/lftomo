import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	BufferSpec,
	DataDomain,
	DomainCompatibilityError,
	PRESERVE,
	Pipeline,
	PipelineState,
	validate_pipeline,
)
from lftomo.pipeline.stages import (
	BronnikovAidedCorrectionFilterStage,
	BronnikovAidedCorrectionStage,
	PaganinPhaseAbsorptionStage,
	PaganinStage,
	MultimaterialStage,
	STAGE_REGISTRY,
	create_stage,
)
from lftomo.util.shapes import ProjOrder, SinoOrder
from tests.sample_data import attach_source_geometry


def _state(array, config):
	return attach_source_geometry(PipelineState(
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
		buffer_spec=BufferSpec(SinoOrder(*array.shape), array.dtype),
	))


def _fixture():
	data = np.full((16, 3, 16), 0.8, dtype=np.float32)
	data[6:10, :, 6:10] = 0.4
	return data


@pytest.mark.parametrize("key", sorted(
	key for key in STAGE_REGISTRY if key.startswith("phase.")
))
def test_registered_phase_stages_explicitly_declare_extent_sensitive_axes(key):
	# Only constructor inputs live here, not a second classification of sensitive
	# algorithms. A newly registered phase Stage automatically joins this test.
	required_params = {
		BronnikovAidedCorrectionStage: {"alpha": 2.0},
		BronnikovAidedCorrectionFilterStage: {"alpha": 2.0},
		MultimaterialStage: {"mu_table": [50, 120], "delta_table": [1e-7, 2.4e-7]},
	}
	kind, algorithm = key.split(".", 1)
	stage = create_stage(StageConfig(
		kind=kind, algorithm=algorithm,
		params=required_params.get(STAGE_REGISTRY[key], {}),
	))
	assert stage.contract.extent_sensitive_axes is not None, (
		f"{key} must explicitly declare StageContract.extent_sensitive_axes "
		"as axis names or (); omission would silently skip the windowed-parity warning"
	)


def test_standard_paganin_stage_is_registered_and_filters_tier1_fixture():
	config = StageConfig(kind="phase", algorithm="paganin", params={"alpha": 0.06})
	working = _fixture()

	result = Pipeline([create_stage(config)]).run(_state(working, config))

	assert STAGE_REGISTRY["phase.paganin"] is PaganinStage
	assert result.working_array is working
	assert np.isfinite(working).all()
	assert working[8, 0, 8] > working[0, 0, 0]


def test_disabled_standard_paganin_preserves_data_and_contract():
	config = StageConfig(
		kind="phase", algorithm="paganin", params={"alpha": 0.06, "enabled": False}
	)
	working = _fixture()
	before = working.copy()
	stage = create_stage(config)

	result = Pipeline([stage]).run(_state(working, config))

	assert result.working_array is working
	np.testing.assert_array_equal(working, before)
	assert stage.contract.produces_domain is PRESERVE
	assert stage.contract.mutates_input is False
	assert stage.contract.required_metadata == ()


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


@pytest.mark.parametrize("stage", [
	PaganinStage(),
	PaganinPhaseAbsorptionStage(),
	BronnikovAidedCorrectionStage(alpha=2.0),
])
def test_enabled_phase_algorithms_declare_terminal_projection_contracts(stage):
	assert stage.contract.accepts_domain == (DataDomain.TRANSMISSION,)
	assert stage.contract.produces_domain is DataDomain.ATTENUATION
	assert stage.contract.accepts == (ProjOrder, SinoOrder)
	assert stage.contract.prefers is ProjOrder
	assert stage.contract.mutates_input is True


@pytest.mark.parametrize("first, second", [
	(PaganinStage(), PaganinPhaseAbsorptionStage()),
	(PaganinPhaseAbsorptionStage(), BronnikovAidedCorrectionStage(alpha=2.0)),
	(BronnikovAidedCorrectionStage(alpha=2.0), PaganinStage()),
])
def test_enabled_phase_algorithms_cannot_chain_after_terminal_conversion(first, second):
	with pytest.raises(DomainCompatibilityError, match=r"1:phase\..*got attenuation"):
		validate_pipeline(
			[first, second],
			_state(_fixture(), StageConfig(kind="phase", algorithm="paganin")).scan,
			initial_layout=ProjOrder,
			initial_domain=DataDomain.TRANSMISSION,
		)
