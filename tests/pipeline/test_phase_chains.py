import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	DataDomain,
	DomainCompatibilityError,
	PRESERVE,
	Pipeline,
	PipelineState,
	validate_pipeline,
)
from lftomo.pipeline.stages import (
	BetaScaleStage,
	BronnikovAidedCorrectionFilterStage,
	BronnikovAidedCorrectionStage,
	NegativeLogStage,
	PaganinFilterStage,
	PaganinPhaseAbsorptionStage,
	PaganinStage,
	STAGE_REGISTRY,
	create_stage,
)
from lftomo.util.shapes import ProjOrder, SinoOrder


def _scan():
	return Scan(
		energy_keV=20,
		pixel_size_um=1.0,
		ssd_cm=10.0,
		n_projections=3,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1.0),
		acquisition=Acquisition(n_projections=3, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


def _fixture(layout):
	data = np.full((16, 3, 16), 0.8, dtype=np.float32)
	data[5:11, :, 5:11] = 0.35
	data[7:9, :, 7:9] = 0.55
	return data if layout is SinoOrder else data.transpose(1, 0, 2).copy()


def _run(stages, data, layout):
	config = StageConfig(kind="phase", algorithm="negative_log")
	state = PipelineState(
		working_array=data,
		scan=_scan(),
		recon=Recon(pipeline=[config]),
		layout=layout,
		domain=DataDomain.TRANSMISSION,
	)
	return Pipeline(
		stages,
		initial_layout=layout,
		initial_domain=DataDomain.TRANSMISSION,
	).run(state)


def _independent_bac(image, alpha, gamma):
	qy = 2 * np.pi * np.fft.fftfreq(image.shape[0])
	qx = 2 * np.pi * np.fft.fftfreq(image.shape[1])
	w2 = np.add.outer(qy**2, qx**2)
	kernel = w2 / (w2 + alpha)
	correction = -np.fft.ifft2(np.fft.fft2(image - 1) * kernel).real
	return image / (1 - gamma * correction)


def test_split_phase_algorithms_are_registered_with_true_domain_effects():
	assert STAGE_REGISTRY["phase.paganin_filter"] is PaganinFilterStage
	assert STAGE_REGISTRY["phase.bac_correction"] is BronnikovAidedCorrectionFilterStage
	assert STAGE_REGISTRY["phase.negative_log"] is NegativeLogStage
	assert STAGE_REGISTRY["phase.beta_scale"] is BetaScaleStage

	first = create_stage(StageConfig(
		kind="phase", id="phase-filter", algorithm="paganin_filter",
	))
	terminal = create_stage(StageConfig(
		kind="phase", id="terminal-log", algorithm="negative_log",
	))

	assert first.occurrence_id == "phase-filter"
	assert first.contract.produces_domain is PRESERVE
	assert terminal.occurrence_id == "terminal-log"
	assert terminal.contract.produces_domain is DataDomain.ATTENUATION


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
def test_paganin_filter_then_negative_log_matches_legacy_fused_stage(layout):
	legacy = _fixture(layout)
	split = legacy.copy()

	_run([PaganinStage(alpha=0.06)], legacy, layout)
	result = _run([
		PaganinFilterStage(alpha=0.06),
		NegativeLogStage(epsilon=1e-8),
	], split, layout)

	# The explicit Stage boundary stores the filtered intermediate at the working
	# array's float32 precision before applying the logarithm.
	np.testing.assert_allclose(split, legacy, rtol=3e-7, atol=3e-8)
	assert result.domain is DataDomain.ATTENUATION


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
def test_split_coupled_phase_absorption_matches_legacy_fused_stage(layout):
	legacy = _fixture(layout)
	split = legacy.copy()

	_run([PaganinPhaseAbsorptionStage(alpha=0.06)], legacy, layout)
	_run([
		PaganinFilterStage(alpha=0.06),
		NegativeLogStage(epsilon=1e-8, maximum=0.99999999),
		BetaScaleStage(),
	], split, layout)

	np.testing.assert_allclose(split, legacy, rtol=3e-7, atol=1e-16)


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
def test_bac_correction_then_negative_log_matches_legacy_fused_stage(monkeypatch, layout):
	alpha = 2.0
	gamma = 1.25
	legacy = _fixture(layout)
	split = legacy.copy()

	class IndependentBAC:
		def __init__(self, _shape, alpha, gamma, device):
			self.alpha = alpha
			self.gamma = gamma

		def __call__(self, image):
			return -np.log(_independent_bac(image, self.alpha, self.gamma))

	monkeypatch.setattr("hotopy.holo.BronnikovAidedCorrection", IndependentBAC)
	_run([BronnikovAidedCorrectionStage(alpha=alpha, gamma=gamma)], legacy, layout)
	_run([
		BronnikovAidedCorrectionFilterStage(alpha=alpha, gamma=gamma),
		NegativeLogStage(epsilon=None),
	], split, layout)

	np.testing.assert_allclose(split, legacy, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
def test_bac_correction_then_negative_log_matches_installed_hotopy(layout):
	pytest.importorskip("hotopy.holo")
	alpha = 2.0
	gamma = 1.25
	legacy = _fixture(layout)
	split = legacy.copy()

	_run([BronnikovAidedCorrectionStage(alpha=alpha, gamma=gamma)], legacy, layout)
	_run([
		BronnikovAidedCorrectionFilterStage(alpha=alpha, gamma=gamma),
		NegativeLogStage(epsilon=None),
	], split, layout)

	np.testing.assert_allclose(split, legacy, rtol=2e-6, atol=2e-7)


def test_multiple_transmission_corrections_compose_before_one_negative_log():
	data = _fixture(ProjOrder)
	configs = [
		StageConfig(
			kind="phase", id="paganin-prefilter", algorithm="paganin_filter",
			params={"alpha": 0.06},
		),
		StageConfig(
			kind="phase", id="bac-correction", algorithm="bac_correction",
			params={"alpha": 2.0, "gamma": 1.25},
		),
		StageConfig(kind="phase", id="attenuation", algorithm="negative_log"),
	]
	stages = [create_stage(config) for config in configs]

	plan = validate_pipeline(
		stages, _scan(), initial_layout=ProjOrder,
		initial_domain=DataDomain.TRANSMISSION,
	)
	result = _run(stages, data, ProjOrder)

	assert [step.output_domain for step in plan.steps] == [
		DataDomain.TRANSMISSION,
		DataDomain.TRANSMISSION,
		DataDomain.ATTENUATION,
	]
	assert result.domain is DataDomain.ATTENUATION
	assert np.isfinite(data).all()


def test_noncommuting_correction_orders_are_both_valid_experiment_candidates():
	first = _fixture(ProjOrder)
	second = first.copy()
	terminal = NegativeLogStage()

	_run([
		PaganinFilterStage(alpha=0.06),
		BronnikovAidedCorrectionFilterStage(alpha=2.0, gamma=1.25),
		terminal,
	], first, ProjOrder)
	_run([
		BronnikovAidedCorrectionFilterStage(alpha=2.0, gamma=1.25),
		PaganinFilterStage(alpha=0.06),
		NegativeLogStage(),
	], second, ProjOrder)

	assert not np.allclose(first, second, rtol=1e-7, atol=1e-8)


def test_terminal_conversion_rejects_later_corrections_and_duplicate_log():
	terminal = NegativeLogStage()
	correction = PaganinFilterStage()

	with pytest.raises(DomainCompatibilityError, match="paganin_filter.*got attenuation"):
		validate_pipeline(
			[terminal, correction], _scan(), initial_layout=ProjOrder,
			initial_domain=DataDomain.TRANSMISSION,
		)
	with pytest.raises(DomainCompatibilityError, match="negative_log.*got attenuation"):
		validate_pipeline(
			[terminal, NegativeLogStage()], _scan(), initial_layout=ProjOrder,
			initial_domain=DataDomain.TRANSMISSION,
		)
	with pytest.raises(DomainCompatibilityError, match="beta_scale.*got transmission"):
		validate_pipeline(
			[BetaScaleStage(), terminal], _scan(), initial_layout=ProjOrder,
			initial_domain=DataDomain.TRANSMISSION,
		)


def test_disabled_correction_is_a_domain_preserving_noop():
	data = _fixture(ProjOrder)
	before = data.copy()
	stage = BronnikovAidedCorrectionFilterStage(alpha=2.0, enabled=False)
	state = PipelineState(
		working_array=data, scan=_scan(),
		recon=Recon(pipeline=[StageConfig(kind="phase", algorithm="bac_correction")]),
		layout=ProjOrder, domain=DataDomain.ATTENUATION,
	)

	result = Pipeline(
		[stage], initial_layout=ProjOrder, initial_domain=DataDomain.ATTENUATION,
	).run(state)

	np.testing.assert_array_equal(data, before)
	assert result.domain is DataDomain.ATTENUATION
	assert stage.contract.produces_domain is PRESERVE


def test_disabled_negative_log_does_not_claim_terminal_conversion():
	data = _fixture(ProjOrder)
	before = data.copy()
	stage = NegativeLogStage(enabled=False)

	result = _run([stage], data, ProjOrder)

	np.testing.assert_array_equal(data, before)
	assert result.domain is DataDomain.TRANSMISSION
	assert stage.contract.produces_domain is PRESERVE
	assert stage.contract.mutates_input is False


def test_split_operations_reject_invalid_unclipped_transmission_values():
	data = np.zeros((1, 4, 4), dtype=np.float32)

	with pytest.raises(ValueError, match="bac_correction produced non-positive"):
		_run([
			BronnikovAidedCorrectionFilterStage(alpha=2.0),
			NegativeLogStage(),
		], data.copy(), ProjOrder)
	with pytest.raises(ValueError, match="requires finite positive transmission"):
		_run([NegativeLogStage(epsilon=None)], data.copy(), ProjOrder)
