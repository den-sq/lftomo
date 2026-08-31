from pathlib import Path

import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import DataDomain, MaterializationKind, Pipeline, PipelineState, validate_pipeline
from lftomo.pipeline.stages import (
	BilateralFilterStage,
	BronnikovAidedCorrectionStage,
	FourierWaveletStripeStage,
	HelicalIndexRemapStage,
	ManualCenterStage,
	MovementCorrectStage,
	PaganinPhaseAbsorptionStage,
	PaganinStage,
	SkewCorrectStage,
	TaperBlendedNormalizeStage,
	ThetaSwitchNormalizeStage,
	WeightedNormalizeStage,
	create_stage,
)
from lftomo.runner import load_recon
from lftomo.util.shapes import ProjOrder, SinoOrder


def scan_config(n_projections=3, rows=2, columns=4):
	return Scan(
		energy_keV=20,
		pixel_size_um=1,
		ssd_cm=10,
		n_projections=n_projections,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1, rows=rows, columns=columns),
		acquisition=Acquisition(n_projections=n_projections, rotation_end_deg=180),
		projections=TIFFStackProjections(
			format="tiff_stack", path="projections", pattern="proj_*.tif",
		),
		# Validation needs presence only; numerical tests replace this with tiny maps.
		metadata={"flat_fields": np.ones((1,), dtype=np.float32)},
	)


def state(array, layout, config=None):
	config = config or StageConfig(kind="test")
	return PipelineState(
		working_array=array,
		scan=scan_config(array.shape[0] if layout is ProjOrder else array.shape[1],
			array.shape[1] if layout is ProjOrder else array.shape[0], array.shape[2]),
		recon=Recon(pipeline=[config]),
		layout=layout,
		domain=DataDomain.TRANSMISSION,
	)


@pytest.mark.parametrize("stage", [
	WeightedNormalizeStage(),
	ThetaSwitchNormalizeStage(),
	TaperBlendedNormalizeStage(),
	PaganinStage(),
	PaganinPhaseAbsorptionStage(),
	BronnikovAidedCorrectionStage(alpha=2),
	BilateralFilterStage(),
	SkewCorrectStage(),
	MovementCorrectStage(),
	HelicalIndexRemapStage(enabled=False),
	ManualCenterStage(value_px=2),
])
def test_projection_local_migrations_advertise_both_layouts(stage):
	assert stage.contract.accepts == (ProjOrder, SinoOrder)


def test_fourier_wavelet_declares_bounded_projection_view_capability():
	contract = FourierWaveletStripeStage(level=2).contract
	assert contract.accepts == (SinoOrder,)
	assert contract.bounded_views_from == (ProjOrder,)


def test_state_layout_drives_normalization_without_constructor_order():
	projection = np.arange(24, dtype=np.float32).reshape(3, 2, 4) + 1
	sinogram = projection.transpose(1, 0, 2).copy()
	flats = np.stack([
		np.full((2, 4), 10, dtype=np.float32),
		np.full((2, 4), 12, dtype=np.float32),
		np.zeros((2, 4), dtype=np.float32),
		np.zeros((2, 4), dtype=np.float32),
	])
	proj_state = state(projection, ProjOrder)
	sino_state = state(sinogram, SinoOrder)
	proj_state.scan.metadata["flat_fields"] = flats.copy()
	sino_state.scan.metadata["flat_fields"] = flats.copy()

	ThetaSwitchNormalizeStage(gain_averaging=True).run(proj_state)
	ThetaSwitchNormalizeStage(gain_averaging=True).run(sino_state)

	np.testing.assert_allclose(projection, sinogram.transpose(1, 0, 2))


def test_state_layout_drives_paganin_without_constructor_order():
	projection = np.linspace(0.2, 1.0, 24, dtype=np.float32).reshape(3, 2, 4)
	sinogram = projection.transpose(1, 0, 2).copy()

	PaganinStage(alpha=0.06).run(state(projection, ProjOrder))
	PaganinStage(alpha=0.06).run(state(sinogram, SinoOrder))

	np.testing.assert_allclose(projection, sinogram.transpose(1, 0, 2), rtol=1e-6, atol=1e-6)


def test_phase_contrast_canonical_plans_projection_order_without_injection():
	path = Path(__file__).parents[2] / "lftomo" / "pipelines" / "phase-contrast-canonical.yaml"
	recon, _ = load_recon(path)
	stages = [create_stage(config) for config in recon.pipeline]
	plan = validate_pipeline(stages, scan_config(1969, 4440, 5008), mode="complete")

	recon_step = next(step for step in plan.steps if step.stage_name == "recon.gridrec")
	assert all(
		step.output_layout is ProjOrder
		for step in plan.steps
		if step.index < recon_step.index
	)
	assert recon_step.input_layout is ProjOrder
	assert recon_step.materializations[0].kind is MaterializationKind.TOMOPY_BOUNDARY


def test_pipeline_preserves_projection_layout_through_manual_center():
	array = np.ones((3, 2, 4), dtype=np.float32)
	result = Pipeline(
		[ManualCenterStage(value_px=2)],
		initial_layout=ProjOrder, initial_domain=DataDomain.ATTENUATION,
	).run(state(array, ProjOrder))

	assert result.layout is ProjOrder
