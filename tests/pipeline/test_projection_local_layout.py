import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import BufferSpec, DataDomain, Pipeline, PipelineState
from lftomo.pipeline.stages import (
	BilateralFilterStage,
	BronnikovAidedCorrectionStage,
	HelicalIndexRemapStage,
	ManualCenterStage,
	MovementCorrectStage,
	NegativeLogStage,
	PaganinPhaseAbsorptionStage,
	PaganinStage,
	SkewCorrectStage,
	TaperBlendedNormalizeStage,
	ThetaSwitchNormalizeStage,
	WeightedNormalizeStage,
)
from lftomo.util.shapes import ProjOrder, SinoOrder
from tests.sample_data import attach_source_geometry


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
	return attach_source_geometry(PipelineState(
		working_array=array,
		scan=scan_config(array.shape[0] if layout is ProjOrder else array.shape[1],
			array.shape[1] if layout is ProjOrder else array.shape[0], array.shape[2]),
		recon=Recon(pipeline=[config]),
		buffer_spec=BufferSpec(layout(*array.shape), array.dtype),
		domain=DataDomain.TRANSMISSION,
	))


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


def test_serial_phase_units_preserve_strided_theta_positions(monkeypatch):
	import lftomo.pipeline.stages.phase as phase_module

	working = np.full((4, 2, 3), 0.5, dtype=np.float32)
	current = state(working, ProjOrder)
	attach_source_geometry(
		current, theta_start=1, theta_stride=2, physical_theta=8,
	)
	positions = []
	local_positions = []
	original = phase_module._unit_index

	def record_index(unit, local_index, shape):
		assert unit.internal.dtype == np.dtype(np.float32)
		positions.append(unit.pos.Theta)
		local_positions.append(local_index[unit.shape.axis_of("Theta")].start)
		return original(unit, local_index, shape)

	monkeypatch.setattr(phase_module, "_unit_index", record_index)
	NegativeLogStage().run(current)

	assert positions == [1, 3, 5, 7]
	assert local_positions == [0, 1, 2, 3]


def test_pipeline_preserves_projection_layout_through_manual_center():
	array = np.ones((3, 2, 4), dtype=np.float32)
	result = Pipeline(
		[ManualCenterStage(value_px=2)],
		initial_layout=ProjOrder, initial_domain=DataDomain.ATTENUATION,
	).run(state(array, ProjOrder))

	assert result.layout is ProjOrder
