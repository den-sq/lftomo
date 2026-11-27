import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import LocalArray, DataDomain, Pipeline, PipelineState
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
	ZingerFilterStage,
)
from lftomo.util.shapes import ProjOrder, SinoOrder, Theta
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
		working_array=LocalArray(array, layout(*array.shape)),
		scan=scan_config(array.shape[0] if layout is ProjOrder else array.shape[1],
			array.shape[1] if layout is ProjOrder else array.shape[0], array.shape[2]),
		recon=Recon(pipeline=[config]),
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
	ZingerFilterStage(),
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


def test_serial_phase_processes_only_assigned_projections_once():
	working = np.full((4, 2, 3), 0.5, dtype=np.float32)
	current = state(working, ProjOrder)
	attach_source_geometry(
		current, theta_start=1, theta_stride=2, physical_theta=8,
	)
	current.work_range = Theta.drange(1, 4, 2)
	expected = working.copy()
	expected[1:4:2] = -np.log(expected[1:4:2])
	assert NegativeLogStage().run(current) is current
	np.testing.assert_allclose(working, expected)


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
def test_zinger_strict_threshold_preserves_bright_edges_and_dark_pixels(layout):
	projection = np.ones((1, 7, 9), dtype=np.float32)
	projection[:, :, 5:] = 3
	projection[0, 2, 2] = 1.5
	projection[0, 4, 2] = np.nextafter(np.float32(1.5), np.float32(np.inf))
	projection[0, 0, 0] = 4
	projection[0, 6, 8] = 9
	projection[0, 2, 7] = 0.2
	expected = projection.copy()
	expected[0, 4, 2] = 1
	expected[0, 0, 0] = 1
	expected[0, 6, 8] = 3
	working = projection if layout is ProjOrder else projection.transpose(1, 0, 2).copy()
	current = state(working, layout)

	assert ZingerFilterStage().run(current) is current

	actual = working if layout is ProjOrder else working.transpose(1, 0, 2)
	np.testing.assert_array_equal(actual, expected)
	assert current.domain is DataDomain.TRANSMISSION


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
def test_zinger_uses_original_plane_medians_before_replacing_neighbors(layout):
	projection = np.array([[[1, 1, 1], [1, 4, 8], [1, 1, 4]]], dtype=np.float32)
	# The middle-right pixel has original median 4. Recomputing after the
	# centre replacement would instead give it median 1.
	expected = np.array([[[1, 1, 1], [1, 1, 4], [1, 1, 4]]], dtype=np.float32)
	working = projection if layout is ProjOrder else projection.transpose(1, 0, 2).copy()

	ZingerFilterStage().run(state(working, layout))

	actual = working if layout is ProjOrder else working.transpose(1, 0, 2)
	np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
@pytest.mark.parametrize("enabled", [False, True])
def test_zinger_respects_strided_assigned_projections_and_disabled_state(layout, enabled):
	projection = np.broadcast_to(
		(np.arange(4, dtype=np.float32) / 10 + 1)[:, None, None], (4, 5, 7),
	).copy()
	projection[:, 2, 3] *= 4
	expected = projection.copy()
	if enabled:
		expected[1:4:2, 2, 3] = expected[1:4:2, 2, 2]
	working = projection if layout is ProjOrder else projection.transpose(1, 0, 2).copy()
	current = state(working, layout)
	attach_source_geometry(current, theta_start=1, theta_stride=2, physical_theta=8)
	current.work_range = Theta.drange(1, 4, 2)

	assert ZingerFilterStage(enabled=enabled).run(current) is current

	actual = working if layout is ProjOrder else working.transpose(1, 0, 2)
	np.testing.assert_array_equal(actual, expected)


def test_pipeline_preserves_projection_layout_through_manual_center():
	array = np.ones((3, 2, 4), dtype=np.float32)
	result = Pipeline(
		[ManualCenterStage(value_px=2)],
		initial_layout=ProjOrder, initial_domain=DataDomain.ATTENUATION,
	).run(state(array, ProjOrder))

	assert result.layout is ProjOrder
