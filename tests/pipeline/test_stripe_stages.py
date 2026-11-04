from contextlib import nullcontext

import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import BufferSpec, Pipeline, PipelineState, run_sinogram_stage
from lftomo.pipeline.stages import STAGE_REGISTRY, create_stage
from lftomo.storage import PipelineRunContext
from lftomo.util.shapes import ProjOrder, SinoOrder
from lftomo.util.window import DataRef
from tests.sample_data import attach_source_geometry


def _state(array, layout, config):
	return PipelineState(
		working_array=array,
		scan=Scan(
			energy_keV=20, pixel_size_um=1,
			n_projections=array.shape[layout.axis_of("Theta")], degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(
				n_projections=array.shape[layout.axis_of("Theta")], rotation_end_deg=180,
			),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
		buffer_spec=BufferSpec(layout(*array.shape), array.dtype),
	)


def _striped_fixture():
	theta = np.linspace(0.8, 1.2, 24, dtype=np.float32)
	x = np.linspace(0.9, 1.1, 24, dtype=np.float32)
	data = theta[None, :, None] * x[None, None, :]
	data = np.repeat(data, 3, axis=0)
	data[:, :, 11] *= 4
	data[:, :, 17] = 0
	return data


@pytest.mark.parametrize(("algorithm", "params"), [
	("vo_sort", {"size": 3}),
	("vo_dead", {"snr": 1, "size": 5, "norm": False}),
	("vo_large", {"snr": 1, "size": 5, "drop_ratio": 0.25}),
])
def test_registered_vo_stage_mutates_tier1_striped_sinogram(algorithm, params):
	config = StageConfig(kind="stripe", algorithm=algorithm, params=params)
	working = _striped_fixture()
	before = working.copy()

	result = Pipeline([create_stage(config)]).run(_state(working, SinoOrder, config))

	stage = create_stage(config)
	assert f"stripe.{algorithm}" in STAGE_REGISTRY
	assert stage.contract.mutates_input is True
	assert stage.contract.window_axes == ("Y",)
	assert stage.contract.required_metadata == ()
	assert stage.contract.consumes_artifacts == ()
	assert stage.contract.produces_artifacts == ()
	assert result.working_array is working
	assert np.isfinite(working).all()
	assert not np.array_equal(working, before)


def test_fourier_wavelet_projection_order_matches_sinogram_order():
	sinogram_order = _striped_fixture()
	projection_order = np.transpose(sinogram_order.copy(), (1, 0, 2))
	sino_config = StageConfig(
		kind="stripe", algorithm="fw", params={"level": 2, "wavelet": "db2"},
	)
	proj_config = sino_config

	create_stage(sino_config).run(_state(sinogram_order, SinoOrder, sino_config))
	proj_stage = create_stage(proj_config)
	proj_state = _state(projection_order, ProjOrder, proj_config)
	proj_stage.run(proj_state)

	assert proj_stage.contract.accepted_layouts == (SinoOrder,)
	np.testing.assert_allclose(projection_order, np.transpose(sinogram_order, (1, 0, 2)))


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
@pytest.mark.parametrize("managed", [False, True])
def test_public_sinogram_helper_uses_canonical_units_and_copies_returned_values(
	monkeypatch, layout, managed,
):
	config = StageConfig(kind="stripe", algorithm="vo_sort", params={"size": 3})
	expected = _striped_fixture()
	stage = create_stage(config)
	expected_state = _state(expected, SinoOrder, config)
	for row in expected:
		stage.run_sinogram(row, expected_state)
	values = _striped_fixture()
	if layout is ProjOrder:
		values = values.transpose(1, 0, 2).copy()
	units = []
	canonical_units = DataRef.units

	def record_units(reference, axis):
		assert axis == "Y"
		for unit, local_index in canonical_units(reference, axis):
			units.append(unit)
			yield unit, local_index

	monkeypatch.setattr(DataRef, "units", record_units)
	with PipelineRunContext() if managed else nullcontext() as context:
		working = values
		if managed:
			working = context.storage.allocate(
				layout(*values.shape), values.dtype, role="normalized",
			)
			working.array()[...] = values
		state = _state(values, layout, config)
		state.working_array = working
		if managed:
			state.storage = context.storage
		state = attach_source_geometry(state, y_start=7, physical_y=10)
		source_window = state.source_window
		buffer_spec = state.buffer_spec
		events_before = context.storage.allocation_events if managed else ()
		kernel = stage.run_sinogram

		def return_corrected_row(row, row_state):
			assert row_state.source_window is units[-1]
			assert row_state.buffer_spec.shape is units[-1].shape
			assert row_state.buffer_spec.dtype == state.buffer_spec.dtype
			assert row_state.artifacts is state.artifacts
			assert np.shares_memory(row, np.asarray(working))
			assert np.shares_memory(row_state.working_array, row)
			replacement = row.copy()
			kernel(replacement, row_state)
			return replacement

		monkeypatch.setattr(stage, "run_sinogram", return_corrected_row)
		assert run_sinogram_stage(stage, state) is state
		assert state.working_array is working
		assert state.source_window is source_window
		assert state.buffer_spec is buffer_spec
		assert [unit.pos.Y for unit in units] == [7, 8, 9]
		assert all(unit.shape.Y == 1 for unit in units)
		if managed:
			assert context.storage.allocation_events == events_before
		if layout is ProjOrder:
			expected = expected.transpose(1, 0, 2)
		np.testing.assert_allclose(np.asarray(working), expected)
