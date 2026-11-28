from uuid import uuid4

import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	LocalArray,
	NumpyBackend,
	Pipeline,
	PipelineRunContext,
	PipelineState,
	Stage,
	StageContract,
)
from lftomo.pipeline.stages.normalize import TaperBlendedNormalizeStage, WeightedNormalizeStage
from lftomo.prep.flat import weighted_normalize
from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import GenericOrder, ProjOrder, SinoOrder, Y
from lftomo.util.window import CTFov, TomoDataSet, TomoWindow
from tests.sample_data import attach_source_geometry


class PassThroughStage(Stage):
	contract = StageContract(accepts=(SinoOrder,), produces=SinoOrder)

	def __init__(self, name):
		self.name = name

	def run(self, state):
		return state


def _scan(flat_fields, *, rows=2, columns=2, projections=3):
	return Scan(
		energy_keV=20,
		pixel_size_um=1.0,
		n_projections=projections,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1.0, rows=rows, columns=columns),
		acquisition=Acquisition(n_projections=projections, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		metadata={"flat_fields": flat_fields},
	)


def _classic(source, flats):
	shape = SinoOrder(Y=2, Theta=3, X=2)
	output_memory = SharedNPArray(f"normalize-output-{uuid4().hex}", shape, np.float32, allocate=True)
	source_memory = SharedNPArray(f"normalize-source-{uuid4().hex}", shape, np.float32, allocate=True)
	flat_memory = SharedNPArray(
		f"normalize-flats-{uuid4().hex}", GenericOrder(*flats.shape), np.float32, allocate=True
	)
	try:
		with source_memory as values:
			values[:] = source
		with flat_memory as values:
			values[:] = flats
		window = TomoWindow(shape, Y(0), Y(0), Y(1), np.float32, np.float32)
		for projection in range(shape.Theta):
			weighted_normalize(output_memory, flat_memory, source_memory, window, projection)
		with output_memory as values:
			return values.copy()
	finally:
		output_memory.shutdown()
		source_memory.shutdown()
		flat_memory.shutdown()


def test_weighted_normalize_stage_matches_classic_function_in_pipeline():
	source = np.arange(12, dtype=np.float32).reshape(2, 3, 2) + 10
	flats = np.array([
		[[6, 8], [10, 12]],
		[[10, 12], [14, 16]],
		[[1, 2], [3, 4]],
		[[2, 3], [4, 5]],
	], dtype=np.float32)
	expected = _classic(source, flats)
	working = source.copy()
	state = attach_source_geometry(PipelineState(
		working_array=LocalArray(working, SinoOrder(*working.shape)),
		scan=_scan(flats),
		recon=Recon(pipeline=[StageConfig(kind="normalize")]),
	))

	result = Pipeline([
		PassThroughStage("read"),
		WeightedNormalizeStage(),
		PassThroughStage("noop"),
	], backend=NumpyBackend()).run(state)

	assert np.asarray(result.working_array) is working
	assert WeightedNormalizeStage.contract.mutates_input is False
	assert WeightedNormalizeStage.contract.window_axes == ("Y",)
	assert WeightedNormalizeStage.contract.consumes_artifacts == ()
	assert WeightedNormalizeStage.contract.produces_artifacts == ("flat_fields",)
	np.testing.assert_allclose(result.working_array, expected, rtol=1e-6)


@pytest.mark.parametrize("layout,flat_origin", [(ProjOrder, "selected"), (SinoOrder, "physical")])
def test_legacy_flat_fields_are_sliced_from_the_canonical_source_window(layout, flat_origin):
	physical = ProjOrder(Theta=8, Y=8, X=2)
	detector_rows = np.arange(physical.Y, dtype=np.float32)
	if flat_origin == "selected":
		detector_rows = detector_rows[1:7]
	gain = np.broadcast_to(20 + 10 * detector_rows[:, None], (len(detector_rows), 2))
	flats = np.stack([gain, gain + 16, np.zeros_like(gain), np.zeros_like(gain)])
	dataset = TomoDataSet.from_range(
		CTFov(8, 180, 1), physical, Y.drange(1, 7, 2), dtype=np.uint16,
		angles=np.linspace(0, np.pi, 8, endpoint=False),
	)
	window = TomoWindow.from_bounds(
		physical, y=(3, 5), theta=(1, 8, 3), halo=(Y(1), Y(2)), dtype=np.uint16,
	)
	working = np.full(layout(Y=5, Theta=3, X=2).as_tuple(), 10, dtype=np.uint16)
	state = PipelineState(
		LocalArray(working, layout(*working.shape)), _scan(flats, rows=8, projections=8),
		Recon(pipeline=[StageConfig(kind="normalize")]),
		dataset=dataset, source_window=window,
	)
	result = WeightedNormalizeStage().run(state)

	assert result.working_array.dtype == np.float32
	np.testing.assert_array_equal(working, 10)
	for index, projection in enumerate((1, 4, 7)):
		expected_gain = 20 + 10 * np.arange(3, 5)[:, None] + 2 * projection
		expected = np.full((5, 2), 10, dtype=np.float32)
		expected[1:3] = 10 / expected_gain
		np.testing.assert_allclose(np.asarray(result.working_array)[layout.point_index(Theta=index)], expected)
	assert result.artifacts["flat_fields"].shape == (4, 2, 2)
	assert state.scan.metadata["flat_fields"].shape == (4, len(detector_rows), 2)


def test_float32_normalization_replaces_raw_handle_after_successful_handoff():
	flats = np.stack([
		np.full((2, 2), 20), np.full((2, 2), 20),
		np.zeros((2, 2)), np.zeros((2, 2)),
	]).astype(np.float32)
	with PipelineRunContext() as context:
		buffer = context.storage.allocate(SinoOrder(Y=2, Theta=3, X=2), np.float32)
		buffer.array()[...] = 10
		state = attach_source_geometry(PipelineState(
			working_array=buffer,
			scan=_scan(flats),
			recon=Recon(pipeline=[StageConfig(kind="normalize")]),
			storage=context.storage,
		))

		result = WeightedNormalizeStage().run(state)

		assert isinstance(result.working_array, SharedNPArray)
		assert result.working_array is not buffer
		assert result.working_array.name != buffer.name
		np.testing.assert_allclose(np.asarray(result.working_array), 0.5)
		assert any(
			event["action"] == "retire" and event["name"] == buffer.name
			for event in context.storage.allocation_events
		)


def test_taper_normalize_projection_batch_uses_global_theta_weights():
	working = np.full((2, 2, 2), 10, dtype=np.uint16)
	flats = np.stack([
		np.full((2, 2), 20), np.full((2, 2), 40),
		np.zeros((2, 2)), np.zeros((2, 2)),
	]).astype(np.float32)
	scan = _scan(flats)
	scan.n_projections = 4
	state = attach_source_geometry(PipelineState(
		working_array=LocalArray(working, ProjOrder(*working.shape)),
		scan=scan,
		recon=Recon(pipeline=[StageConfig(kind="normalize")]),
	), theta_start=2, physical_theta=4)
	stage = TaperBlendedNormalizeStage()

	result = stage.run(state)

	assert stage.contract.accepted_layouts == (ProjOrder, SinoOrder)
	np.testing.assert_allclose(np.asarray(result.working_array)[0], 10 / 30)
	np.testing.assert_allclose(np.asarray(result.working_array)[1], 10 / 35)
	assert "projection_start" not in result.artifacts
	assert "projection_count" not in result.artifacts
