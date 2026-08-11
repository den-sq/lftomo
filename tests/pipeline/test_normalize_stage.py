from uuid import uuid4

import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	NumpyBackend,
	Pipeline,
	PipelineRunContext,
	PipelineState,
	Stage,
	StageContract,
	working_view,
)
from lftomo.pipeline.stages.normalize import TaperBlendedNormalizeStage, WeightedNormalizeStage
from lftomo.prep.flat import weighted_normalize
from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import GenericOrder, ProjOrder, SinoOrder, Y
from lftomo.util.window import TomoWindow


class PassThroughStage(Stage):
	contract = StageContract(expected_input=SinoOrder, produced_output=SinoOrder)

	def __init__(self, name):
		self.name = name

	def run(self, state):
		return state


def _scan(flat_fields):
	return Scan(
		energy_keV=20,
		pixel_size_um=1.0,
		n_projections=3,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1.0, rows=2, columns=2),
		acquisition=Acquisition(n_projections=3, rotation_end_deg=180),
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
	state = PipelineState(
		working_array=working,
		scan=_scan(flats),
		recon=Recon(pipeline=[StageConfig(kind="normalize")]),
	)

	result = Pipeline([
		PassThroughStage("read"),
		WeightedNormalizeStage(),
		PassThroughStage("noop"),
	], backend=NumpyBackend()).run(state)

	assert result.working_array is working
	assert WeightedNormalizeStage.contract.mutates_input is False
	assert WeightedNormalizeStage.contract.window_axes == ("Y",)
	assert WeightedNormalizeStage.contract.consumes_artifacts == ()
	assert WeightedNormalizeStage.contract.produces_artifacts == ("flat_fields",)
	np.testing.assert_allclose(result.working_array, expected, rtol=1e-6)


def test_weighted_normalize_promotes_integer_projections_to_float32():
	working = np.full((2, 3, 2), 10, dtype=np.uint16)
	flats = np.stack([
		np.full((2, 2), 20), np.full((2, 2), 20),
		np.zeros((2, 2)), np.zeros((2, 2)),
	]).astype(np.float32)
	state = PipelineState(
		working_array=working,
		scan=_scan(flats),
		recon=Recon(pipeline=[StageConfig(kind="normalize")]),
	)

	result = WeightedNormalizeStage().run(state)

	assert result.working_array.dtype == np.float32
	np.testing.assert_allclose(result.working_array, 0.5)


def test_float32_normalization_replaces_raw_handle_after_successful_handoff():
	flats = np.stack([
		np.full((2, 2), 20), np.full((2, 2), 20),
		np.zeros((2, 2)), np.zeros((2, 2)),
	]).astype(np.float32)
	with PipelineRunContext() as context:
		buffer = context.storage.allocate(SinoOrder(Y=2, Theta=3, X=2), np.float32)
		buffer.array()[...] = 10
		state = PipelineState(
			working_array=buffer,
			scan=_scan(flats),
			recon=Recon(pipeline=[StageConfig(kind="normalize")]),
			storage=context.storage,
			layout=SinoOrder,
		)

		result = WeightedNormalizeStage().run(state)

		assert isinstance(result.working_array, SharedNPArray)
		assert result.working_array is not buffer
		assert result.working_array.name != buffer.name
		np.testing.assert_allclose(working_view(result), 0.5)
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
	state = PipelineState(
		working_array=working,
		scan=scan,
		recon=Recon(pipeline=[StageConfig(kind="normalize")]),
		artifacts={"projection_start": 2, "projection_count": 4},
	)
	stage = TaperBlendedNormalizeStage(order="projection")

	result = stage.run(state)

	assert stage.contract.expected_input is ProjOrder
	np.testing.assert_allclose(result.working_array[0], 10 / 30)
	np.testing.assert_allclose(result.working_array[1], 10 / 35)
