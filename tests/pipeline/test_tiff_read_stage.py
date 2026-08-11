import numpy as np
import tifffile as tf

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
from lftomo.pipeline.stages import STAGE_REGISTRY, create_stage
from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import ProjOrder, SinoOrder


class NoopStage(Stage):
	name = "noop"
	contract = StageContract(expected_input=SinoOrder, produced_output=SinoOrder)

	def run(self, state):
		return state


def _state(projection_dir, count, size, config):
	return PipelineState(
		working_array=np.empty(0),
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1.0,
			n_projections=count,
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1.0, rows=size, columns=size),
			acquisition=Acquisition(n_projections=count, rotation_end_deg=180),
			projections=TIFFStackProjections(
				format="tiff_stack", path=projection_dir, pattern="projection_*.tif"
			),
		),
		recon=Recon(pipeline=[config]),
	)


def _run(state, *stages):
	with PipelineRunContext() as context:
		state.storage = context.storage
		result = Pipeline(list(stages), backend=NumpyBackend()).run(state)
		assert isinstance(result.working_array, SharedNPArray)
		return result.with_array(working_view(result).copy())


def test_registered_tiff_stack_stage_reads_natural_order_into_pipeline(tmp_path):
	count, size = 4, 5
	projection_dir = tmp_path / "projections"
	projection_dir.mkdir()
	stack = np.arange(count * size * size, dtype=np.uint16).reshape(count, size, size)
	for index in (2, 0, 3, 1):
		tf.imwrite(projection_dir / f"projection_{index}.tif", stack[index])
	config = StageConfig(
		kind="read", algorithm="tiff_stack", params={"order": "sinogram", "threads": 2}
	)

	stage = create_stage(config)
	result = _run(_state(projection_dir, count, size, config), stage, NoopStage())

	assert "read.tiff_stack" in STAGE_REGISTRY
	assert stage.contract.produced_output is SinoOrder
	np.testing.assert_array_equal(result.working_array, np.transpose(stack, (1, 0, 2)))
	assert result.dataset.shape == ProjOrder(Theta=count, Y=size, X=size)
	assert result.source_window.pos == ProjOrder(Theta=0, Y=0, X=0)
	assert result.buffer_spec.layout is SinoOrder


def test_tiff_stack_stage_reads_multi_page_tiff(tmp_path):
	count, size = 3, 4
	stack = np.arange(count * size * size, dtype=np.uint16).reshape(count, size, size)
	path = tmp_path / "projections.tif"
	tf.imwrite(path, stack, photometric="minisblack")
	config = StageConfig(kind="read", algorithm="tiff_stack")

	result = _run(_state(path, count, size, config), create_stage(config))

	np.testing.assert_array_equal(result.working_array, stack)


def test_tiff_stack_stage_reads_bounded_y_window(tmp_path):
	count, size = 2, 6
	projection_dir = tmp_path / "projections"
	projection_dir.mkdir()
	stack = np.arange(count * size * size, dtype=np.uint16).reshape(count, size, size)
	for index, image in enumerate(stack):
		tf.imwrite(projection_dir / f"projection_{index}.tif", image)
	config = StageConfig(
		kind="read", algorithm="tiff_stack", params={"y_start": 2, "y_stop": 4}
	)

	result = _run(_state(projection_dir, count, size, config), create_stage(config))

	np.testing.assert_array_equal(result.working_array, stack[:, 2:4, :])


def test_tiff_stack_stage_reads_bounded_theta_and_y_window(tmp_path):
	count, size = 5, 6
	projection_dir = tmp_path / "projections"
	projection_dir.mkdir()
	stack = np.arange(count * size * size, dtype=np.uint16).reshape(count, size, size)
	for index, image in enumerate(stack):
		tf.imwrite(projection_dir / f"projection_{index}.tif", image)
	config = StageConfig(
		kind="read", algorithm="tiff_stack",
		params={"theta_start": 1, "theta_stop": 4, "y_start": 2, "y_stop": 5},
	)

	result = _run(_state(projection_dir, count, size, config), create_stage(config))

	np.testing.assert_array_equal(result.working_array, stack[1:4, 2:5, :])


def test_tiff_stack_stage_reads_strided_theta_window_with_matching_angles(tmp_path):
	count, size = 6, 4
	projection_dir = tmp_path / "projections"
	projection_dir.mkdir()
	stack = np.arange(count * size * size, dtype=np.uint16).reshape(count, size, size)
	for index, image in enumerate(stack):
		tf.imwrite(projection_dir / f"projection_{index}.tif", image)
	config = StageConfig(
		kind="read", algorithm="tiff_stack",
		params={"theta_start": 1, "theta_stop": 6, "theta_stride": 2},
	)

	result = _run(_state(projection_dir, count, size, config), create_stage(config))

	np.testing.assert_array_equal(result.working_array, stack[1:6:2])
	np.testing.assert_allclose(result.artifacts["angles"], np.deg2rad([30, 90, 150]))
	assert result.source_window.shape.Theta == 5
	assert result.working_array.shape[0] == 3
	assert result.source_window.step == ProjOrder(Theta=2, Y=1, X=1)


def test_legacy_tiff_chunk_parameters_are_not_runtime_stage_state():
	config = StageConfig(kind="read", algorithm="tiff_stack", params={
		"sinogram_chunk_size": 140,
		"chunk_overlap": 30,
	})
	stage = create_stage(config)
	assert not hasattr(stage, "sinogram_chunk_size")
	assert not hasattr(stage, "chunk_overlap")
