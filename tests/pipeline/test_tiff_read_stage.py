import numpy as np
import pytest
import tifffile as tf

from lftomo.config import (
	Acquisition,
	Detector,
	HardwareTimestampAngleSource,
	Recon,
	Scan,
	StageConfig,
	TIFFStackProjections,
)
from lftomo.pipeline import (
	NumpyBackend,
	Pipeline,
	PipelineRunContext,
	PipelineState,
	Stage,
	StageContract,
	resolve_streaming_plan,
)
from lftomo.storage import LocalArray
from lftomo.pipeline.stages import STAGE_REGISTRY, create_stage
from lftomo.pipeline.domains import DataDomain
from lftomo.pipeline.planning import _read_spec
from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import ProjOrder, SinoOrder


class NoopStage(Stage):
	name = "noop"
	contract = StageContract(accepts=(SinoOrder,), produces=SinoOrder)

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


def _irregular_state(projection_dir, scanlog, size, configs):
	return PipelineState(
		working_array=np.empty(0),
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1.0,
			n_projections=5,
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1.0, rows=size, columns=size),
			acquisition=Acquisition(
				n_projections=6,
				rotation_end_deg=180,
				angle_source=HardwareTimestampAngleSource(path=scanlog),
			),
			projections=TIFFStackProjections(
				format="tiff_stack", path=projection_dir, pattern="projection_*.tif"
			),
		),
		recon=Recon(pipeline=configs),
	)


def _write_irregular_fixture(tmp_path, size=4):
	projection_dir = tmp_path / "projections"
	projection_dir.mkdir()
	stack = np.arange(5 * size * size, dtype=np.uint16).reshape(5, size, size)
	for index, image in enumerate(stack):
		tf.imwrite(projection_dir / f"projection_{index}.tif", image)
	scanlog = tmp_path / "scanlog.txt"
	scanlog.write_text(
		"projection\twall_clock\tdelta\thardware\n"
		+ "\n".join(
			f"projection_{index}\tunused\tunused\t{timestamp}"
			for index, timestamp in enumerate([0, 100, 300, 400, 500])
		)
		+ "\n",
		encoding="utf-8",
	)
	return projection_dir, scanlog, stack


def _run(state, *stages):
	if state.dataset is None:
		state.dataset = resolve_streaming_plan(stages, state.scan)
		state.source_window = next(iter(state.dataset))
	with PipelineRunContext() as context:
		state.storage = context.storage
		result = Pipeline(list(stages), backend=NumpyBackend()).run(state)
		assert isinstance(result.working_array, SharedNPArray)
		return result.with_array(
			LocalArray(np.asarray(result.working_array).copy(), result.working_array.shape),
		)


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
	assert stage.contract.produces is SinoOrder
	np.testing.assert_array_equal(result.working_array, np.transpose(stack, (1, 0, 2)))
	assert result.dataset.physical_shape == ProjOrder(Theta=count, Y=size, X=size)
	assert result.source_window.pos == ProjOrder(Theta=0, Y=0, X=0)
	assert result.layout is SinoOrder


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
	planned = _read_spec(create_stage(config), result.scan)
	read_window = result.dataset.full_window().with_pad(["Y"])
	assert planned.shape == read_window.frange.length.to(ProjOrder)
	assert planned.dtype == read_window.dtype
	np.testing.assert_allclose(
		result.dataset.angles[result.source_window.axis_slice("Theta")],
		np.deg2rad([30, 90, 150]),
	)
	assert result.source_window.shape.Theta == 5
	assert result.working_array.shape[0] == 3
	assert result.source_window.step == ProjOrder(Theta=2, Y=1, X=1)
	assert result.dataset.pos == ProjOrder(Theta=1, Y=0, X=0)
	assert result.dataset.shape == ProjOrder(Theta=5, Y=size, X=size)
	assert result.dataset.sample_step == result.source_window.step


def test_irregular_indices_and_angles_share_theta_selection_and_stride(tmp_path):
	projection_dir, scanlog, stack = _write_irregular_fixture(tmp_path)
	config = StageConfig(
		kind="read",
		algorithm="tiff_stack",
		params={"theta_start": 1, "theta_stop": 5, "theta_stride": 2},
	)

	result = _run(
		_irregular_state(projection_dir, scanlog, 4, [config]),
		create_stage(config),
	)

	np.testing.assert_array_equal(result.working_array, stack[1:5:2])
	np.testing.assert_array_equal(result.dataset.acquisition_indices, [0, 1, 3, 4, 5])
	np.testing.assert_allclose(
		np.rad2deg(result.dataset.angles), [0, 30, 90, 120, 150], atol=2e-5,
	)
	np.testing.assert_allclose(
		np.rad2deg(result.dataset.angles[result.source_window.axis_slice("Theta")]),
		[30, 120], atol=2e-5,
	)


def test_irregular_angles_reach_center_and_reconstruction(tmp_path, monkeypatch):
	projection_dir, scanlog, _stack = _write_irregular_fixture(tmp_path)
	read_config = StageConfig(
		kind="read", algorithm="tiff_stack", params={"order": "sinogram"},
	)
	center_config = StageConfig(
		kind="center", algorithm="entropy_local",
		params={"start": 2, "range": 0, "max_expansions": 0},
	)
	recon_config = StageConfig(
		kind="recon", algorithm="gridrec",
		params={"circle_mask": False},
	)
	seen_angles = []

	def recon(data, angles, **kwargs):
		seen_angles.append(np.asarray(angles).copy())
		target = kwargs.get("init_recon")
		if target is None:
			return np.ones((data.shape[0], data.shape[2], data.shape[2]), dtype=np.float32)
		target.fill(1)
		return target

	monkeypatch.setattr("tomopy.recon", recon)
	monkeypatch.setattr("tomopy.circ_mask", lambda data, **_kwargs: data)
	with PipelineRunContext() as context:
		state = _irregular_state(
			projection_dir, scanlog, 4, [read_config, center_config, recon_config],
		)
		state.dataset = resolve_streaming_plan(
			[create_stage(read_config)], state.scan,
		)
		state.source_window = next(iter(state.dataset))
		state.storage = context.storage
		state = Pipeline([create_stage(read_config)], backend=NumpyBackend()).run(state)
		state.domain = DataDomain.ATTENUATION
		result = Pipeline([
			create_stage(center_config), create_stage(recon_config),
		], backend=NumpyBackend()).run(state)

	expected = np.deg2rad([0, 30, 90, 120, 150])
	assert len(seen_angles) == 2
	for angles in seen_angles:
		np.testing.assert_allclose(angles, expected)
	shape = result.working_array.shape
	assert (shape.Y, shape.Z, shape.X) == (4, 4, 4)


def test_irregular_source_count_mismatch_is_still_rejected(tmp_path):
	projection_dir, scanlog, _stack = _write_irregular_fixture(tmp_path)
	config = StageConfig(kind="read", algorithm="tiff_stack")
	state = _irregular_state(projection_dir, scanlog, 4, [config])
	state.scan.n_projections = 4

	with pytest.raises(ValueError, match="scan declares 4.*source contains 5"):
		_run(state, create_stage(config))


def test_legacy_tiff_chunk_parameters_are_not_runtime_stage_state():
	config = StageConfig(kind="read", algorithm="tiff_stack", params={
		"sinogram_chunk_size": 140,
		"chunk_overlap": 30,
	})
	stage = create_stage(config)
	assert not hasattr(stage, "sinogram_chunk_size")
	assert not hasattr(stage, "chunk_overlap")
