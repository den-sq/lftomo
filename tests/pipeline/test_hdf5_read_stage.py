import h5py
import numpy as np
import pytest

from lftomo.config import (
	Acquisition,
	Detector,
	HDF5AngleSource,
	HDF5DxchangeProjections,
	Recon,
	Scan,
	StageConfig,
)
from lftomo.pipeline import NumpyBackend, Pipeline, PipelineRunContext, PipelineState, working_view
from lftomo.pipeline.stages import STAGE_REGISTRY, create_stage
from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import ProjOrder, SinoOrder


def _state(path, count, size, config):
	return PipelineState(
		working_array=np.empty(0),
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1.0,
			n_projections=count,
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1.0, rows=size, columns=size),
			acquisition=Acquisition(n_projections=count, rotation_end_deg=180),
			projections=HDF5DxchangeProjections(format="hdf5_dxchange", path=path),
		),
		recon=Recon(pipeline=[config]),
	)


def _write(path, data, calibration=True, **dataset_kwargs):
	with h5py.File(path, "w") as handle:
		handle.create_dataset("exchange/data", data=data, **dataset_kwargs)
		if calibration:
			handle.create_dataset("exchange/data_white", data=np.full((2, *data.shape[1:]), 8))
			handle.create_dataset("exchange/data_dark", data=np.full((2, *data.shape[1:]), 1))


def _run(state, *stages):
	with PipelineRunContext() as context:
		state.storage = context.storage
		result = Pipeline(list(stages), backend=NumpyBackend()).run(state)
		assert isinstance(result.working_array, SharedNPArray)
		return result.with_array(working_view(result).copy())


def test_registered_hdf5_stage_reads_bounded_sinogram_and_calibration(tmp_path):
	count, size = 4, 6
	data = np.arange(count * size * size, dtype=np.uint16).reshape(count, size, size)
	path = tmp_path / "scan.h5"
	_write(path, data)
	config = StageConfig(kind="read", algorithm="hdf5_dxchange", params={
		"order": "sinogram", "theta_start": 1, "theta_stop": 3,
		"y_start": 2, "y_stop": 5,
	})

	stage = create_stage(config)
	result = _run(_state(path, count, size, config), stage)

	assert "read.hdf5_dxchange" in STAGE_REGISTRY
	assert stage.contract.produced_output is SinoOrder
	np.testing.assert_array_equal(result.working_array, np.transpose(data[1:3, 2:5, :], (1, 0, 2)))
	assert result.artifacts["white_fields"].shape == (2, 3, size)
	assert result.artifacts["dark_fields"].shape == (2, 3, size)
	assert isinstance(result.artifacts["white_fields"], SharedNPArray)
	assert isinstance(result.artifacts["dark_fields"], SharedNPArray)
	assert result.artifacts["hdf5_read_strategy"] == "bounded_scratch"
	assert result.dataset.shape == ProjOrder(Theta=count, Y=size, X=size)
	assert result.source_window.pos == ProjOrder(Theta=1, Y=2, X=0)
	assert result.source_window.shape == ProjOrder(Theta=2, Y=3, X=size)
	assert result.buffer_spec.layout is SinoOrder


def test_hdf5_stage_allows_absent_optional_calibration(tmp_path):
	data = np.ones((2, 3, 3), dtype=np.uint16)
	path = tmp_path / "scan.h5"
	_write(path, data, calibration=False)
	config = StageConfig(kind="read", algorithm="hdf5_dxchange")

	result = _run(_state(path, 2, 3, config), create_stage(config))

	assert "white_fields" not in result.artifacts
	assert "dark_fields" not in result.artifacts
	assert result.artifacts["hdf5_read_strategy"] == "direct"


def test_hdf5_stage_reports_missing_projection_dataset(tmp_path):
	path = tmp_path / "bad.h5"
	with h5py.File(path, "w"):
		pass
	config = StageConfig(kind="read", algorithm="hdf5_dxchange")

	with pytest.raises(ValueError, match="exchange/data.*missing"):
		_run(_state(path, 2, 3, config), create_stage(config))


def test_hdf5_stage_can_require_calibration(tmp_path):
	data = np.ones((2, 3, 3), dtype=np.uint16)
	path = tmp_path / "scan.h5"
	_write(path, data, calibration=False)
	config = StageConfig(
		kind="read", algorithm="hdf5_dxchange", params={"calibration_policy": "required"}
	)

	with pytest.raises(ValueError, match="data_white.*data_dark"):
		_run(_state(path, 2, 3, config), create_stage(config))


def test_hdf5_stage_reads_strided_theta_window_with_matching_angles(tmp_path):
	count, size = 6, 4
	data = np.arange(count * size * size, dtype=np.uint16).reshape(count, size, size)
	path = tmp_path / "scan.h5"
	_write(path, data)
	config = StageConfig(
		kind="read", algorithm="hdf5_dxchange",
		params={"theta_start": 1, "theta_stop": 6, "theta_stride": 2},
	)

	result = _run(_state(path, count, size, config), create_stage(config))

	np.testing.assert_array_equal(result.working_array, data[1:6:2])
	np.testing.assert_allclose(result.artifacts["angles"], np.deg2rad([30, 90, 150]))
	assert result.source_window.shape.Theta == 5
	assert result.working_array.shape[0] == 3
	assert result.source_window.step == ProjOrder(Theta=2, Y=1, X=1)


def test_hdf5_stage_imports_explicit_angles_and_derives_indices(tmp_path):
	count, size = 4, 3
	data = np.ones((count, size, size), dtype=np.uint16)
	path = tmp_path / "scan.h5"
	_write(path, data)
	with h5py.File(path, "a") as handle:
		handle.create_dataset("exchange/theta", data=[0, 36, 108, 144])
	config = StageConfig(kind="read", algorithm="hdf5_dxchange")
	state = _state(path, count, size, config)
	state.scan.acquisition = Acquisition(
		n_projections=5,
		rotation_end_deg=180,
		angle_source=HDF5AngleSource(),
	)

	result = _run(state, create_stage(config))

	np.testing.assert_array_equal(result.artifacts["acquisition_indices"], [0, 1, 3, 4])
	np.testing.assert_allclose(np.rad2deg(result.artifacts["angles"]), [0, 36, 108, 144])


def test_hdf5_stage_uses_bounded_fallback_for_chunked_compressed_source(tmp_path):
	count, size = 5, 6
	data = np.arange(count * size * size, dtype=np.uint16).reshape(count, size, size)
	path = tmp_path / "compressed.h5"
	_write(path, data, chunks=(2, 3, size), compression="gzip")
	config = StageConfig(kind="read", algorithm="hdf5_dxchange")

	result = _run(_state(path, count, size, config), create_stage(config))

	np.testing.assert_array_equal(result.working_array, data)
	assert result.artifacts["hdf5_read_strategy"] == "bounded_scratch"
