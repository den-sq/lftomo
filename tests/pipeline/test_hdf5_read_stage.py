import h5py
import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, HDF5DxchangeProjections, Recon, Scan, StageConfig
from lftomo.pipeline import NumpyBackend, Pipeline, PipelineState
from lftomo.pipeline.stages import STAGE_REGISTRY, create_stage
from lftomo.util.shapes import SinoOrder


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


def _write(path, data, calibration=True):
	with h5py.File(path, "w") as handle:
		handle.create_dataset("exchange/data", data=data)
		if calibration:
			handle.create_dataset("exchange/data_white", data=np.full((2, *data.shape[1:]), 8))
			handle.create_dataset("exchange/data_dark", data=np.full((2, *data.shape[1:]), 1))


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
	result = Pipeline([stage], backend=NumpyBackend()).run(_state(path, count, size, config))

	assert "read.hdf5_dxchange" in STAGE_REGISTRY
	assert stage.contract.produced_output is SinoOrder
	np.testing.assert_array_equal(result.working_array, np.transpose(data[1:3, 2:5, :], (1, 0, 2)))
	assert result.artifacts["white_fields"].shape == (2, 3, size)
	assert result.artifacts["dark_fields"].shape == (2, 3, size)


def test_hdf5_stage_allows_absent_optional_calibration(tmp_path):
	data = np.ones((2, 3, 3), dtype=np.uint16)
	path = tmp_path / "scan.h5"
	_write(path, data, calibration=False)
	config = StageConfig(kind="read", algorithm="hdf5_dxchange")

	result = Pipeline([create_stage(config)]).run(_state(path, 2, 3, config))

	assert "white_fields" not in result.artifacts
	assert "dark_fields" not in result.artifacts


def test_hdf5_stage_reports_missing_projection_dataset(tmp_path):
	path = tmp_path / "bad.h5"
	with h5py.File(path, "w"):
		pass
	config = StageConfig(kind="read", algorithm="hdf5_dxchange")

	with pytest.raises(ValueError, match="exchange/data.*missing"):
		Pipeline([create_stage(config)]).run(_state(path, 2, 3, config))


def test_hdf5_stage_can_require_calibration(tmp_path):
	data = np.ones((2, 3, 3), dtype=np.uint16)
	path = tmp_path / "scan.h5"
	_write(path, data, calibration=False)
	config = StageConfig(
		kind="read", algorithm="hdf5_dxchange", params={"calibration_policy": "required"}
	)

	with pytest.raises(ValueError, match="data_white.*data_dark"):
		Pipeline([create_stage(config)]).run(_state(path, 2, 3, config))
