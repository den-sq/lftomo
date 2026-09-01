import sys
from types import SimpleNamespace

import h5py
import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import BufferSpec, Pipeline, PipelineState
from lftomo.pipeline.stages import HDF5VolumeWriteStage, NeuroglancerWriteStage, STAGE_REGISTRY, create_stage
from lftomo.util.shapes import ProjOrder, ReconOrder, Y
from lftomo.util.window import CTFov, TomoDataSet, TomoWindow


def _state(array, config, output):
	return PipelineState(
		working_array=array,
		buffer_spec=BufferSpec(ReconOrder(*array.shape), array.dtype),
		scan=Scan(
			energy_keV=20,
			pixel_size_um=0.7,
			n_projections=3,
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=0.7),
			acquisition=Acquisition(n_projections=3, rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config], output_path=output),
	)


def _streaming_state(array, config, output, *, start, total_rows):
	state = _state(array, config, output)
	physical = ProjOrder(Theta=1, Y=total_rows, X=array.shape[2])
	state.dataset = TomoDataSet(
		CTFov(1, 180, 1), physical, Y.drange(0, total_rows, array.shape[0]),
		dtype=array.dtype, angles=np.array([0], dtype=np.float32),
	)
	state.source_window = TomoWindow.from_bounds(
		physical, y=(start, start + array.shape[0]), dtype=array.dtype,
	)
	return state


def test_hdf5_writer_writes_volume_and_records_output(tmp_path):
	config = StageConfig(
		kind="write",
		algorithm="hdf5_volume",
		params={"dataset": "exchange/recon", "compression": None},
	)
	working = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
	stage = create_stage(config)
	result = Pipeline([stage]).run(_state(working, config, tmp_path))
	path = tmp_path / "reconstruction.h5"

	assert STAGE_REGISTRY["write.hdf5_volume"] is HDF5VolumeWriteStage
	with h5py.File(path) as handle:
		np.testing.assert_array_equal(handle["exchange/recon"], working)
	assert result.artifacts["output_files"] == [path]


def test_hdf5_writer_places_streaming_interiors_at_global_output_rows(tmp_path):
	config = StageConfig(
		kind="write", algorithm="hdf5_volume",
		params={"dataset": "recon", "compression": None},
	)
	stage = create_stage(config)
	first = _streaming_state(
		np.full((2, 3, 4), 1, np.float32), config, tmp_path, start=0, total_rows=4,
	)
	second = _streaming_state(
		np.full((2, 3, 4), 2, np.float32), config, tmp_path, start=2, total_rows=4,
	)
	stage.run(first)
	stage.run(second)
	with h5py.File(tmp_path / "reconstruction.h5") as handle:
		np.testing.assert_array_equal(handle["recon"][:2], 1)
		np.testing.assert_array_equal(handle["recon"][2:], 2)


def test_neuroglancer_writer_transposes_blocks_for_cloudvolume(tmp_path, monkeypatch):
	calls = {}

	class FakeCloudVolume:
		@staticmethod
		def create_new_info(**kwargs):
			calls["info"] = kwargs
			return kwargs

		def __init__(self, cloudpath, **kwargs):
			calls["cloudpath"] = cloudpath

		def commit_info(self):
			calls["committed"] = True

		def __setitem__(self, key, value):
			calls.setdefault("writes", []).append((key, value.copy()))

	monkeypatch.setitem(sys.modules, "cloudvolume", SimpleNamespace(CloudVolume=FakeCloudVolume))
	config = StageConfig(
		kind="write",
		algorithm="neuroglancer",
		params={"layer_name": "ct", "chunk_xyz": [2, 2, 2], "block_slices": 1},
	)
	working = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
	stage = create_stage(config)
	result = Pipeline([stage]).run(_state(working, config, tmp_path))

	assert STAGE_REGISTRY["write.neuroglancer"] is NeuroglancerWriteStage
	assert calls["info"]["volume_size"] == [4, 3, 2]
	assert calls["info"]["resolution"] == [700, 700, 700]
	assert calls["writes"][0][1].shape == (4, 3, 1)
	np.testing.assert_array_equal(
		np.concatenate([value for _key, value in calls["writes"]], axis=2),
		np.transpose(working, (2, 1, 0)),
	)
	assert result.artifacts["neuroglancer_source"].endswith("/ct")


def test_neuroglancer_writer_offsets_streaming_blocks(tmp_path, monkeypatch):
	calls = {"writes": []}

	class FakeCloudVolume:
		@staticmethod
		def create_new_info(**kwargs):
			calls["info"] = kwargs
			return kwargs

		def __init__(self, cloudpath, **kwargs):
			calls.setdefault("constructors", []).append(kwargs)

		def commit_info(self):
			pass

		def __setitem__(self, key, value):
			calls["writes"].append((key, value.copy()))

	monkeypatch.setitem(sys.modules, "cloudvolume", SimpleNamespace(CloudVolume=FakeCloudVolume))
	config = StageConfig(
		kind="write", algorithm="neuroglancer",
		params={"layer_name": "ct", "block_slices": 2},
	)
	stage = create_stage(config)
	for start, value in ((0, 1), (2, 2)):
		state = _streaming_state(
			np.full((2, 3, 4), value, np.uint16), config, tmp_path,
			start=start, total_rows=4,
		)
		stage.run(state)
	assert calls["info"]["volume_size"] == [4, 3, 4]
	assert calls["writes"][0][0][2] == slice(0, 2)
	assert calls["writes"][1][0][2] == slice(2, 4)
