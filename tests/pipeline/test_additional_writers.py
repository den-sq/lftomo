import sys
from types import SimpleNamespace

import h5py
import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import Pipeline, PipelineState
from lftomo.pipeline.stages import HDF5VolumeWriteStage, NeuroglancerWriteStage, STAGE_REGISTRY, create_stage


def _state(array, config, output):
	return PipelineState(
		working_array=array,
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
	assert result.artifacts["neuroglancer_source"].endswith("/ct")
