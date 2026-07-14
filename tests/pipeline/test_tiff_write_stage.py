import numpy as np
import pytest
import tifffile as tf

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import NumpyBackend, Pipeline, PipelineState
from lftomo.pipeline.stages import STAGE_REGISTRY, create_stage
from lftomo.util.io import np_convert


def _state(array, output, config, sample_id="sample-a"):
	return PipelineState(
		working_array=array,
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1.0,
			n_projections=3,
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1.0),
			acquisition=Acquisition(n_projections=3, rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config], output_path=output, metadata={"sample_id": sample_id}),
	)


def test_registered_tiff_write_stage_round_trips_float_slices(tmp_path):
	array = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) / 10
	config = StageConfig(kind="write", algorithm="tiff_slices")

	result = Pipeline([create_stage(config)], backend=NumpyBackend()).run(
		_state(array, tmp_path, config)
	)

	assert "write.tiff_slices" in STAGE_REGISTRY
	assert [path.name for path in result.artifacts["output_files"]] == [
		"sample-a_00000.tif", "sample-a_00001.tif", "sample-a_00002.tif"
	]
	for index, path in enumerate(result.artifacts["output_files"]):
		np.testing.assert_allclose(tf.imread(path), array[index])


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
def test_tiff_write_stage_supports_integer_template_and_multiprocessing(tmp_path, dtype):
	array = np.linspace(0, 1, 2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
	config = StageConfig(kind="write", algorithm="tiff_slices", params={
		"dtype": dtype,
		"naming_template": "slice-{slice:03d}.tiff",
		"processes": 2,
		"slice_start": 7,
	})

	result = Pipeline([create_stage(config)]).run(_state(array, tmp_path, config))

	assert [path.name for path in result.artifacts["output_files"]] == [
		"slice-007.tiff", "slice-008.tiff"
	]
	for index, path in enumerate(result.artifacts["output_files"]):
		written = tf.imread(path)
		assert written.dtype == np.dtype(dtype)
		np.testing.assert_array_equal(written, np_convert(np.dtype(dtype), array[index]))
