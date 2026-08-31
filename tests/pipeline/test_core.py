import sys

import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import BufferSpec, DaskArrayStub, DaskStubBackend, NumpyBackend, Pipeline, PipelineState, Stage, StageContract
from lftomo.util.shapes import ProjOrder


def scan_config():
	return Scan(
		energy_keV=24,
		pixel_size_um=1.3,
		n_projections=3,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1.3),
		acquisition=Acquisition(n_projections=3, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


def recon_config():
	return Recon(pipeline=[StageConfig(kind="echo")])


class EchoStage(Stage):
	name = "echo"
	contract = StageContract(expected_input=ProjOrder, produced_output=ProjOrder, produces_artifacts=("echo",))

	def run(self, state):
		state.artifacts["echo"] = True
		return state


def test_numpy_backend_runs_echo_pipeline():
	state = PipelineState(
		working_array=np.zeros((3, 4, 5), dtype=np.float32),
		buffer_spec=BufferSpec(ProjOrder(3, 4, 5), np.dtype(np.float32)),
		scan=scan_config(),
		recon=recon_config(),
	)

	result = Pipeline([EchoStage()], backend=NumpyBackend()).run(state)

	assert result.artifacts["echo"] is True
	assert result.backend.name == "numpy"


def test_dask_stub_backend_wraps_without_distributed():
	array = np.zeros((3, 4, 5), dtype=np.float32)
	backend = DaskStubBackend()
	wrapped = backend.wrap_array(array)
	state = PipelineState(
		working_array=wrapped, scan=scan_config(), recon=recon_config(), backend=backend,
		buffer_spec=BufferSpec(ProjOrder(*array.shape), array.dtype),
	)

	result = Pipeline([EchoStage()], backend=backend).run(state)

	assert result.artifacts["echo"] is True
	assert result.backend.name == "dask-stub"
	assert "dask.distributed" not in sys.modules
	assert hasattr(wrapped, "shape") or isinstance(wrapped, DaskArrayStub)
