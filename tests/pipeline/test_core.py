import sys

import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import BufferSpec, DaskArrayStub, DaskStubBackend, NumpyBackend, Pipeline, PipelineState, Stage, StageContract
from lftomo.util.shapes import ProjOrder, SinoOrder, Y


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
	contract = StageContract(
		accepts=(ProjOrder,), produces=ProjOrder, produces_artifacts=("echo",),
	)

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


def test_work_assignment_addresses_only_its_rows_and_survives_layout_change():
	array = np.zeros((3, 5, 4), dtype=np.float32)
	state = PipelineState(
		working_array=array, scan=scan_config(), recon=recon_config(),
		buffer_spec=BufferSpec(ProjOrder(*array.shape), array.dtype),
		work_range=Y.drange(1, 5, 2),
	)

	state.working_array[state.work_index()] += 1
	assert list(state.work_indices("Y")) == [1, 3]
	assert list(state.work_indices("Theta")) == [0, 1, 2]
	assert state.work_index(SinoOrder) == np.s_[1:5:2, :, :]
	np.testing.assert_array_equal(array[:, [1, 3]], 1)
	np.testing.assert_array_equal(array[:, [0, 2, 4]], 0)

	transposed = array.transpose(1, 0, 2)
	result = state.with_array(
		transposed, buffer_spec=BufferSpec(SinoOrder(*transposed.shape), array.dtype),
	)
	result.working_array[result.work_index()] += 2
	np.testing.assert_array_equal(array[:, [1, 3]], 3)
	np.testing.assert_array_equal(array[:, [0, 2, 4]], 0)


def test_unassigned_state_addresses_the_whole_buffer():
	array = np.zeros((3, 5, 4), dtype=np.float32)
	state = PipelineState(
		working_array=array, scan=scan_config(), recon=recon_config(),
		buffer_spec=BufferSpec(ProjOrder(*array.shape), array.dtype),
	)

	assert list(state.work_indices("Y")) == list(range(5))
	assert state.work_index() == np.s_[:, :, :]
	assert state.work_index(SinoOrder) == np.s_[:, :, :]


@pytest.mark.parametrize("assignment", [
	Y.drange(-1, 2, 1), Y.drange(1, 6, 1), Y.drange(0, 2, 0),
	ProjOrder.drange(0, 1, 1),
])
def test_work_assignment_rejects_ranges_that_cannot_safely_index_the_buffer(assignment):
	array = np.zeros((3, 5, 4), dtype=np.float32)
	with pytest.raises(ValueError, match="work_range"):
		PipelineState(
			working_array=array, scan=scan_config(), recon=recon_config(),
			buffer_spec=BufferSpec(ProjOrder(*array.shape), array.dtype), work_range=assignment,
		)
