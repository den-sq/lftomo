import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import Backend, BufferSpec, Pipeline, PipelineState
from lftomo.pipeline.stages import AstraCudaStage, STAGE_REGISTRY, create_stage
from lftomo.util.shapes import ProjOrder, SinoOrder


ALGORITHMS = {
	"fp_cuda": ("FP_CUDA", False),
	"bp_cuda": ("BP_CUDA", False),
	"fbp_cuda": ("FBP_CUDA", False),
	"sirt_cuda": ("SIRT_CUDA", True),
	"sart_cuda": ("SART_CUDA", True),
	"cgls_cuda": ("CGLS_CUDA", True),
	"em_cuda": ("EM_CUDA", True),
}


def _state(array, config, backend=None, layout=ProjOrder):
	return PipelineState(
		working_array=array,
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1,
			n_projections=3,
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(n_projections=3, rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
		backend=backend or Backend("cuda", supports_gpu=True),
		buffer_spec=BufferSpec(layout(*array.shape), array.dtype),
	)


@pytest.mark.parametrize("algorithm,expected", ALGORITHMS.items())
def test_astra_cuda_algorithms_are_registered(algorithm, expected):
	config = StageConfig(kind="recon", algorithm=algorithm, params={"center": 4})
	stage = create_stage(config)

	assert isinstance(stage, AstraCudaStage)
	assert STAGE_REGISTRY[f"recon.{algorithm}"] is type(stage)
	assert stage.method == expected[0]
	assert stage.iterative is expected[1]
	assert stage.contract.resources.requires_gpu is True


def test_astra_stage_passes_bridge_options_and_iterations(monkeypatch):
	config = StageConfig(
		kind="recon",
		algorithm="sirt_cuda",
		params={"center": 4, "iterations": 7, "circle_mask": False},
	)
	working = np.ones((2, 3, 8), dtype=np.float32)
	calls = {}

	def recon(data, angles, **kwargs):
		calls.update(kwargs)
		return np.ones((2, 8, 8), dtype=np.float32)

	monkeypatch.setattr("tomopy.recon", recon)
	backend = Backend("cuda", supports_gpu=True)
	result = Pipeline([create_stage(config)], backend=backend).run(
		_state(working, config, backend, SinoOrder)
	)

	assert calls["algorithm"].__name__ == "astra"
	assert calls["ncore"] == 1
	assert calls["options"] == {
		"proj_type": "cuda",
		"method": "SIRT_CUDA",
		"extra_options": {},
		"num_iter": 7,
	}
	assert result.working_array.shape == (2, 8, 8)


def test_astra_stage_rejects_cpu_backend_before_execution():
	config = StageConfig(kind="recon", algorithm="fbp_cuda", params={"center": 4})
	state = _state(np.ones((3, 2, 8), dtype=np.float32), config, Backend("numpy"))

	with pytest.raises(Exception, match="requires GPU"):
		Pipeline([create_stage(config)]).run(state)
