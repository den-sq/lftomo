from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import lftomo.pipeline.stages.super_resolve as super_resolve_module
from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import LocalArray, Pipeline, PipelineRunContext, PipelineState
from lftomo.pipeline.stages import DRHTStage, LDMStage, STAGE_REGISTRY, create_stage
from lftomo.pipeline.stages.super_resolve import _DRHT_SCRATCH_NAMES
from lftomo.util.shapes import SinoOrder


def _state(array, config, mask):
	return PipelineState(
		working_array=LocalArray(array, SinoOrder(*array.shape)),
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1,
			n_projections=array.shape[1],
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(n_projections=array.shape[1], rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
		artifacts={"availability_mask": mask},
	)


@pytest.mark.parametrize("algorithm,stage_type", [("drht", DRHTStage), ("ldm", LDMStage)])
def test_super_resolution_stages_are_registered(algorithm, stage_type):
	params = {"allow_bicubic_fallback": True} if algorithm == "drht" else {"allow_interpolation_fallback": True}
	stage = create_stage(StageConfig(kind="super_resolve", algorithm=algorithm, params=params))

	assert isinstance(stage, stage_type)
	assert STAGE_REGISTRY[f"super_resolve.{algorithm}"] is stage_type


def test_drht_explicit_fallback_interpolates_missing_angles():
	config = StageConfig(
		kind="super_resolve",
		algorithm="drht",
		params={"allow_bicubic_fallback": True},
	)
	working = np.zeros((1, 3, 2), dtype=np.float32)
	working[:, 0] = 1
	working[:, 2] = 3
	result = Pipeline([create_stage(config)]).run(_state(working, config, [True, False, True]))

	np.testing.assert_allclose(np.asarray(result.working_array)[:, 1], 2)
	assert result.artifacts["super_resolution"]["fallback"] == "interpolation"


class _FakeTensor:
	def __init__(self, values):
		self.values = np.asarray(values)

	def to(self, **_kwargs):
		return self

	def __getitem__(self, index):
		return _FakeTensor(self.values[index])

	def detach(self):
		return self

	def cpu(self):
		return self

	def numpy(self):
		return self.values


def test_drht_reanchors_patch_grid_and_reuses_managed_scratch(monkeypatch):
	config = StageConfig(
		kind="super_resolve",
		algorithm="drht",
		params={"patch_size": 3, "patch_overlap": 1 / 3},
	)
	working = np.arange(2 * 3 * 6, dtype=np.float32).reshape(2, 3, 6)
	stage = create_stage(config)
	fake_torch = SimpleNamespace(
		cuda=SimpleNamespace(is_available=lambda: False),
		device=lambda value: value,
		float32=np.float32,
		from_numpy=_FakeTensor,
		no_grad=nullcontext,
	)
	monkeypatch.setitem(sys.modules, "torch", fake_torch)
	monkeypatch.setattr(stage, "_issues", lambda: [])
	monkeypatch.setattr(
		stage, "_load_model", lambda _torch, _device: lambda tensor: (tensor,) * 4,
	)
	grids = []
	real_grid = super_resolve_module.ChunkGrid
	monkeypatch.setattr(
		super_resolve_module, "ChunkGrid",
		lambda *args, **kwargs: grids.append(real_grid(*args, **kwargs)) or grids[-1],
	)

	with PipelineRunContext(max_workers=1) as context:
		state = _state(working, config, np.ones(3, dtype=bool))
		state.storage = context.storage
		result = stage.run(state)
		output = np.asarray(result.working_array).copy()
		events = context.storage.allocation_events

	np.testing.assert_allclose(output, working)
	assert list(grids[1]) == [slice(0, 3), slice(2, 5), slice(3, 6)]
	assert all(event["role"] != "super_resolution_assembly" for event in events)
	for name in _DRHT_SCRATCH_NAMES:
		role = f"drht_{name}_scratch"
		assert sum(
			event["action"] == "allocate" and event["role"] == role
			for event in events
		) == 1
		assert any(
			event["action"] == "retire" and event["role"] == role
			for event in events
		)


def test_ldm_requires_restored_external_assets_without_fallback():
	config = StageConfig(kind="super_resolve", algorithm="ldm")
	working = np.ones((1, 3, 2), dtype=np.float32)

	with pytest.raises(RuntimeError, match="LDM repo path"):
		Pipeline([create_stage(config)]).run(_state(working, config, [True, True, True]))
