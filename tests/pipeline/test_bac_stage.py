import numpy as np
import builtins
import pytest
import sys
from types import ModuleType

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import DataDomain, PRESERVE, Pipeline, PipelineState
from lftomo.pipeline.stages import BronnikovAidedCorrectionStage, STAGE_REGISTRY, create_stage
from lftomo.util.shapes import ProjOrder, SinoOrder


def _state(array, config, layout=SinoOrder):
	return PipelineState(
		working_array=array,
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
		layout=layout,
	)


@pytest.fixture
def hotopy_holo(monkeypatch):
	"""Provide only the optional module boundary exercised by the BAC Stage."""

	hotopy = ModuleType("hotopy")
	holo = ModuleType("hotopy.holo")
	hotopy.holo = holo
	monkeypatch.setitem(sys.modules, "hotopy", hotopy)
	monkeypatch.setitem(sys.modules, "hotopy.holo", holo)
	return holo


def test_bac_aliases_are_registered_and_process_each_projection(monkeypatch, hotopy_holo):
	config = StageConfig(
		kind="phase",
		algorithm="bronnikov_aided",
		params={"alpha": 2, "gamma": 1.25, "invert": True, "processes": 2},
	)
	working = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
	before = working.copy()
	calls = []

	class FakeBAC:
		def __init__(self, shape, alpha, gamma, device):
			calls.append((shape, alpha, gamma, device))

		def __call__(self, image):
			return image + 2

	monkeypatch.setattr(hotopy_holo, "BronnikovAidedCorrection", FakeBAC, raising=False)
	stage = create_stage(config)
	result = Pipeline([stage]).run(_state(working, config))

	assert STAGE_REGISTRY["phase.bac"] is BronnikovAidedCorrectionStage
	assert STAGE_REGISTRY["phase.bronnikov_aided"] is BronnikovAidedCorrectionStage
	assert len(calls) == working.shape[1]
	assert all(call[:3] == ((3, 5), 2, 1.25) for call in calls)
	assert result.working_array is working
	np.testing.assert_array_equal(working, -(before + 2))


def test_disabled_bac_is_noop(monkeypatch, hotopy_holo):
	config = StageConfig(
		kind="phase",
		algorithm="bac",
		params={"alpha": 2, "enabled": False},
	)
	working = np.ones((2, 3, 4), dtype=np.float32)
	monkeypatch.setattr(
		hotopy_holo,
		"BronnikovAidedCorrection",
		lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not construct")),
		raising=False,
	)

	stage = create_stage(config)
	result = Pipeline([stage]).run(_state(working, config))

	assert result.working_array is working
	np.testing.assert_array_equal(working, 1)
	assert stage.contract.accepts_domain == (
		DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
	)
	assert stage.contract.produces_domain is PRESERVE
	assert stage.contract.mutates_input is False


def test_bac_projection_order_processes_contiguous_projection_images(monkeypatch, hotopy_holo):
	config = StageConfig(
		kind="phase", algorithm="bac",
		params={"alpha": 2, "gamma": 1.25},
	)
	working = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
	before = working.copy()
	shapes = []

	class FakeBAC:
		def __init__(self, shape, **_kwargs):
			shapes.append(shape)

		def __call__(self, image):
			assert image.flags.c_contiguous
			return image + 1

	monkeypatch.setattr(hotopy_holo, "BronnikovAidedCorrection", FakeBAC, raising=False)
	state = _state(working, config, ProjOrder)
	state.scan.n_projections = working.shape[0]
	stage = create_stage(config)

	result = stage.run(state)

	assert ProjOrder in stage.contract.accepted_layouts
	assert shapes == [(3, 5)] * 4
	np.testing.assert_array_equal(result.working_array, before + 1)


def test_bac_reports_missing_optional_dependency(monkeypatch):
	config = StageConfig(kind="phase", algorithm="bac", params={"alpha": 2})
	stage = create_stage(config)
	original_import = builtins.__import__

	def without_hotopy(name, *args, **kwargs):
		if name == "hotopy.holo":
			raise ImportError("not installed")
		return original_import(name, *args, **kwargs)

	monkeypatch.setattr(builtins, "__import__", without_hotopy)
	with pytest.raises(RuntimeError, match="'bac' extra"):
		stage.run(_state(np.ones((1, 1, 2), dtype=np.float32), config))
