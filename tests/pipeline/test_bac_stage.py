import numpy as np
import builtins
import pytest
import sys
from types import ModuleType

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import LocalArray, DataDomain, PRESERVE, Pipeline, PipelineState
from lftomo.pipeline.stages import (
	BronnikovAidedCorrectionFilterStage, BronnikovAidedCorrectionStage,
	STAGE_REGISTRY, create_stage,
)
from lftomo.util.shapes import ProjOrder, SinoOrder
from lftomo.pipeline.stages.phase import BACNonFiniteError
from lftomo.storage import PipelineRunContext
from tests.sample_data import attach_source_geometry


def _state(array, config, layout=SinoOrder):
	shape = layout(*array.shape)
	return attach_source_geometry(PipelineState(
		working_array=LocalArray(array, shape),
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1,
			n_projections=shape.Theta,
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(n_projections=shape.Theta, rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
	))


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
			assert np.shares_memory(image, working)
			return image + 2

	monkeypatch.setattr(hotopy_holo, "BronnikovAidedCorrection", FakeBAC, raising=False)
	stage = create_stage(config)
	result = Pipeline([stage]).run(_state(working, config))

	assert STAGE_REGISTRY["phase.bac"] is BronnikovAidedCorrectionStage
	assert STAGE_REGISTRY["phase.bronnikov_aided"] is BronnikovAidedCorrectionStage
	assert stage.contract.resources.cpu_thread_pool == "torch"
	assert BronnikovAidedCorrectionFilterStage(alpha=2).contract.resources.cpu_thread_pool is None
	assert len(calls) == working.shape[1]
	assert all(call[:3] == ((3, 5), 2, 1.25) for call in calls)
	assert np.asarray(result.working_array) is working
	np.testing.assert_array_equal(working, -(before + 2))


def test_disabled_bac_is_noop(monkeypatch, hotopy_holo):
	original_import = builtins.__import__

	def without_torch(name, *args, **kwargs):
		if name == "torch" or name.startswith("torch."):
			raise AssertionError("disabled BAC must not import Torch")
		return original_import(name, *args, **kwargs)

	monkeypatch.setattr(builtins, "__import__", without_torch)
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

	assert np.asarray(result.working_array) is working
	np.testing.assert_array_equal(working, 1)
	assert stage.contract.accepts_domain == (
		DataDomain.TRANSMISSION, DataDomain.ATTENUATION,
	)
	assert stage.contract.produces_domain is PRESERVE
	assert stage.contract.mutates_input is False
	assert stage.contract.resources.cpu_thread_pool is None


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


class _NonFiniteBAC(BronnikovAidedCorrectionFilterStage):
	def _correct(self, image):
		result = image.copy()
		if image[0, 0] == 2:
			result[0, :2] = [np.nan, np.inf]
		return result


@pytest.mark.system
@pytest.mark.parametrize("managed", [False, True])
def test_bac_failure_keeps_all_window_counts_and_source_indices(managed):
	config = StageConfig(kind="phase", algorithm="bac_correction", params={"alpha": 2})
	values = np.repeat(np.arange(1, 4, dtype=np.float32)[:, None, None], 4, axis=2)
	stage = _NonFiniteBAC(alpha=2)
	with PipelineRunContext(max_workers=2) as context:
		state = _state(values.copy(), config, ProjOrder)
		if managed:
			buffer = context.storage.allocate(ProjOrder(*values.shape), np.float32)
			buffer.array()[...] = values
			state = state.with_array(buffer)
			state.storage = context.storage
		state = attach_source_geometry(state, theta_start=1, theta_stride=2, physical_theta=7)
		with pytest.raises(BACNonFiniteError, match="projection 3 .*2 pixels") as caught:
			if managed:
				context.executor.run_stage(state, stage)
			else:
				stage.run(state)
		qc = caught.value.qc
		assert qc is state.artifacts["bac_qc"]
		assert qc["projections"] == [
			{"projection_index": index, "acquisition_index": index, "nonfinite_count": count}
			for index, count in [(1, 0), (3, 2), (5, 0)]
		]
		assert np.isnan(np.asarray(state.working_array)[1, 0, 0])
		assert np.isinf(np.asarray(state.working_array)[1, 0, 1])


def test_bac_success_records_finite_projection_counts(hotopy_holo):
	class FiniteBAC:
		def __init__(self, *args, **kwargs):
			pass

		def __call__(self, image):
			return image + 0.1

	hotopy_holo.BronnikovAidedCorrection = FiniteBAC
	config = StageConfig(kind="phase", algorithm="bac", params={"alpha": 1.6, "gamma": 1.2})
	state = _state(np.ones((3, 2, 4), dtype=np.float32), config, ProjOrder)
	create_stage(config).run(state)
	assert [item["nonfinite_count"] for item in state.artifacts["bac_qc"]["projections"]] == [0, 0, 0]
