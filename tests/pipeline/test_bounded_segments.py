import threading
import time

import numpy as np

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	DataDomain,
	ExecutionKind,
	Locality,
	PRESERVE,
	Pipeline,
	PipelineState,
	Stage,
	StageContract,
)
from lftomo.pipeline.stages import VoDeadStripeStage, VoSortStripeStage
from lftomo.util.shapes import ProjOrder, SinoOrder


def scan_config(theta=12):
	return Scan(
		energy_keV=20, pixel_size_um=1, n_projections=theta, degrees_per_fov=180,
		detector=Detector(pixel_size_um=1),
		acquisition=Acquisition(n_projections=theta, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


def state(array, layout):
	return PipelineState(
		working_array=array, scan=scan_config(array.shape[0] if layout is ProjOrder else array.shape[1]),
		recon=Recon(pipeline=[StageConfig(kind="test")]),
		layout=layout, domain=DataDomain.ATTENUATION,
	)


class AddSinogramStage(Stage):
	def __init__(self, name, amount, activity=None):
		self.name = name
		self.amount = amount
		self.activity = activity
		self.contract = StageContract(
			accepts=(SinoOrder,), prefers=SinoOrder, produces=PRESERVE,
			accepts_domain=(DataDomain.ATTENUATION,), produces_domain=PRESERVE,
			locality=Locality.SINOGRAM, bounded_views_from=(ProjOrder,),
			mutates_input=True,
		)

	def run_sinogram(self, sinogram, _state):
		if self.activity is not None:
			with self.activity["lock"]:
				self.activity["current"] += 1
				self.activity["peak"] = max(self.activity["peak"], self.activity["current"])
			time.sleep(0.005)
		sinogram += self.amount
		if self.activity is not None:
			with self.activity["lock"]:
				self.activity["current"] -= 1

	def run(self, state):
		raise AssertionError("bounded segment should call run_sinogram, not Stage.run")


class BarrierStage(Stage):
	name = "barrier"
	contract = StageContract(
		accepts=(ProjOrder,), produces=PRESERVE,
		accepts_domain=(DataDomain.ATTENUATION,), produces_domain=PRESERVE,
		locality=Locality.PROJECTION,
	)

	def run(self, state):
		return state


def pipeline(stages, workers=1):
	return Pipeline(
		stages, scan=scan_config(), initial_layout=ProjOrder,
		initial_domain=DataDomain.ATTENUATION, bounded_workers=workers,
	)


def test_planner_groups_adjacent_bounded_stages_and_stops_at_barrier():
	stages = [
		AddSinogramStage("first", 1),
		AddSinogramStage("second", 2),
		BarrierStage(),
		AddSinogramStage("third", 3),
	]
	plan = pipeline(stages).plan

	assert [(item.start, item.stop, item.kind) for item in plan.segments] == [
		(0, 2, ExecutionKind.BOUNDED_SINOGRAM),
		(2, 3, ExecutionKind.SEQUENTIAL),
		(3, 4, ExecutionKind.BOUNDED_SINOGRAM),
	]


def test_segment_gathers_once_per_row_for_whole_chain(monkeypatch):
	import lftomo.pipeline.execution as execution

	array = np.zeros((12, 4, 8), dtype=np.float32)
	calls = []
	original = execution.sinogram_scratch
	monkeypatch.setattr(execution, "sinogram_scratch", lambda data, layout, index:
		calls.append(index) or original(data, layout, index))

	result = pipeline([
		AddSinogramStage("first", 1), AddSinogramStage("second", 2),
	]).run(state(array, ProjOrder))

	assert calls == [0, 1, 2, 3]
	np.testing.assert_array_equal(result.working_array, 3)


def test_ndarray_only_bounded_execution_is_serial_and_hermetic():
	activity = {"lock": threading.Lock(), "current": 0, "peak": 0}
	array = np.zeros((12, 6, 8), dtype=np.float32)

	pipeline([AddSinogramStage("add", 1, activity)], workers=2).run(state(array, ProjOrder))

	assert activity["peak"] == 1


def test_real_vo_chain_matches_contiguous_sinogram_execution():
	theta = np.linspace(0.8, 1.2, 24, dtype=np.float32)
	x = np.linspace(0.9, 1.1, 24, dtype=np.float32)
	sino = np.repeat((theta[None, :, None] * x[None, None, :]), 3, axis=0)
	sino[:, :, 11] *= 4
	projection = sino.transpose(1, 0, 2).copy()
	stages_sino = [
		VoDeadStripeStage(snr=1, size=5, norm=False),
		VoSortStripeStage(size=3),
	]
	stages_proj = [
		VoDeadStripeStage(snr=1, size=5, norm=False),
		VoSortStripeStage(size=3),
	]

	Pipeline(
		stages_sino, scan=scan_config(24), initial_layout=SinoOrder,
		initial_domain=DataDomain.ATTENUATION,
	).run(state(sino, SinoOrder))
	proj_pipeline = Pipeline(
		stages_proj, scan=scan_config(24), initial_layout=ProjOrder,
		initial_domain=DataDomain.ATTENUATION,
	)
	proj_pipeline.run(state(projection, ProjOrder))

	assert proj_pipeline.plan.segments[0].kind is ExecutionKind.BOUNDED_SINOGRAM
	np.testing.assert_allclose(projection, sino.transpose(1, 0, 2), rtol=1e-6, atol=1e-6)
