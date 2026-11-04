import builtins

import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	BufferSpec,
	DataDomain,
	ExecutionKind,
	ExecutionSpanPhase,
	NumpyBackend,
	PRESERVE,
	Pipeline,
	PipelineRunContext,
	PipelineState,
	Stage,
	StageContract,
)
from lftomo.util.shapes import ProjOrder, SinoOrder, X
from lftomo.util.window import TomoWindow


def _scan(theta=6, rows=3, columns=8):
	return Scan(
		energy_keV=20,
		pixel_size_um=1,
		ssd_cm=10,
		n_projections=theta,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1, rows=rows, columns=columns),
		acquisition=Acquisition(n_projections=theta, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


def _recon():
	return Recon(pipeline=[StageConfig(kind="test")], process_workers=2)


def _state(array, *, storage=None, executor=None):
	shape = np.asarray(array).shape
	return PipelineState(
		array,
		_scan(theta=shape[0], rows=shape[1], columns=shape[2]),
		_recon(),
		backend=NumpyBackend(),
		storage=storage,
		executor=executor,
		buffer_spec=BufferSpec(ProjOrder(*shape), np.asarray(array).dtype),
		domain=DataDomain.ATTENUATION,
	)


class _PassStage(Stage):
	name = "test.pass"
	contract = StageContract()

	def run(self, state):
		return state


class _FailStage(_PassStage):
	name = "test.fail"

	def run(self, state):
		raise ValueError("bad stage")


class _InterruptStage(_PassStage):
	name = "test.interrupt"

	def run(self, state):
		raise InterruptedError("stop")


class _AddSinogramStage(Stage):
	def __init__(self, name, amount):
		self.name = name
		self.amount = amount
		self.contract = StageContract(
			accepts=(SinoOrder,),
			prefers=SinoOrder,
			produces=PRESERVE,
			accepts_domain=(DataDomain.ATTENUATION,),
			produces_domain=PRESERVE,
			bounded_views_from=(ProjOrder,),
			mutates_input=True,
		)

	def run(self, state):
		raise AssertionError("the bounded dispatcher owns this Stage")

	def run_sinogram(self, sinogram, _state):
		sinogram += self.amount


class _AllocatingStage(_PassStage):
	name = "test.allocate"

	def run(self, state):
		buffer = state.storage.allocate(X(1), np.uint8, role="telemetry_probe")
		state.storage.retire(buffer, role="telemetry_probe")
		return state


def test_unobserved_pipeline_does_not_load_optional_stage_registry(monkeypatch):
	"""Telemetry opt-out must preserve the lightweight core dependency boundary."""

	original_import = builtins.__import__

	def guarded_import(name, *args, **kwargs):
		if name == "lftomo.pipeline.stages" or name.startswith("lftomo.pipeline.stages."):
			raise AssertionError("unobserved Pipeline loaded the built-in Stage registry")
		return original_import(name, *args, **kwargs)

	monkeypatch.setattr(builtins, "__import__", guarded_import)
	Pipeline([_PassStage()], validate=False).run(
		_state(np.zeros((6, 3, 8), dtype=np.float32))
	)


def test_sequential_span_reports_truthful_single_member_wall_time():
	events = []
	stage_events = []
	state = _state(np.zeros((6, 3, 8), dtype=np.float32))
	state.source_window = TomoWindow.from_bounds(
		ProjOrder(Theta=6, Y=3, X=8), y=(0, 3), dtype=np.float32,
	)
	Pipeline(
		[_PassStage()], validate=False, execution_observer=events.append,
		observer=lambda *event: stage_events.append(event), stage_indices=(7,),
	).run(state)

	assert [event.phase for event in events] == [
		ExecutionSpanPhase.STARTED, ExecutionSpanPhase.COMPLETED,
	]
	completed = events[-1]
	assert completed.span.kind is ExecutionKind.SEQUENTIAL
	assert completed.span.ordinal == 0
	assert completed.span.total == 1
	assert completed.span.member_start == 7
	assert completed.span.member_stop == 8
	assert completed.span.members[0].index == 7
	assert completed.span.members[0].stage_name == "test.pass"
	assert completed.coordinator_wall_seconds is not None
	assert completed.member_timings[0].coordinator_wall_seconds == (
		completed.coordinator_wall_seconds
	)
	assert all(event.source_window is state.source_window for event in events)
	assert all(event[-1] is state.source_window for event in stage_events)


def test_fused_process_span_returns_bounded_member_aggregates_once():
	events = []
	shape = ProjOrder(Theta=6, Y=3, X=8)
	with PipelineRunContext(max_workers=2) as context:
		working = context.storage.allocate(shape, np.float32, role="telemetry_input")
		working.array().fill(0)
		state = _state(working, storage=context.storage, executor=context.executor)
		pipeline = Pipeline(
			[_AddSinogramStage("test.first", 1), _AddSinogramStage("test.second", 2)],
			scan=state.scan,
			initial_layout=ProjOrder,
			initial_domain=DataDomain.ATTENUATION,
			execution_observer=events.append,
		)
		pipeline.run(state)

	assert len(events) == 2
	completed = events[-1]
	assert completed.span.kind is ExecutionKind.BOUNDED_SINOGRAM
	assert [member.stage_name for member in completed.span.members] == [
		"test.first", "test.second",
	]
	assert completed.coordinator_wall_seconds is not None
	assert all(timing.coordinator_wall_seconds is None for timing in completed.member_timings)
	assert [timing.worker.invocation_count for timing in completed.member_timings] == [3, 3]
	assert all(timing.worker.sum_seconds >= timing.worker.maximum_seconds
		for timing in completed.member_timings)
	assert completed.managed_allocation_changes == ()


def test_managed_allocation_changes_are_attached_to_the_owning_span():
	events = []
	with PipelineRunContext(max_workers=1) as context:
		state = _state(
			np.zeros((6, 3, 8), dtype=np.float32),
			storage=context.storage,
			executor=context.executor,
		)
		Pipeline(
			[_AllocatingStage()], validate=False, execution_observer=events.append,
		).run(state)

	changes = events[-1].managed_allocation_changes
	assert [(change.action, change.role) for change in changes] == [
		("allocate", "telemetry_probe"), ("retire", "telemetry_probe"),
	]
	assert all(change.scope == "execution_span" for change in changes)


@pytest.mark.parametrize(("stage", "phase", "error_type"), (
	(_FailStage(), ExecutionSpanPhase.FAILED, "builtins.ValueError"),
	(_InterruptStage(), ExecutionSpanPhase.INTERRUPTED, "builtins.InterruptedError"),
))
def test_failure_and_interruption_retain_span_context(stage, phase, error_type):
	events = []
	with pytest.raises((ValueError, InterruptedError)):
		Pipeline(
			[stage], validate=False, execution_observer=events.append,
		).run(_state(np.zeros((6, 3, 8), dtype=np.float32)))

	assert [event.phase for event in events] == [ExecutionSpanPhase.STARTED, phase]
	terminal = events[-1]
	assert terminal.span.members[0].stage_name == stage.name
	assert terminal.error_type == error_type
	assert terminal.coordinator_wall_seconds is not None
