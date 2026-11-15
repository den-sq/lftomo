"""Stage ownership and managed lifecycle of bounded segment work."""

from dataclasses import replace

import numpy as np
import pytest

from lftomo.pipeline import (
	BoundedWorkRequest, BufferSpec, PipelineRunContext, SinogramStage, StageParams,
	collect_bounded_work,
)
from lftomo.pipeline.execution import run_bounded_sinogram_segment
from lftomo.pipeline.stages import (
	VoDeadStripeStage, VoFamilyStripeStage, VoLargeStripeStage, VoSortStripeStage,
)
from lftomo.pipeline.stages.stripe import VO_MEDIAN_PAD_WORK_ARTIFACT
from lftomo.storage import ManagedBuffer
from lftomo.util.shapes import ProjOrder, SinoOrder
from tests.pipeline.test_stripe_chain_regression import _state
from tests.sample_data import attach_source_geometry


class _RequestedWorkStage(SinogramStage):
	"""A non-Vo consumer that proves the executor follows requests only."""

	class Params(StageParams):
		amount: int = 1
		fail: bool = False

	def bounded_work_requests(self, input_spec):
		return (BoundedWorkRequest(
			"_requested_work", BufferSpec(input_spec.shape.to(SinoOrder), input_spec.dtype),
			"requested_work", fill_value=7,
		),)

	def run_sinogram(self, sinogram, state):
		work = state.artifacts["_requested_work"]
		assert isinstance(work, ManagedBuffer)
		assert work.shape == SinoOrder(1, sinogram.shape[0], sinogram.shape[1])
		assert state.source_window.shape.Y == 1
		assert state.source_window.pos.Y in (7, 8)
		work.array()[0] += self.params.amount
		if self.params.fail:
			raise RuntimeError("requested kernel failure")
		sinogram[...] = work.array()[0]


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
@pytest.mark.parametrize("process", [False, True])
def test_non_vo_requests_share_one_allocation_and_canonical_unit_handles(layout, process):
	shape = ProjOrder(4, 2, 6).to(layout)
	with PipelineRunContext(max_workers=2) as context:
		working = context.storage.allocate(shape, np.float32, role="input")
		state = _state(np.zeros(shape.as_tuple(), np.float32), layout).with_array(working)
		state.storage = context.storage
		attach_source_geometry(state, y_start=7, physical_y=9)
		stages = [_RequestedWorkStage(amount=1), _RequestedWorkStage(amount=2)]
		run_bounded_sinogram_segment(
			state, stages, executor=context.executor if process else None,
		)
		np.testing.assert_array_equal(working.array(), 10)
		assert "_requested_work" not in state.artifacts
		report = state.artifacts["bounded_work_allocations"][0]
		assert report["allocation_count"] == 1
		assert report["buffers"][0]["shape"] == shape.to(SinoOrder).as_dict()
		assert report["managed_bytes"] == shape.nbytes(np.float32)
		events = [event for event in context.storage.allocation_events if event["role"] == "requested_work"]
		assert [event["action"] for event in events] == ["allocate", "retire"]
		run_bounded_sinogram_segment(
			state, stages, executor=context.executor if process else None,
		)
		np.testing.assert_array_equal(working.array(), 10)
		events = [event for event in context.storage.allocation_events if event["role"] == "requested_work"]
		assert [event["action"] for event in events] == ["allocate", "retire", "alias", "retire"]
		assert len({event["name"] for event in events}) == 1


@pytest.mark.parametrize("fail_allocation", [False, True])
def test_segment_failure_removes_and_retires_every_allocated_request(monkeypatch, fail_allocation):
	with PipelineRunContext(max_workers=1) as context:
		shape = ProjOrder(4, 2, 6)
		working = context.storage.allocate(shape, np.float32, role="input")
		state = _state(np.zeros(shape.as_tuple(), np.float32), ProjOrder).with_array(working)
		state.storage = context.storage
		attach_source_geometry(state, y_start=7, physical_y=9)
		stage = _RequestedWorkStage(fail=not fail_allocation)
		if fail_allocation:
			request, = stage.bounded_work_requests(state.buffer_spec)
			monkeypatch.setattr(stage, "bounded_work_requests", lambda _spec: (
				request, replace(request, artifact_key="_second_work", role="second_work"),
			))
			allocate = context.storage.allocate

			def fail_second(*args, **kwargs):
				if kwargs["role"] == "second_work":
					raise RuntimeError("requested allocation failure")
				return allocate(*args, **kwargs)

			monkeypatch.setattr(context.storage, "allocate", fail_second)
		with pytest.raises(RuntimeError, match="requested .* failure"):
			run_bounded_sinogram_segment(state, [stage])
		assert "_requested_work" not in state.artifacts
		assert "_second_work" not in state.artifacts
		events = [event for event in context.storage.allocation_events if event["role"] == "requested_work"]
		assert [event["action"] for event in events] == ["allocate", "retire"]


@pytest.mark.parametrize("stage,half_width", [
	(VoSortStripeStage(size=5), 2),
	(VoSortStripeStage(size=5, dim=2), 0),
	(VoDeadStripeStage(size=7, norm=True), 3),
	(VoDeadStripeStage(size=7, norm=False), 0),
	(VoLargeStripeStage(size=9), 4),
	(VoFamilyStripeStage(size=11), 5),
	(VoFamilyStripeStage(size=11, order=("remove_dead_vo",)), 0),
])
def test_each_vo_stage_owns_inspectable_dtype_and_padding_requests(stage, half_width):
	spec = BufferSpec(ProjOrder(16, 3, 24), np.float32)
	requests = stage.bounded_work_requests(spec)
	assert requests[0].buffer_spec == BufferSpec(SinoOrder(3, 16, 24), np.float32)
	assert requests[1].buffer_spec == BufferSpec(SinoOrder(3, 16, 24), np.uint32)
	assert len(requests) == 2 + bool(half_width)
	if half_width:
		pad = requests[2]
		assert pad.buffer_spec.shape == SinoOrder(3, 16, 24 + 2 * half_width)
		assert pad.fill_value == 0
		assert pad.grow_axes == ("X",)


def test_chain_shares_largest_declared_padding_capacity():
	spec = BufferSpec(ProjOrder(16, 3, 24), np.float32)
	requests = collect_bounded_work([VoLargeStripeStage(size=9), VoSortStripeStage(size=5)], spec)
	assert len(requests) == 3
	assert requests[-1].artifact_key == VO_MEDIAN_PAD_WORK_ARTIFACT
	assert requests[-1].buffer_spec.shape == SinoOrder(3, 16, 32)


def test_vo_request_owner_checks_dtype_capacity_and_median_fallback():
	stage = VoSortStripeStage(size=5)
	with pytest.raises(TypeError, match="floating-point"):
		stage.bounded_work_requests(BufferSpec(ProjOrder(16, 3, 24), np.int32))
	with pytest.raises(ValueError, match="uint32 capacity"):
		stage.bounded_work_requests(BufferSpec(ProjOrder(2**32, 1, 24), np.float32))
	for dtype in (np.float16, np.dtype(np.float32).newbyteorder("S")):
		assert len(stage.bounded_work_requests(BufferSpec(ProjOrder(16, 3, 24), dtype))) == 2


def test_shared_key_rejects_conflicting_initialization():
	spec = BufferSpec(ProjOrder(4, 2, 6), np.float32)
	first = _RequestedWorkStage()
	second = _RequestedWorkStage()
	request, = first.bounded_work_requests(spec)
	second.bounded_work_requests = lambda _spec: (replace(request, fill_value=0),)
	with pytest.raises(ValueError, match="incompatible bounded work"):
		collect_bounded_work([first, second], spec)
