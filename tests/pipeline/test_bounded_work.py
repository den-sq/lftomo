"""Shared scratch request compatibility and failure cleanup contracts."""

from dataclasses import replace

import numpy as np
import pytest

from lftomo.pipeline import (
	BoundedWorkRequest, BufferSpec, PipelineRunContext, SinogramStage, StageParams,
	collect_bounded_work,
)
from lftomo.pipeline.execution import run_bounded_sinogram_segment
from lftomo.pipeline.stages import VoSortStripeStage
from lftomo.util.shapes import ProjOrder, SinoOrder
from tests.pipeline.test_stripe_chain_regression import _state
from tests.sample_data import attach_source_geometry


class _RequestedWorkStage(SinogramStage):
	"""Inject a kernel failure after segment work has been acquired."""

	class Params(StageParams):
		fail: bool = False

	def bounded_work_requests(self, input_spec):
		return (BoundedWorkRequest(
			"_requested_work", BufferSpec(input_spec.shape.to(SinoOrder), input_spec.dtype),
			"requested_work", fill_value=7,
		),)

	def run_sinogram(self, sinogram, state):
		if self.params.fail:
			raise RuntimeError("requested kernel failure")


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
			request, = stage.bounded_work_requests(BufferSpec.from_buffer(state.working_array))
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


def test_shared_requests_use_the_largest_declared_capacity():
	spec = BufferSpec(ProjOrder(4, 2, 6), np.float32)
	first, second = _RequestedWorkStage(), _RequestedWorkStage()
	request = replace(first.bounded_work_requests(spec)[0], grow_axes=("X",))
	larger = replace(request, buffer_spec=BufferSpec(SinoOrder(2, 4, 10), np.float32))
	first.bounded_work_requests = lambda _spec: (request,)
	second.bounded_work_requests = lambda _spec: (larger,)
	assert collect_bounded_work([first, second], spec) == (larger,)
	assert collect_bounded_work([second, first], spec) == (larger,)


def test_vo_requests_reject_unsupported_dtype_and_sort_index_overflow():
	stage = VoSortStripeStage(size=5)
	with pytest.raises(TypeError, match="floating-point"):
		stage.bounded_work_requests(BufferSpec(ProjOrder(16, 3, 24), np.int32))
	with pytest.raises(ValueError, match="uint32 capacity"):
		stage.bounded_work_requests(BufferSpec(ProjOrder(2**32, 1, 24), np.float32))


@pytest.mark.parametrize("conflict", [{"fill_value": 0}, {"required_for_fallback": True}])
def test_shared_key_rejects_conflicting_policy(conflict):
	spec = BufferSpec(ProjOrder(4, 2, 6), np.float32)
	first = _RequestedWorkStage()
	second = _RequestedWorkStage()
	request, = first.bounded_work_requests(spec)
	second.bounded_work_requests = lambda _spec: (replace(request, **conflict),)
	with pytest.raises(ValueError, match="incompatible bounded work"):
		collect_bounded_work([first, second], spec)
