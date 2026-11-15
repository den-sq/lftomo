"""Affine parity oracle: tests/fixtures/phase_affine_reference.json."""

import os
import json
from pathlib import Path

import numpy as np
import pytest

from lftomo.config import StageConfig
from lftomo.pipeline import (
	BufferSpec, DataDomain, Pipeline, PipelineRunContext, PipelineState, validate_pipeline,
)
from lftomo.pipeline.planning import BufferAction
from lftomo.pipeline.stages import (
	PaganinFilterStage, PaganinStage, TIFFStackReadStage, WeightedNormalizeStage,
)
from lftomo.pipeline.stages import phase as phase_stages
from lftomo.pipeline.stages.phase import _paganin_transfer
from lftomo.prep.phase import phase_fft_window, phase_pad, phase_unpad, retrieve_single
from lftomo.storage import ManagedBuffer
from lftomo.util.shapes import Proj, ProjOrder, SinoOrder
from lftomo.util.window import DataRef, TomoWindow
from tests.pipeline.test_process_dispatch import _recon, _scan
from tests.sample_data import attach_source_geometry


@pytest.fixture
def reference():
	with np.load(Path(__file__).parents[1] / "fixtures" / "phase_affine_reference.npz") as record:
		yield record


def _state(values, layout=ProjOrder, *, theta_start=0, theta_stride=1, y_start=0):
	shape = layout(*values.shape)
	return attach_source_geometry(PipelineState(
		values, _scan(theta=7, rows=10, columns=12), _recon(),
		buffer_spec=BufferSpec(shape, values.dtype), domain=DataDomain.TRANSMISSION,
	), theta_start=theta_start, theta_stride=theta_stride, y_start=y_start,
		physical_theta=7, physical_y=10)


@pytest.mark.parametrize("terminal", [False, True])
def test_defaults_preserve_the_original_paganin_bitwise(reference, terminal):
	values = reference["transmission"].copy()
	shape = values.shape[-2:]
	# Frozen pre-#319 operator: full complex NumPy FFT and original constants.
	ky = 2 * np.pi * np.fft.fftfreq(shape[0], d=1e-4)
	kx = 2 * np.pi * np.fft.fftfreq(shape[1], d=1e-4)
	w2 = np.add.outer(ky**2, kx**2)
	wavelength = 2 * np.pi * 6.58211928e-19 * 299792458e2 / 20
	transfer = 1 / (1 + 10 * 0.06 * wavelength * w2 / (4 * np.pi))
	expected = np.real(np.fft.ifft2(np.fft.fft2(values) * transfer))
	if terminal:
		expected = -np.log(np.clip(expected, 1e-8, None))
	expected = expected.astype(values.dtype)
	stage = PaganinStage() if terminal else PaganinFilterStage()
	stage.run(_state(values))
	np.testing.assert_array_equal(values, expected)
	np.testing.assert_array_equal(transfer, _paganin_transfer(
		shape, alpha=0.06, energy_kev=20, pixel_size_um=1, ssd_cm=10,
	))


@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("kind", ["spatial", "scalar"])
def test_affine_retrieval_matches_frozen_fresnel_fixture(reference, pad, kind):
	envelope = reference["envelope"] if kind == "spatial" else 3e-4
	actual = retrieve_single(
		reference["transmission"], 1e-6, 0.01, 1e-9,
		material_alpha=2e-8, envelope=envelope, pad=pad,
	)
	np.testing.assert_array_equal(actual, reference[f"{kind}_{int(pad)}"])
	assert actual.dtype == np.float32


def test_fft_padding_is_owned_by_the_canonical_window():
	window = phase_fft_window((7, 10), True)
	assert isinstance(window, TomoWindow)
	assert window.pad_before == Proj(1, 2)
	assert window.pad_after == Proj(2, 3)
	assert window.with_pad().shape == Proj(10, 15)
	image = np.arange(70).reshape(7, 10)
	canvas = phase_pad(image, window)
	np.testing.assert_array_equal(phase_unpad(canvas, window), image)
	assert np.shares_memory(canvas, phase_unpad(canvas, window))
	assert phase_fft_window((1, 10), True).with_pad().shape == Proj(1, 15)
	from lftomo.runner import _json_value
	from lftomo.util.converter import CONVERTER

	document = json.loads(json.dumps(_json_value({"phase_filter_signature": ("slope", window)})))
	restored = CONVERTER.structure(document["phase_filter_signature"][1], DataRef)
	assert restored.pad_before == window.pad_before
	assert restored.pad_after == window.pad_after


@pytest.mark.parametrize("params", [
	{"material_alpha": 2e-8},
	{"slope": 1e-9, "material_alpha": 2e-8},
	{"envelope_m": 1e-4},
])
def test_affine_configuration_requires_explicit_slope_and_envelope(params):
	with pytest.raises(ValueError, match="requires"):
		PaganinStage(**params)


def test_clip_defaults_are_resolved_once_by_the_parameter_model():
	assert PaganinStage().params.epsilon == 1e-8
	assert PaganinStage(slope=1e-9).params.epsilon == 1e-6
	assert PaganinStage(slope=1e-9, epsilon=None).params.epsilon is None


class _ObservedAffineStage(PaganinStage):
	def run_projection(self, image, state):
		state.artifacts[f"affine_worker_{state.source_window.pos.Theta}"] = os.getpid()
		envelope = state.artifacts["phase_envelope"]
		state.artifacts[f"affine_envelope_{state.source_window.pos.Theta}"] = (
			envelope.name, envelope.shape.Theta, self.params.envelope_m,
			any("envelope_m" in config.params for config in state.recon.pipeline),
		)
		work = {key: state.artifacts[key] for key in (
			phase_stages.PHASE_PADDING_WORK, phase_stages.PHASE_ENVELOPE_WORK,
		)}
		state.artifacts[f"affine_padding_{state.source_window.pos.Theta}"] = tuple(
			(handle.name, handle.shape.Theta, float(handle.array().flat[0]))
			for handle in work.values()
		)
		original = phase_stages.single_material_filter

		def observed_filter(*args, **kwargs):
			assert kwargs["window"] is state.artifacts["phase_filter_signature"][1]
			for argument, key in (
				("padding_work", phase_stages.PHASE_PADDING_WORK),
				("envelope_work", phase_stages.PHASE_ENVELOPE_WORK),
			):
				assert np.shares_memory(kwargs[argument], work[key].array())
			return original(*args, **kwargs)
		phase_stages.single_material_filter = observed_filter
		try:
			return super().run_projection(image, state)
		finally:
			phase_stages.single_material_filter = original


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_projection_dispatch_uses_source_envelope_and_managed_filter(reference, monkeypatch, layout, pad, workers):
	selected = reference["transmission"][1:7:2, 2:7, :].copy()
	if layout is SinoOrder:
		selected = selected.transpose(1, 0, 2).copy()
	stage = _ObservedAffineStage(
		slope=1e-9, material_alpha=2e-8,
		envelope_m=reference["envelope"].tolist(), fft_pad=pad,
	)
	with PipelineRunContext(max_workers=workers) as context:
		padding = {}
		allocate = context.storage.allocate

		def record_allocation(shape, dtype, **kwargs):
			handle = allocate(shape, dtype, **kwargs)
			if kwargs.get("role") in (phase_stages.PHASE_PADDING_WORK, phase_stages.PHASE_ENVELOPE_WORK):
				padding[kwargs["role"]] = handle
				for index, plane in enumerate(handle.array()):
					plane.fill(index + 0.25)
			return handle
		monkeypatch.setattr(context.storage, "allocate", record_allocation)
		working = context.storage.allocate(layout(*selected.shape), selected.dtype, role="normalized")
		working.array()[...] = selected
		state = _state(selected, layout, theta_start=1, theta_stride=2, y_start=2)
		state.working_array = working
		state.storage, state.executor = context.storage, context.executor
		state.scan.geometry.ssd_cm = 1
		state.recon.pipeline = [StageConfig(
			kind="phase", algorithm="paganin", params=stage.params.model_dump(),
		)]
		original_config = state.recon.model_dump()
		result = Pipeline([stage]).run(state)
		actual = np.asarray(result.working_array)
		if layout is SinoOrder:
			actual = actual.transpose(1, 0, 2)
		np.testing.assert_array_equal(actual, reference[f"selected_{int(pad)}"])
		assert isinstance(result.artifacts["phase_filter"], ManagedBuffer)
		assert isinstance(result.artifacts["phase_filter_signature"][1], TomoWindow)
		envelope = result.artifacts["phase_envelope"]
		assert isinstance(envelope, ManagedBuffer)
		assert all(result.artifacts[f"affine_worker_{index}"] != os.getpid() for index in (1, 3, 5))
		assert all(result.artifacts[f"affine_envelope_{index}"] == (
			envelope.name, 1, None, False,
		) for index in (1, 3, 5))
		assert state.recon.model_dump() == original_config
		assert stage.params.envelope_m == reference["envelope"].tolist()
		for local_index, source_index in enumerate((1, 3, 5)):
			assert result.artifacts[f"affine_padding_{source_index}"] == tuple(
				(padding[key].name, 1, local_index + 0.25)
				for key in (phase_stages.PHASE_PADDING_WORK, phase_stages.PHASE_ENVELOPE_WORK)
			)
		requests = stage.bounded_work_requests(state.buffer_spec)
		assert set(padding) == {request.artifact_key for request in requests}
		assert not set(padding).intersection(result.artifacts)
		for request in requests:
			assert padding[request.artifact_key].shape == request.buffer_spec.shape
			assert padding[request.artifact_key].dtype == np.float32
			assert [event["action"] for event in context.storage.allocation_events
				if event["role"] == request.role] == ["allocate", "retire"]
		allocations = [event for event in context.storage.allocation_events
			if event["action"] == "allocate" and event["role"] in ("phase_filter", "phase_envelope")]
		assert [event["role"] for event in allocations] == ["phase_filter", "phase_envelope"]


class _FailingAffineStage(PaganinStage):
	def run_projection(self, image, state):
		raise RuntimeError("projection kernel failed")


def test_failed_projection_retires_managed_padding(reference):
	values = reference["transmission"].copy()
	stage = _FailingAffineStage(slope=1e-9, material_alpha=2e-8, envelope_m=3e-4)
	with PipelineRunContext(max_workers=1) as context:
		state = _state(values)
		state.working_array = context.storage.allocate(state.buffer_spec.shape, values.dtype, role="normalized")
		np.asarray(state.working_array)[...] = values
		state.storage, state.executor = context.storage, context.executor
		with pytest.raises(RuntimeError, match="projection kernel failed"):
			Pipeline([stage]).run(state)
		for request in stage.bounded_work_requests(state.buffer_spec):
			assert request.artifact_key not in state.artifacts
			assert [event["action"] for event in context.storage.allocation_events
				if event["role"] == request.role] == ["allocate", "retire"]


@pytest.mark.parametrize("kind", ["scalar", "spatial"])
def test_serial_affine_stage_prepares_envelope_once(reference, kind):
	values = reference["transmission"].copy()
	state = _state(values)
	state.scan.geometry.ssd_cm = 1
	stage = PaganinStage(
		slope=1e-9, material_alpha=2e-8,
		envelope_m=3e-4 if kind == "scalar" else reference["envelope"].tolist(),
	)
	stage.run(state)
	np.testing.assert_array_equal(values, reference[f"{kind}_1"])
	if kind == "spatial":
		assert state.artifacts["phase_envelope"].shape == values.shape


def test_planning_counts_the_real_frequency_canvas():
	stage = PaganinStage(slope=1e-9, material_alpha=2e-8, envelope_m=[[3e-4]])
	plan = validate_pipeline(
		[TIFFStackReadStage(order="projection"), WeightedNormalizeStage(), stage],
		_scan(theta=7, rows=10, columns=12),
	)
	allocation = next(event for event in plan.allocation_schedule if event.role == "phase_filter")
	assert allocation.bytes == 15 * 10 * np.dtype(np.float64).itemsize
	envelope = next(event for event in plan.allocation_schedule if event.role == "phase_envelope")
	assert envelope.bytes == 7 * 10 * 12 * np.dtype(np.float32).itemsize
	for key in (phase_stages.PHASE_PADDING_WORK, phase_stages.PHASE_ENVELOPE_WORK):
		events = [event for event in plan.allocation_schedule if event.role == key]
		assert [event.action for event in events] == [BufferAction.ALLOCATE, BufferAction.RETIRE]
		assert all(event.bytes == 7 * 15 * 18 * np.dtype(np.float32).itemsize for event in events)
