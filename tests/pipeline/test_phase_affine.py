"""Affine parity oracle: tests/fixtures/phase_affine_reference.json."""

import os
import json
from pathlib import Path

import numpy as np
import pytest

from lftomo.config import StageConfig
from lftomo.pipeline import (
	LocalArray, DataDomain, Pipeline, PipelineRunContext, PipelineState, validate_pipeline,
)
from lftomo.pipeline.stages import (
	PaganinFilterStage, PaganinStage, TIFFStackReadStage, WeightedNormalizeStage,
)
from lftomo.pipeline.stages.phase import _paganin_transfer
from lftomo.prep.phase import (
	phase_fft_window, phase_kernel_radius, phase_pad, phase_unpad, retrieve_single,
	single_material_filter,
)
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
		LocalArray(values, shape), _scan(theta=7, rows=10, columns=12), _recon(),
		domain=DataDomain.TRANSMISSION,
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
		material_alpha=2e-8, envelope=envelope, pad=pad, padding="reference",
	)
	np.testing.assert_array_equal(actual, reference[f"{kind}_{int(pad)}"])
	assert actual.dtype == np.float32


def test_fft_padding_is_owned_by_the_canonical_window():
	window = phase_fft_window((7, 10), True, padding="reference")
	assert isinstance(window, TomoWindow)
	assert window.pad_before == Proj(1, 2)
	assert window.pad_after == Proj(2, 3)
	assert window.with_pad().shape == Proj(10, 15)
	image = np.arange(70).reshape(7, 10)
	canvas = phase_pad(image, window)
	np.testing.assert_array_equal(phase_unpad(canvas, window), image)
	assert np.shares_memory(canvas, phase_unpad(canvas, window))
	assert phase_fft_window((1, 10), True, padding="reference").with_pad().shape == Proj(1, 15)
	from lftomo.runner import _json_value
	from lftomo.util.converter import CONVERTER

	document = json.loads(json.dumps(_json_value({"phase_filter_signature": ("slope", window)})))
	restored = CONVERTER.structure(document["phase_filter_signature"][1], DataRef)
	assert restored.pad_before == window.pad_before
	assert restored.pad_after == window.pad_after


def test_kernel_canvas_covers_eight_decay_lengths_on_each_side():
	# L = sqrt(z*s)/pixel = sqrt(10) pixels; eight lengths need 26 pixels.
	assert phase_kernel_radius(1e-6, 0.01, 1e-9) == 26
	window = phase_fft_window((7, 10), pixel_m=1e-6, distance_m=0.01, slope=1e-9)
	assert window.pad_before == Proj(26, 26)
	assert window.pad_after == Proj(27, 27)
	assert window.with_pad().shape == Proj(60, 63)
	image = np.arange(70).reshape(7, 10)
	canvas = phase_pad(image, window)
	np.testing.assert_array_equal(phase_unpad(canvas, window), image)
	assert canvas[0, 0] == image[0, 0]
	assert canvas[-1, -1] == image[-1, -1]


@pytest.mark.parametrize("rows, expected_before, expected_after, expected_rows", [
	((2, 7), 6, 0, 21),  # The source boundary supplies only two of eight leading rows.
	((20, 27), 0, 0, 23),  # Both sides are covered: retain the non-fast Y length 23.
])
def test_kernel_canvas_supplements_each_realized_source_halo_side(
	rows, expected_before, expected_after, expected_rows,
):
	source = TomoWindow.from_bounds(ProjOrder(3, 40, 10), y=rows, halo=8)
	held = source.with_pad().shape
	window = phase_fft_window(
		(held.Y, held.X), pixel_m=1e-6, distance_m=0.01, slope=1e-10,
		source_window=source,
	)
	assert window.pad_before == Proj(expected_before, 8)
	assert window.pad_after == Proj(expected_after, 9)
	assert window.with_pad().shape == Proj(expected_rows, 27)
	# Reference compatibility ignores real halo when sizing its numerical canvas.
	reference = phase_fft_window((held.Y, held.X), padding="reference", source_window=source)
	plain_reference = phase_fft_window((held.Y, held.X), padding="reference")
	assert reference.pad_before == plain_reference.pad_before
	assert reference.pad_after == plain_reference.pad_after


@pytest.mark.parametrize("pad", [False, True])
def test_slope_filter_preserves_dc_and_retrieval_returns_its_negative_log(pad):
	transmission = np.full((2, 7, 10), 0.8)
	filtered = single_material_filter(transmission, 1e-6, 0.01, 1e-9, pad=pad, dtype=np.float64)
	attenuation = retrieve_single(transmission, 1e-6, 0.01, 1e-9, pad=pad, dtype=np.float64)
	np.testing.assert_allclose(filtered, transmission, rtol=2e-15, atol=0)
	np.testing.assert_allclose(attenuation, -np.log(transmission), rtol=2e-15, atol=0)
	if not pad:
		window = phase_fft_window(transmission.shape[-2:], pad=False)
		assert window.pad_before == window.pad_after == Proj(0, 0)
		assert window.with_pad().shape == Proj(7, 10)


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
		index = state.source_window.source_index("Theta", next(state.work_indices("Theta")))
		state.artifacts[f"affine_worker_{index}"] = os.getpid()
		envelope = state.artifacts["phase_envelope"]
		state.artifacts[f"affine_envelope_{index}"] = (
			envelope.name, envelope.shape.Theta, self.params.envelope_m,
			any("envelope_m" in config.params for config in state.recon.pipeline),
		)
		return super().run_projection(image, state)


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
@pytest.mark.parametrize("pad", [False, True])
def test_projection_dispatch_uses_source_envelope_and_managed_filter(reference, layout, pad):
	selected = reference["transmission"][1:7:2, 2:7, :].copy()
	if layout is SinoOrder:
		selected = selected.transpose(1, 0, 2).copy()
	stage = _ObservedAffineStage(
		slope=1e-9, material_alpha=2e-8,
		envelope_m=reference["envelope"].tolist(), fft_pad=pad, padding="reference",
	)
	with PipelineRunContext(max_workers=2) as context:
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
		allocations = [event for event in context.storage.allocation_events
			if event["action"] == "allocate" and event["role"] in ("phase_filter", "phase_envelope")]
		assert [event["role"] for event in allocations] == ["phase_filter", "phase_envelope"]


@pytest.mark.parametrize("kind", ["scalar", "spatial"])
def test_serial_affine_stage_prepares_envelope_once(reference, kind):
	values = reference["transmission"].copy()
	state = _state(values)
	state.scan.geometry.ssd_cm = 1
	stage = PaganinStage(
		slope=1e-9, material_alpha=2e-8,
		envelope_m=3e-4 if kind == "scalar" else reference["envelope"].tolist(),
		padding="reference",
	)
	stage.run(state)
	np.testing.assert_array_equal(values, reference[f"{kind}_1"])
	if kind == "spatial":
		assert state.artifacts["phase_envelope"].shape == values.shape


def test_planning_counts_the_real_frequency_canvas():
	stage = PaganinStage(slope=1e-9, material_alpha=2e-8, envelope_m=[[3e-4]], padding="reference")
	plan = validate_pipeline(
		[TIFFStackReadStage(order="projection"), WeightedNormalizeStage(), stage],
		_scan(theta=7, rows=10, columns=12),
	)
	allocation = next(event for event in plan.allocation_schedule if event.role == "phase_filter")
	assert allocation.bytes == 15 * 10 * np.dtype(np.float64).itemsize
	envelope = next(event for event in plan.allocation_schedule if event.role == "phase_envelope")
	assert envelope.bytes == 7 * 10 * 12 * np.dtype(np.float32).itemsize
