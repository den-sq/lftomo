"""Phase planning and execution consume the same effective Scan geometry."""

import numpy as np
import pytest

from lftomo.config import Acquisition, ConeGeometry, Detector, Helical, Recon, Scan, TIFFStackProjections
from lftomo.pipeline import BufferSpec, PipelineState, validate_pipeline
from lftomo.storage import LocalArray
from lftomo.pipeline.core import PadRequirement
from lftomo.pipeline.planning import stage_output_spec
from lftomo.pipeline.stages import (
	HelicalIndexRemapStage, PaganinFilterStage, TIFFStackReadStage, WeightedNormalizeStage,
)
from lftomo.pipeline.stages import phase
from lftomo.prep import phase as phase_prep
from lftomo.util.shapes import ProjOrder
from tests.sample_data import attach_source_geometry


def _state():
	data = np.full((3, 8, 8), 0.8, dtype=np.float32)
	return attach_source_geometry(PipelineState(
		working_array=LocalArray(data, ProjOrder(*data.shape)),
		scan=Scan(
			energy_keV=20,
			geometry=ConeGeometry(source_to_sample_m=0.5, sample_to_detector_m=0.5),
			n_projections=3, degrees_per_fov=180,
			detector=Detector(pixel_size_um=10, rows=8, columns=8),
			helical=Helical(enabled=True, pitch_um_per_projection=5),
			acquisition=Acquisition(n_projections=3, rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[]),
	))


def test_phase_filter_and_padding_derive_from_the_same_cone_geometry(monkeypatch):
	state = _state()
	filter_calls, pad_calls = [], []
	transfer, pad = phase._paganin_transfer, phase_prep._phase_pad

	def record_transfer(shape, **parameters):
		filter_calls.append(parameters)
		return transfer(shape, **parameters)

	def record_pad(dimension, pixel_size, distance, energy, **kwargs):
		pad_calls.append((pixel_size, distance, energy))
		return pad(dimension, pixel_size, distance, energy, **kwargs)

	monkeypatch.setattr(phase, "_paganin_transfer", record_transfer)
	monkeypatch.setattr(phase_prep, "_phase_pad", record_pad)
	before, after = PadRequirement(phase_fft_y=True).resolve(state.scan, state.working_array.shape)
	stage = PaganinFilterStage(alpha=0.06)
	phase.prepare_phase_filter(stage, state)
	prepared = state.artifacts["phase_filter"]
	stage.run(state)
	assert state.artifacts["phase_filter"] is prepared
	assert len(filter_calls) == 1
	assert filter_calls[0] == {"alpha": 0.06, "energy_kev": 20, "pixel_size_um": 5, "ssd_cm": 25}
	assert pad_calls == [(5e-4, 25, 20)]
	assert before == after
	assert before.Y == pad(8, 5e-4, 25, 20)
	assert np.isfinite(state.working_array).all()


def test_native_geometry_change_invalidates_the_prepared_phase_filter():
	state = _state()
	stage = PaganinFilterStage(alpha=0.06)
	phase.prepare_phase_filter(stage, state)
	before = state.artifacts["phase_filter"]
	state.scan.geometry.sample_to_detector_m = 1.0
	assert state.scan.magnification == 3
	assert state.scan.pixel_size_um == pytest.approx(10 / 3)
	assert state.scan.ssd_cm == pytest.approx(100 / 3)
	phase.prepare_phase_filter(stage, state)
	after = state.artifacts["phase_filter"]
	assert after is not before
	assert not np.array_equal(before, after)
	np.testing.assert_array_equal(after, phase._paganin_transfer(
		(8, 8), alpha=0.06, energy_kev=20, pixel_size_um=10 / 3, ssd_cm=100 / 3,
	))


def test_slope_kernel_planning_uses_effective_cone_geometry_and_preserves_dc():
	state = _state()
	stage = PaganinFilterStage(slope=1e-9)
	before, after = stage.contract.required_pad.resolve(state.scan, state.working_array.shape)
	# Effective z = 0.25 m and pixel = 5 um give ceil(8*sqrt(10)) = 26 rows.
	assert before.Y == after.Y == 26
	state.scan.metadata["flat_fields"] = np.stack([
		np.ones((8, 8)), np.ones((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)),
	])
	plan = validate_pipeline(
		[TIFFStackReadStage(order="projection"), WeightedNormalizeStage(), stage], state.scan,
	)
	allocation = next(event for event in plan.allocation_schedule if event.role == "phase_filter")
	assert allocation.bytes == 60 * 31 * np.dtype(np.float64).itemsize
	stage.run(state)
	np.testing.assert_allclose(state.working_array, 0.8, rtol=1e-7)
	assert state.artifacts["phase_filter"].shape == (60, 31)


def test_helical_planning_and_execution_use_effective_sample_pitch(monkeypatch):
	state = _state()
	stage = HelicalIndexRemapStage()
	calls = []
	output_geometry = stage.output_geometry

	def record_geometry(source_shape, pitch, source_window):
		calls.append((pitch, source_window))
		return output_geometry(source_shape, pitch, source_window)

	monkeypatch.setattr(stage, "output_geometry", record_geometry)
	planned = stage_output_spec(
		stage, BufferSpec.from_buffer(state.working_array), state.scan,
		dataset=state.dataset, source_window=state.source_window,
	)
	source_window = state.source_window
	result = stage.run(state)
	assert calls == [(1.0, source_window), (1.0, source_window)]
	assert BufferSpec.from_buffer(result.working_array) == planned
	np.testing.assert_array_equal(result.artifacts["helical_row_shifts"], [0, 1, 2])
