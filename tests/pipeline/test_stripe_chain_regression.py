"""Regression guards for ordered, bounded stripe-removal chains."""

from pathlib import Path
import threading
import time

import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import DataDomain, ExecutionKind, Pipeline, PipelineState, PRESERVE
from lftomo.pipeline.stages import (
	AllOfTheStripeStage,
	FourierWaveletStripeStage,
	VoDeadStripeStage,
	VoFamilyStripeStage,
	VoSortStripeStage,
)
from lftomo.runner import load_recon
from lftomo.util.shapes import ProjOrder, SinoOrder


def _scan(theta: int, rows: int = 6, columns: int = 32) -> Scan:
	return Scan(
		energy_keV=20,
		pixel_size_um=1,
		n_projections=theta,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1, rows=rows, columns=columns),
		acquisition=Acquisition(n_projections=theta, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


def _fixture(theta: int = 32, rows: int = 6, columns: int = 32) -> np.ndarray:
	rng = np.random.default_rng(912)
	data = 1 + 0.02 * rng.standard_normal((theta, rows, columns), dtype=np.float32)
	data[:, :, columns // 3] *= 4
	data[:, :, columns // 2] *= 0.1
	return data


def _state(array: np.ndarray, layout: type) -> PipelineState:
	theta = array.shape[0] if layout is ProjOrder else array.shape[1]
	rows = array.shape[1] if layout is ProjOrder else array.shape[0]
	return PipelineState(
		working_array=array,
		scan=_scan(theta, rows, array.shape[2]),
		recon=Recon(pipeline=[StageConfig(kind="test")]),
		layout=layout,
		domain=DataDomain.ATTENUATION,
	)


def _pipeline(stages, layout, *, workers=1):
	return Pipeline(
		stages,
		scan=_scan(32),
		initial_layout=layout,
		initial_domain=DataDomain.ATTENUATION,
		bounded_workers=workers,
	)


def _fw_vo():
	return [
		FourierWaveletStripeStage(level=2, wname="db2", sigma=2),
		VoSortStripeStage(size=5),
	]


def test_r254_canonical_authors_fourier_wavelet_then_vo_sort():
	path = Path(__file__).parents[2] / "lftomo" / "pipelines" / "r254-canonical.yaml"
	recon, _ = load_recon(path)

	assert [
		stage.algorithm for stage in recon.pipeline if stage.kind == "stripe"
	] == ["fw", "vo_sort"]


def test_real_fw_vo_chain_gathers_once_and_matches_contiguous_sinograms(monkeypatch):
	import lftomo.pipeline.execution as execution

	projection = _fixture()
	sinogram = projection.transpose(1, 0, 2).copy()
	scratch = []
	original = execution.sinogram_scratch

	def record_scratch(data, layout, index):
		row = original(data, layout, index)
		scratch.append((index, row.shape, row.nbytes))
		return row

	monkeypatch.setattr(execution, "sinogram_scratch", record_scratch)
	bounded = _pipeline(_fw_vo(), ProjOrder)
	contiguous = _pipeline(_fw_vo(), SinoOrder)
	bounded.run(_state(projection, ProjOrder))
	contiguous.run(_state(sinogram, SinoOrder))

	assert [(segment.start, segment.stop, segment.kind) for segment in bounded.plan.segments] == [
		(0, 2, ExecutionKind.BOUNDED_SINOGRAM),
	]
	assert [item[0] for item in scratch] == list(range(projection.shape[1]))
	assert {item[1] for item in scratch} == {(projection.shape[0], projection.shape[2])}
	assert max(item[2] for item in scratch) < projection.nbytes
	assert all(not step.materializations for step in bounded.plan.steps)
	np.testing.assert_allclose(
		projection, sinogram.transpose(1, 0, 2), rtol=1e-6, atol=1e-6,
	)


def test_reversing_non_commuting_fw_vo_chain_changes_the_result():
	authored = _fixture()
	reversed_order = authored.copy()
	_pipeline(_fw_vo(), ProjOrder).run(_state(authored, ProjOrder))
	_pipeline([
		VoSortStripeStage(size=5),
		FourierWaveletStripeStage(level=2, wname="db2", sigma=2),
	], ProjOrder).run(_state(reversed_order, ProjOrder))

	assert not np.allclose(authored, reversed_order, rtol=1e-6, atol=1e-6)


def test_vo_family_preserves_its_authored_order_and_matches_contiguous(monkeypatch):
	import lftomo.pipeline.stages.stripe as stripe_module

	projection = _fixture()
	sinogram = projection.transpose(1, 0, 2).copy()
	order = ("remove_sort_vo", "remove_dead_vo", "remove_large_vo")
	bounded_stage = VoFamilyStripeStage(order=order, size=5, snr=2, norm=False)
	contiguous_stage = VoFamilyStripeStage(order=order, size=5, snr=2, norm=False)
	_pipeline([bounded_stage], ProjOrder).run(_state(projection, ProjOrder))
	_pipeline([contiguous_stage], SinoOrder).run(_state(sinogram, SinoOrder))
	np.testing.assert_allclose(
		projection, sinogram.transpose(1, 0, 2), rtol=1e-6, atol=1e-6,
	)

	calls = []
	monkeypatch.setattr(stripe_module, "_remove_sort_array", lambda *_args, **_kwargs: calls.append("remove_sort_vo"))
	monkeypatch.setattr(stripe_module, "_remove_dead_array", lambda *_args, **_kwargs: calls.append("remove_dead_vo"))
	monkeypatch.setattr(stripe_module, "_remove_large_array", lambda *_args, **_kwargs: calls.append("remove_large_vo"))
	bounded_stage.run_sinogram(np.ones((8, 8), dtype=np.float32), _state(_fixture(), ProjOrder))
	assert calls == list(order)


def test_full_array_stripe_stage_is_a_predictable_segment_barrier(monkeypatch):
	import lftomo.pipeline.execution as execution

	projection = _fixture()
	gathers = []
	original = execution.sinogram_scratch
	monkeypatch.setattr(execution, "sinogram_scratch", lambda data, layout, index:
		gathers.append(index) or original(data, layout, index))
	monkeypatch.setattr(
		"tomopy.remove_all_stripe",
		lambda data, **_kwargs: np.asarray(data) + np.float32(0.125),
	)
	stages = [
		FourierWaveletStripeStage(level=2, wname="db2", sigma=2),
		AllOfTheStripeStage(snr=2, la_size=5, sm_size=3),
		VoSortStripeStage(size=5),
	]
	pipeline = _pipeline(stages, ProjOrder)
	pipeline.run(_state(projection, ProjOrder))

	assert [(item.start, item.stop, item.kind) for item in pipeline.plan.segments] == [
		(0, 1, ExecutionKind.BOUNDED_SINOGRAM),
		(1, 2, ExecutionKind.SEQUENTIAL),
		(2, 3, ExecutionKind.BOUNDED_SINOGRAM),
	]
	assert gathers == list(range(projection.shape[1])) * 2
	assert np.isfinite(projection).all()


def test_ndarray_real_chain_uses_one_live_scratch():
	activity = {"lock": threading.Lock(), "current": 0, "peak": 0}
	stage = VoDeadStripeStage(snr=2, size=5, norm=False)
	original = stage.run_sinogram

	def tracked(sinogram, state):
		with activity["lock"]:
			activity["current"] += 1
			activity["peak"] = max(activity["peak"], activity["current"])
		time.sleep(0.005)
		try:
			return original(sinogram, state)
		finally:
			with activity["lock"]:
				activity["current"] -= 1

	stage.run_sinogram = tracked
	projection = _fixture(rows=8)
	_pipeline([stage, VoSortStripeStage(size=5)], ProjOrder, workers=2).run(
		_state(projection, ProjOrder)
	)

	assert activity["peak"] == 1
	row_scratch_bytes = projection.shape[0] * projection.shape[2] * projection.dtype.itemsize
	assert activity["peak"] * row_scratch_bytes < projection.nbytes


@pytest.mark.parametrize("stage", [
	FourierWaveletStripeStage(level=2),
	VoDeadStripeStage(size=5),
	VoSortStripeStage(size=5),
	VoFamilyStripeStage(size=5),
])
def test_chain_participant_contracts_preserve_layout_domain_and_mutate(stage):
	contract = stage.contract
	assert contract.accepts == (SinoOrder,)
	assert contract.prefers is SinoOrder
	assert contract.produces is PRESERVE
	assert contract.produces_domain is PRESERVE
	assert contract.bounded_views_from == (ProjOrder,)
	assert contract.mutates_input is True
