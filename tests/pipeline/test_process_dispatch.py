import os
import sys
import time

import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import DataDomain, Pipeline, PipelineRunContext, PipelineState
from lftomo.pipeline.core import Locality, Stage, StageContract
from lftomo.pipeline.dispatch import DispatchKind
from lftomo.pipeline.execution import (
	PipelineExecutor, WorkerTimeoutError, run_bounded_sinogram_segment,
)
from lftomo.pipeline.stages import (
	AffineSkewMovementStage,
	BeamTrackingNormalizeStage,
	BronnikovAidedCorrectionFilterStage,
	BronnikovAidedCorrectionStage,
	HelicalIndexRemapStage,
	LayoutConvertStage,
	NegativeLogStage,
	PaganinFilterStage,
	PaganinStage,
	STAGE_DISPATCH,
	STAGE_REGISTRY,
	TIFFSlicesWriteStage,
	VoCenterStage,
	WeightedNormalizeStage,
	_build_stage_dispatch,
)
from lftomo.storage import ManagedBuffer
from lftomo.util.shapes import ProjOrder, ReconOrder


def _scan(theta=6, rows=2, columns=8):
	return Scan(
		energy_keV=20,
		pixel_size_um=1,
		ssd_cm=10,
		n_projections=theta,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1, rows=rows, columns=columns),
		acquisition=Acquisition(n_projections=theta, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		metadata={"flat_fields": np.stack([
			np.full((rows, columns), 20, np.float32),
			np.full((rows, columns), 40, np.float32),
			np.zeros((rows, columns), np.float32),
			np.zeros((rows, columns), np.float32),
		])},
	)


def _recon(output_path=None):
	return Recon(
		pipeline=[StageConfig(kind="test")], output_path=output_path,
		process_workers=2,
	)


def _write_forever(buffer):
	values = buffer.array()
	while True:
		values[0] += 1
		time.sleep(0.01)


def _set_value(buffer, value):
	buffer.array()[0] = value


def _worker_pid():
	return os.getpid()


def _raise_deterministic_error():
	raise ValueError("deterministic kernel failure")


class _AddSinogramStage(Stage):
	name = "test.add_sinogram"
	contract = StageContract(locality=Locality.SINOGRAM)

	def run(self, state):
		return state

	def run_sinogram(self, sinogram, state):
		sinogram += 3


def test_registry_dispatch_is_mechanically_complete():
	assert set(STAGE_DISPATCH) == set(STAGE_REGISTRY)
	assert all(
		spec.kind is not DispatchKind.EXEMPT or spec.reason
		for spec in STAGE_DISPATCH.values()
	)
	assert STAGE_DISPATCH["normalize.taper_blended"].kind is DispatchKind.PROCESS
	assert STAGE_DISPATCH["recon.gridrec"].kind is DispatchKind.ENGINE
	assert STAGE_DISPATCH["read.hdf5_dxchange"].kind is DispatchKind.THREAD
	with pytest.raises(ValueError, match="requires an explicit dispatch classification"):
		_build_stage_dispatch({**STAGE_REGISTRY, "test.unclassified": _AddSinogramStage})


def test_pool_is_persistent_across_stage_submissions():
	executor = PipelineExecutor(1, context="spawn")
	try:
		first = executor.submit_all(_worker_pid, [()])
		second = executor.submit_all(_worker_pid, [()])
		assert first == second
	finally:
		executor.close()


def test_completed_kernel_error_does_not_discard_healthy_pool():
	executor = PipelineExecutor(1, context="spawn")
	try:
		before = executor.submit_all(_worker_pid, [()])
		with pytest.raises(ValueError, match="deterministic kernel failure"):
			executor.submit_all(_raise_deterministic_error, [()])
		after = executor.submit_all(_worker_pid, [()])
		assert after == before
	finally:
		executor.close()


def test_ndarray_bounded_segment_checks_cancellation_before_each_kernel():
	state = PipelineState(
		np.ones((2, 3, 4), dtype=np.float32),
		_scan(theta=2, rows=3, columns=4), _recon(),
		layout=ProjOrder, domain=DataDomain.ATTENUATION,
	)
	checkpoints = []
	run_bounded_sinogram_segment(
		state,
		[_AddSinogramStage(), _AddSinogramStage()],
		checkpoint=lambda: checkpoints.append(None),
	)
	assert len(checkpoints) == 6


def test_engine_internal_core_budget_is_scoped_to_the_call(monkeypatch):
	stage = VoCenterStage(ncore=7)
	seen = []

	def capture(state):
		seen.append(stage.kwargs["ncore"])
		return state

	monkeypatch.setattr(stage, "run", capture)
	state = PipelineState(
		np.ones((4, 2, 8), dtype=np.float32), _scan(theta=4), _recon(),
		layout=ProjOrder, domain=DataDomain.TRANSMISSION,
	)
	PipelineExecutor(2).run_stage(state, stage)
	assert seen == [2]
	assert stage.kwargs["ncore"] == 7


def test_process_normalization_receives_only_handles_and_matches_serial(monkeypatch):
	scan = _scan()
	shape = ProjOrder(Theta=6, Y=2, X=8)
	with PipelineRunContext(max_workers=2) as context:
		raw = context.storage.allocate(shape, np.uint16, role="raw_staging")
		raw.array()[...] = 10
		state = PipelineState(
			raw, scan, _recon(), storage=context.storage, executor=context.executor,
			layout=ProjOrder, domain=DataDomain.RAW,
			artifacts={"projection_start": 2, "projection_count": 10},
		)
		captured = []
		original = context.executor.submit_all

		def inspect(function, arguments):
			items = list(arguments)
			captured.extend(items)
			return original(function, items)

		monkeypatch.setattr(context.executor, "submit_all", inspect)
		result = context.executor.run_stage(
			state, WeightedNormalizeStage(order="projection"),
		)

		assert all(isinstance(task[0], ManagedBuffer) for task in captured)
		assert all(isinstance(task[4].artifacts["flat_fields"], ManagedBuffer) for task in captured)
		assert all("flat_fields" not in task[4].scan.metadata for task in captured)
		expected = np.empty((6, 2, 8), dtype=np.float32)
		for index in range(6):
			global_index = index + 2
			expected[index] = 10 / (
				20 * (1 - global_index / 10) + 40 * (global_index / 10)
			)
		np.testing.assert_allclose(np.asarray(result.working_array), expected)


def test_process_phase_chain_matches_intentionally_coupled_stage():
	values = np.full((4, 2, 8), 0.8, dtype=np.float32)
	values[:, :, 3:5] = 0.4
	scan = _scan(theta=4)
	expected = values.copy()
	PaganinStage(alpha=0.06, order="projection").run(PipelineState(
		expected, scan, _recon(), layout=ProjOrder, domain=DataDomain.TRANSMISSION,
	))
	with PipelineRunContext(max_workers=2) as context:
		working = context.storage.allocate(
			ProjOrder(Theta=4, Y=2, X=8), np.float32, role="normalized",
		)
		working.array()[...] = values
		state = PipelineState(
			working, scan, _recon(), storage=context.storage, executor=context.executor,
			layout=ProjOrder, domain=DataDomain.TRANSMISSION,
		)
		state = context.executor.run_stage(state, PaganinFilterStage(alpha=0.06, order="projection"))
		state = context.executor.run_stage(state, NegativeLogStage(order="projection"))
		np.testing.assert_allclose(np.asarray(state.working_array), expected, rtol=1e-5, atol=1e-6)


def test_process_bac_filter_then_negative_log_matches_coupled_stage(
	tmp_path, monkeypatch,
):
	package = tmp_path / "hotopy"
	package.mkdir()
	(package / "__init__.py").write_text("", encoding="utf-8")
	(package / "holo.py").write_text(
		"""import numpy as np

class BronnikovAidedCorrection:
	def __init__(self, shape, alpha, gamma, device):
		self.alpha = alpha
		self.gamma = gamma

	def __call__(self, image):
		qy = 2 * np.pi * np.fft.fftfreq(image.shape[0])
		qx = 2 * np.pi * np.fft.fftfreq(image.shape[1])
		w2 = np.add.outer(qy ** 2, qx ** 2)
		correction = -np.fft.ifft2(
			np.fft.fft2(image - 1) * w2 / (w2 + self.alpha)
		).real
		return -np.log(image / (1 - self.gamma * correction))
""",
		encoding="utf-8",
	)
	monkeypatch.syspath_prepend(str(tmp_path))
	monkeypatch.delitem(sys.modules, "hotopy.holo", raising=False)
	monkeypatch.delitem(sys.modules, "hotopy", raising=False)
	values = np.full((4, 8, 8), 0.8, dtype=np.float32)
	values[:, 2:6, 2:6] = 0.4
	scan = _scan(theta=4, rows=8, columns=8)
	expected = values.copy()
	BronnikovAidedCorrectionStage(alpha=2, gamma=1.25).run(PipelineState(
		expected, scan, _recon(), layout=ProjOrder, domain=DataDomain.TRANSMISSION,
	))
	with PipelineRunContext(max_workers=2) as context:
		working = context.storage.allocate(ProjOrder(4, 8, 8), np.float32, role="normalized")
		working.array()[...] = values
		state = PipelineState(
			working, scan, _recon(), storage=context.storage, executor=context.executor,
			layout=ProjOrder, domain=DataDomain.TRANSMISSION,
		)
		state = context.executor.run_stage(
			state, BronnikovAidedCorrectionFilterStage(alpha=2, gamma=1.25),
		)
		state = context.executor.run_stage(state, NegativeLogStage(epsilon=None))
		np.testing.assert_allclose(np.asarray(state.working_array), expected, rtol=2e-6, atol=2e-7)


def test_process_beam_tracking_matches_serial_and_merges_qc():
	scan = _scan(theta=6, rows=64, columns=64)
	gradient = np.linspace(0, 1, 64, dtype=np.float32)[None, :]
	scan.metadata["flat_fields"][0] += gradient
	scan.metadata["flat_fields"][1] += 2 * gradient
	values = np.full((6, 64, 64), 10, dtype=np.uint16)
	params = {"schedule": {"kind": "linear", "alpha": 0, "beta": 1}}
	expected_state = BeamTrackingNormalizeStage(**params).run(PipelineState(
		values.copy(), scan, _recon(), layout=ProjOrder, domain=DataDomain.RAW,
	))
	with PipelineRunContext(max_workers=2) as context:
		working = context.storage.allocate(ProjOrder(6, 64, 64), np.uint16, role="raw_staging")
		working.array()[...] = values
		state = PipelineState(
			working, scan, _recon(), storage=context.storage, executor=context.executor,
			layout=ProjOrder, domain=DataDomain.RAW,
		)
		result = context.executor.run_stage(state, BeamTrackingNormalizeStage(**params))
		np.testing.assert_allclose(
			np.asarray(result.working_array), np.asarray(expected_state.working_array),
		)
		assert result.artifacts["beam_tracking_qc"]["projection_count"] == 6


@pytest.mark.parametrize("stage", [
	LayoutConvertStage(target="sinogram", reason="dispatch parity"),
	AffineSkewMovementStage(
		transition_to="sinogram", pad_x=1, skew_degrees=1.5,
		movement_enabled=True, movement_x=0.25,
	),
	HelicalIndexRemapStage(
		pitch_px_per_projection=0.25, crop_valid=True, order="projection",
	),
])
def test_out_of_place_process_fills_match_serial_and_use_handles(stage, monkeypatch):
	values = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
	scan = _scan(theta=4, rows=6, columns=8)
	expected = stage.run(PipelineState(
		values.copy(), scan, _recon(), layout=ProjOrder,
		domain=DataDomain.TRANSMISSION,
	))
	with PipelineRunContext(max_workers=2) as context:
		working = context.storage.allocate(ProjOrder(4, 6, 8), np.float32, role="normalized")
		working.array()[...] = values
		state = PipelineState(
			working, scan, _recon(), storage=context.storage, executor=context.executor,
			layout=ProjOrder, domain=DataDomain.TRANSMISSION,
		)
		captured = []
		original = context.executor.submit_all

		def inspect(function, arguments):
			items = list(arguments)
			captured.extend(items)
			return original(function, items)

		monkeypatch.setattr(context.executor, "submit_all", inspect)
		result = context.executor.run_stage(state, stage)
		assert isinstance(result.working_array, ManagedBuffer)
		assert all(isinstance(task[0], ManagedBuffer) for task in captured)
		assert all(isinstance(task[1], ManagedBuffer) for task in captured)
		np.testing.assert_allclose(
			np.asarray(result.working_array), np.asarray(expected.working_array),
		)
		if isinstance(stage, AffineSkewMovementStage):
			assert result.source_window is not None
			assert result.source_window.pos == expected.source_window.pos
			assert result.source_window.shape == expected.source_window.shape
			assert result.source_window.source_pos == expected.source_window.source_pos
			assert result.source_window.allocation_shape == expected.source_window.allocation_shape
			assert result.source_window.allocation_shape.to(result.layout) == result.buffer_spec.shape


def test_bounded_process_child_mutation_is_visible_in_parent():
	with PipelineRunContext(max_workers=2) as context:
		working = context.storage.allocate(ProjOrder(4, 3, 5), np.float32, role="normalized")
		working.array().fill(2)
		state = PipelineState(
			working, _scan(theta=4, rows=3, columns=5), _recon(),
			storage=context.storage, executor=context.executor,
			layout=ProjOrder, domain=DataDomain.TRANSMISSION,
		)
		run_bounded_sinogram_segment(
			state, [_AddSinogramStage()], executor=context.executor,
		)
		np.testing.assert_array_equal(working.array(), 5)


def test_tiff_process_dispatch_passes_handles_not_slices(tmp_path, monkeypatch):
	with PipelineRunContext(max_workers=2) as context:
		working = context.storage.allocate(
			ReconOrder(Y=2, Z=4, X=4), np.float32, role="reconstruction",
		)
		working.array()[...] = 1
		state = PipelineState(
			working, _scan(theta=2, rows=2, columns=4), _recon(tmp_path),
			storage=context.storage, executor=context.executor,
			layout=ReconOrder, domain=DataDomain.RECON,
		)
		captured = []

		def serial_capture(function, arguments):
			items = list(arguments)
			captured.extend(items)
			for args in items:
				function(*args)
			return [None] * len(items)

		monkeypatch.setattr(context.executor, "submit_all", serial_capture)
		TIFFSlicesWriteStage(processes=2).run(state)

		assert captured
		assert all(isinstance(task[0], ManagedBuffer) for task in captured)
		assert not any(isinstance(item, np.ndarray) for task in captured for item in task)


def test_timeout_kills_writer_and_recreates_pool():
	with PipelineRunContext(max_workers=1) as context:
		working = context.storage.allocate((1,), np.int64, role="test")
		executor = PipelineExecutor(1, timeout_seconds=0.2, context="spawn")
		try:
			with pytest.raises(WorkerTimeoutError):
				executor.submit_all(_write_forever, [(working,)])
			stopped = int(working.array()[0])
			time.sleep(0.15)
			assert int(working.array()[0]) == stopped

			# Pool recreation includes a fresh interpreter startup; the short
			# deadline above exists only to exercise forced termination.
			executor.timeout_seconds = 2
			executor.submit_all(_set_value, [(working, 99)])
			assert working.array()[0] == 99
		finally:
			executor.close()
