import ast
import inspect
import multiprocessing
import os
from pathlib import Path
import pickle
import signal
import time

import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import PipelineState, Stage, StageContract, working_view
from lftomo.storage import ManagedBuffer, PipelineRunContext, StorageBackend
from lftomo.util.mem import FileBackedNPArray, SharedNPArray, SharedNPManager
from lftomo.util.shapes import ProjOrder


def _scan() -> Scan:
	return Scan(
		energy_keV=24,
		pixel_size_um=1.3,
		n_projections=2,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1.3),
		acquisition=Acquisition(n_projections=2, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


def _recon() -> Recon:
	return Recon(pipeline=[StageConfig(kind="test", algorithm="increment")])


class _IncrementStage(Stage):
	name = "test.increment"
	contract = StageContract(mutates_input=True)

	def run(self, state):
		working_view(state)[...] += 1
		return state


def _run_increment(buffer, scan, recon):
	_IncrementStage().run(PipelineState(buffer, scan, recon))


def _crashing_run(connection):
	if hasattr(os, "setsid"):
		os.setsid()
	with PipelineRunContext() as context:
		buffer = context.storage.allocate((2, 3), np.float32)
		connection.send(buffer.name)
		connection.close()
		while True:
			time.sleep(1)


def _segment_exists(name: str) -> bool:
	from multiprocessing.shared_memory import SharedMemory

	shared_memory_path = Path("/dev/shm") / name.lstrip("/")
	if shared_memory_path.parent.is_dir():
		return shared_memory_path.exists()
	if "track" not in inspect.signature(SharedMemory).parameters:
		raise RuntimeError(
			"safe shared-memory inspection requires /dev/shm or Python 3.13+"
		)
	try:
		segment = SharedMemory(name, track=False)
	except FileNotFoundError:
		return False
	segment.close()
	return True


def test_existing_shared_types_directly_implement_storage_contracts():
	manager = SharedNPManager()
	manager.start()
	try:
		assert isinstance(manager, StorageBackend)
		buffer = manager.allocate((2, 3), np.float32)
		assert isinstance(buffer, ManagedBuffer)
		buffer.array()[...] = np.arange(6, dtype=np.float32).reshape(2, 3)
		fork = manager.fork(buffer, label="test-copy")
		np.testing.assert_array_equal(fork.array(), buffer.array())
		assert fork.name != buffer.name
	finally:
		manager.shutdown()


def test_pickled_handle_is_always_a_non_owning_attachment():
	name = f"lftomo-pickle-{os.getpid()}-{time.time_ns()}"
	owner = SharedNPArray(name, (2, 3), np.float32, allocate=True)
	try:
		clone = pickle.loads(pickle.dumps(owner))
		assert owner.owns_unlink
		assert not clone.owns_unlink
		clone.array()[0, 0] = 7
		clone.close()
		assert owner.array()[0, 0] == 7
	finally:
		owner.shutdown()


def test_file_backed_buffer_reopens_read_only_in_each_process(tmp_path):
	path = tmp_path / "working.npy"
	expected = np.arange(12, dtype=np.float32).reshape(3, 4)
	np.save(path, expected)
	buffer = FileBackedNPArray(path)
	clone = pickle.loads(pickle.dumps(buffer))
	try:
		np.testing.assert_array_equal(clone[1:].array(), expected[1:])
		assert not clone.array().flags.writeable
		with pytest.raises(ValueError, match="read-only"):
			clone.array()[0, 0] = -1
	finally:
		buffer.close()
		clone.close()


def test_real_stage_kernel_in_child_mutates_manager_buffer_in_parent():
	spawn = multiprocessing.get_context("spawn")
	name = None
	with PipelineRunContext() as context:
		buffer = context.storage.allocate(ProjOrder(Theta=2, Y=3, X=4), np.float32)
		name = buffer.name
		buffer.array()[...] = 2
		process = spawn.Process(target=_run_increment, args=(buffer, _scan(), _recon()))
		process.start()
		process.join(timeout=10)
		assert process.exitcode == 0
		np.testing.assert_array_equal(buffer.array(), np.full((2, 3, 4), 3, dtype=np.float32))
	assert name is not None
	assert not _segment_exists(name)


def test_run_context_unlinks_segments_after_exception():
	name = None
	with pytest.raises(RuntimeError, match="deliberate"):
		with PipelineRunContext() as context:
			buffer = context.storage.allocate((2, 3), np.float32)
			name = buffer.name
			assert _segment_exists(name)
			raise RuntimeError("deliberate")
	assert name is not None
	assert not _segment_exists(name)


def test_run_context_rejects_sequential_reentry_after_owned_shutdown():
	context = PipelineRunContext()
	with context:
		pass
	with pytest.raises(RuntimeError, match="cannot be re-entered after exit"):
		context.__enter__()


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGKILL])
def test_killed_process_group_does_not_leak_manager_segments(stop_signal):
	if not hasattr(os, "killpg"):
		pytest.skip("process-group crash cleanup requires POSIX")
	spawn = multiprocessing.get_context("spawn")
	parent, child = spawn.Pipe(duplex=False)
	process = spawn.Process(target=_crashing_run, args=(child,))
	process.start()
	child.close()
	name = parent.recv()
	assert _segment_exists(name)
	os.killpg(process.pid, stop_signal)
	process.join(timeout=10)
	assert not process.is_alive()
	deadline = time.monotonic() + 10
	while _segment_exists(name) and time.monotonic() < deadline:
		time.sleep(0.1)
	assert not _segment_exists(name)


def test_pipeline_has_no_direct_allocators_or_legacy_state_coercions():
	root = Path(__file__).parents[2] / "lftomo"
	legacy = []
	direct_allocators = []
	stage_paths = {
		*(root / "pipeline").rglob("*.py"),
		root / "evaluation" / "stage.py",
	}
	for path in root.rglob("*.py"):
		text = path.read_text(encoding="utf-8")
		if "np.asarray(state.working_array" in text:
			legacy.append(path.relative_to(root))
		if path not in stage_paths:
			continue
		for node in ast.walk(ast.parse(text)):
			if not isinstance(node, ast.Call):
				continue
			name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
			if name == "SharedNPArray" and any(
				keyword.arg == "allocate" and isinstance(keyword.value, ast.Constant)
				and keyword.value.value is True for keyword in node.keywords
			):
				direct_allocators.append(path.relative_to(root))
	assert legacy == []
	assert direct_allocators == []
