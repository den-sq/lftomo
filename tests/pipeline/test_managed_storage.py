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
from lftomo.pipeline import PipelineState, Stage, StageContract
from lftomo.storage import ManagedBuffer, PipelineRunContext, StorageBackend
from lftomo.util.mem import FileBackedNPArray, SharedNPArray, SharedNPManager
from lftomo.util.shapes import ImgSlice, ProjOrder


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
		np.asarray(state.working_array)[...] += 1
		return state


class _UntypedBuffer(ManagedBuffer):
	shape = (2, 3)
	dtype = np.dtype(np.float32)

	def array(self):
		return np.empty(self.shape, dtype=self.dtype)

	def __getitem__(self, view):
		return self


def _run_increment(buffer, scan, recon):
	_IncrementStage().run(PipelineState(buffer, scan, recon))


def _crashing_run(connection):
	if hasattr(os, "setsid"):
		os.setsid()
	with PipelineRunContext() as context:
		buffer = context.storage.allocate(ImgSlice(2, 3), np.float32)
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
		buffer = manager.allocate(ImgSlice(2, 3), np.float32)
		assert isinstance(buffer, ManagedBuffer)
		buffer.array()[...] = np.arange(6, dtype=np.float32).reshape(2, 3)
		fork = manager.fork(buffer, shape=buffer.shape, label="test-copy")
		np.testing.assert_array_equal(fork.array(), buffer.array())
		assert fork.name != buffer.name
	finally:
		manager.shutdown()


def test_pipeline_state_rejects_an_untyped_managed_buffer_at_the_storage_seam():
	with pytest.raises(TypeError, match="ManagedBuffer.shape must be a DataShape"):
		PipelineState(_UntypedBuffer(), _scan(), _recon())


def test_shared_basic_view_keeps_named_extents_without_loading_shape_from_numpy():
	with PipelineRunContext() as context:
		buffer = context.storage.allocate(ProjOrder(Theta=4, Y=5, X=6), np.float32)
		view = buffer[1:3, :, 2:6:2]

		assert view.shape == ProjOrder(Theta=2, Y=5, X=2)


def test_pickled_handle_is_always_a_non_owning_attachment():
	name = f"lftomo-pickle-{os.getpid()}-{time.time_ns()}"
	owner = SharedNPArray(name, ImgSlice(2, 3), np.float32, allocate=True)
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


def test_file_backed_buffer_preserves_typed_shape_through_views_and_pickle(tmp_path):
	path = tmp_path / "working.npy"
	np.save(path, np.arange(24, dtype=np.float32).reshape(2, 3, 4))
	buffer = FileBackedNPArray(path, ProjOrder(2, 3, 4))
	clone = pickle.loads(pickle.dumps(buffer[:, 1:2, :]))
	try:
		assert clone.shape == ProjOrder(2, 1, 4)
		np.testing.assert_array_equal(clone.array(), buffer.array()[:, 1:2, :])
	finally:
		buffer.close()
		clone.close()


def test_file_backed_buffer_rejects_datashape_rank_mismatch(tmp_path):
	path = tmp_path / "working.npy"
	np.save(path, np.zeros((2, 3), dtype=np.float32))

	with pytest.raises(ValueError, match="rank disagrees"):
		FileBackedNPArray(path, shape=ProjOrder(2, 3, 1))


def _assert_numpy_protocol(buffer, expected, *, writeable):
	view = np.asarray(buffer)
	assert view.dtype == expected.dtype
	assert view.flags.writeable is writeable
	assert np.shares_memory(view, buffer.array())
	np.testing.assert_array_equal(view, expected)

	same_dtype = buffer.__array__(dtype=expected.dtype, copy=False)
	assert np.shares_memory(same_dtype, buffer.array())
	with pytest.raises(ValueError, match="dtype conversion.*copy=False"):
		buffer.__array__(dtype=np.float64, copy=False)
	if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
		with pytest.raises(ValueError, match="dtype conversion.*copy=False"):
			np.asarray(buffer, dtype=np.float64, copy=False)
	private = buffer.__array__(copy=True)
	assert not np.shares_memory(private, buffer.array())
	np.testing.assert_array_equal(private, expected)


def test_managed_buffers_expose_an_honest_no_copy_numpy_protocol(tmp_path):
	expected = np.arange(6, dtype=np.float32).reshape(2, 3)
	with PipelineRunContext() as context:
		shared = context.storage.allocate(ImgSlice(*expected.shape), expected.dtype)
		shared.array()[...] = expected
		_assert_numpy_protocol(shared, expected, writeable=True)

	path = tmp_path / "working.npy"
	np.save(path, expected)
	mapped = FileBackedNPArray(path)
	try:
		_assert_numpy_protocol(mapped, expected, writeable=False)
	finally:
		mapped.close()


def test_plain_ndarray_boundary_conversion_is_identity():
	working = np.arange(6, dtype=np.float32).reshape(2, 3)
	assert np.asarray(working) is working


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
			buffer = context.storage.allocate(ImgSlice(2, 3), np.float32)
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


def _owned_python_paths(repository: Path) -> list[Path]:
	"""Return Python files from repository-owned source and test roots only."""

	paths = []
	for name in ("lftomo", "tests", "benchmarks", "B10Scripts", "examples"):
		root = repository / name
		if root.is_dir():
			paths.extend(root.rglob("*.py"))
	return sorted(paths)


def test_working_array_gate_ignores_non_source_python_trees(tmp_path):
	owned = tmp_path / "lftomo" / "owned.py"
	owned.parent.mkdir()
	owned.write_text("value = 1\n", encoding="utf-8")
	vendored = tmp_path / ".venv" / "site-packages" / "vendored.py"
	vendored.parent.mkdir(parents=True)
	vendored.write_text("value = 2\n", encoding="utf-8")

	assert _owned_python_paths(tmp_path) == [owned]


def test_pipeline_has_one_working_array_boundary_idiom():
	repository = Path(__file__).parents[2]
	root = repository / "lftomo"
	legacy_accessor = "_".join(("working", "view"))
	legacy = []
	boundary_bypasses = []
	direct_allocators = []
	stage_paths = {
		*(root / "pipeline").rglob("*.py"),
		root / "evaluation" / "stage.py",
	}
	for path in _owned_python_paths(repository):
		text = path.read_text(encoding="utf-8")
		tree = ast.parse(text)
		if legacy_accessor in text:
			legacy.append(path.relative_to(repository))
		for node in ast.walk(tree):
			if not isinstance(node, ast.Call):
				continue
			working_argument = (
				node.args
				and isinstance(node.args[0], ast.Attribute)
				and node.args[0].attr == "working_array"
			)
			if (
				isinstance(node.func, ast.Attribute)
				and node.func.attr == "array"
				and isinstance(node.func.value, ast.Attribute)
				and node.func.value.attr == "working_array"
			):
				boundary_bypasses.append(path.relative_to(repository))
			if (
				working_argument
				and isinstance(node.func, ast.Name)
				and node.func.id == "buffer_view"
			):
				boundary_bypasses.append(path.relative_to(repository))
			if (
				working_argument
				and isinstance(node.func, ast.Attribute)
				and isinstance(node.func.value, ast.Name)
				and node.func.value.id == "np"
				and node.func.attr == "array"
			):
				boundary_bypasses.append(path.relative_to(repository))
		if path not in stage_paths:
			continue
		for node in ast.walk(tree):
			if not isinstance(node, ast.Call):
				continue
			name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
			if name == "SharedNPArray" and any(
				keyword.arg == "allocate" and isinstance(keyword.value, ast.Constant)
				and keyword.value.value is True for keyword in node.keywords
			):
				direct_allocators.append(path.relative_to(root))
	for path in (repository / "docs").rglob("*.md"):
		if legacy_accessor in path.read_text(encoding="utf-8"):
			legacy.append(path.relative_to(repository))
	assert legacy == []
	assert boundary_bypasses == []
	assert direct_allocators == []
