import ast
from contextlib import nullcontext
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

import numpy as np
import psutil
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import DataDomain, PipelineState, Stage, StageContract
from lftomo.pipeline.core import BoundedWorkRequest, BufferSpec, ProjectionStage, StageParams
from lftomo.pipeline.dispatch import DispatchKind, DispatchSpec
from lftomo.storage import LocalArray, ManagedBuffer, PipelineRunContext, StorageBackend
from lftomo.util.mem import FileBackedNPArray, SharedNPArray, SharedNPManager
from lftomo.util.shapes import ImgSlice, ProjOrder, SinoOrder


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


class _ProjectionScratchStage(ProjectionStage):
	name = "test.projection_scratch"
	dispatch = DispatchSpec(DispatchKind.PROCESS, kernel="run_projection_kernel", per_projection=True)
	contract = StageContract(mutates_input=True)

	class Params(StageParams):
		fail: bool = False

	def bounded_work_requests(self, input_spec):
		return (BoundedWorkRequest(
			"_test_work", BufferSpec(input_spec.shape, np.float32), "test_projection_work",
			required_for_fallback=True,
		),)

	def run_projection(self, image, state):
		handle = state.artifacts["_test_work"]
		assigned = handle[state.work_index(handle.shape)]
		work = assigned.array()[0]
		np.multiply(image, 2, out=work)
		if self.params.fail:
			raise RuntimeError("projection kernel failed")
		image[...] = work


@pytest.mark.system
@pytest.mark.parametrize("execution", ["local", "managed_serial", "process"])
@pytest.mark.parametrize("fail", [False, True])
def test_projection_declared_work_is_retired_after_dispatch_even_on_failure(execution, fail):
	managed = execution != "local"
	with PipelineRunContext(max_workers=2) if managed else nullcontext() as context:
		shape = ProjOrder(2, 3, 4)
		buffer = context.storage.allocate(shape, np.float32) if managed else LocalArray(
			np.empty(shape.as_tuple(), dtype=np.float32), shape,
		)
		buffer.array().fill(7)
		state = PipelineState(
			buffer, _scan(), _recon(), storage=context.storage if managed else None,
			executor=context.executor if managed else None,
		)
		stage = _ProjectionScratchStage(fail=fail)
		run = (lambda: context.executor.run_stage(state, stage)) if execution == "process" else (
			lambda: stage.run(state)
		)
		if fail:
			with pytest.raises(RuntimeError, match="projection kernel failed"):
				run()
		else:
			result = run()
			np.testing.assert_array_equal(np.asarray(result.working_array), 14)
		assert "_test_work" not in state.artifacts
		if managed:
			work_events = [event for event in context.storage.allocation_events
				if event["role"] == "test_projection_work"]
			assert [event["action"] for event in work_events] == ["allocate", "retire"]


@pytest.mark.system
@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
def test_zinger_work_capacity_and_results_do_not_depend_on_process_count(layout):
	from lftomo.pipeline.stages.filter import ZingerFilterStage

	theta, rows, columns = np.indices((3, 7, 11), dtype=np.float32)
	values = np.exp(-0.4 * np.exp(-((columns - 5) ** 2 + (rows - 3) ** 2) / 20)
		* (1 + 0.1 * np.sin(theta))).astype(np.float32)
	for index in ((0, 0, 0), (1, 3, 5), (2, 6, 10)):
		values[index] *= 4
	if layout is SinoOrder:
		values = values.transpose(1, 0, 2).copy()
	shape = layout(*values.shape)
	serial = PipelineState(
		LocalArray(values.copy(), shape), _scan(), _recon(), domain=DataDomain.TRANSMISSION,
	)
	ZingerFilterStage().run(serial)
	expected = np.asarray(serial.working_array).copy()
	assert np.count_nonzero(expected != values) == 3
	assert not any(isinstance(value, (LocalArray, ManagedBuffer)) for value in serial.artifacts.values())
	reports = []
	for workers in (1, 2):
		with PipelineRunContext(max_workers=workers) as context:
			buffer = context.storage.allocate(shape, values.dtype)
			buffer.array()[...] = values
			state = PipelineState(
				buffer, _scan(), _recon(), storage=context.storage, executor=context.executor,
				domain=DataDomain.TRANSMISSION,
			)
			result = context.executor.run_stage(state, ZingerFilterStage())
			np.testing.assert_array_equal(np.asarray(result.working_array), expected)
			assert not any(isinstance(value, (LocalArray, ManagedBuffer))
				for value in result.artifacts.values())
			report, = result.artifacts["bounded_work_allocations"]
			reports.append(report)
			assert report["unit_axis"] == "Theta"
			assert report["units"] == 3
			assert {work["role"]: work["bytes"] for work in report["buffers"]} == {
				"zinger_median": 3 * 7 * 11 * 4,
				"zinger_threshold": 3 * 11 * 4,
				"zinger_mask": 3 * 11,
			}
			for role in ("zinger_median", "zinger_threshold", "zinger_mask"):
				events = [event for event in context.storage.allocation_events if event["role"] == role]
				assert [event["action"] for event in events] == ["allocate", "retire"]
	assert reports[0] == reports[1]


class _UntypedBuffer(ManagedBuffer):
	shape = (2, 3)
	dtype = np.dtype(np.float32)

	def array(self):
		return np.empty(self.shape, dtype=self.dtype)

	def __getitem__(self, view):
		return self


def _run_increment(buffer, scan, recon):
	_IncrementStage().run(PipelineState(buffer, scan, recon))


def _crashing_run(connection, directory, retain):
	if hasattr(os, "setsid"):
		os.setsid()
	with PipelineRunContext(intermediate_root=directory, retain_intermediates=retain) as context:
		buffer = context.storage.allocate(ImgSlice(2, 3), np.float32)
		intermediate = context.create_intermediate(ProjOrder(2, 3, 4), np.float32)
		intermediate.array()[...] = 17
		intermediate.flush()
		connection.send((buffer.name, intermediate.path))
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
	with pytest.raises(TypeError, match="TypedBuffer.shape must be a DataShape"):
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


def test_writable_file_backed_views_reopen_in_workers_and_preserve_existing_files(tmp_path):
	path = tmp_path / "working.npy"
	expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
	buffer = FileBackedNPArray.create(path, ProjOrder(2, 3, 4), np.float32)
	try:
		buffer.array()[...] = expected
		buffer.flush()
		view = buffer[:, 1:2, :]
		clone = pickle.loads(pickle.dumps(view))
		try:
			assert clone.shape == ProjOrder(2, 1, 4)
			_assert_numpy_protocol(clone, expected[:, 1:2, :], writeable=True)
			process = multiprocessing.get_context("spawn").Process(
				target=_run_increment, args=(clone, _scan(), _recon()),
			)
			process.start()
			process.join(timeout=10)
			assert process.exitcode == 0
		finally:
			clone.close()
			view.close()
		buffer.close()
		expected[:, 1:2, :] += 1
		np.testing.assert_array_equal(buffer.array(), expected)
		with pytest.raises(FileExistsError):
			FileBackedNPArray.create(path, ProjOrder(2, 3, 4), np.float32)
		np.testing.assert_array_equal(np.load(path), expected)
	finally:
		buffer.close()


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


@pytest.mark.system
@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_run_context_intermediates_are_fresh_with_explicit_retention(tmp_path, retain, fail):
	root = tmp_path / "intermediates"
	previous_directory = None
	for _ in range(2):
		context = PipelineRunContext(intermediate_root=root, retain_intermediates=retain)
		assert context.intermediate_directory is None
		assert context.intermediate_paths == ()
		try:
			with context:
				buffer = context.create_intermediate(ProjOrder(2, 3, 4), np.float32, name="working")
				directory = context.intermediate_directory
				assert directory.parent == root
				assert directory != previous_directory
				assert context.intermediate_paths == (buffer.path,)
				buffer.array()[...] = 7
				with pytest.raises(FileExistsError):
					context.create_intermediate(ProjOrder(2, 3, 4), np.float32, name="working")
				if fail:
					raise RuntimeError("deliberate")
		except RuntimeError as error:
			assert fail and str(error) == "deliberate"
		assert directory.exists() is retain
		assert context.intermediate_paths == (buffer.path,)
		if retain:
			np.testing.assert_array_equal(np.load(buffer.path), np.full((2, 3, 4), 7))
		previous_directory = directory
	with pytest.raises(RuntimeError, match="active PipelineRunContext"):
		context.create_intermediate(ProjOrder(2, 3, 4), np.float32)


def test_run_context_intermediate_directory_is_lazy_and_names_stay_inside_it(tmp_path):
	root = tmp_path / "unused"
	with PipelineRunContext(intermediate_root=root) as context:
		assert context.intermediate_directory is None
		with pytest.raises(ValueError, match="filename without directories"):
			context.create_intermediate(ProjOrder(2, 3, 4), np.float32, name="../escaped")
	assert not root.exists()


def test_run_context_cleans_intermediates_when_executor_shutdown_fails(tmp_path, monkeypatch):
	context = PipelineRunContext(intermediate_root=tmp_path)
	with pytest.raises(RuntimeError, match="executor shutdown failed"):
		with context:
			buffer = context.create_intermediate(ProjOrder(2, 3, 4), np.float32)
			mapping = buffer.array()
			shared = context.storage.allocate(ImgSlice(2, 3), np.float32)

			def fail_shutdown():
				raise RuntimeError("executor shutdown failed")

			monkeypatch.setattr(context.executor, "close", fail_shutdown)
	assert mapping._mmap.closed
	assert not context.intermediate_directory.exists()
	assert not _segment_exists(shared.name)


@pytest.mark.system
@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize("retain", [False, True])
def test_killed_process_group_cleans_owned_storage_and_honors_retention(tmp_path, stop_signal, retain):
	if not hasattr(os, "killpg"):
		pytest.skip("process-group crash cleanup requires POSIX")
	spawn = multiprocessing.get_context("spawn")
	parent, child = spawn.Pipe(duplex=False)
	user_file = tmp_path / "unrelated.txt"
	user_file.write_text("keep")
	process = spawn.Process(target=_crashing_run, args=(child, tmp_path, retain))
	process.start()
	child.close()
	name, intermediate = parent.recv()
	assert _segment_exists(name)
	assert intermediate.exists()
	os.killpg(process.pid, stop_signal)
	process.join(timeout=10)
	assert not process.is_alive()
	deadline = time.monotonic() + 10
	while (_segment_exists(name) or (not retain and intermediate.parent.exists())) and time.monotonic() < deadline:
		time.sleep(0.1)
	assert not _segment_exists(name)
	assert intermediate.parent.exists() is retain
	if retain:
		np.testing.assert_array_equal(np.load(intermediate), np.full((2, 3, 4), 17))
	assert user_file.read_text() == "keep"


@pytest.mark.system
@pytest.mark.parametrize("start_method", ["default", "spawn"])
def test_fatal_owner_exit_releases_manager_workers_and_pipes(tmp_path, start_method):
	if sys.platform != "linux":
		pytest.skip("fatal-owner cleanup inspection uses Linux shared-memory paths")
	owner = subprocess.Popen(
		[sys.executable, "-m", "tests.pipeline.process_helpers", start_method, str(tmp_path)],
		cwd=Path(__file__).parents[2],
		stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
		start_new_session=True,
	)
	owner_process = psutil.Process(owner.pid)
	descendants = {}
	resources = {}
	resource_path = tmp_path / "resources.json"

	def running(process):
		try:
			return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
		except psutil.NoSuchProcess:
			return False

	try:
		deadline = time.monotonic() + 30
		while "worker_pids" not in resources and time.monotonic() < deadline:
			for process in owner_process.children(recursive=True):
				descendants[process.pid] = process
			if resource_path.exists():
				resources = json.loads(resource_path.read_text(encoding="utf-8"))
			assert owner.poll() is None, "standalone run owner exited before workers were ready"
			time.sleep(0.01)
		assert "worker_pids" in resources, "standalone run workers did not become ready"
		manager = psutil.Process(resources["manager_pid"])
		workers = [psutil.Process(pid) for pid in resources["worker_pids"]]
		for process in owner_process.children(recursive=True):
			descendants[process.pid] = process
		assert len(set(resources["worker_pids"])) == 2
		assert all(running(process) for process in [manager, *workers])
		expected_method = (
			multiprocessing.get_all_start_methods()[0] if start_method == "default" else "spawn"
		)
		assert resources["manager_start_method"] == expected_method
		if expected_method == "forkserver":
			assert manager.ppid() != owner.pid
		assert _segment_exists(resources["segment"])

		# Kill only the owning process: each detached child must notice its death.
		killed_at = time.monotonic()
		owner.send_signal(signal.SIGBUS)
		deadline = time.monotonic() + 10
		while (
			any(running(process) for process in [manager, *workers])
			or _segment_exists(resources["segment"])
		) and time.monotonic() < deadline:
			time.sleep(0.01)
		assert not any(running(process) for process in [manager, *workers])
		assert not _segment_exists(resources["segment"])
		_stdout, stderr = owner.communicate(timeout=max(0.1, deadline - time.monotonic()))
		print(f"{start_method}: owner-exit cleanup and pipe EOF in {time.monotonic() - killed_at:.3f}s")
		assert owner.returncode == -signal.SIGBUS
		if stderr:
			print(stderr, end="")
		assert all(
			not (Path("/dev/shm") / f"sem.{name.lstrip('/')}").exists()
			for name in resources["semaphores"]
		)
	finally:
		if owner.poll() is None:
			try:
				for process in owner_process.children(recursive=True):
					descendants[process.pid] = process
			except psutil.NoSuchProcess:
				pass
			owner.kill()
		for process in descendants.values():
			if running(process):
				try:
					process.kill()
				except psutil.NoSuchProcess:
					pass
		owner.wait(timeout=5)
		deadline = time.monotonic() + 5
		while any(running(process) for process in descendants.values()) and time.monotonic() < deadline:
			time.sleep(0.01)
		owner.stdout.close()
		owner.stderr.close()
		if resource_path.exists():
			resources = json.loads(resource_path.read_text(encoding="utf-8"))
		if resources.get("segment"):
			(Path("/dev/shm") / resources["segment"].lstrip("/")).unlink(missing_ok=True)
		for name in resources.get("semaphores", ()):
			(Path("/dev/shm") / f"sem.{name.lstrip('/')}").unlink(missing_ok=True)
		remaining = [process.pid for process in descendants.values() if running(process)]
		if remaining:
			message = f"test-owned processes survived cleanup: {remaining}"
			if sys.exc_info()[1] is not None:
				sys.exc_info()[1].add_note(message)
			else:
				pytest.fail(message)


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
	parallel_work_apis = []
	runtime_specs = []
	retired_work_apis = {
		"work_specs", "uses_vo_work", "_prepare_vo_work", "_scope_vo_work_to_unit", "_release_vo_work",
	}
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
			identifier = getattr(node, "id", None) or getattr(node, "attr", None) or getattr(node, "name", None)
			if identifier in retired_work_apis:
				parallel_work_apis.append(f"{path.relative_to(repository)}:{node.lineno} {identifier}")
			if isinstance(node, ast.ClassDef) and node.name in {"PipelineState", "_WorkerState"}:
				for member in node.body:
					if (
						isinstance(member, ast.AnnAssign)
						and isinstance(member.target, ast.Name)
						and member.target.id == "buffer_spec"
					):
						runtime_specs.append(f"{path.relative_to(repository)}:{member.lineno}")
			if isinstance(node, ast.Attribute) and node.attr == "buffer_spec":
				receiver = getattr(node.value, "id", "")
				if (
					receiver in {"state", "initial", "current", "result", "payload", "parent"}
					or receiver.endswith("_state")
					or isinstance(node.ctx, ast.Store)
				):
					runtime_specs.append(f"{path.relative_to(repository)}:{node.lineno}")
			if not isinstance(node, ast.Call):
				continue
			call_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
			if (
				call_name in {"PipelineState", "_WorkerState", "with_array", "commit_output"}
				and any(keyword.arg == "buffer_spec" for keyword in node.keywords)
			):
				runtime_specs.append(f"{path.relative_to(repository)}:{node.lineno}")
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
	assert parallel_work_apis == []
	assert runtime_specs == []
