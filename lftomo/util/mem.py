from dataclasses import replace
from contextlib import ExitStack
from functools import cached_property
from multiprocessing.shared_memory import SharedMemory, ShareableList
from multiprocessing.managers import SharedMemoryManager, SharedMemoryServer, BaseManager, dispatch
from multiprocessing import util as multiprocessing_util
import os
from pathlib import Path
import signal
from sys import platform, stdout
import threading
from time import sleep
from typing import TextIO

import numpy as np

from lftomo.storage import (
	basic_view_shape,
	FileBackedNPArray,
	ManagedBuffer,
	StorageBackend,
	buffer_view,
	is_advanced_index,
	select_buffer_alias,
)

from .shapes import DataShape
from .log import log, LOG
from .process import watch_parent_exit

MEM_INTERVAL_TIMER = 1.0


class _RunStorageServer(SharedMemoryServer):
	"""Extend the existing manager's owner-death cleanup to run directories."""

	public = SharedMemoryServer.public + [
		"create_intermediate_directory", "release_intermediate_directory",
	]

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._intermediate_directories: set[Path] = set()
		self._directory_lock = threading.RLock()
		self._directories_closed = False

	def create_intermediate_directory(self, _connection, root, retain):
		# Create and register under one lock in the surviving manager process:
		# an owner killed before the RPC returns cannot strand a fresh directory.
		with self._directory_lock:
			if self._directories_closed:
				raise RuntimeError("storage owner has already exited")
			directory = StorageBackend.create_intermediate_directory(
				self, Path(root), retain=retain,
			)
			if not retain:
				self._intermediate_directories.add(directory)
			return str(directory)

	def release_intermediate_directory(self, _connection, directory):
		with self._directory_lock:
			directory = Path(directory)
			if directory not in self._intermediate_directories:
				raise ValueError("directory is not owned by this storage manager")
			try:
				StorageBackend.release_intermediate_directory(self, directory)
			except FileNotFoundError:
				pass
			self._intermediate_directories.remove(directory)

	def cleanup_intermediates(self):
		with self._directory_lock:
			self._directories_closed = True
			with ExitStack() as cleanup:
				for directory in tuple(self._intermediate_directories):
					cleanup.callback(self.release_intermediate_directory, None, directory)

	def shutdown(self, connection):
		try:
			self.cleanup_intermediates()
		finally:
			super().shutdown(connection)


class SharedNPManager(SharedMemoryManager, StorageBackend):
	""" Manages shared memory numpy arrays. """

	_Server = _RunStorageServer

	def create_intermediate_directory(self, root: Path, *, retain: bool) -> Path:
		with self._Client(self._address, authkey=self._authkey) as connection:
			return Path(dispatch(
				connection, None, "create_intermediate_directory", (str(root), retain),
			))

	def release_intermediate_directory(self, directory: Path) -> None:
		with self._Client(self._address, authkey=self._authkey) as connection:
			dispatch(connection, None, "release_intermediate_directory", (str(directory),))

	@staticmethod
	def check_capacity(planned_peak_bytes: int) -> None:
		"""Reject a known Linux tmpfs shortfall before lazily allocated pages SIGBUS."""

		if platform != "linux":
			return
		try:
			space = os.statvfs("/dev/shm")
		except OSError as error:
			raise OSError(f"cannot check shared memory capacity at /dev/shm: {error}") from error
		available = space.f_bavail * space.f_frsize
		if planned_peak_bytes > available:
			raise MemoryError(
				f"insufficient shared memory: planned peak {planned_peak_bytes} bytes "
				f"({planned_peak_bytes / 1024**2:.1f} MiB), available /dev/shm space "
				f"{available} bytes ({available / 1024**2:.1f} MiB); reduce window sizes "
				"(--window-rows/--window-theta/--window-columns) or increase /dev/shm "
				"(Docker: --shm-size)"
			)

	@classmethod
	def _run_server(cls, registry, address, authkey, serializer, writer,
					initializer=None, initargs=()):
		"""Run the manager outside the client process group with a death watch.

		Experiment timeouts kill the isolated trial's complete process group. The
		manager must live just long enough to unlink its tracked segments and
		intermediate directories after the owning client dies, including when that
		client receives ``SIGKILL``.

		``SharedMemoryManager`` exposes no public owner-death hook.  This override
		therefore deliberately mirrors CPython's ``BaseManager._run_server`` and
		uses its ``shared_memory_context`` cleanup object; the crash-cleanup tests
		exercise that assumption across the supported Python versions.
		"""

		signal.signal(signal.SIGINT, signal.SIG_IGN)
		if hasattr(os, "setsid"):
			try:
				os.setsid()
			except OSError:
				pass
		if initializer is not None:
			initializer(*initargs)
		server = cls._Server(registry, address, authkey, serializer)
		cleanup_lock = threading.Lock()
		cleanup_started = False

		def unlink_once():
			"""Remove tracked storage before any abnormal manager exit."""

			nonlocal cleanup_started
			with cleanup_lock:
				if cleanup_started:
					return
				cleanup_started = True
				try:
					server.cleanup_intermediates()
				finally:
					# CPython-private counterpart to SharedMemoryManager.shutdown().
					server.shared_memory_context.unlink()

		def cleanup_after_sigterm(_signum, _frame):
			# A process-tree reaper may target this detached server explicitly.
			# Ignore repeated delivery while its synchronous cleanup completes.
			signal.signal(signal.SIGTERM, signal.SIG_IGN)
			try:
				unlink_once()
			finally:
				os._exit(0)

		signal.signal(signal.SIGTERM, cleanup_after_sigterm)

		def cleanup_after_owner_exit():
			try:
				unlink_once()
			finally:
				os._exit(0)

		watch_parent_exit(cleanup_after_owner_exit)
		writer.send(server.address)
		writer.close()
		multiprocessing_util.info("manager serving at %r", server.address)
		server.serve_forever()

	def __init__(self, *args,
					queue_mem: list[tuple[DataShape, np.dtype] | tuple[tuple[DataShape, np.dtype]]] | None = None,
					enable_aliasing: bool = True,
					enable_engine_aliasing: bool = True,
					**kwargs):
		SharedMemoryManager.__init__(self, *args, **kwargs)
		self.enable_aliasing = enable_aliasing
		self.enable_engine_aliasing = enable_engine_aliasing
		self._physical: dict[str, dict] = {}
		self._retired: dict[str, list[str]] = {}
		self._events: list[dict] = []
		self._minimum_capacities: dict[str, int] = {}

		self.__mem_queue = []
		for mem in queue_mem or ():
			if isinstance(mem[0], tuple):
				self.__mem_queue.append([mem[0][0], mem[0][1]] + list(mem[1:]))
			else:
				self.__mem_queue.append([mem[0], mem[1]])

	def SharedNPArray(self, shape: DataShape, dtype: np.dtype, *create_with: tuple[DataShape, np.dtype]):
		full_set = [(shape, dtype)] + list(create_with)
		if any(not isinstance(item_shape, DataShape) for item_shape, _dtype in full_set):
			raise TypeError("shared allocations require DataShape instances")
		size = max(item_shape.nbytes(item_dtype) for item_shape, item_dtype in full_set)
		mem = self.SharedMemory(size)
		self._physical[mem.name] = {
			"capacity": int(size), "active_roles": set(), "pins": set(),
		}
		result = [SharedNPArray(mem.name, shape, dtype) for (shape, dtype) in full_set]
		return result[0] if len(result) == 1 else result

	def allocate(self, shape: DataShape, dtype, *, aliases=(), role=None, reuse_from=(),
			alias_from=None, proof=None):
		"""Allocate manager-owned storage using only named, proved alias paths."""

		if not isinstance(shape, DataShape):
			raise TypeError("shared allocations require a DataShape")
		dtype = np.dtype(dtype)
		required = shape.nbytes(dtype)
		name = None
		alias_kind = None
		if self.enable_aliasing:
			engine_ok = proof != "tomopy_input_materialization" or self.enable_engine_aliasing
			candidate = None if alias_from is None else self._physical.get(alias_from.name)
			current = (
				alias_from.name
				if candidate is not None and engine_ok else None
			)
			name, alias_kind = select_buffer_alias(
				required, self._retired, reuse_from,
				capacity=lambda key: self._physical[key]["capacity"],
				pinned=lambda key: bool(self._physical[key]["pins"]),
				alias_from=current, proof=proof,
			)
		if name is None:
			reserved = max(required, self._minimum_capacities.get(role, 0))
			if reserved > required:
				full_set = [(shape, dtype)] + list(aliases)
				memory = self.SharedMemory(reserved)
				self._physical[memory.name] = {
					"capacity": int(reserved), "active_roles": set(), "pins": set(),
				}
				buffers = [
					SharedNPArray(memory.name, item_shape, item_dtype)
					for item_shape, item_dtype in full_set
				]
				result = buffers[0] if len(buffers) == 1 else buffers
			else:
				result = self.SharedNPArray(shape, dtype, *tuple(aliases))
			buffers = result if isinstance(result, list) else [result]
			for index, buffer in enumerate(buffers):
				buffer._role = role if index == 0 else None
			if role is not None:
				self._physical[buffers[0].name]["active_roles"].add(role)
			self._events.append({
				"action": "allocate", "role": role, "name": buffers[0].name,
				"bytes": self._physical[buffers[0].name]["capacity"], "alias": None,
			})
			return result

		for values in self._retired.values():
			while name in values:
				values.remove(name)
		buffer = SharedNPArray(name, shape, dtype, role=role)
		if role is not None:
			self._physical[name]["active_roles"].add(role)
		self._events.append({
			"action": "alias", "role": role, "name": name,
			"bytes": required, "alias": alias_kind,
		})
		return buffer

	def reserve(self, role: str, capacity: int) -> None:
		"""Set the minimum capacity for the next physical allocation of a role."""

		if capacity < 0:
			raise ValueError("reserved storage capacity cannot be negative")
		self._minimum_capacities[role] = max(
			self._minimum_capacities.get(role, 0), int(capacity),
		)

	def fork(self, source, *, shape: DataShape, label: str):
		"""Return a manager-owned copy of an existing buffer."""

		source_view = buffer_view(source)
		result = self.allocate(shape, source_view.dtype, role=label)
		result.array()[...] = source_view
		return result

	def retire(self, buffer, *, role=None):
		"""Record retirement without closing or unlinking the run-owned segment."""

		name = buffer.name
		record = self._physical.get(name)
		if record is None:
			return
		resolved = role or getattr(buffer, "role", None)
		if resolved is not None:
			record["active_roles"].discard(resolved)
		if not record["active_roles"]:
			retired_role = resolved or "working"
			if name not in self._retired.setdefault(retired_role, []):
				self._retired[retired_role].append(name)
		self._events.append({
			"action": "retire", "role": resolved, "name": name,
			"bytes": record["capacity"], "alias": None,
		})

	def pin(self, buffer, *, owner: str):
		name = getattr(buffer, "name", None)
		record = self._physical.get(name)
		if record is not None:
			record["pins"].add(owner)
			self._events.append({
				"action": "pin", "role": getattr(buffer, "role", None),
				"name": name, "bytes": record["capacity"], "alias": owner,
			})

	def unpin(self, buffer, *, owner: str):
		name = getattr(buffer, "name", None)
		record = self._physical.get(name)
		if record is not None:
			record["pins"].discard(owner)
			self._events.append({
				"action": "unpin", "role": getattr(buffer, "role", None),
				"name": name, "bytes": record["capacity"], "alias": owner,
			})

	@property
	def allocation_events(self) -> tuple[dict, ...]:
		return tuple(dict(event) for event in self._events)

	def clear_queue(self):
		ar_mem = []
		while len(self.__mem_queue) > 0:
			new_mem = self.SharedNPArray(*self.__mem_queue.pop(0))
			ar_mem += new_mem if isinstance(new_mem, list) else [new_mem]
		return ar_mem

	def __enter__(self):
		this = [BaseManager.__enter__(self)]
		return tuple(this + self.clear_queue())

	def __exit__(self, *args, **kwargs):
		super().__exit__(*args, **kwargs)


class SharedNPArray(ManagedBuffer):
	def __init__(self, name: str, shape: DataShape, dtype: np.dtype,
					views: list = None, *, allocate: bool = False,
					role: str | None = None):
		if not isinstance(shape, DataShape):
			raise TypeError("SharedNPArray.shape must be a DataShape")
		self.__name = name
		self.__shape = shape
		self.__dtype = np.dtype(dtype)
		self.__views = [] if views is None else views
		self.__shutdown = False
		self.__allocated = allocate
		self._role = role
		self.__sm = None
		if allocate:
			self.__sm = SharedMemory(name=name, create=True, size=self.size())
		else:
			self.__sm = SharedMemory(name=name)

	def size(self, swap: DataShape | None = None):
		""" Size of array in memory.

			:param swap: Named extents overriding shape parameters for a custom size.
			:retval: Size in bytes.
		"""
		if swap is None:
			calc_shape = self.shape
		else:
			calc_shape = replace(self.shape, **swap.as_dict())
		return calc_shape.nbytes(self.__dtype)

	def __load(self) -> np.ndarray:
		"""Allocates a numpy array in the shared memory."""
		if self.__sm.buf is not None:
			self.__ar = np.ndarray(
				shape=self.__shape.as_tuple(), dtype=self.dtype, buffer=self.__sm.buf,
			)
			if self.has_views:
				ar = self.__ar
				for view in self.__views:
					ar = ar[view]
				return ar
			else:
				return self.__ar
		else:
			log.write("Shared Memory", f"Shared Memory {self.__name} Was Destroyed or Never Created",
						log_level=LOG.ERROR)
			raise FileNotFoundError(f"Shared Memory {self.__name} Was Destroyed or Never Created")

	def array(self):
		return self.__load()

	def __enter__(self):
		return self.__load()

	def shutdown(self):
		if self.__shutdown:
			return
		if self.__sm is None:
			self.__shutdown = True
			return
		try:
			self.__sm.close()
		finally:
			if self.__allocated:
				try:
					self.__sm.unlink()
				except FileNotFoundError:
					pass
			self.__shutdown = True

	close = shutdown

	def __del__(self):
		if not getattr(self, "_SharedNPArray__shutdown", True):
			try:
				self.shutdown()
			except (AttributeError, BufferError, FileNotFoundError):
				pass

	def __exit__(self, *args, **kwargs):
		if hasattr(self, "_SharedNPArray__ar"):
			del self.__ar

	def __reduce__(self):
		"""Pickle as a non-owning attachment, never with unlink authority."""

		return type(self), (self.__name, self.__shape, self.__dtype, self.__views), {"_role": self._role}

	def __getitem__(self, view):
		# Test if it would create a copy. non-ideal but we shouldn't be calling this much anwyay!
		if is_advanced_index(view):
			raise IndexError("This would be an advanced index and create copies, so should not be used here.")
		else:
			return SharedNPArray(
				self.__name, self.__shape, self.__dtype, self.__views + [view], role=self._role,
			)

	def __view_shape(self) -> DataShape:
		"""Apply basic views using only declared named shape classes."""

		return basic_view_shape(self.__shape, self.__views)

	@cached_property
	def has_views(self):
		return len(self.__views) > 0

	@property
	def name(self):
		return self.__name

	@property
	def role(self):
		return self._role

	@property
	def dtype(self):
		return self.__dtype

	@property
	def owns_unlink(self) -> bool:
		"""Whether this process-local handle has unlink authority."""

		return self.__allocated

	@cached_property
	def shape(self) -> DataShape:
		return self.__shape if not self.__views else self.__view_shape()


def cleanup_mem(*shm_objects):
	""" Close and unlink shared memory objects.

		:param shm_objects: Shared memory objects to shut down.
	"""
	for shm in shm_objects:
		if isinstance(shm, SharedNPArray):
			shm.shutdown()
			del shm
		elif isinstance(shm, ShareableList):
			shm.shm.close()
			shm.shm.unlink()
		elif isinstance(shm, SharedMemory):
			shm.close()
			shm.unlink()


def exit_cleanly(step: str, *shm_objects, return_code: int = 0, statement: str = '', log_level: LOG = LOG.TIME,
					out: TextIO = stdout, throw: Exception = None):
	""" Exit while cleaning up shared memory.

		:param step: Step of reconstruction process we are exiting during.
		:param shm_objects: Shared memory objects to shut down.
		:param return_code: Process return code to send.
	"""
	log.write(step, statement, log_level, out)
	cleanup_mem(*shm_objects)

	sleep(MEM_INTERVAL_TIMER)
	log.footer(error=throw)

	exit(return_code)


def mem_monitor(mem_file, mem_store, pid):
	with open(mem_file, "w") as out, mem_store as mem_branch:
		with mem_branch as last_step_arr:
			last_step = last_step_arr.tobytes().decode()

			while last_step.strip() not in ["COMPLETED", "ERRORED"]:
				last_step = last_step_arr.tobytes().decode()
				log.write(last_step, out=out, pid=pid)
				sleep(MEM_INTERVAL_TIMER)
