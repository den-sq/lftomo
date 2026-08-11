from dataclasses import astuple, is_dataclass, replace, asdict, fields
from functools import cached_property
from multiprocessing.shared_memory import SharedMemory, ShareableList
from multiprocessing.managers import SharedMemoryManager, BaseManager
from multiprocessing import util as multiprocessing_util
import os
import signal
from sys import stdout
import threading
from time import sleep
from typing import TextIO

import numpy as np

from lftomo.storage import (
	FileBackedNPArray,
	ManagedBuffer,
	StorageBackend,
	buffer_view,
	is_advanced_index,
)

from .shapes import DataShape
from .log import log, LOG

MEM_INTERVAL_TIMER = 1.0


def _shape_tuple(shape) -> tuple[int, ...]:
	return tuple(astuple(shape)) if is_dataclass(shape) else tuple(shape)


class SharedNPManager(SharedMemoryManager, StorageBackend):
	""" Manages shared memory numpy arrays. """

	@classmethod
	def _run_server(cls, registry, address, authkey, serializer, writer,
					initializer=None, initargs=()):
		"""Run the manager outside the client process group with a death watch.

		Experiment timeouts kill the isolated trial's complete process group. The
		manager must live just long enough to unlink its tracked segments after the
		owning client dies, including when that client receives ``SIGKILL``.

		``SharedMemoryManager`` exposes no public owner-death hook.  This override
		therefore deliberately mirrors CPython's ``BaseManager._run_server`` and
		uses its ``shared_memory_context`` cleanup object; the crash-cleanup tests
		exercise that assumption across the supported Python versions.
		"""

		signal.signal(signal.SIGINT, signal.SIG_IGN)
		owner_pid = os.getppid()
		if hasattr(os, "setsid"):
			try:
				os.setsid()
			except OSError:
				pass
		if initializer is not None:
			initializer(*initargs)
		server = cls._Server(registry, address, authkey, serializer)

		def cleanup_after_owner_exit():
			while os.getppid() == owner_pid:
				sleep(0.1)
			try:
				# CPython-private counterpart to SharedMemoryManager.shutdown().
				server.shared_memory_context.unlink()
			finally:
				os._exit(0)

		threading.Thread(target=cleanup_after_owner_exit, daemon=True).start()
		writer.send(server.address)
		writer.close()
		multiprocessing_util.info("manager serving at %r", server.address)
		server.serve_forever()

	def __init__(self, *args,
					queue_mem: list[tuple[DataShape, np.dtype] | tuple[tuple[DataShape, np.dtype]]] | None = None,
					**kwargs):
		SharedMemoryManager.__init__(self, *args, **kwargs)

		self.__mem_queue = []
		for mem in queue_mem or ():
			if isinstance(mem[0], tuple):
				self.__mem_queue.append([mem[0][0], mem[0][1]] + list(mem[1:]))
			else:
				self.__mem_queue.append([mem[0], mem[1]])

	def SharedNPArray(self, shape: DataShape, dtype: np.dtype, *create_with: tuple[DataShape, np.dtype]):
		full_set = [(shape, dtype)] + list(create_with)
		size = max([np.prod(_shape_tuple(shape)) * np.dtype(dtype).itemsize for (shape, dtype) in full_set])
		mem = self.SharedMemory(size)
		result = [SharedNPArray(mem.name, shape, dtype) for (shape, dtype) in full_set]
		return result[0] if len(result) == 1 else result

	def allocate(self, shape, dtype, *, aliases=()):
		"""Allocate manager-owned storage, optionally with logical aliases."""

		return self.SharedNPArray(shape, dtype, *tuple(aliases))

	def fork(self, source, *, label: str):
		"""Return a manager-owned copy of an existing buffer."""

		source_view = buffer_view(source)
		result = self.allocate(getattr(source, "shape", source_view.shape), source_view.dtype)
		result.array()[...] = source_view
		return result

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
					views: list = None, *, allocate: bool = False):
		self.__name = name
		self.__shape = shape
		self.__dtype = np.dtype(dtype)
		self.__views = [] if views is None else views
		self.__shutdown = False
		self.__allocated = allocate
		self.__sm = None
		if allocate:
			self.__sm = SharedMemory(name=name, create=True, size=self.size())
		else:
			self.__sm = SharedMemory(name=name)

	def size(self, swap=None):
		""" Size of array in memory.

			:param swap: Tuple overriding shape parameters, for a custom size.
			:retval: Size in bytes.
		"""
		if swap is None:
			calc_shape = self.shape
		elif is_dataclass(self.shape):
			calc_shape = replace(self.shape, **asdict(swap))
		else:
			calc_shape = swap
		return int(np.prod(_shape_tuple(calc_shape), dtype=np.float64) * self.__dtype.itemsize)

	def __load(self) -> np.ndarray:
		"""Allocates a numpy array in the shared memory."""
		if self.__sm.buf is not None:
			self.__ar = np.ndarray(shape=_shape_tuple(self.__shape), dtype=self.dtype, buffer=self.__sm.buf)
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

	def __array__(self, dtype=None, copy=None):
		"""Compatibility conversion; Stage code uses ``working_view`` explicitly."""

		result = self.array()
		if dtype is not None:
			result = result.astype(dtype, copy=False)
		if copy:
			result = result.copy()
		return result

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

		return type(self), (self.__name, self.__shape, self.__dtype, self.__views)

	def __getitem__(self, view):
		# Test if it would create a copy. non-ideal but we shouldn't be calling this much anwyay!
		if is_advanced_index(view):
			raise IndexError("This would be an advanced index and create copies, so should not be used here.")
		else:
			return SharedNPArray(self.__name, self.__shape, self.__dtype, self.__views + [view])

	@cached_property
	def has_views(self):
		return len(self.__views) > 0

	@property
	def name(self):
		return self.__name

	@property
	def dtype(self):
		return self.__dtype

	@property
	def owns_unlink(self) -> bool:
		"""Whether this process-local handle has unlink authority."""

		return self.__allocated

	@cached_property
	def shape(self):
		if len(self.__views) == 0:
			return self.__shape
		else:
			# non-ideal but we shouldn't be calling this much anwyay!
			shape = self.__load().shape
			if is_dataclass(self.__shape) and len(shape) == len(fields(self.__shape)):
				shape = type(self.__shape)(*shape)
			elif is_dataclass(self.__shape):
				shape = self.__shape.gen(fields(self.__shape)[-len(shape):])(*shape)
			return shape


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
