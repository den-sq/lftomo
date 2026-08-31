"""Storage contracts and run-scoped lifecycle coordination."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np

from lftomo.util.shapes import DataShape, Proj, Theta, X, XSlice, Y, YSlice, Z


_VIEW_SHAPES = {
	("Y", "X"): Proj,
	("Theta", "X"): YSlice,
	("Theta", "Y"): XSlice,
	("Theta",): Theta,
	("Y",): Y,
	("X",): X,
	("Z",): Z,
}


class ManagedBuffer(ABC):
	"""A reopenable array buffer with explicit storage ownership."""

	@property
	@abstractmethod
	def shape(self) -> DataShape:
		"""Return the logical array shape."""

	@property
	@abstractmethod
	def dtype(self) -> np.dtype:
		"""Return the logical array dtype."""

	@abstractmethod
	def array(self) -> np.ndarray:
		"""Open and return the process-local NumPy view."""

	@abstractmethod
	def __getitem__(self, view: Any) -> "ManagedBuffer":
		"""Return a reopenable basic-index view without materializing a copy."""

	def close(self) -> None:
		"""Release this process's attachment, if one is open."""


def is_advanced_index(index: Any) -> bool:
	"""Return whether ``index`` would materialize rather than preserve a view."""

	if isinstance(index, tuple):
		return any(
			isinstance(item, (list, np.ndarray)) and not isinstance(item, slice)
			for item in index
		)
	return isinstance(index, (list, np.ndarray)) and not isinstance(index, slice)


def basic_view_shape(shape: DataShape, views: list[Any]) -> DataShape:
	"""Apply basic array views while retaining the canonical named axes."""

	names = list(shape.names)
	values = list(shape.as_tuple())
	for view in views:
		indices = view if isinstance(view, tuple) else (view,)
		if Ellipsis in indices:
			position = indices.index(Ellipsis)
			missing = len(values) - (len(indices) - 1)
			indices = indices[:position] + (slice(None),) * missing + indices[position + 1:]
		indices += (slice(None),) * (len(values) - len(indices))
		if len(indices) != len(values) or any(index is None for index in indices):
			raise TypeError("new-axis ManagedBuffer views have no canonical DataShape")
		selected = [
			(name, len(range(*index.indices(size))))
			for name, size, index in zip(names, values, indices, strict=True)
			if not isinstance(index, (int, np.integer))
		]
		names = [name for name, _value in selected]
		values = [value for _name, value in selected]
	shape_type = type(shape) if tuple(names) == shape.names else _VIEW_SHAPES.get(tuple(names))
	if shape_type is None:
		raise TypeError(f"no canonical DataShape for basic view axes {tuple(names)!r}")
	return shape_type(*values)


class FileBackedNPArray(ManagedBuffer):
	"""Reopenable, read-only ``.npy`` mapping suitable for worker processes."""

	def __init__(self, path: str | Path, shape: DataShape | None = None,
			views: list[Any] | None = None):
		self.__path = Path(path)
		self.__views = [] if views is None else views
		self.__base_shape = shape
		self.__mapping = None
		probe = np.load(self.__path, mmap_mode="r", allow_pickle=False)
		try:
			if shape is not None:
				if len(shape.as_tuple()) != probe.ndim:
					raise ValueError(
						"FileBackedNPArray DataShape rank disagrees with the NPY header"
					)
				if shape.as_tuple() != probe.shape:
					raise ValueError(
						"FileBackedNPArray DataShape extents disagree with the NPY header"
					)
			view = probe
			for index in self.__views:
				view = view[index]
			self.__shape = (
				view.shape if shape is None else basic_view_shape(shape, self.__views)
			)
			self.__dtype = view.dtype
		finally:
			mmap = getattr(probe, "_mmap", None)
			if mmap is not None:
				mmap.close()

	@property
	def path(self) -> Path:
		return self.__path

	@property
	def shape(self) -> DataShape | tuple[int, ...]:
		return self.__shape

	@property
	def dtype(self) -> np.dtype:
		return self.__dtype

	def array(self) -> np.ndarray:
		if self.__mapping is None:
			self.__mapping = np.load(self.__path, mmap_mode="r", allow_pickle=False)
		view = self.__mapping
		for index in self.__views:
			view = view[index]
		view.flags.writeable = False
		return view

	def __array__(self, dtype=None, copy=None) -> np.ndarray:
		"""Return the mapped view used by NumPy at PipelineState boundaries."""

		result = self.array()
		if dtype is not None:
			if copy is False and np.dtype(dtype) != result.dtype:
				raise ValueError("dtype conversion cannot be honored with copy=False")
			result = result.astype(dtype, copy=False)
		if copy:
			result = result.copy()
		return result

	def __getitem__(self, view: Any) -> "FileBackedNPArray":
		if is_advanced_index(view):
			raise IndexError(
				"This would be an advanced index and create copies, so should not be used here."
			)
		return type(self)(self.__path, self.__base_shape, self.__views + [view])

	def close(self) -> None:
		if self.__mapping is not None:
			mmap = getattr(self.__mapping, "_mmap", None)
			if mmap is not None:
				mmap.close()
			self.__mapping = None

	shutdown = close

	def __enter__(self) -> np.ndarray:
		return self.array()

	def __exit__(self, *args, **kwargs) -> None:
		self.close()

	def __reduce__(self):
		return type(self), (self.__path, self.__base_shape, self.__views)


class StorageBackend(ABC):
	"""Allocator used by a pipeline run for owned working storage."""

	@abstractmethod
	def allocate(
		self,
		shape: DataShape,
		dtype: np.dtype,
		*,
		aliases: tuple[tuple[Any, np.dtype], ...] = (),
		role: str | None = None,
		reuse_from: tuple[str, ...] = (),
		alias_from: ManagedBuffer | None = None,
		proof: str | None = None,
	) -> ManagedBuffer | list[ManagedBuffer]:
		"""Allocate storage, optionally using an explicitly proved named alias."""

	@abstractmethod
	def fork(self, source: Any, *, shape: DataShape, label: str) -> ManagedBuffer:
		"""Allocate an owned copy of ``source``."""

	@abstractmethod
	def retire(self, buffer: ManagedBuffer, *, role: str | None = None) -> None:
		"""Mark a successfully replaced logical buffer as no longer live."""

	@abstractmethod
	def pin(self, buffer: ManagedBuffer, *, owner: str) -> None:
		"""Prevent a buffer from participating in alias reuse for ``owner``."""

	@abstractmethod
	def unpin(self, buffer: ManagedBuffer, *, owner: str) -> None:
		"""Release one external liveness pin."""


class PipelineRunContext:
	"""Own the storage backend for the complete lifetime of a pipeline run.

	The context is intentionally separate from ``PipelineState``: state threads
	the actual buffer handle and an allocator reference, while this object owns
	manager startup and shutdown. Callers that provide an existing backend retain
	responsibility for its lifetime.
	"""

	def __init__(self, storage: StorageBackend | None = None, *,
			enable_aliasing: bool = True, enable_engine_aliasing: bool = True,
			max_workers: int = 2, worker_timeout_seconds: float | None = None,
			multiprocessing_context: str = "spawn"):
		self.storage = storage
		self._owns_storage = storage is None
		self.enable_aliasing = enable_aliasing
		self.enable_engine_aliasing = enable_engine_aliasing
		self.max_workers = max_workers
		self.worker_timeout_seconds = worker_timeout_seconds
		self.multiprocessing_context = multiprocessing_context
		self.executor = None
		self._entered = False
		self._spent = False

	def __enter__(self) -> Self:
		if self._entered:
			raise RuntimeError("PipelineRunContext is already active")
		if self._spent:
			raise RuntimeError("PipelineRunContext cannot be re-entered after exit")
		self._entered = True
		if self.storage is None:
			# Local import avoids a cycle between the contracts and concrete backend.
			from lftomo.util.mem import SharedNPManager

			manager = SharedNPManager(
				enable_aliasing=self.enable_aliasing,
				enable_engine_aliasing=self.enable_engine_aliasing,
			)
			manager.start()
			self.storage = manager
		from lftomo.pipeline.execution import PipelineExecutor

		self.executor = PipelineExecutor(
			self.max_workers, timeout_seconds=self.worker_timeout_seconds,
			context=self.multiprocessing_context,
		)
		# ``state.executor`` is the explicit Pipeline path.  The storage link also
		# lets directly invoked Stages (and states created before context entry)
		# discover the same run-owned executor without a second pool.
		setattr(self.storage, "pipeline_executor", self.executor)
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_value: BaseException | None,
		traceback: TracebackType | None,
	) -> None:
		try:
			if self.executor is not None:
				self.executor.close()
			if self._owns_storage and self.storage is not None:
				shutdown = getattr(self.storage, "shutdown", None)
				if shutdown is not None:
					shutdown()
		finally:
			self.executor = None
			self._entered = False
			self._spent = True


def buffer_view(buffer: Any) -> np.ndarray:
	"""Return a process-local NumPy view of a pipeline buffer."""

	if isinstance(buffer, ManagedBuffer):
		return buffer.array()
	if isinstance(buffer, np.ndarray):
		return buffer
	return np.asarray(buffer)
