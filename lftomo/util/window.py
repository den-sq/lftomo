from collections import namedtuple
from copy import copy
from dataclasses import asdict, replace, fields
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Self, Sequence, Iterator
import warnings

import numpy as np

from .document import (
	DocumentCodec,
	DocumentField,
	array_from_document,
	array_to_document,
	decode_document,
	encode_document,
)
from .shapes import DataShape, DataRange, Interval, IntIter, ReconOrder, SliceIter, TFIter, Y

# Field of view during data acquisition.
CTFov = namedtuple("CTFov", ['Theta', 'Degrees', "Count"])


def _dtype_to_document(value) -> str | None:
	return None if value is None else np.dtype(value).str


def _dtype_from_document(value) -> np.dtype | None:
	if value is not None and not isinstance(value, str):
		raise TypeError("dtype document must be a string or null")
	return None if value is None else np.dtype(value)


def _slices_to_document(value) -> list[dict[str, int | None]] | None:
	return None if value is None else [
		{"start": item.start, "stop": item.stop, "step": item.step} for item in value
	]


def _slices_from_document(value) -> tuple[slice, ...] | None:
	if value is None:
		return None
	if not isinstance(value, list) or any(
		not isinstance(item, dict) or set(item) != {"start", "stop", "step"}
		or any(component is not None and type(component) is not int for component in item.values())
		for item in value
	):
		raise ValueError("memory-index document is invalid")
	return tuple(slice(item["start"], item["stop"], item["step"]) for item in value)


def _fov_to_document(value: CTFov) -> dict[str, int | str | None]:
	return {
		"theta": None if value.Theta is None else int(value.Theta),
		"degrees": repr(float(value.Degrees)),
		"count": int(value.Count),
	}


def _fov_from_document(value) -> CTFov:
	if not isinstance(value, dict) or set(value) != {"theta", "degrees", "count"} \
			or not isinstance(value["degrees"], str) or type(value["count"]) is not int \
			or value["theta"] is not None and type(value["theta"]) is not int:
		raise ValueError("FOV document is invalid")
	return CTFov(value["theta"], float(value["degrees"]), value["count"])


def _readonly_array_from_document(value) -> np.ndarray:
	result = array_from_document(value)
	result.setflags(write=False)
	return result


def _int_from_document(value) -> int:
	if type(value) is not int:
		raise TypeError("integer document field is invalid")
	return value


_SHAPE_CODEC = DocumentCodec(lambda value: value.to_document(), DataShape.from_document)
_DTYPE_CODEC = DocumentCodec(_dtype_to_document, _dtype_from_document)
_SLICES_CODEC = DocumentCodec(_slices_to_document, _slices_from_document)
_FOV_CODEC = DocumentCodec(_fov_to_document, _fov_from_document)
_ARRAY_CODEC = DocumentCodec(array_to_document, _readonly_array_from_document)
_INT_CODEC = DocumentCodec(int, _int_from_document)

_DATA_REF_DOCUMENT_FIELDS = (
	DocumentField("shape", _SHAPE_CODEC, "_DataRef__shape"),
	DocumentField("pad_before", _SHAPE_CODEC, "_DataRef__pad_before"),
	DocumentField("pad_after", _SHAPE_CODEC, "_DataRef__pad_after"),
	DocumentField("pos", _SHAPE_CODEC, "_DataRef__pos"),
	DocumentField("step", _SHAPE_CODEC, "_DataRef__step"),
	DocumentField("dtype", _DTYPE_CODEC, "_DataRef__dtype"),
	DocumentField("offset", _INT_CODEC, "_DataRef__offset"),
	DocumentField("memory_index", _SLICES_CODEC, "_DataRef__memory_index"),
)
_TOMO_WINDOW_DOCUMENT_FIELDS = _DATA_REF_DOCUMENT_FIELDS + (
	DocumentField("work_dtype", _DTYPE_CODEC, "_TomoWindow__work_dtype"),
)
_TOMO_DATASET_DOCUMENT_FIELDS = _DATA_REF_DOCUMENT_FIELDS + (
	DocumentField("physical_shape", _SHAPE_CODEC, "_TomoDataSet__physical_shape"),
	DocumentField("sample_step", _SHAPE_CODEC, "_TomoDataSet__sample_step"),
	DocumentField("work_dtype", _DTYPE_CODEC, "_TomoDataSet__work_dtype"),
	DocumentField("fov", _FOV_CODEC, "_TomoDataSet__fov"),
	DocumentField("angles", _ARRAY_CODEC, "_TomoDataSet__angles"),
	DocumentField("acquisition_indices", _ARRAY_CODEC, "_TomoDataSet__acquisition_indices"),
)


class DataRef(object):
	_document_fields = _DATA_REF_DOCUMENT_FIELDS

	def __init__(self, shape: DataShape, pad: DataShape, pos: DataShape,
					step: DataShape, dtype: type = None, offset: int = 0,
					*, pad_after: DataShape = None):
		self.__pos = pos
		self.__shape = shape
		self.__pad_before = pad
		self.__pad_after = pad if pad_after is None else type(pad)(**asdict(pad_after))
		self.__step = step
		self.__dtype = dtype
		self.__offset = offset
		self.__memory_index = None

	@property
	def index(self) -> tuple[slice]:
		if self.__memory_index is not None:
			return self.__memory_index
		start = list(asdict(self.pos.merge(self.shape.make_with(0))).values())
		stop = list(asdict(self.shape + self.pos).values())
		step = list(asdict(self.step.merge(self.shape.make_with(1))).values())

		return tuple([slice(start[i], stop[i], step[i]) for i in range(len(start))])

	@property
	def shape(self) -> DataShape:
		return self.__shape

	@property
	def pos(self) -> DataShape:
		return self.__pos

	@property
	def step(self) -> DataShape:
		return self.__step

	@property
	def frange(self) -> DataRange:
		return DataRange(self.pos, (self.shape + self.pos).extract(self.pos), self.step)

	@property
	def pad(self) -> DataShape:
		"""Return the legacy symmetric value, rejecting an ambiguous asymmetric reference."""
		if self.pad_before != self.pad_after:
			raise ValueError("asymmetric padding has no single value; use pad_before and pad_after")
		return self.pad_before

	@property
	def pad_before(self) -> DataShape:
		return self.__pad_before

	@property
	def pad_after(self) -> DataShape:
		return self.__pad_after

	@property
	def dtype(self) -> type:
		return self.__dtype

	def __str__(self) -> str:
		return f"{self.shape} ({self.dtype}) at {self.pos} stepping by {self.step}"

	@property
	def itemsize(self) -> int:
		return np.dtype(self.dtype).itemsize

	def axis_slice(self, name: str) -> slice:
		return self.index[self.shape.axis_of(name)]

	def interval(self, name: str) -> Interval:
		"""Return the half-open covered interval for one semantic axis."""
		self.shape.axis_of(name)
		axis_type = next(
			shape_type for shape_type in DataShape.__subclasses__()
			if shape_type.names == (name,)
		)
		start = int(getattr(self.pos, name, 0))
		return Interval(axis_type(start), axis_type(start + int(getattr(self.shape, name))))

	def to_document(self) -> dict[str, object]:
		return encode_document(self, type(self)._document_fields)

	@classmethod
	def from_document(cls, document: dict[str, object]) -> Self:
		if not isinstance(document, dict) or not isinstance(document.get("type"), str):
			raise ValueError(f"expected {cls.__name__} document")
		targets = (cls,) if cls is not DataRef else (DataRef, TomoWindow, TomoDataSet)
		target = next((item for item in targets if item.__name__ == document["type"]), None)
		if target is None:
			raise ValueError(f"unknown or incompatible DataRef type {document['type']!r}")
		result = decode_document(target, document, target._document_fields)
		result._validate_document()
		return result

	def _validate_document(self) -> None:
		pass

	def nbytes(self) -> int:
		return self.shape.nbytes(self.dtype)

	@property
	def step_shape(self) -> DataShape:
		ss = replace(self.shape, **asdict(self.step))
		return ss

	@property
	def stride(self) -> DataShape:
		vals = list(asdict(self.shape).values()) + [1]
		return type(self.shape)(*[self.itemsize * np.prod(vals[x + 1:]) for x in range(len(vals) - 1)])

	def with_pad(self, pad_dim: Sequence = None, shrink=False) -> Self:
		# DataShape values are immutable by convention; re-init replaces every
		# DataRef field while retaining the concrete subclass.
		new_ref = copy(self)

		if pad_dim is None:
			pad_dim = [field.name for field in fields(self.pad_before)]
		try:
			kept_before = replace(self.pad_before, **{key: 0 for key in pad_dim})
			kept_after = replace(self.pad_after, **{key: 0 for key in pad_dim})
			shift_before = self.pad_before - kept_before
			shift_after = self.pad_after - kept_after
		except TypeError as te:
			te.add_note("No padding found for dimension.")
			raise te

		DataRef.__init__(new_ref, self.shape + shift_before + shift_after, kept_before,
						self.pos - shift_before.extract(self.pos), self.step, self.dtype, self.__offset,
						pad_after=kept_after)
		return new_ref

	def rescope(self, pos: DataShape = None, pad: DataShape = None, dtype: type = None,
					shape: DataShape = None, *, pad_after: DataShape = None) -> Self:
		new_ref = copy(self)

		new_shape = replace(self.shape, **asdict(shape)) if shape is not None else self.shape
		new_pos = (
			pos if pos is not None and set(self.pos.names) < set(pos.names)
			else replace(self.pos, **asdict(pos)) if pos is not None else self.pos
		)
		new_dtype = dtype if dtype is not None else self.dtype

		if pad is not None:
			new_pad_before = replace(self.pad_before, **asdict(pad))
			new_pad_after = replace(self.pad_after, **asdict(pad if pad_after is None else pad_after))
		else:
			new_pad_before = self.pad_before
			new_pad_after = self.pad_after if pad_after is None else replace(self.pad_after, **asdict(pad_after))

		pad_shift_before = new_pad_before - self.pad_before
		pad_shift_after = new_pad_after - self.pad_after
		position_fields = {field.name for field in fields(new_pos)}
		new_pos += replace(new_pos.make_with(0), **{
			key: value for key, value in asdict(pad_shift_before).items()
			if key in position_fields
		})
		new_shape -= (pad_shift_before + pad_shift_after)

		DataRef.__init__(new_ref, new_shape, new_pad_before, new_pos, self.step, new_dtype, self.__offset,
						pad_after=new_pad_after)
		return new_ref

	def with_output_padding(self, padding: DataShape) -> Self:
		"""Expand published output coordinates; unlike source padding, do not clip."""

		if any(value < 0 for value in asdict(padding).values()):
			raise ValueError("output padding cannot be negative")
		expansion = replace(self.shape.make_with(0), **asdict(padding))
		return self.rescope(
			pos=self.pos - expansion.extract(self.pos),
			shape=self.shape + expansion + expansion,
		)

	def with_crop(self, before: DataShape, after: DataShape) -> Self:
		"""Return the positive interval left after an asymmetric axis crop."""

		zero = self.shape.make_with(0)
		before = replace(zero, **asdict(before))
		after = replace(zero, **asdict(after))
		if any(value < 0 for value in (*before.as_tuple(), *after.as_tuple())):
			raise ValueError("crop extents cannot be negative")
		shape = self.shape - before - after
		if any(value < 1 for value in shape.as_tuple()):
			raise ValueError("crop must retain a positive extent on every axis")
		position = self.pos + before.extract(self.pos)
		return self.rescope(pos=position, shape=shape)

	def int_iter(self) -> Iterator:
		return self.frange.get_iter(conv=IntIter)

	def step_iter(self) -> Iterator:
		return self.frange.get_iter()

	def slice_iter(self) -> Iterator:
		return self.frange.get_iter(conv=partial(SliceIter, shape=self.shape))

	def units(self, axis: str) -> Iterator[Self]:
		"""Yield one-sample refs with local memory indices and global positions."""

		axis_index = self.shape.axis_of(axis)
		position = self.pos.as_dict()
		step = self.step.as_dict().get(axis, 1)
		start = position.get(axis, 0)
		for memory_position in range(getattr(self.shape, axis)):
			unit_position = replace(
				self.shape.make_with(0),
				**{**position, axis: start + memory_position * step},
			)
			unit = self.rescope(
				pos=unit_position,
				shape=replace(self.shape, **{axis: 1}),
			)
			index = [slice(None)] * len(self.shape.names)
			index[axis_index] = slice(memory_position, memory_position + 1)
			unit.__memory_index = tuple(index)
			yield unit


class TomoWindow(DataRef):
	_document_fields = _TOMO_WINDOW_DOCUMENT_FIELDS

	def __init__(self, shape: DataShape, pad: DataShape, pos: DataShape, step: DataShape,
				dtype: type, work_dtype: type, *, pad_after: DataShape = None):
		super().__init__(shape, pad, pos, step, dtype, pad_after=pad_after)
		self.__work_dtype = work_dtype

	@classmethod
	def from_bounds(
		cls,
		physical_shape: DataShape,
		*,
		y: tuple[int, int] | slice | None = None,
		theta: tuple[int, int] | tuple[int, int, int] | slice | None = None,
		halo: int | DataShape | tuple[int | DataShape, int | DataShape] = 0,
		dtype: type = None,
		work_dtype: type = np.float32,
	) -> Self:
		"""Construct a source-coordinate window and record its realized halo.

		``shape`` records the covered source-coordinate span while ``step``
		records sampling within it.  Consequently a strided Theta selection has
		``shape.Theta == stop - start`` even though ``frange.length.Theta`` is
		the number of samples allocated in memory.  A two-item ``halo`` supplies
		independent before/after requirements for streaming boundaries.
		"""

		def bounds(value, size: int, name: str, *, stride: bool) -> tuple[int, int, int]:
			if value is None:
				start, stop, step = 0, size, 1
			elif isinstance(value, slice):
				start = 0 if value.start is None else value.start
				stop = size if value.stop is None else value.stop
				step = 1 if value.step is None else value.step
			else:
				if len(value) not in ((2, 3) if stride else (2,)):
					raise ValueError(f"{name} bounds must contain start/stop" + ("[/stride]" if stride else ""))
				start, stop = value[:2]
				step = value[2] if len(value) == 3 else 1
			if not 0 <= start < stop <= size or step < 1 or (not stride and step != 1):
				raise ValueError(f"invalid {name} bounds [{start}, {stop}) with stride {step}")
			return int(start), int(stop), int(step)

		theta_start, theta_stop, theta_step = bounds(
			theta, physical_shape.Theta, "Theta", stride=True,
		)
		y_start, y_stop, _ = bounds(y, physical_shape.Y, "Y", stride=False)
		shape = type(physical_shape)(
			Theta=theta_stop - theta_start, Y=y_stop - y_start, X=physical_shape.X,
		)
		pos = type(physical_shape)(Theta=theta_start, Y=y_start, X=0)
		step = type(physical_shape)(Theta=theta_step, Y=1, X=1)
		def canonical_halo(value: int | DataShape) -> DataShape:
			return replace(
				physical_shape.make_with(0),
				**(Y(value).as_dict() if isinstance(value, int) else value.as_dict()),
			)

		if isinstance(halo, tuple):
			if len(halo) != 2:
				raise ValueError("window halo pair must contain before and after values")
			requested_before, requested_after = map(canonical_halo, halo)
		else:
			requested_before = requested_after = canonical_halo(halo)
		if any(
			value < 0
			for requested in (requested_before, requested_after)
			for value in requested.as_tuple()
		):
			raise ValueError("window halo cannot be negative")
		pad_before, pad_after = _resolved_padding(
			pos, shape, requested_before, requested_after, physical_shape,
		)
		return cls(
			shape, pad_before, pos, step, dtype, work_dtype, pad_after=pad_after,
		)

	@property
	def internal(self):
		return self.rescope(pos=self.pad_before.extract(self.pos), dtype=self.__work_dtype)

	def recon_shape(self) -> ReconOrder:
		"""Return the square physical reconstruction shape for the core window.

		Published/cropped Z and X extents remain reconstruction-stage policy. The
		result is meaningful only for a window that will actually be reconstructed.
		"""

		return ReconOrder.of(self.internal.shape)


def _resolved_padding(pos: DataShape, shape: DataShape, pad_before: DataShape, pad_after: DataShape,
						physical_shape: DataShape = None) -> tuple[DataShape, DataShape]:
	if physical_shape is None:
		return pad_before, pad_after

	physical_pos = replace(physical_shape.make_with(0), **asdict(pos))
	physical_core_shape = replace(physical_shape, **asdict(shape))
	physical_stop = physical_pos + physical_core_shape
	zero = physical_shape.make_with(0)
	if not physical_pos >= zero or not physical_stop <= physical_shape:
		raise ValueError("window output must remain within the physical source shape")

	requested_before = replace(zero, **asdict(pad_before))
	requested_after = replace(zero, **asdict(pad_after))
	available_before = physical_pos
	available_after = physical_shape - physical_stop
	realized_before = type(physical_shape).param_min(requested_before, available_before)
	realized_after = type(physical_shape).param_min(requested_after, available_after)
	return realized_before.extract(pad_before), realized_after.extract(pad_after)


class TomoWindowIter(TFIter):
	def __init__(self, dr: DataRange, shape: DataShape, pad: DataShape,
					dtype: type, work_dtype: type, substep: DataShape = None,
					pad_after: DataShape = None, physical_shape: DataShape = None):
		self.shape = shape
		self.pad_before = pad
		self.pad_after = pad if pad_after is None else type(pad)(**asdict(pad_after))
		self.dtype = dtype
		self.work_dtype = work_dtype
		self.start = dr.start
		self.stop = dr.stop
		self.substep = substep if substep is not None else dr.step.make_with(1)
		self.physical_shape = physical_shape
		self.__step_class = type(dr.step)

	def __call__(self, pos) -> tuple:
		# DataRange iteration supplies only the scheduled axes.  Retain the
		# selected coordinates on every other axis in each emitted window.
		pos = replace(self.start, **asdict(self.__step_class(*pos)))
		step_end = (self.shape + pos).extract(pos)

		if step_end < self.stop:
			winshape = self.shape
		else:
			# Standard shape would extend beyond the range, so cut it down.
			winshape = replace(self.shape, **{k: v for k, v in asdict(self.stop - pos).items()
											if v < asdict(self.shape)[k]})

		pad_before, pad_after = _resolved_padding(
			pos, winshape, self.pad_before, self.pad_after, self.physical_shape,
		)
		return TomoWindow(winshape, pad_before, pos, self.substep, self.dtype, self.work_dtype,
						pad_after=pad_after)


class TomoDataSet(DataRef):
	""" Set of windows for slicing data into smaller portions.

		Iterable for grabbing each window in order.
	"""
	_document_fields = _TOMO_DATASET_DOCUMENT_FIELDS

	def __init__(self, fov: CTFov, shape: DataShape, over: DataRange | DataShape,
					angle_steps: np.ndarray = None, dtype: type = np.int16, work_dtype: type = np.float32,
					*, angles: np.ndarray = None, acquisition_indices: np.ndarray = None,
					pad: DataShape = None, pad_after: DataShape = None,
					sample_step: DataShape = None):
		""" Initializes DataWindowSet.

			:param ds_shape: Shape of the overall dataset.
			:param over: Either a full range to iterate over or just the per-step iteration.
			:param fov: Field of view details of the image, for angle calculation.
			:param dtype: Type of data being iterated over.
			:param angle_steps: # of angle steps to take for each projection, handling dropped frames.
		"""
		self.__physical_shape = shape
		unit_step = shape.make_with(1)
		self.__sample_step = (
			unit_step if sample_step is None
			else replace(unit_step, **asdict(sample_step))
		)
		if any(value < 1 for value in self.__sample_step.as_tuple()):
			raise ValueError("source sampling steps must be positive")

		if isinstance(over, DataShape):
			pos = over.make_with(0)
			shape = shape
			step = over
		elif isinstance(over, DataRange):
			for axis, start in asdict(over.start).items():
				if start < 0 or not hasattr(self.physical_shape, axis):
					raise ValueError(f"output range starts outside physical axis {axis}")
			for axis, stop in asdict(over.stop).items():
				if not hasattr(self.physical_shape, axis) or stop > getattr(self.physical_shape, axis):
					raise ValueError(f"output range stops outside physical axis {axis}")
			pos = over.start
			shape = replace(shape, **asdict(over.stop - over.start))
			step = over.step
		else:
			print(f"Invalid over {over}:{type(over)} - must be step size (DataShape) or a full range (DataRange)")
			raise TypeError(f"Invalid over {over}:{type(over)} - must be step size (DataShape) or a full range (DataRange)")

		zero = shape.make_with(0)
		pad_before = zero if pad is None else replace(zero, **asdict(pad))
		resolved_pad_after = (
			pad_before if pad_after is None else replace(zero, **asdict(pad_after))
		)
		super().__init__(
			shape=shape, pad=pad_before, pos=pos, step=step, dtype=dtype,
			pad_after=resolved_pad_after,
		)

		self.__work_dtype = work_dtype

		self.__fov = fov

		if angles is not None:
			if angle_steps is not None:
				raise ValueError("angles and angle_steps are mutually exclusive")
			if len(angles) != self.physical_shape.Theta:
				raise ValueError("one source angle is required for every projection")
			self.__angles = np.asarray(angles, dtype=np.float32)
		elif angle_steps is None:
			warnings.warn(
				"TomoDataSet FOV angle synthesis is deprecated; pass acquisition-model angles",
				DeprecationWarning,
				stacklevel=2,
			)
			if self.physical_shape.Theta == fov.Theta * fov.Count:
				self.__angles = np.array(np.linspace(0, np.deg2rad(fov.Degrees), fov.Theta, False))
			else:
				raise Exception("Missing Angle Steps for handling Missing Frames")
		else:
			self.__angles = np.array([step * np.deg2rad(fov.Degrees) / fov.Count for step in angle_steps])
		if acquisition_indices is None:
			acquisition_indices = np.arange(self.physical_shape.Theta, dtype=np.int64)
		if len(acquisition_indices) != self.physical_shape.Theta:
			raise ValueError("one acquisition index is required for every projection")
		self.__acquisition_indices = np.asarray(acquisition_indices, dtype=np.int64)
		# Geometry copies share these arrays, so construction makes their
		# immutability convention enforceable instead of cloning them per rescope.
		self.__angles.setflags(write=False)
		self.__acquisition_indices.setflags(write=False)

	@property
	def angles(self):
		return self.__angles

	@property
	def acquisition_indices(self):
		return self.__acquisition_indices

	@property
	def fov(self):
		return self.__fov

	@property
	def sample_step(self) -> DataShape:
		"""Sampling within each emitted window, distinct from its Y schedule."""
		return self.__sample_step

	@property
	def physical_shape(self) -> DataShape:
		"""Full source shape, independent of the selected output range in ``frange``."""
		return self.__physical_shape

	def _validate_document(self) -> None:
		if any(value < 1 for value in self.sample_step.as_tuple()):
			raise ValueError("source sampling steps must be positive")
		if len(self.angles) != self.physical_shape.Theta \
				or len(self.acquisition_indices) != self.physical_shape.Theta:
			raise ValueError("TomoDataSet provenance length does not match its physical Theta shape")

	def full_window(self) -> TomoWindow:
		pad_before, pad_after = _resolved_padding(
			self.pos, self.shape, self.pad_before, self.pad_after, self.physical_shape,
		)
		return TomoWindow(self.shape, pad_before, self.pos, self.sample_step, self.dtype, self.__work_dtype,
						pad_after=pad_after)

	def __iter__(self) -> Iterator:
		partial_twi = partial(TomoWindowIter, shape=self.step_shape, pad=self.pad_before,
								pad_after=self.pad_after, physical_shape=self.physical_shape,
								dtype=self.dtype, work_dtype=self.__work_dtype,
								substep=self.sample_step)
		return self.frange.get_iter(partial_twi)

	def iter_with(self, *, wsize: DataShape = None, substep: DataShape = None,
					pad: DataShape = None, pad_after: DataShape = None):
		"""Iterate output windows with padding realized against ``physical_shape``."""
		window = self.step_shape
		if wsize is not None:
			window = replace(window, **asdict(wsize))
			frange = replace(self.frange, step=wsize)
		else:
			frange = self.frange

		pad_before = self.pad_before if pad is None else replace(self.pad_before, **asdict(pad))
		if pad_after is None:
			resolved_pad_after = self.pad_after if pad is None else replace(self.pad_after, **asdict(pad))
		else:
			resolved_pad_after = replace(self.pad_after, **asdict(pad_after))

		return frange.get_iter(partial(TomoWindowIter, shape=window, pad=pad_before,
											pad_after=resolved_pad_after, physical_shape=self.physical_shape,
											dtype=self.dtype, work_dtype=self.__work_dtype,
											substep=self.sample_step if substep is None else substep))

	@classmethod
	def from_disk(cls, source: PathLike, fov: CTFov, shape_type: type, over: DataRange | DataShape,
					angle_steps: np.ndarray = None, work_dtype: type = np.float32) -> Self:
		""" Creates a TomoWindowSet from a multi-page tiff or h5 file, or a set of tiff files.

			:param source: Path to input
			:param shape_type: Type for the shape of the data (ProjOrder, SinoOrder, ReconOrder).
			:param window_size: Size of the window (in the y dimension) that will be created when moving through the data.
			:param data_range: Size of the window (in the theta/Z dimension) based on the files.
			:param fov: Field of view details of the image, for handling helical reconstruction.
			:param angle_steps: # of angle steps to take for each projection, handling dropped frames.
			:retval: DataWindowSet for the files.
			"""

		# Dataset discovery is the only window operation that needs storage-format
		# libraries.  Keep them out of pure planning/import paths.
		import h5py
		from natsort import natsorted
		import tifffile as tf

		source = Path(source)
		if source.is_dir():
			source_files = natsorted(source.glob("**/*.tif*"))
			with tf.TiffFile(source_files[0]) as tif:
				img_shape = tif.pages[0].shape
				img_dtype = tif.pages[0].dtype
				length = len(source_files)
		elif source.suffix == ".h5":
			h5_ds = h5py.File(source, 'r')['exchange']['data']
			img_shape = h5_ds.shape[1], h5_ds.shape[2]
			img_dtype = h5_ds.dtype
			length = len(h5_ds)
		elif source.suffix in [".tif", ".tiff"]:
			with tf.TiffFile(source) as tif:
				img_shape = tif.pages[0].shape
				img_dtype = tif.pages[0].dtype
				length = len(tif.pages)
		else:
			print("Could not ID file type. exiting.")

		if fov.Theta is None:
			if "Theta" in [field.name for field in fields(shape_type)]:
				fov = fov._replace(Theta=length)

		new_shape = shape_type(length, *img_shape)
		return cls(fov, new_shape, over, angle_steps, img_dtype, work_dtype)
