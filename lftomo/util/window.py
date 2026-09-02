from collections import namedtuple
from dataclasses import InitVar, asdict, dataclass, field, fields, replace
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Self, Sequence, Iterator
import warnings

import numpy as np

from .shapes import DataShape, DataRange, Interval, IntIter, ReconOrder, SliceIter, TFIter, Y

# Field of view during data acquisition.
CTFov = namedtuple("CTFov", ['Theta', 'Degrees', "Count"])


@dataclass(frozen=True, eq=False)
class DataRef(object):
	"""A globally positioned reference to typed data geometry."""

	shape: DataShape
	pad_before: DataShape
	pos: DataShape
	step: DataShape
	dtype: np.dtype | None = None
	offset: int = field(default=0, kw_only=True)
	pad_after: DataShape | None = field(default=None, kw_only=True)

	def __post_init__(self):
		object.__setattr__(self, "dtype", None if self.dtype is None else np.dtype(self.dtype))
		object.__setattr__(
			self, "pad_after",
			self.pad_before if self.pad_after is None
			else type(self.pad_before)(**self.pad_after.as_dict()),
		)
		if type(self.offset) is not int:
			raise TypeError("data offset must be an integer")

	@property
	def index(self) -> tuple[slice]:
		start = list(self.pos.merge(self.shape.make_with(0)).as_dict().values())
		stop = list((self.shape + self.pos).as_dict().values())
		step = list(self.step.merge(self.shape.make_with(1)).as_dict().values())

		return tuple([slice(start[i], stop[i], step[i]) for i in range(len(start))])

	@property
	def frange(self) -> DataRange:
		return DataRange(self.pos, (self.shape + self.pos).extract(self.pos), self.step)

	@property
	def pad(self) -> DataShape:
		"""Return the legacy symmetric value, rejecting an ambiguous asymmetric reference."""
		if self.pad_before != self.pad_after:
			raise ValueError("asymmetric padding has no single value; use pad_before and pad_after")
		return self.pad_before

	def __str__(self) -> str:
		dtype = None if self.dtype is None else self.dtype.type
		return f"{self.shape} ({dtype}) at {self.pos} stepping by {self.step}"

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

	def nbytes(self) -> int:
		return self.shape.nbytes(self.dtype)

	@property
	def step_shape(self) -> DataShape:
		ss = replace(self.shape, **asdict(self.step))
		return ss

	@property
	def stride(self) -> DataShape:
		return self.shape.stride(self.dtype)

	def with_pad(self, pad_dim: Sequence = None, shrink=False) -> Self:
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

		return replace(
			self,
			shape=self.shape + shift_before + shift_after,
			pad_before=kept_before,
			pad_after=kept_after,
			pos=self.pos - shift_before.extract(self.pos),
		)

	def rescope(self, pos: DataShape = None, pad: DataShape = None, dtype: type = None,
					shape: DataShape = None, *, pad_after: DataShape = None) -> Self:
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

		return replace(
			self, shape=new_shape, pad_before=new_pad_before, pad_after=new_pad_after,
			pos=new_pos, dtype=new_dtype,
		)

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

	def int_iter(self, axis: str | None = None) -> Iterator[int]:
		"""Iterate one semantic axis across this reference's sample positions."""

		if axis is None:
			if len(self.step.names) != 1:
				return self.frange.get_iter(conv=IntIter)
			axis = self.step.names[0]
		bounds = self.interval(axis)
		return DataRange(bounds.start, bounds.stop, self.step.extract(bounds.start)).get_iter(IntIter)

	def step_iter(self) -> Iterator:
		return self.frange.get_iter()

	def slice_iter(self) -> Iterator:
		return self.frange.get_iter(conv=partial(SliceIter, shape=self.shape))

	def units(self, axis: str) -> Iterator[tuple[Self, tuple[slice, ...]]]:
		"""Yield globally positioned units with their source-buffer-local slices."""

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
			yield unit, tuple(index)


@dataclass(frozen=True, eq=False)
class TomoWindow(DataRef):
	work_dtype: np.dtype = np.dtype(np.float32)

	def __post_init__(self):
		super().__post_init__()
		object.__setattr__(self, "work_dtype", np.dtype(self.work_dtype))

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
		return self.rescope(pos=self.pad_before.extract(self.pos), dtype=self.work_dtype)

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


@dataclass(frozen=True, eq=False)
class TomoDataSet(DataRef):
	""" Set of windows for slicing data into smaller portions.

		Iterable for grabbing each window in order.
	"""
	work_dtype: np.dtype = np.dtype(np.float32)
	physical_shape: DataShape = field(kw_only=True)
	sample_step: DataShape = field(kw_only=True)
	fov: CTFov = field(kw_only=True)
	angles: np.ndarray | None = field(default=None, kw_only=True)
	acquisition_indices: np.ndarray | None = field(default=None, kw_only=True)
	angle_steps: InitVar[np.ndarray | None] = field(default=None, kw_only=True)

	def __post_init__(self, angle_steps: np.ndarray | None):
		super().__post_init__()
		object.__setattr__(self, "work_dtype", np.dtype(self.work_dtype))
		if any(value < 1 for value in self.sample_step.as_tuple()):
			raise ValueError("source sampling steps must be positive")
		for axis, start in self.pos.as_dict().items():
			if start < 0 or not hasattr(self.physical_shape, axis):
				raise ValueError(f"output range starts outside physical axis {axis}")
		for axis, stop in (self.pos + self.shape).as_dict().items():
			if not hasattr(self.physical_shape, axis) or stop > getattr(self.physical_shape, axis):
				raise ValueError(f"output range stops outside physical axis {axis}")

		if self.angles is not None:
			if angle_steps is not None:
				raise ValueError("angles and angle_steps are mutually exclusive")
			angles = np.asarray(self.angles)
		elif angle_steps is None:
			warnings.warn(
				"TomoDataSet FOV angle synthesis is deprecated; pass acquisition-model angles",
				DeprecationWarning,
				stacklevel=3,
			)
			if self.physical_shape.Theta != self.fov.Theta * self.fov.Count:
				raise Exception("Missing Angle Steps for handling Missing Frames")
			angles = np.linspace(
				0, np.deg2rad(self.fov.Degrees), self.fov.Theta, endpoint=False,
			)
		else:
			angles = np.asarray(angle_steps) * np.deg2rad(self.fov.Degrees) / self.fov.Count
		acquisition_indices = (
			np.arange(self.physical_shape.Theta, dtype=np.int64)
			if self.acquisition_indices is None
			else np.asarray(self.acquisition_indices, dtype=np.int64)
		)
		if len(angles) != self.physical_shape.Theta \
				or len(acquisition_indices) != self.physical_shape.Theta:
			raise ValueError("TomoDataSet provenance length does not match its physical Theta shape")
		angles.setflags(write=False)
		acquisition_indices.setflags(write=False)
		object.__setattr__(self, "angles", angles)
		object.__setattr__(self, "acquisition_indices", acquisition_indices)

	@classmethod
	def from_range(cls, fov: CTFov, shape: DataShape, over: DataRange | DataShape,
					angle_steps: np.ndarray = None, dtype: type = np.int16,
					work_dtype: type = np.float32, *, angles: np.ndarray = None,
					acquisition_indices: np.ndarray = None, pad: DataShape = None,
					pad_after: DataShape = None, sample_step: DataShape = None) -> Self:
		""" Initializes DataWindowSet.

			:param ds_shape: Shape of the overall dataset.
			:param over: Either a full range to iterate over or just the per-step iteration.
			:param fov: Field of view details of the image, for angle calculation.
			:param dtype: Type of data being iterated over.
			:param angle_steps: # of angle steps to take for each projection, handling dropped frames.
		"""
		physical_shape = shape
		unit_step = shape.make_with(1)
		sample_step = (
			unit_step if sample_step is None
			else replace(unit_step, **asdict(sample_step))
		)

		if isinstance(over, DataShape):
			pos = over.make_with(0)
			selected_shape = shape
			step = over
		elif isinstance(over, DataRange):
			for axis, start in asdict(over.start).items():
				if start < 0 or not hasattr(physical_shape, axis):
					raise ValueError(f"output range starts outside physical axis {axis}")
			for axis, stop in asdict(over.stop).items():
				if not hasattr(physical_shape, axis) or stop > getattr(physical_shape, axis):
					raise ValueError(f"output range stops outside physical axis {axis}")
			pos = over.start
			selected_shape = replace(shape, **asdict(over.stop - over.start))
			step = over.step
		else:
			raise TypeError(f"Invalid over {over}:{type(over)} - must be step size (DataShape) or a full range (DataRange)")

		zero = selected_shape.make_with(0)
		pad_before = zero if pad is None else replace(zero, **asdict(pad))
		resolved_pad_after = (
			pad_before if pad_after is None else replace(zero, **asdict(pad_after))
		)

		return cls(
			selected_shape, pad_before, pos, step, dtype, work_dtype,
			pad_after=resolved_pad_after, physical_shape=physical_shape,
			sample_step=sample_step, fov=fov,
			angles=None if angles is None else np.asarray(angles, dtype=np.float32),
			angle_steps=angle_steps, acquisition_indices=acquisition_indices,
		)

	def full_window(self) -> TomoWindow:
		pad_before, pad_after = _resolved_padding(
			self.pos, self.shape, self.pad_before, self.pad_after, self.physical_shape,
		)
		return TomoWindow(self.shape, pad_before, self.pos, self.sample_step, self.dtype, self.work_dtype,
						pad_after=pad_after)

	def __iter__(self) -> Iterator:
		partial_twi = partial(TomoWindowIter, shape=self.step_shape, pad=self.pad_before,
								pad_after=self.pad_after, physical_shape=self.physical_shape,
								dtype=self.dtype, work_dtype=self.work_dtype,
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
											dtype=self.dtype, work_dtype=self.work_dtype,
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
		return cls.from_range(fov, new_shape, over, angle_steps, img_dtype, work_dtype)
