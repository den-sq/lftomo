from collections import namedtuple
from copy import deepcopy
from dataclasses import asdict, replace, fields
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Self, Sequence, Iterator

import numpy as np

from .shapes import DataShape, DataRange, IntIter, SliceIter, TFIter

# Field of view during data acquisition.
CTFov = namedtuple("CTFov", ['Theta', 'Degrees', "Count"])


class DataRef(object):
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

	@property
	def index(self) -> tuple[slice]:
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

	@property
	def step_shape(self) -> DataShape:
		ss = replace(self.shape, **asdict(self.step))
		return ss

	@property
	def stride(self) -> DataShape:
		vals = list(asdict(self.shape).values()) + [1]
		return type(self.shape)(*[self.itemsize * np.prod(vals[x + 1:]) for x in range(len(vals) - 1)])

	def with_pad(self, pad_dim: Sequence = None, shrink=False) -> Self:
		new_ref = deepcopy(self)

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
		new_ref = deepcopy(self)

		new_shape = replace(self.shape, **asdict(shape)) if shape is not None else self.shape
		new_pos = replace(self.pos, **asdict(pos)) if pos is not None else self.pos
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

	def int_iter(self) -> Iterator:
		return self.frange.get_iter(conv=IntIter)

	def step_iter(self) -> Iterator:
		return self.frange.get_iter()

	def slice_iter(self) -> Iterator:
		return self.frange.get_iter(conv=partial(SliceIter, shape=self.shape))


class TomoWindow(DataRef):
	def __init__(self, shape: DataShape, pad: DataShape, pos: DataShape, step: DataShape,
					dtype: type, work_dtype: type, *, pad_after: DataShape = None):
		super().__init__(shape, pad, pos, step, dtype, pad_after=pad_after)
		self.__work_dtype = work_dtype

	@property
	def internal(self):
		return self.rescope(pos=self.pad_before.extract(self.pos), dtype=self.__work_dtype)


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
		self.stop = dr.stop
		self.substep = substep if substep is not None else dr.step.make_with(1)
		self.physical_shape = physical_shape
		self.__step_class = type(dr.step)

	def __call__(self, pos) -> tuple:
		pos = self.__step_class(*pos)
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

	def __init__(self, fov: CTFov, shape: DataShape, over: DataRange | DataShape,
					angle_steps: np.ndarray = None, dtype: type = np.int16, work_dtype: type = np.float32,
					*, angles: np.ndarray = None, acquisition_indices: np.ndarray = None):
		""" Initializes DataWindowSet.

			:param ds_shape: Shape of the overall dataset.
			:param over: Either a full range to iterate over or just the per-step iteration.
			:param fov: Field of view details of the image, for angle calculation.
			:param dtype: Type of data being iterated over.
			:param angle_steps: # of angle steps to take for each projection, handling dropped frames.
		"""
		self.__physical_shape = shape

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

		super().__init__(shape=shape, pad=shape.make_with(0), pos=pos, step=step, dtype=dtype)

		self.__work_dtype = work_dtype

		self.__fov = fov

		if angles is not None:
			if angle_steps is not None:
				raise ValueError("angles and angle_steps are mutually exclusive")
			if len(angles) != shape.Theta:
				raise ValueError("one source angle is required for every projection")
			self.__angles = np.asarray(angles, dtype=np.float32)
		elif angle_steps is None:
			if shape.Theta == fov.Theta * fov.Count:
				self.__angles = np.array(np.linspace(0, np.deg2rad(fov.Degrees), fov.Theta, False))
			else:
				raise Exception("Missing Angle Steps for handling Missing Frames")
		else:
			self.__angles = np.array([step * np.deg2rad(fov.Degrees) / fov.Count for step in angle_steps])
		if acquisition_indices is None:
			acquisition_indices = np.arange(shape.Theta, dtype=np.int64)
		if len(acquisition_indices) != shape.Theta:
			raise ValueError("one acquisition index is required for every projection")
		self.__acquisition_indices = np.asarray(acquisition_indices, dtype=np.int64)

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
	def physical_shape(self) -> DataShape:
		"""Full source shape, independent of the selected output range in ``frange``."""
		return self.__physical_shape

	def full_window(self) -> TomoWindow:
		pad_before, pad_after = _resolved_padding(
			self.pos, self.shape, self.pad_before, self.pad_after, self.physical_shape,
		)
		return TomoWindow(self.shape, pad_before, self.pos, self.step, self.dtype, self.__work_dtype,
						pad_after=pad_after)

	def __iter__(self) -> Iterator:
		partial_twi = partial(TomoWindowIter, shape=self.step_shape, pad=self.pad_before,
								pad_after=self.pad_after, physical_shape=self.physical_shape,
								dtype=self.dtype, work_dtype=self.__work_dtype)
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
											dtype=self.dtype, work_dtype=self.__work_dtype, substep=substep))

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
