from collections import namedtuple
from copy import deepcopy
from dataclasses import asdict, replace, fields
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Self, Sequence, Iterator

import h5py
from natsort import natsorted
import numpy as np
import tifffile as tf

from .shapes import DataShape, DataRange, IntIter, SliceIter, TFIter

# Field of view during data acquisition.
CTFov = namedtuple("CTFov", ['Theta', 'Degrees', "Count"])


class DataRef(object):
	def __init__(self, shape: DataShape, pad: DataShape, pos: DataShape,
					step: DataShape, dtype: type = None, offset: int = 0):

		pad_shift = pos.param_max(pos, pad) - pos
		if pad_shift:
			print("Shifting position and shape for padding.")

		self.__pos = pos + pad_shift
		self.__shape = shape - pad_shift  # If padding reallocates position, we shrink.
		self.__pad = pad
		self.__step = step
		self.__dtype = dtype
		self.__offset = offset

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
		return self.__pad

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

	def with_pad(self, pad_dim: Sequence = None, shrink=False) -> Self:
		new_ref = deepcopy(self)
		if pad_dim is None:
			pad_dim = [field.name for field in fields(self.__pad)]
		try:
			kept = replace(self.__pad, **{key: 0 for key in pad_dim})
		except ValueError as ve:
			ve.add_note("No padding found for dimension.")

		shift = self.pad - kept
		new_ref.__shape = new_ref.__shape + shift + shift  # maybe I do need that int math
		new_ref.__pad = kept
		new_ref.__pos = self.pos - shift.extract(self.pos) 	# removing padding so don't need to recheck dim.
		return new_ref

	def rescope(self, pos: DataShape = None, pad: DataShape = None, dtype: type = None, shape: DataShape = None) -> Self:
		new_ref = deepcopy(self)

		if pos is not None:
			new_ref.__pos = replace(self.pos, **asdict(pos))
		if pad is not None:
			pad_shift = pad + (-self.pad)
			new_ref.__pad = replace(self.pad, **asdict(pad))
			if pos is None:
				new_ref.__pos += pad_shift
		if dtype is not None:
			new_ref.__dtype = dtype
		if shape is not None:
			new_ref.__shape = replace(self.shape, **asdict(shape))
		elif pad is not None:
			new_ref.__shape -= (pad_shift + pad_shift)

		pos_pad_shift = new_ref.__pos.param_max(new_ref.__pos, new_ref.__pad) - new_ref.__pos
		if pos_pad_shift:
			print(f"Shifting position and shape to allow for padding by {pos_pad_shift}")
			new_ref.__pos += pos_pad_shift
			new_ref.__shape -= pos_pad_shift

		return new_ref

	def int_iter(self) -> Iterator:
		return self.frange.get_iter(conv=IntIter)

	def step_iter(self) -> Iterator:
		return self.frange.get_iter()

	def slice_iter(self) -> Iterator:
		return self.frange.get_iter(conv=partial(SliceIter, shape=self.shape))


class TomoWindow(DataRef):
	def __init__(self, shape: DataShape, pad: DataShape, pos: DataShape, step: DataShape,
					dtype: type, work_dtype: type):
		super().__init__(shape, pad, pos, step, dtype)
		self.__work_dtype = work_dtype

	@property
	def internal(self):
		return self.rescope(pos=self.pad.extract(self.pos), dtype=self.__work_dtype)


class TomoWindowIter(TFIter):
	def __init__(self, dr: DataRange, shape: DataShape, pad: DataShape,
					dtype: type, work_dtype: type, substep: DataShape = None):
		self.shape = shape
		self.pad = pad
		self.dtype = dtype
		self.work_dtype = dtype
		self.stop = dr.stop
		self.substep = substep if substep is not None else dr.step.make_with(1)

	def __call__(self, pos) -> tuple:
		step_end = (self.shape + pos + self.pad).extract(pos)

		if step_end < self.stop:
			winshape = self.shape
		else:
			# Standard shape would extend beyond the range, so cut it down.
			winshape = replace(self.shape, **{k: v for k, v in asdict(self.stop - pos - self.pad.extract(pos)).items()
											if v < asdict(self.shape)[k]})

		return TomoWindow(winshape, self.pad, pos, self.substep, self.dtype, self.work_dtype)


class TomoDataSet(DataRef):
	""" Set of windows for slicing data into smaller portions.

		Iterable for grabbing each window in order.
	"""

	def __init__(self, fov: CTFov, shape: DataShape, over: DataRange | DataShape,
					angle_steps: np.ndarray = None, dtype: type = np.int16, work_dtype: type = np.float32):
		""" Initializes DataWindowSet.

			:param ds_shape: Shape of the overall dataset.
			:param over: Either a full range to iterate over or just the per-step iteration.
			:param fov: Field of view details of the image, for angle calculation.
			:param dtype: Type of data being iterated over.
			:param angle_steps: # of angle steps to take for each projection, handling dropped frames.
		"""
		if isinstance(over, DataShape):
			pos = over.make_with(0)
			shape = shape
			step = over
		elif isinstance(over, DataRange):
			pos = over.start
			shape = replace(shape, **asdict(over.stop - over.start))
			step = over.step
		else:
			print(f"Invalid over {type(over)} - must be step size (single shape) or a full range (start, stop, step)")

		super().__init__(shape=shape, pad=shape.make_with(0), pos=pos, step=step, dtype=dtype)

		self.__work_dtype = work_dtype

		self.__fov = fov
		self.__angle_steps = angle_steps
		if shape.Theta == fov.Theta * fov.Count:
			self.__angles = np.array(np.linspace(0, np.deg2rad(fov.Degrees), fov.Theta, False))
		else:
			if angle_steps is None:
				raise Exception("Missing Angle Steps for handling Missing Frames")
			else:
				self.__angles = np.array([step * np.deg2rad(fov.Degrees) / fov.Count for step in angle_steps])

	@property
	def angles(self):
		return self.__angles

	@property
	def fov(self):
		return self.__fov

	def full_window(self) -> TomoWindow:
		return TomoWindow(self.shape, self.pad, self.pos, self.step, self.dtype, self.__work_dtype)

	def __iter__(self) -> Iterator:
		partial_twi = partial(TomoWindowIter, shape=self.step_shape, pad=self.pad, dtype=self.dtype,
								work_dtype=self.__work_dtype)
		return self.frange.get_iter(partial_twi)

	def iter_with(self, *, wsize: DataShape = None, substep: DataShape = None):
		window = self.step_shape
		if wsize is not None:
			window = replace(window, **asdict(wsize))
		self.frange.get_iter(partial(TomoWindowIter, window, self.pad, self.dtype, self.__work_dtype, substep))

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


def get_tilt_pad(width: int, rot_center: float, tilt: float):
	# Remvoing minimum pad
	return max(abs(int(np.ceil(max(width - rot_center, rot_center) * np.sin(tilt)))), 0)
