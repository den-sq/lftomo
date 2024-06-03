from collections import namedtuple
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import Self

from natsort import natsorted
import numpy as np
import tifffile as tf

SinoOrder = namedtuple("SinoOrder", ['Y', 'Theta', 'X'])
ProjOrder = namedtuple("ProjOrder", ['Theta', 'Y', 'X'])
ReconOrder = namedtuple("ReconOrder", ['Y', 'Z', 'X'])

CTFov = namedtuple("CTFov", ['Theta', 'Degrees', "Count"])
ImgShape = namedtuple("ImgShape", ['Y', 'X'])


class DataWindow(object):
	""" Window of data currently being worked with. """

	def __init__(self, shape: namedtuple, pos: int, pad: ImgShape, dtype: type = None, step: int = None):
		self.__shape = shape
		self.__pad = pad
		self.__pos = pos
		self.__dtype = dtype
		self.__window_type = type(self.__shape)
		if step is None:
			self.__step = 1
		else:
			self.__step = step

	def __str__(self):
		return f"{self.__shape} ({self.__dtype}) at {self.__pos}"

	@property
	def start(self) -> int:
		return self.__pos

	@property
	def stop(self) -> int:
		return self.__pos + self.__shape.Y * self.__step

	@property
	def step(self) -> int:
		return self.__step

	@property
	def shape(self) -> namedtuple:
		return self.__shape

	def recon_shape(self) -> namedtuple:
		return ReconOrder(Y=self.__shape.Y, Z=self.__shape.X, X=self.__shape.X)

	@property
	def __len__(self) -> int:
		return self.__shape.Y

	@property
	def dtype(self) -> type:
		return self.__dtype

	@property
	def itemsize(self) -> int:
		return self.__dtype.itemsize

	@property
	def internal(self) -> Self:
		""" Get the window for internal memory, positioned at 0.

			:retval: DataWindow expanded and repositioned by padding.
		"""
		return DataWindow(self.__shape, self.__pad.Y, self.__pad, self.__dtype)

	@property
	def fresh(self) -> Self:
		return DataWindow(self.__shape, self.__pos + self.__pad.Y, ImgShape(0, self.__pad.X),
							self.__dtype, self.__step)

	@property
	def retained(self) -> Self:
		return DataWindow(self.__shape._replace(Y=self.__pad.Y * 2),
							self.__pos - self.__pad.Y, ImgShape(0, self.__pad.X),
							self.__dtype, self.__step)

	@property
	def y_pad(self) -> Self:
		""" Get the window padded in the Y dimension.

			:retval: DataWindow expanded and repositioned by padding.
		"""
		return DataWindow(self.__shape._replace(Y=self.__shape.Y + 2 * self.__pad.Y),
							self.__pos - self.__pad.Y, self.__pad._replace(Y=0), 
							self.__dtype, self.__step)

	@property
	def xy_pad(self) -> Self:
		""" Get the window padded in the X and Y dimension.

			:retval: DataWindow expanded and repositioned by padding.
		"""
		return DataWindow(self.__shape._replace(Y=self.__shape.Y + 2 * self.__pad.Y),
							self.__pos - self.__pad.Y, ImgShape(0, 0), 
							self.__dtype, self.__step)

	def sino(self) -> slice:
		""" Gets the slice of the window in the sinogram domain.
		"""
		return np.s_[self.__pos: self.__pos + self.__shape.Y * self.__step:self.__step, :, :]

	def proj(self) -> slice:
		""" Gets the slice of the window in the projection domain. """
		return np.s_[:, self.__pos: self.__pos + self.__shape.Y * self.__step:self.__step, :]

	def Y(self) -> range:
		""" Gets the slice of the window in the Y dimension. """
		return np.s_[self.__pos:self.__pos + self.__shape.Y * self.__step:self.__step]


class _DataWindowIter(Iterator):
	def __init__(self, data_shape: SinoOrder, data_range: range, pad: ImgShape, dtype: type):
		self.__iter = iter(data_range)
		self.__pos = 0
		self.__data_shape = data_shape
		self.__pad = pad
		self.__dtype = dtype

	def __next__(self):
		self.__pos = next(self.__iter)
		return DataWindow(self.__data_shape, self.__pos, self.__pad, self.__dtype)


class DataWindowSet(object):
	""" Set of windows for slicing data into smaller portions.

		Iterable for grabbing each window in order.
	"""

	def __init__(self, data_shape: namedtuple, window_size: int, data_range: range, fov: CTFov,
					dtype: type = None, angle_steps: np.ndarray = None):
		""" Initializes DataWindowSet.

			:param data_shape: Shape of the overall dataset.
			:param window_size: Size of the window (in the y dimension) that will be created when moving through the data.
			:param data_range: Size of the window (in the theta/Z dimension) based on the files.
			:param fov: Field of view details of the image, for handling helical reconstruction.
			:param dtype: Type of data being iterated over.
			:param angle_steps: # of angle steps to take for each projection, handling dropped frames.
		"""

		if not hasattr(data_shape, 'Y'):
			raise TypeError("data_shape must have a y-dimension to iterate through.")

		self.__data_shape = data_shape
		if data_range is not None:
			self.__data_range = data_range
		else:
			self.__data_range = range(0, self.__data_shape.Y, window_size)

		self.__fov = fov
		self.__pad_shape = ImgShape(0, 0)
		self.__window_size = window_size
		self.__dtype = dtype

		if data_shape.Theta == fov.Theta * fov.Count:
			self.__angles = np.array(np.linspace(0, np.deg2rad(fov.Degrees), fov.Theta, False))
		else:
			if angle_steps is None:
				raise Exception("Missing Angle Steps for handling Missing Frames")
			else:
				self.__angles = np.array([step * np.deg2rad(fov.Degrees) / fov.Count for step in angle_steps])

	def __iter__(self) -> _DataWindowIter:
		return _DataWindowIter(self.__data_shape._replace(Y=self.__window_size),
								self.__data_range, self.__pad_shape, self.__dtype)

	def update_padding(self, pad_shape: ImgShape):
		""" Updates padding values to those of pad_shape.
			None values are ignored (e.g. equivalent is not updated).

			:param pad_shape: NamedTuple (Y and X) with new values (integers).
		"""
		self.__pad_shape = ImgShape(
				pad_shape.Y if pad_shape.Y is not None else self.__pad_shape.Y,
				pad_shape.X if pad_shape.X is not None else self.__pad_shape.X)

	@property
	def window_size(self):
		return self.__window_size

	@property
	def data_shape(self):
		return self.__data_shape

	@property
	def window_shape(self):
		return self.__data_shape._replace(Y=self.__window_size)

	@property
	def angles(self):
		return self.__angles

	@property
	def img_shape(self):
		return ImgShape(self.__data_shape.Y, self.__data_shape.X)

	@property
	def dtype(self):
		return self.__dtype

	def __len__(self):
		return len(range(0, self.__data_range.stop, self.__window_size))

	def full_window(self) -> DataWindow:
		return DataWindow(self.__data_shape, self.__data_range.start, self.__pad_shape, self.__dtype, self.__data_range.step)

	def at_interval(self, interval: int) -> Self:
		""" Todo: Fix Angle Stuff
		"""

		start = self.__data_range.start
		stop = self.__data_range.stop

		return DataWindowSet(self.__data_shape._replace(Y=interval), 1,
								range(start, stop, int((stop - start) / interval)), self.__fov,
								self.__dtype)

	def from_folder(source: PathLike, shape_type: type, window_size: int, fov: CTFov, data_range: range = None,
					angle_steps: np.ndarray = None) -> Self:
		""" Creates a DataWindowSet from a folder of files.

			Currently limited to .tiff files.

			:param source: Path to folder of .tiff files.
			:param shape_type: Type for the shape of the data (ProjOrder, SinoOrder, ReconOrder).
			:param window_size: Size of the window (in the y dimension) that will be created when moving through the data.
			:param data_range: Size of the window (in the theta/Z dimension) based on the files.
			:param fov: Field of view details of the image, for handling helical reconstruction.
			:param angle_steps: # of angle steps to take for each projection, handling dropped frames.
			:retval: DataWindowSet for the files.
			"""
		source_files = natsorted(Path(source).glob("**/*.tif*"))
		with tf.TiffFile(source_files[0]) as tif:
			base_image = tif.pages[0]
			img_shape = ImgShape(base_image.shape[0], base_image.shape[1])

			# Iteration is over Y dimension, so if no data range is specified go over entire Y dimension.
			if data_range is None:
				data_range = range(0, img_shape.Y, window_size)

			if hasattr(shape_type, "Theta"):
				if fov.Theta is None:
					fov = fov._replace(Theta=len(source_files))
				return DataWindowSet(shape_type(Theta=len(source_files), Y=img_shape.Y, X=img_shape.X),
									min(window_size, img_shape.Y), data_range, fov, base_image.dtype, angle_steps)
			elif hasattr(shape_type, "Z"):
				return DataWindowSet(shape_type(Z=len(source_files), Y=img_shape.Y, X=img_shape.X),
									min(window_size, img_shape.Y), data_range, fov, base_image.dtype, angle_steps)
			else:
				raise TypeError("Data Shape Must Contain Y, X, and Theta or Z")


def get_tilt_pad(width: int, rot_center: float, tilt: float):
	# Remvoing minimum pad
	return max(abs(int(np.ceil(max(width - rot_center, rot_center) * np.sin(tilt)))), 0)
