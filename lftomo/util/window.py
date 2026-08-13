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
					step: DataShape, dtype: type = None, offset: int = 0):

		pad_shift = pos.param_max(pos, pad) - pos
		self._set_geometry(
			shape - pad_shift,
			pad,
			pos + pad_shift,
			step,
			dtype,
			offset,
		)

	def _set_geometry(self, shape: DataShape, pad: DataShape, pos: DataShape,
					step: DataShape, dtype: type = None, offset: int = 0) -> None:
		"""Set geometry without applying legacy lower-bound pad adjustment.

		Subclasses with an authoritative bounding scope use this seam to preserve
		the requested logical extent and derive clipping separately.
		"""

		self.__pos = pos
		self.__shape = shape
		self.__pad = pad
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

	@property
	def stride(self) -> DataShape:
		vals = list(asdict(self.shape).values()) + [1]
		return type(self.shape)(*[self.itemsize * np.prod(vals[x + 1:]) for x in range(len(vals) - 1)])

	def with_pad(self, pad_dim: Sequence = None, shrink=False) -> Self:
		new_ref = deepcopy(self)

		if pad_dim is None:
			pad_dim = [field.name for field in fields(self.__pad)]
		try:
			kept = replace(self.__pad, **{key: 0 for key in pad_dim})
			shift = self.pad - kept
		except TypeError as te:
			te.add_note("No padding found for dimension.")
			raise te

		DataRef.__init__(new_ref, self.shape + shift + shift, kept, self.pos - shift.extract(self.pos),
						self.step, self.dtype, self.__offset)
		return new_ref

	def rescope(self, pos: DataShape = None, pad: DataShape = None, dtype: type = None, shape: DataShape = None) -> Self:
		new_ref = deepcopy(self)

		new_shape = replace(self.shape, **asdict(shape)) if shape is not None else self.shape
		new_pos = replace(self.pos, **asdict(pos)) if pos is not None else self.pos
		new_dtype = dtype if dtype is not None else self.dtype

		if pad is not None:
			new_pad = replace(self.pad, **asdict(pad))
			pad_shift = pad + (-self.pad)
			new_pos += pad_shift
			new_shape -= (pad_shift + pad_shift)
		else:
			new_pad = self.pad

		DataRef.__init__(new_ref, new_shape, new_pad, new_pos, self.step, new_dtype, self.__offset)
		return new_ref

	def int_iter(self) -> Iterator:
		return self.frange.get_iter(conv=IntIter)

	def step_iter(self) -> Iterator:
		return self.frange.get_iter()

	def slice_iter(self) -> Iterator:
		return self.frange.get_iter(conv=partial(SliceIter, shape=self.shape))


class TomoWindow(DataRef):
	def __init__(self, shape: DataShape, pad: DataShape, pos: DataShape, step: DataShape,
					dtype: type, work_dtype: type, *, bounds: DataRange | None = None,
					dataset: "TomoDataSet | None" = None):
		"""Describe one logical core and its bounded source allocation.

		``shape`` and ``pos`` always describe the requested logical core. ``pad``
		is the symmetric requirement supplied by the caller. The realized source
		extent is derived by clipping that request to ``bounds``; it is never a
		second independently supplied region.
		"""

		if any(value < 0 for value in asdict(pad).values()):
			raise ValueError("window padding cannot be negative")
		self._set_geometry(shape, pad, pos, step, dtype)
		self.__work_dtype = work_dtype
		self.__dataset = dataset
		zero = shape.make_with(0)
		core_pos = replace(zero, **asdict(pos))
		requested = replace(zero, **asdict(pad))
		core_stop = core_pos + shape
		if bounds is None:
			bound_start = zero
			# Directly constructed legacy windows have always clipped at zero but
			# had no upper-bound authority. Preserve that behavior while making
			# their logical core explicit.
			bound_stop = core_stop + requested
		else:
			bound_start = replace(zero, **asdict(bounds.start))
			bound_stop = replace(core_stop, **asdict(bounds.stop))
		if not (bound_start <= core_pos and core_stop <= bound_stop):
			raise ValueError("window core must be contained by its authoritative bounds")
		self.__bounds = DataRange(bound_start, bound_stop, shape.make_with(1))
		self.__source_pos = type(shape).param_max(bound_start, core_pos - requested)
		self.__source_stop = type(shape).param_min(bound_stop, core_stop + requested)

	@property
	def work_dtype(self) -> type:
		return self.__work_dtype

	@property
	def dataset(self) -> "TomoDataSet | None":
		return self.__dataset

	@property
	def bounds(self) -> DataRange:
		return self.__bounds

	@property
	def core_pos(self) -> DataShape:
		return self.pos

	@property
	def core_shape(self) -> DataShape:
		return self.shape

	@property
	def source_pos(self) -> DataShape:
		return self.__source_pos

	@property
	def source_shape(self) -> DataShape:
		return self.__source_stop - self.__source_pos

	@property
	def realized_pad_before(self) -> DataShape:
		zero = self.shape.make_with(0)
		core_pos = replace(zero, **asdict(self.pos))
		return core_pos - self.__source_pos

	@property
	def realized_pad_after(self) -> DataShape:
		zero = self.shape.make_with(0)
		core_stop = replace(zero, **asdict(self.pos)) + self.shape
		return self.__source_stop - core_stop

	@staticmethod
	def _sample_shape(shape: DataShape, step: DataShape) -> DataShape:
		values = asdict(shape)
		for name, stride in asdict(step).items():
			if stride < 1:
				raise ValueError("window steps must be positive")
			values[name] = -(-values[name] // stride)
		return type(shape)(**values)

	@property
	def allocation_shape(self) -> DataShape:
		"""Contiguous buffer shape required for the realized source extent."""

		return self._sample_shape(self.source_shape, self.step)

	@property
	def internal(self):
		"""Return the logical core in contiguous buffer-local coordinates."""

		zero = self.shape.make_with(0)
		core_pos = replace(zero, **asdict(self.pos))
		offset = core_pos - self.source_pos
		local = asdict(offset)
		for name, stride in asdict(self.step).items():
			if local[name] % stride:
				raise ValueError("window core is not aligned to its source stride")
			local[name] //= stride
		local_pos = type(self.pos)(**{
			name: local[name] for name in asdict(self.pos)
		})
		return DataRef(
			self._sample_shape(self.shape, self.step),
			self.shape.make_with(0),
			local_pos,
			self.shape.make_with(1),
			self.__work_dtype,
		)

	def with_pad(self, pad_dim: Sequence = None, shrink=False) -> Self:
		"""Return the bounded source view derived from selected pad dimensions."""

		if pad_dim is None:
			pad_dim = [field.name for field in fields(self.pad)]
		unknown = set(pad_dim) - {field.name for field in fields(self.pad)}
		if unknown:
			raise TypeError("No padding found for dimension.")
		zero = self.shape.make_with(0)
		selected = replace(
			zero,
			**{
				name: value if name in pad_dim else 0
				for name, value in asdict(replace(zero, **asdict(self.pad))).items()
			},
		)
		core_pos = replace(zero, **asdict(self.pos))
		core_stop = core_pos + self.shape
		source_pos = type(self.shape).param_max(self.bounds.start, core_pos - selected)
		source_stop = type(self.shape).param_min(self.bounds.stop, core_stop + selected)
		source_shape = source_stop - source_pos
		return TomoWindow(
			source_shape,
			zero,
			source_pos,
			self.step,
			self.dtype,
			self.__work_dtype,
			bounds=DataRange(source_pos, source_stop, self.shape.make_with(1)),
			dataset=self.__dataset,
		)

	def rescope(self, pos: DataShape = None, pad: DataShape = None,
					dtype: type = None, shape: DataShape = None) -> Self:
		"""Return another logical core resolved against the same bounds."""

		new_shape = replace(self.shape, **asdict(shape)) if shape is not None else self.shape
		new_pos = replace(self.pos, **asdict(pos)) if pos is not None else self.pos
		new_pad = replace(self.pad, **asdict(pad)) if pad is not None else self.pad
		return TomoWindow(
			new_shape, new_pad, new_pos, self.step,
			self.dtype if dtype is None else dtype, self.__work_dtype,
			bounds=self.__bounds, dataset=self.__dataset,
		)

	def expanded(self, pad: DataShape) -> Self:
		"""Return generated output geometry expanded symmetrically by ``pad``.

		Unlike a source halo, generated output padding is not clipped to the input
		dataset. The returned object is still the canonical owner of the resulting
		allocation and coordinate translation.
		"""

		if any(value < 0 for value in asdict(pad).values()):
			raise ValueError("window expansion cannot be negative")
		zero = self.shape.make_with(0)
		expansion = replace(zero, **asdict(pad))
		core_pos = replace(zero, **asdict(self.pos)) - expansion
		core_shape = self.shape + expansion + expansion
		core_stop = core_pos + core_shape
		return TomoWindow(
			core_shape, zero, core_pos, self.step, self.dtype, self.__work_dtype,
			bounds=DataRange(core_pos, core_stop, self.shape.make_with(1)),
			dataset=self.__dataset,
		)

	def rebind(self, dataset: "TomoDataSet") -> Self:
		"""Resolve this same canonical geometry against an authoritative dataset."""

		return dataset.window(
			self.shape,
			pos=self.pos,
			pad=self.pad,
			step=self.step,
			bounds=self.bounds,
		)


class TomoWindowIter(TFIter):
	def __init__(self, dr: DataRange, shape: DataShape, pad: DataShape,
					dtype: type, work_dtype: type, substep: DataShape = None):
		self.shape = shape
		self.pad = pad
		self.dtype = dtype
		self.work_dtype = work_dtype
		self.stop = dr.stop
		self.substep = substep if substep is not None else dr.step.make_with(1)
		self.__step_class = type(dr.step)

	def __call__(self, pos) -> tuple:
		pos = self.__step_class(*pos)
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
					angle_steps: np.ndarray = None, dtype: type = np.int16, work_dtype: type = np.float32,
					*, angles: np.ndarray = None, acquisition_indices: np.ndarray = None):
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

	def full_window(self) -> TomoWindow:
		zero = self.shape.make_with(0)
		dataset_pos = replace(zero, **asdict(self.pos))
		return TomoWindow(
			self.shape, self.pad, self.pos, self.step, self.dtype, self.__work_dtype,
			bounds=DataRange(
				dataset_pos, dataset_pos + self.shape, self.shape.make_with(1),
			),
			dataset=self,
		)

	def window(self, shape: DataShape, *, pos: DataShape = None,
					pad: DataShape = None, step: DataShape = None,
					bounds: DataRange = None) -> TomoWindow:
		"""Resolve a logical window against this dataset's selected source scope."""

		core_shape = replace(self.shape, **asdict(shape))
		zero = self.shape.make_with(0)
		dataset_pos = replace(zero, **asdict(self.pos))
		core_pos = dataset_pos if pos is None else replace(dataset_pos, **asdict(pos))
		requested = zero if pad is None else replace(zero, **asdict(pad))
		resolved_step = core_shape.make_with(1) if step is None else replace(
			core_shape.make_with(1), **asdict(step),
		)
		dataset_stop = dataset_pos + self.shape
		if bounds is None:
			resolved_bounds = DataRange(
				dataset_pos, dataset_stop, self.shape.make_with(1),
			)
		else:
			bound_start = replace(dataset_pos, **asdict(bounds.start))
			bound_stop = replace(dataset_stop, **asdict(bounds.stop))
			if not (dataset_pos <= bound_start and bound_stop <= dataset_stop):
				raise ValueError("window bounds must be contained by the dataset")
			resolved_bounds = DataRange(
				bound_start, bound_stop, self.shape.make_with(1),
			)
		return TomoWindow(
			core_shape, requested, core_pos, resolved_step,
			self.dtype, self.__work_dtype,
			bounds=resolved_bounds, dataset=self,
		)

	def __iter__(self) -> Iterator:
		partial_twi = partial(TomoWindowIter, shape=self.step_shape, pad=self.pad, dtype=self.dtype,
								work_dtype=self.__work_dtype)
		return self.frange.get_iter(partial_twi)

	def iter_with(self, *, wsize: DataShape = None, substep: DataShape = None):
		window = self.step_shape
		if wsize is not None:
			window = replace(window, **asdict(wsize))
			frange = replace(self.frange, step=wsize)
		else:
			frange = self.frange

		return frange.get_iter(partial(TomoWindowIter, shape=window, pad=self.pad, dtype=self.dtype,
											work_dtype=self.__work_dtype, substep=substep))

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
