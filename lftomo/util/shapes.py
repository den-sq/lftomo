"""
Module containing shapes of data.
"""
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, replace, make_dataclass, Field, InitVar
from functools import cached_property, reduce
import operator
import sys
from typing import ClassVar, Self, Sequence, Union

import numpy as np


class DataShape(ABC):
	__slots__ = ()
	names: ClassVar[tuple[str, ...]] = ()

	@classmethod
	def _install_accessors(cls, names: Sequence[str]) -> None:
		cls.names = tuple(names)
		getter = operator.attrgetter(*cls.names)
		if len(cls.names) == 1:
			cls.as_tuple = lambda self, _getter=getter: (_getter(self),)
		else:
			cls.as_tuple = lambda self, _getter=getter: _getter(self)

	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		annotations = cls.__dict__.get("__annotations__", {})
		if isinstance(annotations, dict) and annotations:
			names = tuple(annotations)
		elif "__dataclass_fields__" in cls.__dict__:
			names = tuple(cls.__dataclass_fields__)
		else:
			names = ()
		if names:
			cls._install_accessors(names)

	@classmethod
	def axis_of(cls, name: str) -> int:
		return cls.names.index(name)

	@classmethod
	def point_index(cls, **indices: int | slice) -> tuple[int | slice, ...]:
		for name in indices:
			cls.axis_of(name)
		return tuple(indices.get(name, slice(None)) for name in cls.names)

	def as_dict(self) -> dict[str, int]:
		return dict(zip(self.names, self.as_tuple()))

	def as_tuple(self) -> tuple[int, ...]:
		return tuple(getattr(self, name) for name in self.names)

	@staticmethod
	def _shape_type(target):
		shape_type = target if isinstance(target, type) else type(target)
		if not issubclass(shape_type, DataShape):
			raise TypeError(f"{shape_type!r} is not a DataShape type")
		return shape_type

	def to(self, target: type | Self) -> Self:
		try:
			dec = self._shape_type(target)
			return dec(**self.as_dict())
		except TypeError as te:
			te.add_note(f"{sys._getframe(0).f_code.co_name}: Must target another DataShape or Datashape Type.")
			raise te

	def extract(self, target: type | Self) -> Self:
		try:
			dec = self._shape_type(target)
			source = self.as_dict()
			return dec(**{name: source[name] for name in dec.names})
		except TypeError as te:
			te.add_note(f"{sys._getframe(0).f_code.co_name}: Must target another Datashape or Datashape Type.")
			raise te
		except KeyError as ke:
			ke.add_note(f"{sys._getframe(0).f_code.co_name}: Target field(s) must exist in source.")
			raise ke

	def cut(self, target: type | Self) -> Self:
		try:
			target_type = self._shape_type(target)
			target_fields = target_type.names
			dec = self.gen([name for name in self.names if name not in target_fields])
			source = self.as_dict()
			return dec(**{k: v for k, v in source.items() if k not in target_fields})
		except TypeError as te:
			te.add_note(f"{sys._getframe(0).f_code.co_name}: Must target another Datashape or Datashape Type.")
			raise te
		except IndexError as ie:
			ie.add_note(f"{sys._getframe(0).f_code.co_name}: Must leave at least 1 field.")
			raise ie

	def merge(self, *targets: type | Self, retain=True) -> Self:
		merge_data = [target.as_dict() for target in targets]
		if retain:
			merge_data += [self.as_dict()]
		else:
			merge_data = [self.as_dict()] + merge_data
		new = reduce(operator.or_, merge_data)
		return self.gen(list(new))(**new)

	def transpose(self, target: type | Self) -> list[int]:
		""" Generates transpose matrix for compatiable with np.transpose and the like."""
		try:
			dec = self._shape_type(target)
			if len(self.names) != len(dec.names):
				raise ValueError(f"{sys._getframe(0).f_code.co_name}: Source and Target must have same number of fields.")

			dest = dec(**self.as_dict())
			return [self.axis_of(name) for name in dest.names]
		except TypeError as te:
			te.add_note(f"{sys._getframe(0).f_code.co_name}: Target must be a datashape or datashape type with the same fields.")
			raise te

	def transposed_view(self, array: np.ndarray, target: type | Self) -> np.ndarray:
		return np.transpose(array, self.transpose(target))

	def prod(self) -> int:
		return int(np.prod(self.as_tuple(), dtype=np.float64))

	def max_field(self) -> int:
		return max(self.as_tuple())

	def nbytes(self, dtype: type | np.dtype) -> int:
		return self.prod() * np.dtype(dtype).itemsize

	def is_sub(self, target: type | Self) -> Self:
		target_type = self._shape_type(target)
		return set(self.names).issubset(target_type.names)

	def __mop(self, other: Self, op) -> Self:
		""" Operator function for operators where order does not matter (add, mult). """
		if type(self) is type(other):
			return type(self)(*map(op, self.as_tuple(), other.as_tuple()))
		names = other.names if other.is_sub(self) else self.names
		return replace(self, **{name: op(getattr(self, name), getattr(other, name)) for name in names})

	def __op(self, other: Self, op) -> Self:
		""" Operator function for operators where order does matter (sub, div, mod). """
		try:
			if type(self) is type(other):
				return type(self)(*map(op, self.as_tuple(), other.as_tuple()))
			return replace(self, **{name: op(getattr(self, name), getattr(other, name)) for name in other.names})
		except AttributeError as ae:
			ae.add_note(f"{op} Failed - Fields of other parameter must be in first.")
			raise ae

	def __cp(self, other: Self, cp) -> Self:
		try:
			if type(self) is type(other):
				return all(map(cp, self.as_tuple(), other.as_tuple()))
			return all(cp(getattr(self, name), getattr(other, name)) for name in other.names)
		except AttributeError as ae:
			ae.add_note(f"{cp} Failed - Fields of other parameter must be in first.")
			raise ae

	def __add__(self, other: Self) -> Self:
		return self.__mop(other, operator.add)

	def __sub__(self, other: Self) -> Self:
		return self.__op(other, operator.sub)

	def __mul__(self, other: Self) -> Self:
		return self.__mop(other, operator.mul)

	def __floordiv__(self, other: Self) -> Self:
		return self.__op(other, operator.floordiv)

	def __mod__(self, other: Self) -> Self:
		return self.__op(other, operator.mod)

	def __lt__(self, other: Self) -> Self:
		return self.__cp(other, operator.lt)

	def __gt__(self, other: Self) -> Self:
		return self.__cp(other, operator.gt)

	def __le__(self, other: Self) -> Self:
		return self.__cp(other, operator.le)

	def __ge__(self, other: Self) -> Self:
		return self.__cp(other, operator.ge)

	def __neg__(self) -> Self:
		return type(self)(*map(operator.neg, self.as_tuple()))

	def __bool__(self) -> Self:
		return any(self.as_tuple())

	@classmethod
	def __param_op(cls, op, *shapes: Self) -> Self:
		fields_dict = [shape.names for shape in shapes]
		try:
			return cls(**{name: op([getattr(shapes[i], name) for i in range(len(fields_dict)) if name in fields_dict[i]])
									for name in cls.names})
		except ValueError as ve:
			ve.add_note(f"{op} Failed - All fields of {cls} must be in at least one argument.")
			raise ve

	@classmethod
	def param_min(cls, *data_shapes: Self) -> Self:
		return cls.__param_op(min, *data_shapes)

	@classmethod
	def param_max(cls, *data_shapes: Self) -> Self:
		return cls.__param_op(max, *data_shapes)

	@staticmethod
	def gen(fields: list[Field] | list[str]) -> Self:
		try:
			names = fields if isinstance(fields[0], str) else [field.name for field in fields]
			shape_type = make_dataclass("_cg_" + "_".join(names), names, bases=(DataShape,))
			shape_type._install_accessors(names)
			return shape_type
		except IndexError as ie:
			ie.add_note(f"{sys._getframe(0).f_code.co_name}: Must leave at least 1 field.")
			raise ie

	@classmethod
	def make_with(cls, val: int) -> Self:
		return cls(*([val] * len(cls.names)))

	@classmethod
	def _coerce(cls, value: Self | int | Sequence[int]) -> Self:
		return (
			value if isinstance(value, cls)
			else cls(*value) if isinstance(value, Sequence)
			else cls.make_with(value)
		)

	@classmethod
	def drange(cls, start: Self | int | Sequence[int],
					stop: Self | int | Sequence[int],
					step: Self | int | Sequence[int]):
		return DataRange(cls._coerce(start), cls._coerce(stop), cls._coerce(step))

	@classmethod
	def interval(cls, start: Self | int | Sequence[int],
					 stop: Self | int | Sequence[int]):
		"""Construct bounds without asserting an iteration stride."""
		return DataRange(cls._coerce(start), cls._coerce(stop))

	@classmethod
	def range_at_interval(cls, start: Self | int | Sequence[int],
					stop: Self | int | Sequence[int],
					interval: Self | int | Sequence[int]):
		istart = start if isinstance(start, cls) else cls(*start) if isinstance(start, Sequence) else cls.make_with(start)
		istop = stop if isinstance(stop, cls) else cls(*stop) if isinstance(stop, Sequence) else cls.make_with(stop)
		interval = interval if isinstance(interval, cls) else cls(*interval) if isinstance(interval, Sequence) else cls.make_with(interval) 	# noqa:E501
		istep = -((istop - istart) // -interval).extract(interval)
		return DataRange(istart, istop, istep)


@dataclass(slots=True)
class ProjOrder(DataShape): Theta: int; Y: int; X: int 	 # noqa: E701,E702
@dataclass(slots=True)
class SinoOrder(DataShape): Y: int; Theta: int; X: int 	 # noqa: E701,E702
@dataclass(slots=True)
class ReverseSinoOrder(DataShape): X: int; Theta: int; Y: int 	 # noqa: E701,E702
@dataclass(slots=True)
class ImageStack(DataShape): Z: int; Y: int; X: int 	 # noqa: E701,E702


@dataclass(slots=True)
class ReconOrder(DataShape):
	Y: int; Z: int; X: int 	 # noqa: E701,E702

	@classmethod
	def of(cls, source: DataShape):
		try:
			return cls(Y=source.Y, Z=source.X, X=source.X)
		except AttributeError as ae:
			ae.add_note("Source shape should have X and Y dimensions.")
			raise ae


@dataclass(slots=True)
class ReverseReconOrder(DataShape): X: int; Z: int; Y: int 	 # noqa: E701,E702


@dataclass(slots=True)
class ImgSlice(DataShape): Y: int; X: int 	 	# noqa: E701,E702
@dataclass(slots=True)
class Proj(DataShape): Y: int; X: int 	 		# noqa: E701,E702
@dataclass(slots=True)
class YSlice(DataShape): Theta: int; X: int 	# noqa: E701,E702
@dataclass(slots=True)
class XSlice(DataShape): Theta: int; Y: int 	# noqa: E701,E702


@dataclass(slots=True)
class Y(DataShape): Y: int 	 # noqa: E701,E702
@dataclass(slots=True)
class X(DataShape): X: int 	 # noqa: E701,E702
@dataclass(slots=True)
class Theta(DataShape): Theta: int 	 # noqa: E701,E702
@dataclass(slots=True)
class Z(DataShape): Z: int 	 # noqa: E701,E702


# If we don't actually know the dimensions of a 3D array.
# Avoids X/Y/Z to prevent being used where it shouldn't.
@dataclass(slots=True)
class GenericOrder(DataShape): A: int; B: int; C: int 	 # noqa: E701,E702


# Deprecated: retained for import compatibility only.
# Removal: delete in the next breaking release; do not add new consumers.
NPString = namedtuple("NPString", 'T')


@dataclass(eq=False)
class DataRange(object):
	start: DataShape; stop: DataShape; step: DataShape | None = None 	 # noqa: E701,E702

	def __post_init__(self):
		if not self.stop.is_sub(self.start):
			raise ValueError("Stop fields must be in Start.")
		if self.step is not None and not self.step.is_sub(self.stop):
			raise ValueError("Step and Stop fields must be in Start.")

	def start_stop(self, name: str) -> tuple[int, int]:
		"""Return the half-open bounds for one semantic axis."""
		self.start.axis_of(name)
		self.stop.axis_of(name)
		return int(getattr(self.start, name)), int(getattr(self.stop, name))

	def __eq__(self, other):
		if isinstance(other, tuple):
			return len(self.start.names) == 1 and self.start_stop(self.start.names[0]) == other
		if not isinstance(other, DataRange):
			return NotImplemented
		return self.start == other.start and self.stop == other.stop and self.step == other.step

	def _iteration_step(self) -> DataShape:
		if self.step is None:
			raise TypeError("an interval has no iteration stride")
		return self.step

	def get_iter(self, conv: 'TFIter' = None):
		return _DataRangeIter(self.start, self.stop, self._iteration_step(), conv)

	@cached_property
	def shape(self) -> DataShape:
		""" Shape of the area iterated over. """
		shape = self.stop - self.start.extract(self.stop)
		return shape if self.step is None else shape.extract(self.step)

	@cached_property
	def length(self) -> DataShape:
		""" Number of steps in each dimension of the iteration. """
		return -(self.shape // -self._iteration_step())

	def __len__(self) -> int:
		""" Total number of steps in the iteration. """
		return self.length.prod()

	def __iter__(self) -> '_DataRangeIter':
		return _DataRangeIter(self.start, self.stop, self._iteration_step())


@dataclass
class _DataRangeIter(DataRange):
	conv: InitVar[callable] = None

	def __post_init__(self, conv: callable):
		self.__it = np.ndindex(self.length.as_tuple()) 			# nditerator
		if conv is not None:
			self.conv = conv(self)
		else:
			self.conv = self.__pos

		self.__step_start = self.start.extract(self.step).as_tuple()
		self.__step = self.step.as_tuple()
		self.__step_f = self.step.names

	def __iter__(self):
		return self

	def __pos(self, step_pos):
		return replace(self.start, **{self.__step_f[i]: step_pos[i] for i in range(len(self.__step_f))})

	def __next__(self):
		# Get index of step (next.(self__it)), find the total traversal (operator.mul)
		# and then the position in the step dimension (operator.add)
		step_pos = tuple(map(operator.add, self.__step_start,
						map(operator.mul, self.__step, next(self.__it))))

		return self.conv(step_pos)


class TFIter(ABC):
	@abstractmethod
	def __call__(self, pos) -> DataShape:
		return pos


class IntIter(TFIter):
	""" TODO: 1d only """
	def __init__(self, dr: DataRange):
		# Increment the first field by 1 to enable clean finishes.
		# increment
		self.__extended_block = (dr.length + dr.start).as_tuple()

	def __call__(self, pos) -> int:
		return np.ravel_multi_index(pos, self.__extended_block)


class ContigMemIter(TFIter):
	def __init__(self, dr: DataRange, offset: int, shape: DataShape, stride: DataShape,
					jump: list[Union[str, Field]]):
		# increment
		self.offset = offset
		self.stride = stride

		self.__drshape = dr.shape.as_dict()
		self.__drstep = dr.step.as_dict()
		self.__shape = shape.as_dict()

		self.__step_stride = stride.extract(dr.step).as_tuple()
		self.__step_per_addr, self.__contig_volume = self.__contig_calc(jump)
		self.__counter = 0
		self.__cached_val = None

	@property
	def contig_stride(self):
		return self.__contig_volume

	def __contig_calc(self, break_fields):
		stride_dict = self.stride.as_dict()
		stride_keys = list(stride_dict.keys())

		break_fields = [f if not isinstance(f, Field) else f.name for f in break_fields]
		contig_field = None
		step_per_addr = None

		for field in reversed(stride_keys):
			if field in break_fields:
				# Break fields are in separate files (or are otherwise non-contiguous), so anything in the shape before the
				# last of them is non-contiguous.
				break
			if field in self.__drshape:
				if self.__drstep[field] != 1:
					# Not contiguous from here if it's more than one step per read, so it's this field and end.
					contig_field = field
					step_per_addr = 1
					break
				else:
					contig_field = field
					step_per_addr = self.__drshape[field]
					continue

			# Otherwise, this field is fully contiguous in memory.
			contig_field = field
			step_per_addr = None

		if contig_field is None:
			raise IndexError(f"Break fields must not include last field of shape; {break_fields} in {stride_keys}")

		if step_per_addr is not None:
			return step_per_addr, step_per_addr * stride_dict[contig_field]
		else:
			return 1, self.__shape[contig_field] * stride_dict[contig_field]

	def __call__(self, step_pos) -> int:
		# Break Fields offset issue?
		if self.__counter % self.__step_per_addr == 0:
			self.__cached_val = np.sum(tuple(map(operator.mul, step_pos, self.__step_stride)), dtype=int) + self.offset
		self.__counter += 1
		return self.__cached_val


class MemAddressIter(TFIter):
	def __init__(self, dr: DataRange, offset: int, stride: DataShape):
		# Increment the first field by 1 to enable clean finishes.
		self.offset = offset
		self.stride = stride
		self.__step_stride = stride.extract(dr.step).as_tuple()

	def __call__(self, step_pos) -> int:
		# Break Fields offset issue?
		return np.sum(tuple(map(operator.mul, step_pos, self.__step_stride))) + self.offset


class SliceIter(TFIter):
	def __init__(self, dr: DataRange, shape: DataShape):
		self.__base = {name: slice(None, None, None) for name in shape.names}
		self.__step_f = dr.step.names

	def __call__(self, pos) -> tuple:
		# print(f"{val}|{self.__base.keys()}")
		return tuple([base if key not in self.__step_f else pos[self.__step_f.index(key)]
						for key, base in self.__base.items()])
