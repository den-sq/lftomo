"""
Module containing shapes of data.
"""
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, asdict, replace, fields, astuple, make_dataclass, Field, InitVar
from functools import cached_property
import operator
from typing import Self, Sequence

import numpy as np


class DataShape(ABC):
	def to(self, target: type | Self) -> Self:
		try:
			dec = target if isinstance(target, type) else type(target)
			return dec(**asdict(self))
		except TypeError as te:
			te.add_note("Must target another DataShape.")
			raise te

	def extract(self, target: type | Self) -> Self:
		try:
			dec = target if isinstance(target, type) else type(target)
			source = asdict(self)
			return dec(**{field.name: source[field.name] for field in fields(target)})
		except ValueError as ve:
			ve.add_note("Target field(s) must exist in source.")
			raise ve

	def cut(self, target: type | Self) -> Self:
		try:
			dec = self.gen(fields[target])
			source = asdict(self)
			return dec(**{k: v for k, v in source.items() if k not in asdict(target)})
		except ValueError as ve:
			ve.add_note("Must leave at least 1 field.")
			raise ve

	def transpose(self, target: type | Self) -> tuple:
		""" Generates transpose matrix for compatiable with np.transpose and the like."""
		try:
			if len(fields(self)) != len(fields(target)):
				raise ValueError("Source and Target must have same number of fields.")
			dec = target if isinstance(target, type) else type(target)

			source = asdict(self)
			dest = dec(**source)
			return [list(source).index(key) for key in asdict(dest).keys()]
		except ValueError as ve:
			ve.add_note("Source and Target must have same fields.")
			raise ve
		except TypeError as te:
			te.add_note("Must target another DataShape.")
			raise te

	def is_sub(self, target: type | Self) -> Self:
		return all([field.name in [field.name for field in fields(target)] for field in fields(self)])

	def __mop(self, other: Self, op) -> Self:
		""" Operator function for operators where order does not matter (add, mult). """
		try:
			return replace(self, **{f.name: op(getattr(self, f.name), getattr(other, f.name)) for f in fields(other)})
		except AttributeError:
			return replace(self, **{f.name: op(getattr(self, f.name), getattr(other, f.name)) for f in fields(self)})

	def __op(self, other: Self, op) -> Self:
		""" Operator function for operators where order does matter (sub, div, mod). """
		try:
			return replace(self, **{f.name: op(getattr(self, f.name), getattr(other, f.name)) for f in fields(other)})
		except AttributeError as ae:
			ae.add_note(f"{op} Failed - Fields of other parameter must be in first.")
			raise ae

	def __cp(self, other: Self, cp) -> Self:
		try:
			return all([cp(getattr(self, f.name), getattr(other, f.name)) for f in fields(other)])
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
		return type(self)(**{f.name: -getattr(self, f.name) for f in fields(self)})

	def __bool__(self) -> Self:
		return any(asdict(self).values())

	@classmethod
	def __param_op(cls, op, *shapes: Self) -> Self:
		fields_dict = [[f.name for f in fields(shape)] for shape in shapes]
		try:
			return cls(**{f.name: op([getattr(shapes[i], f.name) for i in range(len(fields_dict)) if f.name in fields_dict[i]])
									for f in fields(cls)})
		except AttributeError as ae:
			ae.add_note(f"{op} Failed - All fields of {cls} must be in at least one argument.")
			raise ae

	@classmethod
	def param_min(cls, *data_shapes: Self) -> Self:
		return cls.__param_op(min, *data_shapes)

	@classmethod
	def param_max(cls, *data_shapes: Self) -> Self:
		return cls.__param_op(max, *data_shapes)

	@staticmethod
	def gen(fields: list[Field] | list[str]) -> Self:
		names = fields if type(fields[0]) is str else [field.name for field in fields]
		return make_dataclass("_cg_" + "_".join(names), names, bases=(DataShape,))

	@classmethod
	def make_with(cls, val: int) -> Self:
		return cls(*([val] * len(fields(cls))))

	@classmethod
	def drange(cls, start: Self | int | Sequence[int],
					stop: Self | int | Sequence[int],
					step: Self | int | Sequence[int]):
		return DataRange(
				start if isinstance(start, cls) else cls(*start) if isinstance(start, Sequence) else cls.make_with(start),
				stop if isinstance(stop, cls) else cls(*stop) if isinstance(stop, Sequence) else cls.make_with(stop),
				step if isinstance(step, cls) else cls(*step) if isinstance(step, Sequence) else cls.make_with(step))

	@classmethod
	def range_at_interval(cls, start: Self | int | Sequence[int],
					stop: Self | int | Sequence[int],
					interval: Self | int | Sequence[int]):
		istart = start if isinstance(start, cls) else cls(*start) if isinstance(start, Sequence) else cls.make_with(start)
		istop = stop if isinstance(stop, cls) else cls(*stop) if isinstance(stop, Sequence) else cls.make_with(stop)
		interval = interval if isinstance(interval, cls) else cls(*interval) if isinstance(interval, Sequence) else cls.make_with(interval) 	# noqa:E501
		istep = -((istart - istop) // -interval).extract(interval)
		return DataRange(istart, istop, istep)


@dataclass
class ProjOrder(DataShape): Theta: int; Y: int; X: int 	 # noqa: E701,E702
@dataclass
class SinoOrder(DataShape): Y: int; Theta: int; X: int 	 # noqa: E701,E702
@dataclass
class BackwardsSinoOrder(DataShape): X: int; Theta: int; Y: int 	 # noqa: E701,E702


@dataclass
class ReconOrder(DataShape):
	Y: int; Z: int; X: int 	 # noqa: E701,E702

	@classmethod
	def of(cls, source: DataShape):
		try:
			return cls(Y=source.Y, Z=source.X, X=source.X)
		except AttributeError as ae:
			ae.add_note("Source shape should have X and Y dimensions.")
			raise ae


@dataclass
class ImgSlice(DataShape): Y: int; X: int 	 	# noqa: E701,E702
@dataclass
class Proj(DataShape): Y: int; X: int 	 		# noqa: E701,E702
@dataclass
class YSlice(DataShape): Theta: int; X: int 	# noqa: E701,E702


@dataclass
class Y(DataShape): Y: int 	 # noqa: E701,E702
@dataclass
class X(DataShape): X: int 	 # noqa: E701,E702
@dataclass
class Theta(DataShape): Theta: int 	 # noqa: E701,E702
@dataclass
class Z(DataShape): Z: int 	 # noqa: E701,E702


# If we don't actually know the dimensions of a 3D array.
# Avoids X/Y/Z to prevent being used where it shouldn't.
@dataclass
class GenericOrder(DataShape): A: int; B: int; C: int 	 # noqa: E701,E702


# ????
NPString = namedtuple("NPString", 'T')


@dataclass
class DataRange(object):
	start: DataShape; stop: DataShape; step: DataShape 	 # noqa: E701,E702

	def __post_init__(self):
		if not (self.step.is_sub(self.stop) and self.stop.is_sub(self.start)):
			raise ValueError("Step and Stop fields must be in Start.")

		for key, val in asdict(self.step).items():
			if val == 0:
				raise ValueError(f"Field {key} has a 0 step and would not progress.")

	def get_iter(self, conv: 'TFIter' = None):
		self.iter_conv = conv
		fetched_iter = self.__iter__()
		del self.iter_conv
		return fetched_iter

	@cached_property
	def length(self) -> DataShape:
		dist = (self.stop - self.start.extract(self.stop)).extract(self.step) 	# Distance traveled in range.
		return -(dist // -self.step)									# Steps to travel; ceiling division to handle partial areas.

	@cached_property
	def __len__(self) -> int:
		s_len = self.length
		return np.ravel_multi_index(astuple(s_len - s_len.make_with(1)), astuple(s_len))

	def __iter__(self) -> '_DataRangeIter':
		return _DataRangeIter(self.start, self.stop, self.step,
							None if not hasattr(self, "iter_conv") else self.iter_conv)


@dataclass
class _DataRangeIter(DataRange):
	conv: InitVar[callable] = None

	def __post_init__(self, conv: callable):
		self.at = -1 	# Raveled Index
		if conv is not None:
			self.conv = conv(self)

		# Increment the first field by 1 to enable unraveling the last index.
		# annoying oneliner bit : + DataShape.gen([fields(self.step_dist)[0]])(1))
		self._step_block = list(asdict(self.length).values())
		self._step_block[0] += 1

	def __iter__(self):
		return self

	def __next__(self):
		self.at += 1
		travel = type(self.step)(*np.unravel_index(self.at, self._step_block))
		pos = self.start + self.step * travel

		if pos < self.stop:
			return pos if self.conv is None else self.conv(pos)
		else:
			raise StopIteration


class TFIter(ABC):
	@abstractmethod
	def __call__(self, pos) -> DataShape:
		return pos


class IntIter(TFIter):
	def __init__(self, dr: DataRange):
		# Increment the first field by 1 to enable clean finishes.
		# Do we need this here? don't think so  increment = DataShape.gen([fields(dr.len))[0]])(1)
		# increment
		self.__extended_block = astuple(dr.length + dr.start)

	def __call__(self, pos) -> int:
		return np.ravel_multi_index(astuple(pos), self.__extended_block)


class SliceIter:
	def __init__(self, dr: DataRange, shape: DataShape):
		self.__base = {field.name: slice(None, None, None) for field in fields(shape)}

	def __call__(self, pos) -> tuple:
		val = asdict(pos)
		print(f"{val}|{self.__base.keys()}")
		return tuple([base if key not in val else val[key] for key, base in self.__base.items()])
