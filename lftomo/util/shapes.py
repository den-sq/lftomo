"""
Module containing shapes of data.
"""
from abc import ABC
from collections import namedtuple
from dataclasses import dataclass, asdict, replace, fields, astuple
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

	def cut(self, target: type | Self) -> Self:
		try:
			dec = target if isinstance(target, type) else type(target)
			source = asdict(self)
			return dec(**{field.name: source[field.name] for field in fields(target)})
		except ValueError as ve:
			ve.add_note("Target field(s) must exist in source.")
			raise ve

	def transpose(self, target: type | Self) -> tuple:
		try:
			dec = target if isinstance(target, type) else type(target)
			source = asdict(self)
			dest = dec(**source)
			if len(source) != len(dest):
				raise ValueError("Source and Target must have same number of fields.")
			return [list(source).index(key) for key in asdict(dest).keys()]
		except ValueError as ve:
			ve.add_note("Source and Target must have same fields.")
			raise ve
		except TypeError as te:
			te.add_note("Must target another DataShape.")
			raise te

	def __op(self, other: Self, op) -> Self:
		try:
			return replace(self, **{key: op(asdict(self)[key], other_val) for key, other_val in asdict(other).items()})
		except ValueError as ve:
			ve.add_note(f"{op} Failed - Fields of other parameter must be in first.")
			raise ve

	def __cp(self, other: Self, cp) -> Self:
		try:
			return all([cp(asdict(self)[key], other_val) for key, other_val in asdict(other).items()])
		except ValueError as ve:
			ve.add_note(f"{cp} Failed - Fields of other parameter must be in first.")
			raise ve

	def __add__(self, other: Self) -> Self:
		return self.__op(other, operator.add)

	def __sub__(self, other: Self) -> Self:
		return self.__op(other, operator.sub)

	def __mul__(self, other: Self) -> Self:
		return self.__op(other, operator.mul)

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
		return replace(self, **{key: -value for key, value in asdict(self).items()})

	def is_sub(self, target: type | Self) -> Self:
		return all([field.name in [field.name for field in fields(target)] for field in fields(self)])

	@classmethod
	def make_with(cls, val: int) -> Self:
		return cls(*([val] * len(fields(cls))))

	@classmethod
	def drange(cls, start: Self | int | Sequence[int],
					stop: Self | int | Sequence[int],
					step: Self | int | Sequence[int]):
		return DataRange(
				start if isinstance(start, cls) else cls(*start) if isinstance(start, Sequence) else cls.make_with(start),
				stop if isinstance(start, cls) else cls(*stop) if isinstance(stop, Sequence) else cls.make_with(stop),
				step if isinstance(start, cls) else cls(*step) if isinstance(step, Sequence) else cls.make_with(step))


@dataclass
class DataRange:
	start: DataShape; stop: DataShape; step: DataShape 	 # noqa: E701,E702

	def __post_init__(self):
		if not (self.step.is_sub(self.stop) and self.stop.is_sub(self.start)):
			raise ValueError("Step and Stop fields must be in Start.")

		for key, val in asdict(self.step).items():
			if val == 0:
				raise ValueError(f"Field {key} has a 0 step and would not progress.")

	def __iter__(self):
		dist = (self.stop - self.start.cut(self.stop)).cut(self.step) 	# Distance traveled in range.
		step_dist = -(dist // -self.step)								# Steps to travel; ceiling division to handle partial areas.

		return _DataRangeIter(self.start, self.stop, self.step, step_dist)


@dataclass
class _DataRangeIter:
	pos: DataShape
	stop: DataShape
	step: DataShape
	step_dist: DataShape

	def __post_init__(self):
		self.at = -1 	# Raveled Index

		# Extra padding in first dimension for clean halt; Probably a cleaner way to do this.
		self.step_block = astuple(self.step_dist + type(self.step_dist)(*[1] + [0] * (len(fields(self.step_dist)) - 1)))

	def __next__(self):
		self.at += 1
		travel = type(self.step)(*np.unravel_index(self.at, self.step_block))
		pos = self.pos + self.step * travel
		if pos < self.stop:
			return pos
		else:
			raise StopIteration


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
class ImgSlice(DataShape): Y: int; X: int 	 # noqa: E701,E702
@dataclass
class YSlice(DataShape): Theta: int; X: int 	 # noqa: E701,E702


@dataclass
class Y(DataShape): Y: int 	 # noqa: E701,E702
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
