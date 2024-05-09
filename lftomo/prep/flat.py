from collections import namedtuple
from enum import Enum

import numpy as np
import tifffile as tf

FlatPair = namedtuple("FlatPair", ['Index', 'Offset'])


class FLAT(Enum):
	PREGAIN = FlatPair(0, -1)
	POSTGAIN = FlatPair(1, 1)
	PREDARK = FlatPair(2, -2)
	POSTDARK = FlatPair(3, 2)

	def __str__(self):
		return str(self.name.lower())

	def opp(self):
		tens = self._value_.Index // 2
		ones = self._value_.Index % 2
		return FLAT(2 * tens + 1 - ones)

	def __getitem__(self):
		return self._value_.Index

	@property
	def index(self):
		return self._value_.Index

	@property
	def offset(self):
		return self._value_.Offset


def calculate_shift(flats):
	interval = flats.shape[1] // 20
	horiz_range = 20
	flats_map = np.divide(
		np.divide(flats[1], flats[0]),
		np.divide(flats[3], flats[2])
	)

	lines = flats_map[list(range(0, flats.shape[1], 20)), horiz_range:-horiz_range]

	# Cleanup for loop
	for x in range(0, lines.shape[0]):
		temp = np.add(flats_map[horiz_range: -horiz_range], lines[x, :])
		sums = np.std(temp, axis=0)
	pass
	# generate division
	# choose n lines
	# For each line
	# 	Sum against every line in the division, getting:
	# 		average intensity value
	# 		deviation from average intensity value
	# 		Whether random line had higher or lower intensity value
	# For Now:
	# 	Print lines, step with lowest devation when lower or higher


def shifting_flat(data_mem, flat_mem):
	pass


def test_shift():
	dim = tf.TiffFile("predark_median.tif").pages[0].shape
	ar = np.zeros((4, dim[0], dim[1]))
	ar[0] = tf.imread("predark_median.tif")
	ar[1] = tf.imread("pregain_median.tif")
	ar[0] = tf.imread("postdark_median.tif")
	ar[1] = tf.imread("postgain_median.tif")
	calculate_shift(ar)
