from collections import namedtuple
from enum import Enum

import numpy as np
import tifffile as tf


from lftomo.util.mem import SharedNP
from lftomo.util.window import DataWindow

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


# Normalziation cacluation using weighted pre/post.
def weighted_normalize(sino_mem: SharedNP, flats_mem: SharedNP, sino_read: SharedNP,
						window: DataWindow, projection: int, pre_sub: bool = False):
	"""Normalizes single projection using gain and dark flats, weighing pre and post darks based on projection #.

		:param sino_mem: Shared memory metadata for sinogram.
		:param flats_mem: Shared memory metadata for flats (gains, darks).
		:param sino_read: Shared memory metadata for input data.
		:param window: Sinogram Data Window.
		:param projection: Projection # to normalize.
		:param pre_sub: Whether darks have already been subtracted from flats.
	"""

	with sino_mem[window.internal.Y()] as sino, sino_read[window.internal.Y()] as sino_sm, flats_mem as flats:
		dark_map = np.average(flats[FLAT.PREDARK.index:FLAT.POSTDARK.index + 1, window.Y(), :], axis=0,
								weights=[window.shape.Theta - projection, projection])
		if pre_sub:
			gain_map = np.average(flats[FLAT.PREGAIN.index:FLAT.POSTGAIN.index + 1, window.Y(), :], axis=0,
									weights=[window.shape.Theta - projection, projection])
		else:
			gain_map = np.subtract(np.average(flats[FLAT.PREGAIN.index:FLAT.POSTGAIN.index + 1, window.Y(), :], axis=0,
									weights=[window.shape.Theta - projection, projection]), dark_map)
		gain_map[gain_map == 0.0] = 0.000000001
		temp = np.subtract(sino_sm[:, projection, :].astype(sino_mem.dtype), dark_map)

		sino[:, projection, :] = np.divide(temp, gain_map)


def shift_normalize():
	pass


def calculate_shift(flats):
	# interval = flats.shape[1] // 20
	horiz_range = 20
	flats_map = np.divide(
		np.divide(flats[1], flats[0]),
		np.divide(flats[3], flats[2])
	)

	lines = flats_map[list(range(0, flats.shape[1], 20)), horiz_range:-horiz_range]

	# Cleanup for loop
	for x in range(0, lines.shape[0]):
		# temp = np.add(flats_map[horiz_range: -horiz_range], lines[x, :])
		# sums = np.std(temp, axis=0)
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
