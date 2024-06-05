from collections import namedtuple
from enum import Enum

import numpy as np
import tifffile as tf


from lftomo.util.mem import SharedNP
from lftomo.util.window import DataWindow, ImgShape


FlatPair = namedtuple("FlatPair", ['Index', 'Offset'])
Distance = namedtuple("Distance", ['Y', 'X'])


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


def calculate_shift(flats: np.ndarray, scope: ImgShape = ImgShape(Y=20, X=20),
					interval: int = 20, presub: bool = False):
	""" Calculate location of beam movement.

		:param flats: Flatfield data.
		:param scope: Range of maximum beam movement to detect.
		:param interval: Interval between lines examined for movement.
		:param presub: Whether darks are presubtracted from gains.
	"""
	if not presub:
		flats[FLAT.PREGAIN] -= flats[FLAT.PREDARK]
		flats[FLAT.POSTGAIN] -= flats[FLAT.POSTDARK]

	# generate division
	flats_map = np.divide(flats[FLAT.POSTGAIN], flats[FLAT.PREGAIN])
	flats_scope = np.s_[scope.Y: -scope.Y, scope.X: -scope.Y]

	# Array for map ranges.
	pos_range = [Distance(Y=y, X=x) for y in range(-scope.Y, scope.Y)
									for x in range(-scope.X, scope.X)]

	# Array for lines to spot check.
	line_intervals = np.array(range(scope.Y, flats.shape[0] - scope.Y, interval))

	# Completely blown out areas - No change at all
	defect_mask = flats_map == 1.0

	# Shifted beam map array.
	linemap = np.empty((scope.Y * 2, scope.X * 2,
						flats.shape[0] - scope.Y * 2, flats.shape[1] - scope.X * 2))

	# Results (Average and Intensity) array.
	dist_change = np.empty((scope.Y * 2, scope.X * 2, 1, 1))

	def make_linemap(pos: Distance):
		""" Assigns entry in 4d-array (linemap) with beam map shifted in X-Y direction.
			Base is beam map limited by scope.

			:param pos: Y,X position ranging from -scope to scope.
		"""
		linemap[pos.Y + scope.Y, pos.X + scope.X, :, :] = flats_map[
				pos.Y + scope.Y: flats_map.shape[0] + pos.Y - scope.Y,
				pos.X + scope.X: flats_map.shape[1] + pos.X - scope.X]

	map(make_linemap, pos_range)

	def calc_linemap(pos: Distance):
		""" Calculates accuracy of inverted linemaps """
		pos_scope = np.s_[pos.Y + scope.Y, pos.X + scope.X]

		temp = np.add(flats_map[flats_scope][defect_mask[flats_scope]],
						linemap[pos_scope][defect_mask[pos_scope]])

		dist_change[pos.Y + scope.Y, pos.X + scope.X] = (np.average(temp), np.std(temp))

	map(calc_linemap, pos_range)

	results = np.argsort(dist_change, axis=3)

	with open("test_out.txt", "w") as result_text:
		result_text.write(results)


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
