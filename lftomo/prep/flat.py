from collections import namedtuple
from enum import Enum
from pathlib import Path

import numpy as np
import tifffile as tf

from lftomo.util.mem import SharedNP
from lftomo.util.window import DataWindow, ImgShape
from lftomo.util.log import log


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
		temp = np.subtract(sino_sm[:, projection, :].astype(sino_mem.dtype), dark_map)

		gain_zero = (gain_map == 0).reshape(gain_map.shape)
		if gain_zero.sum() > 0:
			gain_map[gain_zero] = np.min(np.abs(gain_map[np.invert(gain_zero)]))

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
		flats[FLAT.PREGAIN.index] -= flats[FLAT.PREDARK.index]
		flats[FLAT.POSTGAIN.index] -= flats[FLAT.POSTDARK.index]

	# generate division
	flats[FLAT.PREGAIN.index][flats[FLAT.PREGAIN.index] == 0] = 0.000000001
	flats_map = np.divide(flats[FLAT.POSTGAIN.index], flats[FLAT.PREGAIN.index])
	flats_scope = np.s_[scope.Y: -scope.Y: interval, scope.X: -scope.X]

	log.write("Flat Shift", "Map and Scope Created.")

	# Array for map ranges.
	pos_range = [Distance(Y=y, X=x) for y in range(-scope.Y, scope.Y)
									for x in range(-scope.X, scope.X)]

	# Completely blown out areas - No change at all
	defect_mask = flats_map != 1.0

	# Results (Average and Standard Deviation) arrays.
	dist_avg = np.empty((scope.Y * 2, scope.X * 2))
	dist_stddev = np.empty((scope.Y * 2, scope.X * 2))

	log.write("Flat Shift", f"Map and Scope Created {flats_map[flats_scope].shape}"
							f"{defect_mask[flats_scope].shape}, {np.count_nonzero(defect_mask)}")

	# Mask Test
	test_scope = np.s_[0: flats_map.shape[0] - 2 * scope.Y: interval, 0: flats_map.shape[1] - 2 * scope.X]
	test = np.logical_and(defect_mask[flats_scope], defect_mask[test_scope])
	log.write("Flat Shift", f"{np.count_nonzero(test)}")

	out_base = Path("..", "data", "output", "flat_gen")
	out_base.mkdir(parents=True, exist_ok=True)

	def calc_linemap(pos: Distance):
		""" Calculates accuracy of inverted linemaps """
		pos_scope = np.s_[pos.Y + scope.Y, pos.X + scope.X]
		local_scope = np.s_[pos.Y + scope.Y: flats_map.shape[0] + pos.Y - scope.Y:interval,
								pos.X + scope.X: flats_map.shape[1] + pos.X - scope.X]
		local_mask = np.logical_and(defect_mask[local_scope], defect_mask[pos_scope])

		temp = np.add(flats_map[flats_scope][local_mask], flats_map[local_scope][local_mask])
		tf.imwrite(out_base.joinpath(f"sum_{pos.Y}_{pos.X}.tif"),
					np.add(flats_map[flats_scope], flats_map[local_scope]))

		dist_avg[pos_scope] = np.average(temp)
		dist_stddev[pos_scope] = np.std(temp)

	list(map(calc_linemap, pos_range))

	log.write("Flat Shift", "Linemap Calculated")

	results = np.argsort(dist_stddev, axis=None).reshape(40, 40)

	log.write("Flat Finding", f"{results.shape}")

	with open("test_other.txt", "w") as change_data:
		for y in range(0, scope.Y * 2):
			for x in range(0, scope.X * 2):
				change_data.write(f"{y - scope.Y}:{x - scope.X}:{dist_stddev[y,x]}:{results[y,x]}\n")


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
