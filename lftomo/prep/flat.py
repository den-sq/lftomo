from collections import namedtuple
from enum import Enum
from pathlib import Path

import numpy as np
import tifffile as tf

from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import DataRange, ImgSlice, SinoOrder, Theta
from lftomo.util.window import DataRef, TomoWindow
from lftomo.util.log import log


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


def _weighted_normalize_projection(target: np.ndarray, source: np.ndarray, flats: np.ndarray,
									projection: TomoWindow, scope: TomoWindow,
									pre_sub: bool = False) -> None:
	"""Weight a source projection relative to its source-coordinate scope."""
	position = projection.interval("Theta").start - scope.interval("Theta").start
	weights = [scope.shape.Theta - position.Theta, position.Theta]
	dark_map = np.average(
		flats[FLAT.PREDARK.index:FLAT.POSTDARK.index + 1], axis=0, weights=weights
	)
	gain_map = np.average(
		flats[FLAT.PREGAIN.index:FLAT.POSTGAIN.index + 1], axis=0, weights=weights
	)
	if not pre_sub:
		gain_map = np.subtract(gain_map, dark_map)
	temp = np.subtract(source.astype(target.dtype), dark_map)

	gain_zero = gain_map == 0
	if gain_zero.any():
		nonzero = np.abs(gain_map[np.invert(gain_zero)])
		if nonzero.size == 0:
			raise ValueError("weighted normalization gain map contains only zeros")
		gain_map[gain_zero] = np.min(nonzero)

	target[...] = np.divide(temp, gain_map)


# Normalziation cacluation using weighted pre/post.
def weighted_normalize(sino_mem: SharedNPArray, flats_mem: SharedNPArray, sino_read: SharedNPArray,
						window: TomoWindow, projection: int, pre_sub: bool = False):
	"""Normalizes single projection using gain and dark flats, weighing pre and post darks based on projection #.

		:param sino_mem: Shared memory metadata for sinogram.
		:param flats_mem: Shared memory metadata for flats (gains, darks).
		:param sino_read: Shared memory metadata for input data.
		:param window: Detector window in either ProjOrder or SinoOrder; buffers remain Y-first.
		:param projection: Dense local projection coordinate to normalize, not a packed ordinal.
		:param pre_sub: Whether darks have already been subtracted from flats.
	"""

	internal = window.internal
	index = SinoOrder.point_index(Y=internal.axis_slice("Y"), Theta=projection)
	with sino_mem[index] as sino, sino_read[index] as sino_sm, flats_mem as flats:
		_weighted_normalize_projection(
			sino,
			sino_sm,
			flats[:, window.axis_slice("Y"), :],
			window.rescope(
				pos=(window.interval("Theta").start + Theta(projection)).merge(window.pos),
				shape=Theta(1),
			),
			window,
			pre_sub,
		)


def shift_normalize():
	pass


def calculate_shift(flats: np.ndarray, scope: ImgSlice = ImgSlice(Y=20, X=20),
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
	interior = DataRef(
		ImgSlice(*flats_map.shape), ImgSlice(0, 0), ImgSlice(0, 0),
		ImgSlice(Y=interval, X=1), flats_map.dtype,
	).rescope(pad=scope)

	log.write("Flat Shift", "Map and Scope Created.")

	# Candidate beam displacements in detector coordinates.
	positions = DataRange(-interior.pad_before, interior.pad_after, ImgSlice(1, 1))

	# Completely blown out areas - No change at all
	defect_mask = flats_map != 1.0

	# Results (Average and Standard Deviation) arrays.
	dist_avg = np.empty(positions.length.as_tuple())
	dist_stddev = np.empty(positions.length.as_tuple())

	log.write("Flat Shift", f"Map and Scope Created {flats_map[interior.index].shape}"
							f"{defect_mask[interior.index].shape}, {np.count_nonzero(defect_mask)}")

	# Mask Test
	test_scope = interior.rescope(pos=ImgSlice(0, 0))
	test = np.logical_and(defect_mask[interior.index], defect_mask[test_scope.index])
	log.write("Flat Shift", f"{np.count_nonzero(test)}")

	out_base = Path("..", "data", "output", "flat_gen")
	out_base.mkdir(parents=True, exist_ok=True)

	def calc_linemap(pos: ImgSlice):
		""" Calculates accuracy of inverted linemaps """
		pos_scope = (pos - positions.start).as_tuple()
		local_scope = interior.rescope(pos=interior.pos + pos)
		local_mask = np.logical_and(defect_mask[local_scope.index], defect_mask[pos_scope])

		temp = np.add(flats_map[interior.index][local_mask], flats_map[local_scope.index][local_mask])
		tf.imwrite(out_base.joinpath(f"sum_{pos.Y}_{pos.X}.tif"),
					np.add(flats_map[interior.index], flats_map[local_scope.index]))

		dist_avg[pos_scope] = np.average(temp)
		dist_stddev[pos_scope] = np.std(temp)

	for position in positions:
		calc_linemap(position)

	log.write("Flat Shift", "Linemap Calculated")

	results = np.argsort(dist_stddev, axis=None).reshape(positions.length.as_tuple())

	log.write("Flat Finding", f"{results.shape}")

	with open("test_other.txt", "w") as change_data:
		for position in positions:
			index = (position - positions.start).as_tuple()
			change_data.write(f"{position.Y}:{position.X}:{dist_stddev[index]}:{results[index]}\n")


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
