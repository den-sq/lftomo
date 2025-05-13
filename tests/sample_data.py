import numpy as np

from lftomo.util.shapes import ProjOrder, Y, ImgSlice
from lftomo.util.window import DataRef, TomoDataSet, CTFov


def generate_dataref():
	shape = ProjOrder(Theta=1900, Y=24, X=2048)
	pos = Y(1024)
	step = Y(1)
	pad = ImgSlice(Y=2, X=4)
	dtype = np.int16
	offset = 1024

	return DataRef(shape, pad, pos, step, dtype, offset)


def generate_dataset():
	fov = CTFov(1969, 180, 1)
	shape = ProjOrder(Theta=1969, Y=240, X=2048)
	over = Y(24)
	dtype = np.int16
	work_dtype = np.float32

	return TomoDataSet(fov, shape, over, dtype=dtype, work_dtype=work_dtype)


def disk_dataset():
	fov = CTFov(Theta=None, Degrees=180, Count=1)
	over = Y("24")
	work_dtype = np.float32

	return TomoDataSet.from_disk("data/input/Sam19/", fov, ProjOrder, over, None, work_dtype)
