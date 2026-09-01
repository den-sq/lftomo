import numpy as np

from lftomo.util.shapes import ProjOrder, Y, ImgSlice
from lftomo.util.window import DataRef, TomoDataSet, CTFov, TomoWindow


def attach_source_geometry(
	state, *, theta_start=0, theta_stride=1, y_start=0,
	physical_theta=None, physical_y=None,
):
	"""Attach canonical full-source metadata and current-window coordinates in tests."""

	shape = state.buffer_spec.shape.to(ProjOrder)
	physical_theta = state.scan.n_projections if physical_theta is None else physical_theta
	physical_y = y_start + shape.Y if physical_y is None else physical_y
	physical = ProjOrder(physical_theta, physical_y, shape.X)
	angles = np.linspace(0, np.pi, physical_theta, endpoint=False)
	state.dataset = TomoDataSet.from_range(
		CTFov(physical_theta, 180, 1), physical, Y(physical.Y),
		dtype=state.buffer_spec.dtype, angles=angles,
	)
	theta_stop = theta_start + (shape.Theta - 1) * theta_stride + 1
	state.source_window = TomoWindow.from_bounds(
		physical,
		y=(y_start, y_start + shape.Y),
		theta=(theta_start, theta_stop, theta_stride),
		dtype=state.buffer_spec.dtype,
	)
	return state


def generate_dataref():
	shape = ProjOrder(Theta=1900, Y=24, X=2048)
	pos = Y(1024)
	step = Y(1)
	pad = ImgSlice(Y=2, X=4)
	dtype = np.int16
	offset = 1024

	return DataRef(shape, pad, pos, step, dtype, offset=offset)


def generate_dataset():
	fov = CTFov(1969, 180, 1)
	shape = ProjOrder(Theta=1969, Y=240, X=2048)
	over = Y(24)
	dtype = np.int16
	work_dtype = np.float32

	return TomoDataSet.from_range(
		fov, shape, over, dtype=dtype, work_dtype=work_dtype,
		angles=np.linspace(0, np.deg2rad(fov.Degrees), shape.Theta, endpoint=False),
	)


def disk_dataset():
	fov = CTFov(Theta=None, Degrees=180, Count=1)
	over = Y("24")
	work_dtype = np.float32

	return TomoDataSet.from_disk("data/input/Sam19/", fov, ProjOrder, over, None, work_dtype)
