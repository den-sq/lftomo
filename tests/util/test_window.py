from functools import partial
import numpy as np

from lftomo.util.shapes import ProjOrder, Y, ImgSlice, IntIter, SliceIter, Proj
from lftomo.util.window import TomoWindow, TomoWindowIter, CTFov
from tests.sample_data import generate_dataref, generate_dataset, disk_dataset


def test_file_load():
	tds = disk_dataset()

	assert tds.shape == ProjOrder(Theta=1501, Y=2048, X=2048)
	print(tds.fov)
	assert tds.fov == CTFov(Theta=1501, Degrees=180, Count=1)
	assert (tds.angles == np.array(np.linspace(0, np.deg2rad(180), 1501, False))).all()


def test_tilt_pad():
	pass


def test_full_dummy_run():
	pass


def test_dataset_methods():
	tds = generate_dataset()

	# assert tds.angles == np.array(np.linspace(0, np.deg2rad(180), 1969, False))
	assert tds.shape == ProjOrder(Theta=1969, Y=240, X=2048)
	assert tds.frange == Y.drange(0, 240, 24)
	assert tds.pad == ProjOrder(0, 0, 0)

	rtds = tds.rescope(pad=Y(2))

	assert rtds.pad == ProjOrder(Theta=0, Y=2, X=0)
	assert rtds.frange == Y.drange(2, 238, 24)
	assert rtds.shape == ProjOrder(Theta=1969, Y=236, X=2048)

	std_list = list(iter(tds))
	rs_list = list(iter(rtds))

	iter_part = partial(TomoWindowIter, shape=ProjOrder(Theta=1969, Y=240, X=2048), pad=ProjOrder(Theta=0, Y=2, X=0),
										dtype=np.int16, work_dtype=np.float32)

	# assert tds.full_window() == TomoWindow(ProjOrder(Theta=1969, Y=240, X=2048), ProjOrder(Theta=0, Y=0, X=0),
	# 											Y(0), Y(24), dtype=np.int16, work_dtype=np.float32)
	assert iter(tds) == Y.drange(0, 240, 24).get_iter(iter_part)
	assert iter(rtds) == Y.drange(2, 238, 24).get_iter(iter_part)
	print([(f.shape, f.pos) for f in rs_list])
	print([(ProjOrder(Theta=1969, Y=min(24, 236 - x), X=2048), Y(max(x, 2))) for x in range(2, 236, 24)])
	assert [(f.shape, f.pos) for f in std_list] == [(ProjOrder(Theta=1969, Y=min(24, 240 - x), X=2048), Y(x))
												for x in range(0, 240, 24)]
	assert [(f.shape, f.pos) for f in rs_list] == [(ProjOrder(Theta=1969, Y=min(24, 236 - x), X=2048), Y(max(x, 2)))
												for x in range(2, 238, 24)]


def test_1d_window_iter():
	ttwi = partial(TomoWindowIter, shape=ProjOrder(Theta=1900, Y=24, X=2048), pad=Proj(Y=2, X=4),
					dtype=np.int16, work_dtype=np.float32)
	alt = Y.drange(2, 90, 24).get_iter(ttwi)

	alt_list = list(alt)

	assert iter(alt) == alt

	assert [(win.shape, win.pos) for win in alt_list[:-1]] == \
			[(ProjOrder(Theta=1900, Y=24, X=2048), Y(y)) for y in range(2, 74, 24)]
	assert (alt_list[-1].shape, alt_list[-1].pos) == (ProjOrder(Theta=1900, Y=14, X=2048), Y(74))


def test_2d_window_iter():
	# 2D Window
	twitwo_manual = [(ProjOrder(Theta=1900, Y=24, X=24), Proj(Y=2, X=600)),
					(ProjOrder(Theta=1900, Y=24, X=24), Proj(Y=2, X=624)),
					(ProjOrder(Theta=1900, Y=24, X=18), Proj(Y=2, X=648)),
					(ProjOrder(Theta=1900, Y=18, X=24), Proj(Y=26, X=600)),
					(ProjOrder(Theta=1900, Y=18, X=24), Proj(Y=26, X=624)),
					(ProjOrder(Theta=1900, Y=18, X=18), Proj(Y=26, X=648))]

	ttwi_two = partial(TomoWindowIter, shape=ProjOrder(Theta=1900, Y=24, X=24), pad=Proj(Y=2, X=4),
						dtype=np.int16, work_dtype=np.float32)
	alt_two = Proj.drange((2, 600), (46, 670), (24, 24)).get_iter(ttwi_two)

	alt_two_list = list(alt_two)

	print([(win.shape, win.pos) for win in alt_two_list])

	assert iter(alt_two) == alt_two
	assert [(win.shape, win.pos) for win in alt_two_list] == twitwo_manual


def test_tomo_window():
	tw = TomoWindow(ProjOrder(Theta=1900, Y=24, X=2048), ImgSlice(Y=2, X=4),
					Y(1026), Y(1), np.int16, np.float32)

	twi = tw.internal

	assert twi.pos == Y(2)
	assert twi.dtype == np.float32


def test_data_ref_fields():
	df = generate_dataref()

	assert df.shape == ProjOrder(Theta=1900, Y=24, X=2048)
	assert df.pos == Y(1024)
	assert df.pad == ImgSlice(Y=2, X=4)
	assert df.step == Y(1)
	assert df.dtype == np.int16
	assert df.itemsize == 2
	assert str(df) == f"{ProjOrder(Theta=1900, Y=24, X=2048)} ({np.int16}) at {Y(1024)} stepping by {Y(1)}"
	assert df.step_shape == ProjOrder(Theta=1900, Y=1, X=2048)


def test_data_ref_iters():
	df = generate_dataref()

	drange = Y.drange(1024, 1048, 1)
	drange_int_iter = Y.drange(1024, 1048, 1).get_iter(IntIter)
	drange_slice_iter = Y.drange(1024, 1048, 1).get_iter(partial(SliceIter, shape=ProjOrder(Theta=1900, Y=24, X=2048)))

# 	convert objects differ.
# 	assert df.step_iter() == iter(drange)
# 	assert df.int_iter() == drange_int_iter
# 	assert df.slice_iter() == drange_slice_iter

	assert list(df.frange) == [Y(x) for x in range(1024, 1048, 1)]
	assert list(df.step_iter()) == [Y(x) for x in range(1024, 1048, 1)]
	assert list(drange) == [Y(x) for x in range(1024, 1048, 1)]

	assert list(df.int_iter()) == list(range(1024, 1048, 1))
	assert list(drange_int_iter) == list(range(1024, 1048, 1))

	assert list(df.slice_iter()) == list([np.s_[:, x, :] for x in range(1024, 1048, 1)])
	assert list(drange_slice_iter) == list([np.s_[:, x, :] for x in range(1024, 1048, 1)])


def test_data_ref_padding():
	df = generate_dataref()

	dfp = df.with_pad(["Y"])
	print(dfp.shape)
	assert dfp.shape == ProjOrder(Theta=1900, Y=28, X=2048)
	assert dfp.frange == Y.drange(1022, 1050, 1)

	dffp = df.with_pad()
	print(dffp.shape)
	assert dffp.shape == ProjOrder(Theta=1900, Y=28, X=2056)
	assert dffp.frange == Y.drange(1022, 1050, 1)


def test_data_ref_rescope():
	df = generate_dataref()

	dfr = df.rescope(pos=Y(0), dtype=np.float32)
	assert dfr.frange == Y.drange(2, 24, 1)					# Move the pos and reduced the shape due to Y=2 padding.
	assert dfr.itemsize == 4
	assert dfr.shape == ProjOrder(1900, 22, 2048)

	dfr = df.rescope(shape=ProjOrder(1900, 14, 2048))
	assert dfr.frange == Y.drange(1024, 1038, 1)
	assert dfr.itemsize == 2
	assert dfr.shape == ProjOrder(1900, 14, 2048)
