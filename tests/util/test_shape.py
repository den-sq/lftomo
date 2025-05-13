from dataclasses import astuple
from lftomo.util.shapes import ProjOrder, SinoOrder, ImgSlice, Y, DataRange, ReconOrder


def test_shape_casting():
	so = SinoOrder(Y=12, Theta=50, X=100)
	po = ProjOrder(Theta=50, Y=12, X=100)
	ro = ReconOrder(Y=12, X=100, Z=100)
	im = ImgSlice(Y=12, X=100)
	yv = Y(Y=12)

	assert po == so.to(ProjOrder)

	assert im == so.extract(ImgSlice)
	assert im == po.extract(ImgSlice)
	assert yv == so.extract(Y)
	assert yv == po.extract(Y)
	assert yv == im.extract(Y)
	assert ro == ReconOrder.of(so)
	assert ro == ReconOrder.of(po)
	assert astuple(po.cut(Y)) == (50, 100)

	so_to_po = [1, 0, 2]
	po_to_so = [1, 0, 2]

	assert so_to_po == so.transpose(po)
	assert po_to_so == po.transpose(so)


def test_shape_math():
	# 1D Math
	assert Y(5) - Y(3) == Y(2)
	assert ImgSlice(Y=5, X=3) - Y(3) == ImgSlice(Y=2, X=3)
	assert SinoOrder(Y=7, Theta=12, X=3) - Y(1) == SinoOrder(Y=6, Theta=12, X=3)

	assert Y(5) + Y(3) == Y(8)
	assert ImgSlice(Y=4, X=3) + Y(3) == ImgSlice(Y=7, X=3)
	assert SinoOrder(Y=7, Theta=12, X=3) + Y(1) == SinoOrder(Y=8, Theta=12, X=3)

	assert Y(5) * Y(3) == Y(15)
	assert ImgSlice(Y=4, X=3) * Y(4) == ImgSlice(Y=16, X=3)
	assert SinoOrder(Y=7, Theta=12, X=3) * Y(1) == SinoOrder(Y=7, Theta=12, X=3)

	assert Y(5) // Y(3) == Y(1)
	assert ImgSlice(Y=4, X=3) // Y(4) == ImgSlice(Y=1, X=3)
	assert SinoOrder(Y=7, Theta=12, X=3) // Y(1) == SinoOrder(Y=7, Theta=12, X=3)

	assert Y(5) % Y(3) == Y(2)
	assert ImgSlice(Y=4, X=3) % Y(4) == ImgSlice(Y=0, X=3)
	assert SinoOrder(Y=7, Theta=12, X=3) % Y(5) == SinoOrder(Y=2, Theta=12, X=3)

	# Multi-D Math
	assert ImgSlice(Y=5, X=3) - ImgSlice(Y=3, X=1) == ImgSlice(Y=2, X=2)
	assert SinoOrder(Y=7, Theta=12, X=3) - ImgSlice(Y=5, X=3) == SinoOrder(Y=2, Theta=12, X=0)
	assert ProjOrder(Theta=15, Y=9, X=4) - SinoOrder(Y=6, Theta=9, X=1) == ProjOrder(Theta=6, Y=3, X=3)


def test_shape_compare():
	# 1D Compare
	assert Y(5) > Y(3)
	assert not ImgSlice(Y=5, X=3) > Y(8)
	assert SinoOrder(Y=7, Theta=12, X=3) > Y(6)

	assert Y(5) >= Y(5)
	assert ImgSlice(Y=5, X=3) >= Y(4)
	assert not SinoOrder(Y=7, Theta=12, X=3) >= Y(8)

	assert Y(5) <= Y(5)
	assert not ImgSlice(Y=5, X=3) <= Y(4)
	assert SinoOrder(Y=7, Theta=12, X=3) <= Y(8)

	assert not Y(5) < Y(2)
	assert ImgSlice(Y=5, X=3) < Y(12)
	assert SinoOrder(Y=7, Theta=12, X=3) < Y(8)

	# Multi-D Compare
	assert not ImgSlice(Y=5, X=3) > ImgSlice(Y=4, X=4)
	assert SinoOrder(Y=7, Theta=12, X=3) > ProjOrder(Theta=11, Y=6, X=2)
	assert not ProjOrder(Theta=12, Y=11, X=7) > SinoOrder(Y=7, Theta=12, X=3)


def test_shape_func():
	# Flat Creation
	po = ProjOrder.make_with(2)
	assert po == ProjOrder(2, 2, 2)

	# Range Creation
	dr = Y.drange(0, 5, 1)
	alt_dr = DataRange(Y(0), Y(5), Y(1))
	assert dr == alt_dr

	dr_seq = ImgSlice.drange((0, 0), (4, 2), (1, 1))
	match_seq = DataRange(ImgSlice(Y=0, X=0), ImgSlice(Y=4, X=2), ImgSlice(Y=1, X=1))
	assert dr_seq == match_seq

	daft_seq = Y.drange(Y(0), Y(5), Y(1))
	assert daft_seq == dr


def test_range_1D():
	basic_1d = DataRange(Y(0), Y(6), Y(1))
	basic_1d_list = list(basic_1d)
	match = [Y(x) for x in range(0, 6)]

	assert basic_1d_list == match

	dr_pos = DataRange(ProjOrder(Theta=30, Y=10, X=5), Y(30), Y(2))
	dr_pos_list = list(dr_pos)
	pos_list_ext = [ProjOrder(Theta=30, Y=y, X=5) for y in range(10, 30, 2)]

	assert dr_pos_list == pos_list_ext

	extra_space = Y.drange(0, 12, 5)
	extra_space_list = list(extra_space)
	extra_space_match = [Y(0), Y(5), Y(10)]

	assert extra_space_list == extra_space_match


def test_range_2D():
	basic_2d = ImgSlice.drange((0, 0), (4, 2), (1, 1))
	basic_2d_list = list(basic_2d)
	match = sum([[ImgSlice(Y=y, X=0), ImgSlice(Y=y, X=1)] for y in range(0, 4)], [])

	print(basic_2d_list)
	print(match)

	assert basic_2d_list == match


def test_range_3D():
	basic_3d = ProjOrder.drange((10, 5, 1), (12, 9, 9), (1, 2, 4))
	basic_3d_list = list(basic_3d)
	match = sum(sum([[[ProjOrder(theta, y, x) for x in range(1, 9, 4)] for y in range(5, 9, 2)]
						for theta in range(10, 12, 1)], []), [])

	print(basic_3d_list)
	print(match)

	assert basic_3d_list == match

	space_3d = ProjOrder.drange((0, 0, 0), (3, 3, 3), (2, 2, 2))
	space_3d_list = list(space_3d)
	space_3d_match = sum(sum([[[ProjOrder(theta, y, x) for x in range(0, 3, 2)] for y in range(0, 3, 2)]
								for theta in range(0, 3, 2)], []), [])

	assert space_3d_list == space_3d_match
