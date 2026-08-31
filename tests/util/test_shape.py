import pytest

import numpy as np

from dataclasses import astuple, asdict
from functools import partial
from timeit import repeat
import lftomo.util.shapes as shape_module
from lftomo.util.shapes import ImageStack, ProjOrder, ReverseReconOrder, ReverseSinoOrder, SinoOrder, ImgSlice, Y, DataRange, ReconOrder, Proj, Theta, YSlice, XSlice
from lftomo.util.shapes import IntIter, ContigMemIter, MemAddressIter, SliceIter


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

	# Cut
	tx = so.cut(Y)
	assert tx.Theta == 50
	assert tx.X == 100
	assert "Y" not in asdict(tx)

	# Extract
	pj = so.extract(Proj)
	assert pj.Y == 12
	assert pj.X == 100
	assert "Theta" not in asdict(pj)


def test_image_stack_has_spatial_xyz_fields():
	assert astuple(ImageStack(Z=3, Y=5, X=7)) == (3, 5, 7)


def test_reverse_orders_reverse_their_forward_twins():
	assert SinoOrder(Y=5, Theta=7, X=11).transpose(ReverseSinoOrder) == [2, 1, 0]
	assert ReconOrder(Y=5, Z=7, X=11).transpose(ReverseReconOrder) == [2, 1, 0]


@pytest.mark.parametrize("layout", [ProjOrder, ProjOrder(Theta=2, Y=3, X=4)])
def test_named_axis_accessors_work_on_shape_classes_and_instances(layout):
	assert layout.names == ("Theta", "Y", "X")
	assert layout.axis_of("Y") == 1
	assert layout.point_index(Theta=1, X=2) == (1, slice(None), 2)
	with pytest.raises(ValueError):
		layout.axis_of("Z")


def test_shape_aggregates_and_transposed_view_share_the_named_axis_model():
	shape = ProjOrder(Theta=2, Y=3, X=4)
	array = np.arange(shape.prod()).reshape(shape.as_tuple())
	view = shape.transposed_view(array, SinoOrder)

	assert shape.as_dict() == {"Theta": 2, "Y": 3, "X": 4}
	assert shape.max_field() == 4
	assert shape.nbytes(np.float32) == 96
	assert view.shape == SinoOrder(Y=3, Theta=2, X=4).as_tuple()
	assert np.shares_memory(array, view)
	assert not hasattr(shape, "__dict__")
	assert hasattr(DataRange(Y(0), Y(1), Y(1)), "__dict__")


def test_shape_shallow_tuple_and_same_class_operator_fast_paths(monkeypatch):
	shape = ProjOrder(Theta=2, Y=3, X=4)
	globals_ = {"shape": shape, "astuple": astuple}
	shallow = min(repeat("shape.as_tuple()", globals=globals_, number=100_000, repeat=5))
	deep = min(repeat("astuple(shape)", globals=globals_, number=100_000, repeat=5))
	assert deep >= shallow * 5

	monkeypatch.setattr(shape_module, "replace", lambda *_args, **_kwargs: pytest.fail("slow mixed-shape path"))
	assert shape + ProjOrder(Theta=5, Y=7, X=11) == ProjOrder(Theta=7, Y=10, X=15)


def test_cast_errors():
	so = SinoOrder(Y=24, Theta=1501, X=2048)
	po = ProjOrder(Y=24, Theta=1501, X=2048)
	ro = ReconOrder(Y=24, X=2048, Z=2048)
	pj = Proj(Y=24, X=2048)
	tx = YSlice(Theta=1501, X=2048)

	with pytest.raises(TypeError) as te:
		so.to(['Y', 'Theta', 'X'])
	print(f"to: {te.__dict__}")

	with pytest.raises(TypeError) as te:
		so.extract("Kumquat")
	print(f"extract: {te}")

	with pytest.raises(KeyError) as ke:
		pj.extract(Theta)
	print(f"extract fields: {ke}")

	with pytest.raises(TypeError) as te:
		so.cut(["Y", "Theta", "X"])
	print(f"cut (type): {te}")

	with pytest.raises(IndexError) as ie:
		so.cut(ProjOrder)
	print(f"cut (values): {ie}")

	with pytest.raises(TypeError) as te:
		po.transpose(ro)
	print(f"transpose: {te}")

	with pytest.raises(ValueError) as ve:
		po.transpose(pj)
	print(f"transpose field count: {ve}")

	with pytest.raises(AttributeError) as ae:
		ReconOrder.of(tx)
	print(f"bad recon cast: {ae}")


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


def test_shape_operator_error():
	po = ProjOrder(Y=24, Theta=1501, X=2048)
	pj = Proj(Y=24, X=2048)

	with pytest.raises(AttributeError) as ae:
		pj < po
	print(f"Comparison Count: {ae}")

	with pytest.raises(AttributeError) as ae:
		pj // po
	print(f"Comparison Count: {ae}")

	with pytest.raises(ValueError) as ve:
		po.param_min(pj)
	print(f"Comparison Count: {ve}")


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

	with pytest.raises(ValueError) as ve:
		DataRange(Y(5), Y(25), Proj(1, 1))
	print(f"Oversized Step: {ve}")

	with pytest.raises(ValueError) as ve:
		DataRange(Y(5), Proj(25, 5), Y(1))
	print(f"Oversized Stop: {ve}")


def test_range_at_interval():
	interval_range = Y.range_at_interval(0, 24, 4)
	std_range = Y.drange(0, 24, 6)

	assert std_range == interval_range


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


def test_custom_iter_1d():
	shape = ProjOrder(Theta=1501, Y=2048, X=2048)
	win_shape = ProjOrder(Theta=1501, Y=24, X=2048)
	sino_shape = SinoOrder(Theta=1501, Y=24, X=2048)
	itemsize = np.dtype(np.uint16).itemsize
	vals = list(asdict(shape).values()) + [1]
	stride = type(shape)(*[itemsize * np.prod(vals[x + 1:]) for x in range(len(vals) - 1)])
	vals = list(asdict(sino_shape).values()) + [1]
	sino_stride = type(sino_shape)(*[itemsize * np.prod(vals[x + 1:]) for x in range(len(vals) - 1)])

	drange_int_iter = Y.drange(1024, 1048, 1).get_iter(IntIter)
	drange_slice_iter = Y.drange(1024, 1048, 1).get_iter(partial(SliceIter, shape=win_shape))
	drange_mem_iter = Y.drange(1024, 1048, 1).get_iter(partial(MemAddressIter, offset=1024, stride=stride))
	drange_contig_iter = Y.drange(1024, 1048, 1).get_iter(
							partial(ContigMemIter, offset=1024, stride=stride, shape=win_shape, jump=["Theta"]))

	int_iter_list = list(drange_int_iter)
	slice_iter_list = list(drange_slice_iter)
	mem_iter_list = list(drange_mem_iter)

	assert int_iter_list == list(range(1024, 1048, 1))
	assert slice_iter_list == list([np.s_[:, x, :] for x in range(1024, 1048, 1)])

# 	print(mem_iter_list)
# 	print([1024 + (2048 * 2) * i for i in range(1024, 1048)])

	assert mem_iter_list == [1024 + (2048 * 2) * i for i in range(1024, 1048)]

	print(drange_contig_iter.conv.contig_stride)

	assert drange_contig_iter.conv.contig_stride == 24 * 2048 * 2
	drange_contig_iter_list = list(drange_contig_iter)

	print(drange_contig_iter_list)

	assert list(np.unique(drange_contig_iter_list)) == [1024 + (2048 * 2) * 1024]

	drange_sino_contig_iter = Y.drange(1024, 1048, 1).get_iter(
							partial(ContigMemIter, offset=1024, stride=sino_stride, shape=sino_shape, jump=["Theta"]))

	print(drange_sino_contig_iter.conv.contig_stride)

	assert drange_sino_contig_iter.conv.contig_stride == 2048 * 2
	drange_sino_contig_iter_list = list(drange_sino_contig_iter)

	print(drange_sino_contig_iter_list)
	print([1024 + (1501 * 2048 * 2) * i for i in range(1024, 1048)])

	print(len(drange_sino_contig_iter_list))
	print(len([1024 + (1501 * 2048 * 2) * i for i in range(1024, 1048)]))

	assert list(np.unique(drange_sino_contig_iter_list)) == [1024 + (2048 * 1501 * 2) * i for i in range(1024, 1048)]

	with pytest.raises(IndexError) as ie:
		Y.drange(1024, 1048, 1).get_iter(partial(ContigMemIter, offset=1024, stride=sino_stride, shape=sino_shape,
											jump=["X"]))
	print(f"Break Field last in shape: {ie}")

	drange_skip_contig_iter = Y.drange(1024, 1048, 2).get_iter(
							partial(ContigMemIter, offset=1024, stride=stride, shape=win_shape, jump=["Theta"]))
	assert drange_skip_contig_iter.conv.contig_stride == 2048 * 2

	skip_contig_iter_list = list(drange_skip_contig_iter)
	assert skip_contig_iter_list == [1024 + (2048 * 2) * i for i in range(1024, 1048, 2)]


def test_custom_iter_2d():
	shape = ProjOrder(Theta=1501, Y=2048, X=2048)
	win_shape = ProjOrder(Theta=1501, Y=24, X=2048)
	sino_shape = SinoOrder(Theta=1501, Y=24, X=2048)
	itemsize = np.dtype(np.uint16).itemsize
	vals = list(asdict(shape).values()) + [1]
	stride = type(shape)(*[itemsize * np.prod(vals[x + 1:]) for x in range(len(vals) - 1)])
	vals = list(asdict(sino_shape).values()) + [1]
	sino_stride = type(sino_shape)(*[itemsize * np.prod(vals[x + 1:]) for x in range(len(vals) - 1)])

	start = XSlice(Theta=0, Y=1024)
	stop = XSlice(Theta=1501, Y=1048)
	step = XSlice(Theta=1, Y=1)

	drange_int_iter = XSlice.drange(start, stop, step).get_iter(IntIter)
	drange_slice_iter = XSlice.drange(start, stop, step).get_iter(partial(SliceIter, shape=win_shape))
	drange_mem_iter = XSlice.drange(start, stop, step).get_iter(partial(MemAddressIter, offset=1024, stride=stride))
	drange_contig_iter = XSlice.drange(start, stop, step).get_iter(
							partial(ContigMemIter, offset=1024, stride=stride, shape=win_shape, jump=["Theta"]))
	drange_sino_contig_iter = XSlice.drange(start, stop, step).get_iter(
							partial(ContigMemIter, offset=1024, stride=sino_stride, shape=sino_shape, jump=["Theta"]))

	int_iter_list = list(drange_int_iter)
	slice_iter_list = list(drange_slice_iter)
	mem_iter_list = list(drange_mem_iter)
	contig_iter_list = list(np.unique(list(drange_contig_iter)))
	sino_contig_iter_list = list(np.unique(list(drange_sino_contig_iter)))

	print(int_iter_list[:50])

	assert slice_iter_list == list([np.s_[y, x, :] for y in range(0, 1501) for x in range(1024, 1048)])

	assert mem_iter_list == [1024 + (2048 * 2048 * 2) * theta + (2048 * 2) * y for theta in range(0, 1501)
								for y in range(1024, 1048)]

	assert contig_iter_list == [1024 + (2048 * 2048 * 2) * theta + (2048 * 2) * 1024 for theta in range(0, 1501)]

	print(sino_contig_iter_list[:50])
	print([1024 + (24 * 2048 * 2) * y + (2048 * 2) * theta for y in range(1024, 1048)
										for theta in range(0, 1501)][:50])

	assert sino_contig_iter_list == [1024 + (1501 * 2048 * 2) * y + (2048 * 2) * theta for y in range(1024, 1048)
										for theta in range(0, 1501)]
