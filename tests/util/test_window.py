from functools import partial
import numpy as np
import pytest

from lftomo.util.shapes import ProjOrder, ReconOrder, SinoOrder, Y, X, ImgSlice, IntIter, SliceIter, Proj
from lftomo.util.window import TomoWindow, TomoWindowIter, CTFov, TomoDataSet, DataRef
from tests.sample_data import generate_dataref, generate_dataset, disk_dataset


def _uniform_angles(count):
	return np.linspace(0, np.pi, count, endpoint=False)


@pytest.mark.integration
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


def test_creation():
	fov = CTFov(1501, 180, 1)
	shape = ProjOrder(Theta=1501, Y=2048, X=2048)
	over = Y.drange(0, 1048, 24)
	dtype = np.int16
	work_dtype = np.float32

	tds = TomoDataSet(
		fov, shape, over, dtype=dtype, work_dtype=work_dtype,
		angles=_uniform_angles(shape.Theta),
	)

	assert tds.pos == Y(0)
	assert tds.shape == ProjOrder(Theta=1501, Y=1048, X=2048)
	assert tds.step == Y(24)

	tdr = DataRef(shape, Y(2), Y(0), Y(1), dtype)

	assert tdr.pos == Y(0)
	assert tdr.shape == shape
	assert tdr.interval("Y") == Y.interval(0, 2048)
	assert DataRef.from_document(tdr.to_document()).to_document() == tdr.to_document()

	tw = tds.full_window()

	assert tw.pos == Y(0)
	assert tw.step == ProjOrder(1, 1, 1)
	assert tw.shape == ProjOrder(1501, 1048, 2048)
	assert tw.internal.dtype == np.dtype(work_dtype)
	assert TomoWindow.from_document(tw.to_document()).to_document() == tw.to_document()
	restored = TomoDataSet.from_document(tds.to_document())
	assert restored.to_document() == tds.to_document()
	assert not restored.angles.flags.writeable and not restored.acquisition_indices.flags.writeable

	# Missing angle steps
	with pytest.warns(DeprecationWarning, match="FOV angle synthesis"):
		with pytest.raises(Exception):
			TomoDataSet(CTFov(1969, 180, 1), shape, over, dtype=dtype, work_dtype=work_dtype)

	# Passing Angles
	tds = TomoDataSet(fov, shape, over, angle_steps=np.arange(0, 1501, 2),
						dtype=dtype, work_dtype=work_dtype)

	assert all(tds.angles == [i * np.deg2rad(fov.Degrees) for i in range(0, 1501, 2)])

	with pytest.raises(TypeError):
		tds = TomoDataSet(fov, shape, 24, dtype=dtype, work_dtype=work_dtype)


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
	assert [(f.shape, f.pos) for f in rs_list] == [(ProjOrder(Theta=1969, Y=min(24, 238 - x), X=2048), Y(max(x, 2)))
												for x in range(2, 238, 24)]


def test_alt_iter():
	tds = generate_dataset()

	large_window = tds.iter_with(wsize=Y(48))
	large_window_list = list(large_window)

	print([(x.shape, x.pos) for x in large_window_list])

	assert [x.shape for x in large_window_list] == [ProjOrder(Theta=1969, Y=48, X=2048)] * 5


def test_1d_window_iter():
	ttwi = partial(TomoWindowIter, shape=ProjOrder(Theta=1900, Y=24, X=2048), pad=Proj(Y=2, X=4),
					dtype=np.int16, work_dtype=np.float32)
	alt = Y.drange(2, 90, 24).get_iter(ttwi)

	alt_list = list(alt)

	assert iter(alt) == alt

	assert [(win.shape, win.pos) for win in alt_list[:-1]] == \
			[(ProjOrder(Theta=1900, Y=24, X=2048), Y(y)) for y in range(2, 74, 24)]
	assert (alt_list[-1].shape, alt_list[-1].pos) == (ProjOrder(Theta=1900, Y=16, X=2048), Y(74))
	assert alt_list[-1].with_pad(["Y"]).index[1] == slice(72, 92, 1)


def test_2d_window_iter():
	# 2D Window
	twitwo_manual = [(ProjOrder(Theta=1900, Y=24, X=24), Proj(Y=2, X=600)),
					(ProjOrder(Theta=1900, Y=24, X=24), Proj(Y=2, X=624)),
					(ProjOrder(Theta=1900, Y=24, X=22), Proj(Y=2, X=648)),
					(ProjOrder(Theta=1900, Y=20, X=24), Proj(Y=26, X=600)),
					(ProjOrder(Theta=1900, Y=20, X=24), Proj(Y=26, X=624)),
					(ProjOrder(Theta=1900, Y=20, X=22), Proj(Y=26, X=648))]

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
	assert isinstance(twi, TomoWindow)


@pytest.mark.parametrize(
	("shape", "axis", "pos", "step", "positions", "indices"),
	[
		(
			ProjOrder(3, 2, 4), "Theta", ProjOrder(5, 10, 0), ProjOrder(2, 1, 1),
			[5, 7, 9], [np.s_[0:1, :, :], np.s_[1:2, :, :], np.s_[2:3, :, :]],
		),
		(
			SinoOrder(2, 3, 4), "Y", SinoOrder(10, 5, 0), SinoOrder(3, 2, 1),
			[10, 13], [np.s_[0:1, :, :], np.s_[1:2, :, :]],
		),
	],
)
def test_data_ref_units_pair_local_indices_with_global_strided_positions(
	shape, axis, pos, step, positions, indices,
):
	units = list(DataRef(shape, shape.make_with(0), pos, step, np.float32).units(axis))

	assert [getattr(unit.pos, axis) for unit in units] == positions
	assert [unit.index for unit in units] == indices
	assert all(unit.step == step and getattr(unit.shape, axis) == 1 for unit in units)


def test_window_from_bounds_records_edge_clamped_halo_and_theta_stride():
	physical = ProjOrder(12, 10, 8)
	top = TomoWindow.from_bounds(
		physical, y=(0, 4), theta=(2, 11, 3), halo=3,
		dtype=np.uint16, work_dtype=np.float32,
	)
	bottom = TomoWindow.from_bounds(
		physical, y=(7, 10), theta=(0, 12), halo=4,
		dtype=np.uint16, work_dtype=np.float32,
	)
	full = TomoWindow.from_bounds(physical, y=(0, 10), halo=5)

	assert (top.shape, top.pos, top.step) == (
		ProjOrder(9, 4, 8), ProjOrder(2, 0, 0), ProjOrder(3, 1, 1),
	)
	assert (top.pad_before, top.pad_after) == (ProjOrder(0, 0, 0), ProjOrder(0, 3, 0))
	assert (bottom.pad_before, bottom.pad_after) == (ProjOrder(0, 4, 0), ProjOrder(0, 0, 0))
	assert (full.pad_before, full.pad_after) == (physical.make_with(0), physical.make_with(0))
	assert top.frange.length == ProjOrder(3, 4, 8)


def test_recon_shape_uses_unpadded_core_geometry():
	window = TomoWindow(
		ProjOrder(Theta=4, Y=10, X=12), Y(2), Y(5), Y(1),
		np.uint16, np.float32, pad_after=Y(3),
	)

	assert window.recon_shape() == ReconOrder(Y=10, Z=12, X=12)


def test_dataset_geometry_arrays_are_read_only():
	dataset = generate_dataset()

	with pytest.raises(ValueError, match="read-only"):
		dataset.angles[0] = 0
	with pytest.raises(ValueError, match="read-only"):
		dataset.acquisition_indices[0] = 0


def test_window_coordinate_operations_share_the_existing_dataref_model():
	window = TomoWindow(
		ProjOrder(Theta=4, Y=100, X=6), Y(10), Y(100), Y(1),
		np.uint16, np.float32,
	)

	assert window.index == np.s_[0:4:1, 100:200:1, 0:6:1]

	read_window = window.with_pad(["Y"])
	assert read_window.index == np.s_[0:4:1, 90:210:1, 0:6:1]
	assert read_window.pad == Y(0)

	internal = window.internal
	assert internal.index == np.s_[0:4:1, 10:110:1, 0:6:1]
	assert internal.dtype == np.dtype(np.float32)

	rescoped = window.rescope(pos=Y(30), shape=Y(40))
	assert rescoped.index == np.s_[0:4:1, 30:70:1, 0:6:1]
	assert list(rescoped.int_iter()) == list(range(30, 70))
	assert list(rescoped.slice_iter()) == [np.s_[:, y, :] for y in range(30, 70)]

	cropped = window.with_crop(X(1), X(2))
	assert cropped.shape.X == 3


def test_nondivisible_selection_emits_a_short_final_unpadded_window():
	selection = Y.drange(100, 1000, 400)
	dataset = TomoDataSet(
		CTFov(4, 180, 1), ProjOrder(Theta=4, Y=1000, X=6), selection,
		dtype=np.uint16, work_dtype=np.float32, angles=_uniform_angles(4),
	)

	windows = list(dataset)

	assert dataset.frange == selection
	assert [window.pos for window in windows] == [Y(100), Y(500), Y(900)]
	assert [window.shape.Y for window in windows] == [400, 400, 100]
	assert [window.index for window in windows] == [
		np.s_[0:4:1, 100:500:1, 0:6:1],
		np.s_[0:4:1, 500:900:1, 0:6:1],
		np.s_[0:4:1, 900:1000:1, 0:6:1],
	]
	assert windows[-1].internal.index == np.s_[0:4:1, 0:100:1, 0:6:1]


def test_data_ref_supports_asymmetric_padding_without_a_wrapper_type():
	window = TomoWindow(
		ProjOrder(Theta=4, Y=5, X=6), Y(2), Y(10), Y(1),
		np.uint16, np.float32, pad_after=Y(3),
	)

	assert window.pad_before == Y(2)
	assert window.pad_after == Y(3)
	with pytest.raises(ValueError, match="no single value"):
		_ = window.pad

	assert window.with_pad(["Y"]).index == np.s_[0:4:1, 8:18:1, 0:6:1]
	assert window.internal.index == np.s_[0:4:1, 2:7:1, 0:6:1]

	rebalanced = window.rescope(pad=Y(1), pad_after=Y(4))
	assert (rebalanced.pos.Y, rebalanced.shape.Y) == (9, 5)
	assert rebalanced.with_pad(["Y"]).index == window.with_pad(["Y"]).index
	assert rebalanced.internal.index == np.s_[0:4:1, 1:6:1, 0:6:1]


def test_output_padding_expands_coordinates_without_becoming_source_padding():
	window = TomoWindow(
		ProjOrder(Theta=4, Y=5, X=6), ProjOrder(0, 2, 1), ProjOrder(0, 10, 0),
		ProjOrder(1, 1, 1), np.uint16, np.float32,
		pad_after=ProjOrder(0, 3, 2),
	)

	expanded = window.with_output_padding(X(4))

	assert expanded.pos == ProjOrder(Theta=0, Y=10, X=-4)
	assert expanded.shape == ProjOrder(Theta=4, Y=5, X=14)
	assert expanded.pad_before == ProjOrder(Theta=0, Y=2, X=1)
	assert expanded.pad_after == ProjOrder(Theta=0, Y=3, X=2)
	assert expanded.with_pad(["Y"]).index == np.s_[0:4:1, 8:18:1, -4:10:1]
	assert expanded.with_pad().index == np.s_[0:4:1, 8:18:1, -5:12:1]
	assert expanded.internal.index == np.s_[0:4:1, 2:7:1, 1:15:1]


def test_dataset_keeps_physical_shape_separate_from_output_range():
	selection = Y.drange(100, 900, 400)
	dataset = TomoDataSet(
		CTFov(4, 180, 1), ProjOrder(Theta=4, Y=1000, X=6), selection,
		dtype=np.uint16, work_dtype=np.float32, angles=_uniform_angles(4),
	)

	assert dataset.physical_shape == ProjOrder(Theta=4, Y=1000, X=6)
	assert dataset.shape == ProjOrder(Theta=4, Y=800, X=6)
	assert dataset.frange == selection

	windows = list(dataset.iter_with(wsize=Y(400), pad=Y(10)))
	assert [(window.pos.Y, window.shape.Y) for window in windows] == [(100, 400), (500, 400)]
	assert [window.with_pad(["Y"]).index[1] for window in windows] == [
		slice(90, 510, 1), slice(490, 910, 1),
	]
	assert [window.internal.index[1] for window in windows] == [
		slice(10, 410, 1), slice(10, 410, 1),
	]


def test_dataset_clips_padding_only_at_physical_boundaries():
	dataset = TomoDataSet(
		CTFov(4, 180, 1), ProjOrder(Theta=4, Y=1000, X=6),
		Y.drange(0, 1000, 400), dtype=np.uint16, work_dtype=np.float32,
		angles=_uniform_angles(4),
	)

	windows = list(dataset.iter_with(wsize=Y(400), pad=Y(10)))
	first, middle, final = windows

	assert (first.pos.Y, first.shape.Y) == (0, 400)
	assert (first.pad_before.Y, first.pad_after.Y) == (0, 10)
	assert first.with_pad(["Y"]).index[1] == slice(0, 410, 1)
	assert first.internal.index[1] == slice(0, 400, 1)

	assert (middle.pos.Y, middle.shape.Y) == (400, 400)
	assert (middle.pad_before.Y, middle.pad_after.Y) == (10, 10)
	assert middle.with_pad(["Y"]).index[1] == slice(390, 810, 1)

	assert (final.pos.Y, final.shape.Y) == (800, 200)
	assert (final.pad_before.Y, final.pad_after.Y) == (10, 0)
	assert final.with_pad(["Y"]).index[1] == slice(790, 1000, 1)
	assert final.internal.index[1] == slice(10, 210, 1)


def test_nondivisible_selected_edge_keeps_short_output_when_padding_clips():
	dataset = TomoDataSet(
		CTFov(4, 180, 1), ProjOrder(Theta=4, Y=1000, X=6),
		Y.drange(100, 1000, 400), dtype=np.uint16, work_dtype=np.float32,
		angles=_uniform_angles(4),
	)

	final = list(dataset.iter_with(wsize=Y(400), pad=Y(10)))[-1]

	assert (final.pos.Y, final.shape.Y) == (900, 100)
	assert (final.pad_before.Y, final.pad_after.Y) == (10, 0)
	assert final.index[1] == slice(900, 1000, 1)
	assert final.with_pad(["Y"]).index[1] == slice(890, 1000, 1)
	assert final.internal.index[1] == slice(10, 110, 1)


def test_dataset_resolves_explicit_asymmetric_padding():
	dataset = TomoDataSet(
		CTFov(4, 180, 1), ProjOrder(Theta=4, Y=1000, X=6),
		Y.drange(100, 500, 400), dtype=np.uint16, work_dtype=np.float32,
		angles=_uniform_angles(4),
	)

	window = next(dataset.iter_with(wsize=Y(400), pad=Y(20), pad_after=Y(5)))

	assert (window.pad_before.Y, window.pad_after.Y) == (20, 5)
	assert window.with_pad(["Y"]).index[1] == slice(80, 505, 1)
	assert window.internal.index[1] == slice(20, 420, 1)


def test_selected_range_must_fit_physical_source_shape():
	with pytest.raises(ValueError, match="stops outside physical axis Y"):
		TomoDataSet(
			CTFov(4, 180, 1), ProjOrder(Theta=4, Y=1000, X=6),
			Y.drange(100, 1001, 400), dtype=np.uint16, work_dtype=np.float32,
			angles=_uniform_angles(4),
		)


def test_from_disk_retains_physical_shape_for_selected_output(tmp_path):
	h5py = pytest.importorskip("h5py")
	source = tmp_path / "selected.h5"
	with h5py.File(source, "w") as h5:
		h5.create_dataset("exchange/data", shape=(4, 10, 6), dtype=np.uint16)

	with pytest.warns(DeprecationWarning, match="FOV angle synthesis"):
		dataset = TomoDataSet.from_disk(
			source, CTFov(4, 180, 1), ProjOrder, Y.drange(2, 9, 3),
			work_dtype=np.float32,
		)

	assert dataset.physical_shape == ProjOrder(Theta=4, Y=10, X=6)
	assert dataset.shape == ProjOrder(Theta=4, Y=7, X=6)
	assert dataset.frange == Y.drange(2, 9, 3)


def test_data_ref_fields():
	df = generate_dataref()

	assert df.shape == ProjOrder(Theta=1900, Y=24, X=2048)
	assert df.pos == Y(1024)
	assert df.pad == ImgSlice(Y=2, X=4)
	assert df.step == Y(1)
	assert df.dtype == np.int16
	assert df.itemsize == 2
	assert df.nbytes() == 1900 * 24 * 2048 * 2
	assert df.axis_slice("Y") == df.index[ProjOrder.axis_of("Y")]
	assert str(df) == f"{ProjOrder(Theta=1900, Y=24, X=2048)} ({np.int16}) at {Y(1024)} stepping by {Y(1)}"
	assert df.step_shape == ProjOrder(Theta=1900, Y=1, X=2048)
	assert df.stride == ProjOrder(Theta=98304, Y=4096, X=2)


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

	int_iter_list = list(drange_int_iter)
	assert list(df.int_iter()) == list(range(1024, 1048, 1))
	assert int_iter_list == list(df.int_iter())

	slice_iter_list = list(drange_slice_iter)
	assert list(df.slice_iter()) == list([np.s_[:, x, :] for x in range(1024, 1048, 1)])
	assert slice_iter_list == list(df.slice_iter())


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

	with pytest.raises(TypeError):
		dfp = df.with_pad(["Kumquat"])


def test_data_ref_rescope():
	df = generate_dataref()

	dfr = df.rescope(pos=Y(0), dtype=np.float32)
	assert dfr.frange == Y.drange(0, 24, 1)
	assert dfr.itemsize == 4
	assert dfr.shape == ProjOrder(1900, 24, 2048)

	dfr = df.rescope(shape=ProjOrder(1900, 14, 2048))
	assert dfr.frange == Y.drange(1024, 1038, 1)
	assert dfr.itemsize == 2
	assert dfr.shape == ProjOrder(1900, 14, 2048)
