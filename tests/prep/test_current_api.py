from uuid import uuid4

import numpy as np
import pytest

from lftomo.prep.flat import weighted_normalize
from lftomo.prep.phase import _phase_pad, _phase_retrieval, filter_create
from lftomo.prep import flat, stripe
from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import DataRange, GenericOrder, ImgSlice, ProjOrder, SinoOrder, Y
from lftomo.util.window import DataRef, TomoWindow


def _shared(shape, dtype=np.float32):
	return SharedNPArray(f"lftomo-test-{uuid4().hex}", shape, dtype, allocate=True)


def _window(shape):
	return TomoWindow(shape, Y(0), Y(0), Y(1), np.float32, np.float32)


@pytest.mark.parametrize("order,y_start,y_size,before,after,y_step,pre_sub", [
	pytest.param(SinoOrder, 0, 8, 0, 0, 1, False, id="sino-full"),
	pytest.param(ProjOrder, 0, 8, 0, 0, 1, True, id="proj-full-presub"),
	pytest.param(SinoOrder, 2, 4, 0, 0, 2, True, id="sino-subset-strided-presub"),
	pytest.param(ProjOrder, 2, 4, 0, 0, 2, False, id="proj-subset-strided"),
	pytest.param(SinoOrder, 3, 2, 1, 2, 1, False, id="sino-subset-halo"),
	pytest.param(ProjOrder, 3, 2, 1, 2, 1, True, id="proj-subset-halo-presub"),
])
def test_weighted_normalize_uses_current_memory_and_window_api(
	monkeypatch, order, y_start, y_size, before, after, y_step, pre_sub,
):
	window = TomoWindow(
		order(Y=y_size, Theta=3, X=4), order(Y=before, Theta=0, X=0),
		order(Y=y_start, Theta=10, X=0), order(Y=y_step, Theta=1, X=1),
		np.float32, np.float32, pad_after=order(Y=after, Theta=0, X=0),
	)
	# The classic API always receives Y-first arrays, independently of window order.
	output = _shared(window.internal.with_pad().shape.to(SinoOrder))
	source = _shared(output.shape)
	flats = _shared(GenericOrder(4, 8, 4))
	calls = []
	views = []
	kernel = flat._weighted_normalize_projection
	getitem = SharedNPArray.__getitem__

	def record_view(memory, index):
		view = getitem(memory, index)
		views.append(view)
		return view

	def record(target, original, fields, projection, projection_count, pre_sub):
		assert isinstance(projection, int)
		assert isinstance(projection_count, int)
		assert np.shares_memory(target, views[0].array())
		assert np.shares_memory(original, views[1].array())
		assert np.shares_memory(fields, flats.array())
		calls.append((projection, projection_count))
		return kernel(target, original, fields, projection, projection_count, pre_sub)

	monkeypatch.setattr(flat, "_weighted_normalize_projection", record)
	monkeypatch.setattr(SharedNPArray, "__getitem__", record_view)
	try:
		output.array()[...] = -7
		source.array()[...] = np.arange(output.shape.prod()).reshape(output.shape.as_tuple()) + 11
		rows = np.arange(8, dtype=np.float32)[:, None]
		flats.array()[...] = np.stack([
			np.broadcast_to(20 + rows, (8, 4)), np.broadcast_to(26 + rows, (8, 4)),
			np.full((8, 4), 1), np.full((8, 4), 4),
		])
		expected = output.array().copy()
		dark = 2
		gain = 22 + rows[y_start:y_start + y_size:y_step] - (0 if pre_sub else dark)
		local_rows = slice(before, before + y_size, y_step)
		expected[local_rows, 1, :] = (source.array()[local_rows, 1, :] - dark) / gain

		weighted_normalize(output, flats, source, window, projection=1, pre_sub=pre_sub)

		np.testing.assert_allclose(output.array(), expected)
		assert [view.name for view in views] == [output.name, source.name]
		assert calls == [(1, 3)]
	finally:
		for view in views:
			view.shutdown()
		output.shutdown()
		source.shutdown()
		flats.shutdown()


@pytest.mark.parametrize("presub", [False, True])
def test_calculate_shift_reshapes_for_asymmetric_scope(tmp_path, monkeypatch, presub):
	# Simulate smooth, shifted illumination over a nonuniform detector dark map.
	y, x = np.mgrid[:10, :13]
	dark = (1 + y / 20 + x / 30).astype(np.float32)
	pre_gain = (100 + 30 * np.exp(-((y - 5) ** 2 + (x - 6) ** 2) / 25)).astype(np.float32)
	post_gain = (115 + 30 * np.exp(-((y - 4) ** 2 + (x - 5) ** 2) / 25)).astype(np.float32)
	flats = np.stack([pre_gain + dark, post_gain + dark, dark, dark])
	if presub:
		flats[:2] -= flats[2:]
	before = flats.copy()
	gain = before[:2] if presub else before[:2] - before[2:]
	flat_map = gain[1] / gain[0]
	center = flat_map[2:-2:2, 3:-3]
	expected_stddev = np.empty((4, 6))
	expected_images = {}
	for dy in range(-2, 2):
		for dx in range(-3, 3):
			image = center + flat_map[dy + 2:dy + 8:2, dx + 3:dx + 10]
			expected_images[f"sum_{dy}_{dx}.tif"] = image
			expected_stddev[dy + 2, dx + 3] = np.std(image)

	indices = []
	ranges = []
	index = DataRef.index.fget
	iterate = DataRange.__iter__

	def record_index(ref):
		indices.append((ref.pos, ref.shape, ref.step))
		return index(ref)

	def record_range(ref):
		ranges.append(ref)
		return iterate(ref)

	monkeypatch.setattr(DataRef, "index", property(record_index))
	monkeypatch.setattr(DataRange, "__iter__", record_range)
	work = tmp_path / "work"
	work.mkdir()
	monkeypatch.chdir(work)
	flat.calculate_shift(flats, scope=ImgSlice(Y=2, X=3), interval=2, presub=presub)

	np.testing.assert_array_equal(flats[:2], gain)
	np.testing.assert_array_equal(flats[2:], before[2:])
	rows = np.loadtxt("test_other.txt", delimiter=":")
	np.testing.assert_array_equal(rows[:, :2], [[y, x] for y in range(-2, 2) for x in range(-3, 3)])
	np.testing.assert_allclose(rows[:, 2], expected_stddev.ravel())
	np.testing.assert_array_equal(rows[:, 3], np.argsort(expected_stddev, axis=None))
	for name, expected in expected_images.items():
		np.testing.assert_array_equal(flat.tf.memmap(tmp_path / "data/output/flat_gen" / name, mode="r"), expected)
	assert len(ranges) == 2
	assert ranges[0] is ranges[1]
	assert ranges[0].start == ImgSlice(-2, -3)
	assert ranges[0].stop == ImgSlice(2, 3)
	assert all(shape == ImgSlice(6, 7) and step == ImgSlice(2, 1) for _, shape, step in indices)
	assert {(pos.Y, pos.X) for pos, _, _ in indices} == {(y, x) for y in range(4) for x in range(6)}


def test_phase_filter_uses_current_shape_and_memory_api():
	name = f"lftomo-test-{uuid4().hex}"
	phase_filter = filter_create(
		name,
		np.float32,
		ImgSlice(Y=8, X=8),
		alpha=1e-3,
		pixel_size=1e-4,
		dist=10.0,
		energy=20.0,
	)

	try:
		assert phase_filter.shape == ImgSlice(Y=8, X=8)
		with phase_filter as values:
			assert np.isfinite(values).all()
			assert values.max() == 1.0
	finally:
		phase_filter.shutdown()


def test_phase_retrieval_uses_dist_then_energy_for_height_padding():
	shape = SinoOrder(Y=4, Theta=1, X=4)
	pixel_size, dist, energy = 1e-3, 2.0, 20.0
	pad = _phase_pad(shape.Y, pixel_size, dist, energy)
	assert (pad, _phase_pad(shape.Y, pixel_size, energy, dist)) == (1, 2)
	work = _shared(shape)
	phase_filter = filter_create(
		f"lftomo-test-{uuid4().hex}", np.float32, shape,
		1e-3, pixel_size, dist, energy, tp=True, height_pad=True,
	)

	try:
		with work as values, phase_filter as filter_values:
			values[:] = np.linspace(0.1, 0.9, num=16, dtype=np.float32).reshape(4, 1, 4)
			padded = np.pad(values[:, 0, :], ((pad, pad), (0, 0)), mode="edge")
			filtered = np.fft.fft2(padded) * filter_values
			expected = np.real(np.fft.ifft2(filtered[pad:-pad, :]))

		_phase_retrieval(
			np.s_[:, 0, :], work, phase_filter,
			pixel_size, energy, dist, tp=True, height_pad=True, thread_fallback=1,
		)

		assert phase_filter.shape == ImgSlice(Y=shape.Y + 2 * pad, X=shape.X)
		with work as values:
			np.testing.assert_allclose(values[:, 0, :], expected, rtol=1e-6, atol=1e-7)
	finally:
		phase_filter.shutdown()
		work.shutdown()


def test_stripe_window_dispatch_uses_current_internal_iteration(monkeypatch):
	shape = SinoOrder(Y=2, Theta=5, X=3)
	memory = _shared(shape)
	calls = []

	class InlinePool:
		def __init__(self, _proc_count):
			pass

		def __enter__(self):
			return self

		def __exit__(self, *_args):
			pass

		def starmap(self, function, arguments):
			for argument in arguments:
				function(*argument)

	def record_call(_memory, index, size, dim):
		calls.append((int(index), size, dim))

	monkeypatch.setattr(stripe, "Pool", InlinePool)
	monkeypatch.setattr(stripe, "_remove_sort_vo", record_call)

	try:
		stripe.remove_sort_vo(_window(shape), memory, size=3, dim=1, proc_count=1)
		assert calls == [(0, 3, 1), (1, 3, 1)]
	finally:
		memory.shutdown()
