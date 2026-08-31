from uuid import uuid4

import numpy as np

from lftomo.prep.flat import weighted_normalize
from lftomo.prep.phase import _phase_pad, _phase_retrieval, filter_create
from lftomo.prep import flat, stripe
from lftomo.util.mem import SharedNPArray
from lftomo.util.shapes import GenericOrder, ImgSlice, SinoOrder, Y
from lftomo.util.window import TomoWindow


def _shared(shape, dtype=np.float32):
	return SharedNPArray(f"lftomo-test-{uuid4().hex}", shape, dtype, allocate=True)


def _window(shape):
	return TomoWindow(shape, Y(0), Y(0), Y(1), np.float32, np.float32)


def test_weighted_normalize_uses_current_memory_and_window_api():
	shape = SinoOrder(Y=2, Theta=2, X=2)
	output = _shared(shape)
	source = _shared(shape)
	flats = _shared(GenericOrder(A=4, B=2, C=2))

	try:
		with source as source_array:
			source_array[:] = 11.0
		with flats as flat_array:
			flat_array[0] = 5.0  # pre-gain
			flat_array[1] = 7.0  # post-gain
			flat_array[2] = 1.0  # pre-dark
			flat_array[3] = 1.0  # post-dark

		weighted_normalize(output, flats, source, _window(shape), projection=1)

		with output as normalized:
			np.testing.assert_allclose(normalized[:, 1, :], 2.0)
	finally:
		output.shutdown()
		source.shutdown()
		flats.shutdown()


def test_calculate_shift_reshapes_for_asymmetric_scope(tmp_path, monkeypatch):
	flats = np.ones((len(flat.FLAT), 8, 10), dtype=np.float32)
	flats[flat.FLAT.POSTGAIN.index] = 2.0
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr(flat.tf, "imwrite", lambda *_args, **_kwargs: None)
	flat.calculate_shift(flats, scope=ImgSlice(Y=2, X=3), interval=1, presub=True)


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
