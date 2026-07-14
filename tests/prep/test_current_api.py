from uuid import uuid4

import numpy as np

from lftomo.prep.flat import weighted_normalize
from lftomo.prep.phase import filter_create
from lftomo.prep import stripe
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
