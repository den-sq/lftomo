import bottleneck as bn
import numpy as np
import pytest
from scipy.signal import medfilt2d

from lftomo.prep import stripe


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("axis", "size"), [
	(0, 1), (0, 3), (0, 7),
	(1, 1), (1, 3), (1, 7),
])
def test_bottleneck_line_median_exactly_matches_centered_scipy_borders(dtype, axis, size):
	rng = np.random.default_rng(209)
	values = rng.standard_normal((9, 11)).astype(dtype)
	kernel = (size, 1) if axis == 0 else (1, size)

	actual = stripe._line_median(values, size, axis)
	expected = medfilt2d(values, kernel)

	assert actual.dtype == values.dtype
	np.testing.assert_array_equal(actual, expected)
	np.testing.assert_array_equal(
		actual.take([0, -1], axis=axis), expected.take([0, -1], axis=axis),
	)


def test_managed_pad_plane_is_reused_across_different_line_sizes_and_axes():
	rng = np.random.default_rng(210)
	values = rng.standard_normal((9, 11)).astype(np.float32)
	pad_work = np.zeros((9, 11 + 12), dtype=np.float32)

	large = stripe._line_median(values, 13, axis=1, pad_work=pad_work)
	sort = stripe._line_median(values.T, 3, axis=0, pad_work=pad_work.T)

	np.testing.assert_array_equal(large, medfilt2d(values, (1, 13)))
	np.testing.assert_array_equal(sort, medfilt2d(values.T, (3, 1)))
	assert np.count_nonzero(pad_work[:, :6]) == 0
	assert np.count_nonzero(pad_work[:, 17:]) == 0


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_line_input_retains_scipy_path(monkeypatch, nonfinite):
	values = np.arange(35, dtype=np.float32).reshape(5, 7)
	values[2, 3] = nonfinite
	monkeypatch.setattr(
		stripe, "_move_median",
		lambda *_args, **_kwargs: pytest.fail("non-finite input reached Bottleneck"),
	)

	actual = stripe._line_median(values, 3, axis=1)
	expected = medfilt2d(values, (1, 3))

	np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_unsupported_dtype_retains_scipy_path(monkeypatch):
	values = np.arange(35, dtype=np.uint8).reshape(5, 7)
	monkeypatch.setattr(
		stripe, "_move_median",
		lambda *_args, **_kwargs: pytest.fail("unsupported dtype reached Bottleneck"),
	)

	np.testing.assert_array_equal(
		stripe._line_median(values, 3, axis=1), medfilt2d(values, (1, 3)),
	)


def test_vo_large_and_dim1_sort_use_bottleneck_but_dim2_sort_stays_on_scipy(monkeypatch):
	rng = np.random.default_rng(211)
	values = rng.standard_normal((9, 11)).astype(np.float32)
	calls = []
	scipy_calls = []
	original = bn.move_median
	original_scipy = stripe.medfilt2d

	def record(*args, **kwargs):
		calls.append(kwargs["axis"])
		return original(*args, **kwargs)

	def record_scipy(array, kernel):
		scipy_calls.append(kernel)
		return original_scipy(array, kernel)

	monkeypatch.setattr(stripe, "_move_median", record)
	monkeypatch.setattr(stripe, "medfilt2d", record_scipy)
	stripe._remove_large_array(values.copy(), size=3, norm=False)
	stripe._remove_sort_array(values.copy(), size=3, dim=1)
	stripe._remove_sort_array(values.copy(), size=3, dim=2)

	assert calls == [1, 0]
	assert scipy_calls == [(3, 3)]
