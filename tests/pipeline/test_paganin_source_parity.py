"""Characterization contract for lftomo issue #199."""

import numpy as np
import pytest

from lftomo.pipeline.stages.phase import _paganin_filter, _paganin_transfer
from tests.fixtures.paganin_source import (
	SOURCE_COMMIT,
	phase_fixture,
	source_filter,
	source_transfer,
)


CASES = (
	pytest.param((32, 48), 0.02, 20.0, 0.637, 11.0, id="tuning-low-alpha"),
	pytest.param((32, 48), 0.06, 20.0, 0.637, 11.0, id="tuning-reference-alpha"),
	pytest.param((32, 48), 0.20, 20.0, 0.637, 11.0, id="tuning-high-alpha"),
	pytest.param((31, 47), 0.06, 20.0, 0.637, 11.0, id="held-out-odd-shape"),
	pytest.param((33, 64), 0.08, 25.45, 0.7046, 15.0, id="held-out-energy-geometry"),
)


def _parameters(alpha, energy, pixel, distance):
	return {
		"alpha": alpha, "energy_kev": energy,
		"pixel_size_um": pixel, "ssd_cm": distance,
	}


def test_source_fixture_pins_audited_commit_and_even_grid_behavior():
	assert SOURCE_COMMIT == "f601fc995ffbfc8757447c992cbb3f5fd1e36b19"
	parameters = _parameters(0.06, 20.0, 0.637, 11.0)
	source = source_transfer((32, 48), **parameters)
	current = _paganin_transfer((32, 48), **parameters)

	# FFT bins always contain DC; the source's even N-1 centered construction
	# instead has no zero coordinate and normalizes four half-bin maxima to one.
	assert current[0, 0] == 1
	assert np.count_nonzero(current == 1) == 1
	assert np.count_nonzero(source == 1) == 4
	np.testing.assert_allclose(
		[source[index] for index in ((0, 0), (0, 1), (1, 0), (1, 1), (16, 24), (31, 47))],
		[1.0, 0.98871076, 0.97442496, 0.96370262, 0.13696811, 1.0],
		rtol=1e-7, atol=0,
	)


@pytest.mark.parametrize("shape,alpha,energy,pixel,distance", CASES)
def test_source_and_current_discretizations_are_deliberately_not_equivalent(
	shape, alpha, energy, pixel, distance,
):
	parameters = _parameters(alpha, energy, pixel, distance)
	image = phase_fixture(shape)
	source = source_filter(image, **parameters)
	current = _paganin_filter(image, _paganin_transfer(shape, **parameters))
	relative_l2 = np.linalg.norm(current - source) / np.linalg.norm(source)

	assert np.isfinite(source).all()
	assert np.isfinite(current).all()
	assert relative_l2 > 1e-3
