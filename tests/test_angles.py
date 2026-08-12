import numpy as np
import pytest

from lftomo.angles import (
	angles_from_indices,
	indices_from_angles,
	indices_from_hardware_timestamps,
	validate_acquisition_indices,
)


def test_hardware_gaps_are_applied_cumulatively():
	indices = indices_from_hardware_timestamps(
		[0, 100, 300, 400, 500],
		recorded_count=5,
		planned_count=6,
	)
	angles = angles_from_indices(
		indices,
		planned_count=6,
		rotation_start_deg=0,
		rotation_end_deg=180,
	)

	np.testing.assert_array_equal(indices, [0, 1, 3, 4, 5])
	np.testing.assert_allclose(np.rad2deg(angles), [0, 30, 90, 120, 150], atol=2e-5)


@pytest.mark.parametrize(
	"indices,message",
	[
		([[0, 1]], "one-dimensional"),
		([0, 2, 2], "strictly increasing"),
		([-1, 1, 2], "cannot be negative"),
		([0, 1, 4], "outside 4 planned slots"),
	],
)
def test_acquisition_index_validation(indices, message):
	with pytest.raises(ValueError, match=message):
		validate_acquisition_indices(indices, recorded_count=3, planned_count=4)


def test_explicit_angles_must_match_the_declared_slot_convention():
	with pytest.raises(ValueError, match="inconsistent"):
		indices_from_angles(
			np.deg2rad([0, 40, 90]),
			recorded_count=3,
			planned_count=4,
			rotation_start_deg=0,
			rotation_end_deg=180,
		)
