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


def test_hardware_count_deficit_without_gap_is_assigned_to_the_tail():
	indices = indices_from_hardware_timestamps(
		[0, 100, 200, 300, 400],
		recorded_count=5,
		planned_count=6,
	)

	np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])


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


@pytest.mark.parametrize("angles,message", [
	([0, 0, 180], "acquisition indices must be strictly increasing"),
	([-45, 90, 180], "acquisition indices cannot be negative"),
	([0, 90, 179.98], "acquisition index 1969 is outside 1969 planned slots"),
])
def test_endpoint_hint_preserves_other_angle_validation_errors(angles, message):
	with pytest.raises(ValueError) as error:
		indices_from_angles(
			np.deg2rad(angles),
			recorded_count=len(angles),
			planned_count=1969,
			rotation_start_deg=0,
			rotation_end_deg=180,
		)
	assert str(error.value) == message
