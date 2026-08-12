from pathlib import Path

import numpy as np
import pytest

from lftomo.angles import (
	angles_from_indices,
	hardware_timestamps,
	indices_from_hardware_timestamps,
)
from lftomo.config import HardwareTimestampAngleSource, Scan


pytestmark = pytest.mark.tier2


def test_aaa388_hardware_timestamps_identify_two_dropped_slots(
	dataset_dropped_frame_tiff: Path,
):
	template = Path(__file__).with_name("tier2_metadata") / "dropped_frame_tiff.yaml"
	scan = Scan.from_yaml(template.read_text(encoding="utf-8"))
	source = scan.acquisition.angle_source
	assert isinstance(source, HardwareTimestampAngleSource)
	timestamps = hardware_timestamps(
		dataset_dropped_frame_tiff / source.path,
		column=source.timestamp_column,
		delimiter=source.delimiter,
		header_rows=source.header_rows,
	)
	indices = indices_from_hardware_timestamps(
		timestamps,
		recorded_count=scan.n_projections,
		planned_count=scan.acquisition.n_projections,
	)
	angles = angles_from_indices(
		indices,
		planned_count=scan.acquisition.n_projections,
		rotation_start_deg=scan.acquisition.rotation_start_deg,
		rotation_end_deg=scan.acquisition.rotation_end_deg,
	)
	files = list((dataset_dropped_frame_tiff / scan.projections.path).glob(scan.projections.pattern))

	assert len(files) == scan.n_projections == 3935
	assert len(indices) == len(angles) == 3935
	assert indices[0] == 0
	assert indices[-1] == 3936
	assert set(range(3937)) - set(indices) == {2837, 2888}
	assert np.all(np.diff(indices) > 0)
