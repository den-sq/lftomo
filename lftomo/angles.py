"""Validated acquisition-index and projection-angle provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def validate_acquisition_indices(
	indices,
	*,
	recorded_count: int,
	planned_count: int,
) -> np.ndarray:
	"""Return a validated one-dimensional index for each recorded projection."""

	values = np.asarray(indices)
	if values.ndim != 1:
		raise ValueError("acquisition indices must be one-dimensional")
	if len(values) != recorded_count:
		raise ValueError(
			f"acquisition indices contain {len(values)} values for {recorded_count} "
			"recorded projections"
		)
	if not np.issubdtype(values.dtype, np.integer):
		if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
			raise ValueError("acquisition indices must contain integers")
	values = values.astype(np.int64, copy=False)
	if np.any(values < 0):
		raise ValueError("acquisition indices cannot be negative")
	if len(values) > 1 and np.any(np.diff(values) <= 0):
		raise ValueError("acquisition indices must be strictly increasing")
	if len(values) and values[-1] >= planned_count:
		raise ValueError(
			f"acquisition index {values[-1]} is outside {planned_count} planned slots"
		)
	return values


def angles_from_indices(
	indices,
	*,
	planned_count: int,
	rotation_start_deg: float,
	rotation_end_deg: float,
) -> np.ndarray:
	"""Map planned-slot indices to endpoint-exclusive angles in radians."""

	step = np.deg2rad(rotation_end_deg - rotation_start_deg) / planned_count
	start = np.deg2rad(rotation_start_deg)
	return np.asarray(start + np.asarray(indices, dtype=np.float64) * step, dtype=np.float32)


def indices_from_angles(
	angles_radians,
	*,
	recorded_count: int,
	planned_count: int,
	rotation_start_deg: float,
	rotation_end_deg: float,
) -> np.ndarray:
	"""Recover and validate planned-slot indices under the acquisition convention."""

	angles = np.asarray(angles_radians, dtype=np.float64)
	if angles.ndim != 1 or len(angles) != recorded_count:
		raise ValueError(
			f"angles must contain one one-dimensional value for each of {recorded_count} "
			"recorded projections"
		)
	if not np.all(np.isfinite(angles)):
		raise ValueError("angles must be finite")
	start = np.deg2rad(rotation_start_deg)
	step = np.deg2rad(rotation_end_deg - rotation_start_deg) / planned_count
	if step == 0:
		raise ValueError("rotation start and end angles must differ")
	indices = np.rint((angles - start) / step).astype(np.int64)
	indices = validate_acquisition_indices(
		indices, recorded_count=recorded_count, planned_count=planned_count,
	)
	expected = angles_from_indices(
		indices,
		planned_count=planned_count,
		rotation_start_deg=rotation_start_deg,
		rotation_end_deg=rotation_end_deg,
	)
	tolerance = max(1e-6, abs(step) * 1e-4)
	if not np.allclose(angles, expected, rtol=0, atol=tolerance):
		raise ValueError("angles are inconsistent with acquisition indices and convention")
	return indices


def hardware_timestamps(
	path: str | Path,
	*,
	column: int = 3,
	delimiter: str = "\t",
	header_rows: int = 1,
) -> np.ndarray:
	"""Read integer hardware timestamps without interpreting wall-clock fields."""

	path = Path(path)
	values = []
	with path.open("r", encoding="utf-8") as stream:
		for line_number, line in enumerate(stream, start=1):
			if line_number <= header_rows or not line.strip():
				continue
			fields = line.rstrip("\r\n").split(delimiter)
			if column >= len(fields):
				raise ValueError(
					f"hardware timestamp column {column} is missing at {path}:{line_number}"
				)
			try:
				values.append(int(fields[column]))
			except ValueError as error:
				raise ValueError(
					f"invalid hardware timestamp at {path}:{line_number}"
				) from error
	if not values:
		raise ValueError(f"hardware timestamp source contains no records: {path}")
	return np.asarray(values, dtype=np.int64)


def indices_from_hardware_timestamps(
	timestamps,
	*,
	recorded_count: int,
	planned_count: int,
) -> np.ndarray:
	"""Convert timestamp gaps to cumulative planned-slot indices.

	The endpoint-calibrated nominal interval makes every interval contribute at
	least one slot and a near-double interval contribute two.  Cumulative sums,
	not per-row local offsets, keep all projections after a drop shifted.
	"""

	values = np.asarray(timestamps)
	if values.ndim != 1 or len(values) != recorded_count:
		raise ValueError(
			f"hardware timestamps contain {len(values)} values for {recorded_count} "
			"recorded projections"
		)
	if len(values) == 1:
		return validate_acquisition_indices(
			[0], recorded_count=recorded_count, planned_count=planned_count,
		)
	deltas = np.diff(values.astype(np.float64))
	if np.any(deltas <= 0):
		raise ValueError("hardware timestamps must be strictly increasing")
	nominal = float(values[-1] - values[0]) / (planned_count - 1)
	if nominal <= 0:
		raise ValueError("hardware timestamps do not span the planned acquisition")
	increments = np.rint(deltas / nominal).astype(np.int64)
	if np.any(increments < 1):
		raise ValueError("hardware timestamp interval resolved to fewer than one slot")
	indices = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(increments)))
	return validate_acquisition_indices(
		indices, recorded_count=recorded_count, planned_count=planned_count,
	)
