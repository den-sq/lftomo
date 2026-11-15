"""One-way adapters from the versioned plain-data phase calibration contract."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite

from lftomo.config.models import Scan, StageConfig


def _require_scan_match(field: str, exported: float, scanned: float | None) -> None:
	# Allow unit-conversion roundoff, not a physical calibration tolerance.
	if not isfinite(exported) or scanned is None or not isclose(exported, scanned, rel_tol=1e-12, abs_tol=0.0):
		raise ValueError(f"phase calibration {field} mismatch: export has {exported!r}, scan has {scanned!r}")


def multimaterial_config(calibration: Mapping, scan: Scan, **params) -> tuple[Scan, StageConfig]:
	"""Validate calibration compatibility and return the same scan with stage config.

	The export is an input boundary, never a second runtime table or geometry.
	Its target energy and native geometry must match the authoritative scan;
	the source calibration energy may differ after a transfer. Numeric matching
	allows only unit-conversion roundoff (relative tolerance 1e-12, absolute 0).
	The multimaterial driver explicitly fits its ordered coefficients through
	the origin and preserves the producer's weights, including for affine
	composition exports. Neither input is modified.
	"""
	if calibration.get("schema") != "mctutil.phase.material_table" \
			or calibration.get("schema_version") != 1:
		raise ValueError("unsupported phase material-table schema/version; expected mctutil.phase.material_table v1")
	if (calibration.get("transfer") or {}).get("extreme_weights"):
		raise ValueError("phase calibration transfer has extreme_weights; use a validated target-energy table")
	if {"mu_table", "delta_table", "weights"}.intersection(params):
		raise ValueError("calibration owns material coefficients and weights; do not supply duplicate parameters")
	_require_scan_match("effective_energy_kev", calibration["effective_energy_kev"], scan.energy_keV)
	geometry = calibration["geometry"]
	kind = geometry["kind"]
	if kind not in ("parallel", "cone"):
		raise ValueError("phase calibration geometry must be parallel or cone")
	if kind != scan.geometry.kind:
		raise ValueError(f"phase calibration geometry.kind mismatch: export has {kind!r}, scan has {scan.geometry.kind!r}")
	if kind == "parallel":
		_require_scan_match("geometry.pixel_m", geometry["pixel_m"], scan.geometry.pixel_size_um * 1e-6)
		distance_m = None if scan.geometry.ssd_cm is None else scan.geometry.ssd_cm * 1e-2
		_require_scan_match("geometry.distance_m", geometry["distance_m"], distance_m)
	else:
		for field in ("source_to_sample_m", "sample_to_detector_m"):
			_require_scan_match(f"geometry.{field}", geometry[field], getattr(scan.geometry, field))
		_require_scan_match("geometry.detector_pixel_m", geometry["detector_pixel_m"], scan.detector.pixel_size_um * 1e-6)
	table = calibration["material_table"]
	weights = calibration["calibration"]["parameters"].get("weights")
	return scan, StageConfig(kind="phase", algorithm="multimaterial", params={
		"mu_table": table["mu"], "delta_table": table["delta"], "weights": weights, **params,
	})
