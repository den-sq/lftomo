"""One-way adapters from the versioned plain-data phase calibration contract."""

from __future__ import annotations

from collections.abc import Mapping

from lftomo.config.models import Scan, StageConfig


def multimaterial_config(calibration: Mapping, scan: Scan, **params) -> tuple[Scan, StageConfig]:
	"""Apply an exported calibration's energy/geometry and create stage config.

	The export is an input boundary, never a second runtime table or geometry.
	The multimaterial driver explicitly fits its ordered coefficients through
	the origin and preserves the producer's weights, including for affine
	composition exports. Other acquisition metadata comes from ``scan``.
	"""
	if calibration.get("schema") != "mctutil.phase.material_table" \
			or calibration.get("schema_version") != 1:
		raise ValueError("unsupported phase material-table schema/version; expected mctutil.phase.material_table v1")
	if (calibration.get("transfer") or {}).get("extreme_weights"):
		raise ValueError("phase calibration transfer has extreme_weights; use a validated target-energy table")
	if {"mu_table", "delta_table", "weights"}.intersection(params):
		raise ValueError("calibration owns material coefficients and weights; do not supply duplicate parameters")
	geometry = calibration["geometry"]
	data = scan.model_dump()
	data.pop("pixel_size_um", None)
	data.pop("ssd_cm", None)
	data["energy_keV"] = calibration["effective_energy_kev"]
	if geometry["kind"] == "parallel":
		data["geometry"] = {
			"kind": "parallel", "pixel_size_um": geometry["pixel_m"] * 1e6,
			"ssd_cm": geometry["distance_m"] * 100,
		}
	elif geometry["kind"] == "cone":
		data["geometry"] = {
			"kind": "cone", "source_to_sample_m": geometry["source_to_sample_m"],
			"sample_to_detector_m": geometry["sample_to_detector_m"],
		}
		data["detector"]["pixel_size_um"] = geometry["detector_pixel_m"] * 1e6
	else:
		raise ValueError("phase calibration geometry must be parallel or cone")
	resolved_scan = Scan.model_validate(data)
	table = calibration["material_table"]
	weights = calibration["calibration"]["parameters"].get("weights")
	return resolved_scan, StageConfig(kind="phase", algorithm="multimaterial", params={
		"mu_table": table["mu"], "delta_table": table["delta"], "weights": weights, **params,
	})
