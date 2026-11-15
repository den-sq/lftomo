"""Physical scan geometry has one owner across parsing, use and persistence."""

from copy import deepcopy

import pytest

from lftomo.config import ConeGeometry, ParallelGeometry, Scan


def _scan_document():
	return {
		"energy_keV": 20,
		"pixel_size_um": 5,
		"ssd_cm": 25,
		"n_projections": 3,
		"degrees_per_fov": 180,
		"detector": {"pixel_size_um": 10, "rows": 8, "columns": 8},
		"acquisition": {"n_projections": 3, "rotation_end_deg": 180},
		"projections": {"format": "tiff_stack", "path": "projections"},
	}


def _reference_fresnel_scaling(R1, R2, pixel_det):
	# mctutil phase_tomo.fresnel_scaling at d05f5b06a891d55500b2c76c3c77dc65e58199c0.
	# This reference stays local: lftomo does not import the producer package.
	M = (R1 + R2) / R1
	return R1 * R2 / (R1 + R2), M, pixel_det / M


def test_parallel_geometry_preserves_legacy_serialization_without_storing_scalar_copies():
	document = _scan_document()
	original = deepcopy(document)
	scan = Scan.model_validate(document)
	assert document == original
	assert isinstance(scan.geometry, ParallelGeometry)
	assert "pixel_size_um" not in scan.__dict__
	assert "ssd_cm" not in scan.__dict__
	assert scan.pixel_size_um == 5
	assert scan.ssd_cm == 25
	assert scan.magnification == 1
	expected = {
		**document,
		"n_fov": 1,
		"helical": {"enabled": False, "start_um": None, "stop_um": None, "pitch_um_per_projection": None},
		"detector": {**document["detector"], "name": None},
		"acquisition": {**document["acquisition"], "rotation_start_deg": 0.0, "exposure_ms": None},
		"projections": {**document["projections"], "pattern": "*.tif*"},
		"metadata": {},
	}
	assert scan.model_dump(mode="json") == expected
	assert Scan.model_validate(expected) == scan
	assert Scan.from_yaml(scan.to_yaml()) == scan
	assert scan.model_copy(update={"metadata": {"sample_id": "sample"}}).geometry is scan.geometry


def test_explicit_parallel_geometry_has_the_same_legacy_wire_format():
	document = _scan_document()
	scan = Scan.model_validate(document)
	document["geometry"] = {
		"kind": "parallel", "pixel_size_um": document.pop("pixel_size_um"), "ssd_cm": document.pop("ssd_cm"),
	}
	assert Scan.model_validate(document) == scan
	assert Scan.model_validate(document).model_dump(mode="json") == scan.model_dump(mode="json")


@pytest.mark.parametrize("R1,R2,pixel_um", [(0.5, 0.5, 10.0), (1.0, 0.1, 0.637), (0.25, 0.75, 20.0)])
def test_cone_geometry_matches_reference_and_persists_only_native_inputs(R1, R2, pixel_um):
	document = _scan_document()
	document.pop("pixel_size_um")
	document.pop("ssd_cm")
	document["detector"]["pixel_size_um"] = pixel_um
	document["geometry"] = {"kind": "cone", "source_to_sample_m": R1, "sample_to_detector_m": R2}
	scan = Scan.model_validate(document)
	distance_m, magnification, sample_pixel_m = _reference_fresnel_scaling(R1, R2, pixel_um * 1e-6)
	assert scan.magnification == pytest.approx(magnification)
	assert scan.ssd_cm == pytest.approx(distance_m * 100)
	assert scan.pixel_size_um == pytest.approx(sample_pixel_m * 1e6)
	if R1 == R2 == 0.5:
		# Hand check: equal half-metre arms give M=2, z_eff=0.25m and half-size pixels.
		assert (scan.magnification, scan.ssd_cm, scan.pixel_size_um) == (2.0, 25.0, 5.0)
	serialized = scan.model_dump(mode="json")
	assert serialized["geometry"] == document["geometry"]
	assert not {"pixel_size_um", "ssd_cm", "magnification"} & serialized.keys()
	assert not {"pixel_size_um", "ssd_cm", "magnification"} & scan.__dict__.keys()
	assert Scan.model_validate(serialized) == scan
	assert Scan.from_yaml(scan.to_yaml()) == scan
	with pytest.raises(AttributeError, match="no setter"):
		scan.pixel_size_um = 9
	with pytest.raises(AttributeError, match="no setter"):
		scan.ssd_cm = 9


@pytest.mark.parametrize("field,value", [("pixel_size_um", 5), ("ssd_cm", 25)])
def test_cone_geometry_rejects_competing_effective_input(field, value):
	document = _scan_document()
	document.pop("pixel_size_um")
	document.pop("ssd_cm")
	document["geometry"] = {"kind": "cone", "source_to_sample_m": 0.5, "sample_to_detector_m": 0.5}
	document[field] = value
	with pytest.raises(ValueError, match="provide geometry or legacy"):
		Scan.model_validate(document)


@pytest.mark.parametrize("field", ["source_to_sample_m", "sample_to_detector_m"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_cone_geometry_rejects_distances_that_do_not_define_finite_scaling(field, value):
	parameters = {"source_to_sample_m": 0.5, "sample_to_detector_m": 0.5, field: value}
	with pytest.raises(ValueError, match=field):
		ConeGeometry(**parameters)
