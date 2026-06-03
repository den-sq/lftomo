import importlib

import pytest

pydantic = pytest.importorskip("pydantic")
yaml = pytest.importorskip("yaml")

from lftomo.config import (  # noqa: E402
	Acquisition,
	Detector,
	HDF5DxchangeProjections,
	Helical,
	Machine,
	Recon,
	Scan,
	Session,
	StageConfig,
	SweepConfig,
	TIFFStackProjections,
)


def roundtrip(model):
	return type(model).from_yaml(model.to_yaml())


def test_machine_roundtrip():
	model = Machine(name="yarrkha", storage_roots={"data": "/data"}, threads=32, memory_gb=256)

	assert roundtrip(model) == model


def test_session_roundtrip():
	model = Session(sample_id="sam19", operator="Sorotassu", location="Berkeley 8.3.2")

	assert roundtrip(model) == model


@pytest.mark.parametrize(
	"projections",
	[
		TIFFStackProjections(format="tiff_stack", path="projections"),
		HDF5DxchangeProjections(format="hdf5_dxchange", path="scan.h5"),
	],
)
def test_scan_roundtrip(projections):
	model = Scan(
		energy_keV=24,
		pixel_size_um=1.3,
		ssd_cm=12.5,
		n_projections=1501,
		degrees_per_fov=180,
		n_fov=1,
		helical=Helical(enabled=False),
		detector=Detector(pixel_size_um=1.3, rows=2048, columns=2048, name="pco.edge"),
		acquisition=Acquisition(n_projections=1501, rotation_end_deg=180),
		projections=projections,
	)

	assert roundtrip(model) == model


def test_scan_rejects_projection_count_mismatch():
	with pytest.raises(ValueError, match="n_projections"):
		Scan(
			energy_keV=24,
			pixel_size_um=1.3,
			n_projections=1501,
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1.3),
			acquisition=Acquisition(n_projections=10, rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		)


def test_scan_rejects_unknown_projection_format():
	with pytest.raises(ValueError):
		Scan.model_validate(
			{
				"energy_keV": 24,
				"pixel_size_um": 1.3,
				"n_projections": 1501,
				"degrees_per_fov": 180,
				"detector": {"pixel_size_um": 1.3},
				"acquisition": {"n_projections": 1501, "rotation_end_deg": 180},
				"projections": {"format": "raw_binary", "path": "data.raw"},
			}
		)


def test_recon_roundtrip():
	model = Recon(
		pipeline_template="b10",
		pipeline=[
			StageConfig(kind="read", algorithm="tiff_stack"),
			StageConfig(kind="phase", algorithm="paganin", params={"alpha": 1e-3}),
			StageConfig(kind="recon", algorithm="gridrec", sweep=SweepConfig(kind="list", params={"center": [1024, 1025]})),
		],
		output_path="recon",
	)

	assert roundtrip(model) == model
