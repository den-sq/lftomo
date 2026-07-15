import importlib

import pytest


CURRENT_IMPORTS = [
	"lftomo",
	"lftomo.calibration",
	"lftomo.util.io",
	"lftomo.util.log",
	"lftomo.util.mem",
	"lftomo.util.shapes",
	"lftomo.util.window",
]

RUNTIME_DEPENDENCIES = [
	"click",
	"h5py",
	"natsort",
	"numexpr",
	"numpy",
	"pywt",
	"scipy",
	"skimage",
	"tifffile",
	"tomopy",
]

PREP_IMPORTS = [
	"lftomo.prep.flat",
	"lftomo.prep.phase",
	"lftomo.prep.stripe",
]


@pytest.mark.parametrize("module_name", CURRENT_IMPORTS)
def test_current_modules_import(module_name):
	for dependency in RUNTIME_DEPENDENCIES:
		pytest.importorskip(dependency)
	importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", PREP_IMPORTS)
@pytest.mark.xfail(reason="prep modules still use old lftomo API names; tracked by lftomo#14")
def test_prep_modules_import_after_api_port(module_name):
	importlib.import_module(module_name)
