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

@pytest.mark.parametrize("module_name", CURRENT_IMPORTS)
def test_current_modules_import(module_name):
	for dependency in RUNTIME_DEPENDENCIES:
		pytest.importorskip(dependency)
	importlib.import_module(module_name)
