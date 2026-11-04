import cv2
import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import BufferSpec, Pipeline, PipelineState
from lftomo.pipeline.stages import BilateralFilterStage, STAGE_REGISTRY, create_stage
from lftomo.util.shapes import SinoOrder
from tests.sample_data import attach_source_geometry


def _state(array, config):
	return attach_source_geometry(PipelineState(
		working_array=array,
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1,
			n_projections=array.shape[1],
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(n_projections=array.shape[1], rotation_end_deg=180),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
		buffer_spec=BufferSpec(SinoOrder(*array.shape), array.dtype),
	))


def test_bilateral_stage_is_registered_and_filters_each_projection(monkeypatch):
	config = StageConfig(
		kind="filter",
		algorithm="bilateral",
		params={"diameter": 5, "sigma_color": 2, "sigma_space": 3},
	)
	working = np.ones((4, 3, 6), dtype=np.float32)
	calls = []

	def bilateral(image, diameter, sigma_color, sigma_space, borderType):
		calls.append((
			image.dtype, diameter, sigma_color, sigma_space, borderType,
			np.shares_memory(image, working),
		))
		return image + 1

	monkeypatch.setattr("cv2.bilateralFilter", bilateral)
	stage = create_stage(config)
	result = Pipeline([stage]).run(_state(working, config))

	assert STAGE_REGISTRY["filter.bilateral"] is BilateralFilterStage
	assert len(calls) == 3
	assert all(call[0] == np.float32 for call in calls)
	assert all(call[1:5] == (5, 2, 3, cv2.BORDER_DEFAULT) and call[5] for call in calls)
	assert result.working_array is working
	assert result.working_array.dtype == np.float32
	np.testing.assert_array_equal(working, 2)


def test_bilateral_rejects_noncanonical_work_dtype():
	config = StageConfig(kind="filter", algorithm="bilateral")
	with pytest.raises(TypeError, match="canonical float32 work dtype"):
		create_stage(config).run(_state(
			np.ones((4, 3, 6), dtype=np.float64), config,
		))


def test_disabled_bilateral_stage_is_noop(monkeypatch):
	config = StageConfig(
		kind="filter", algorithm="bilateral", params={"enabled": False}
	)
	working = np.ones((2, 3, 4), dtype=np.float32)
	monkeypatch.setattr(
		"cv2.bilateralFilter",
		lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
	)

	result = Pipeline([create_stage(config)]).run(_state(working, config))

	assert result.working_array is working


def test_float32_work_buffer_is_bit_exact_with_public_conversion():
	config = StageConfig(kind="filter", algorithm="bilateral")
	source = np.linspace(0, 1, 4 * 3 * 6, dtype=np.float64).reshape(4, 3, 6)
	expected = np.empty(source.shape, dtype=np.float32)
	for projection in range(source.shape[1]):
		expected[:, projection] = cv2.bilateralFilter(
			source[:, projection].astype(np.float32), 3, 1, 1,
		)
	working = source.astype(np.float32)

	create_stage(config).run(_state(working, config))

	np.testing.assert_array_equal(working, expected)
