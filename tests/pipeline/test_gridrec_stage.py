import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import BufferSpec, Pipeline, PipelineState
from lftomo.pipeline.stages import GridrecStage, STAGE_REGISTRY, create_stage
from lftomo.util.shapes import ProjOrder, ReconOrder, SinoOrder
from lftomo.util.window import TomoWindow


def _state(array, config, artifacts=None, layout=ProjOrder):
	shape = layout(*array.shape)
	return PipelineState(
		working_array=array,
		scan=Scan(
			energy_keV=20,
			pixel_size_um=1,
			n_projections=3,
			degrees_per_fov=180,
			detector=Detector(pixel_size_um=1),
			acquisition=Acquisition(n_projections=3, rotation_start_deg=10, rotation_end_deg=190),
			projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		),
		recon=Recon(pipeline=[config]),
		artifacts={} if artifacts is None else artifacts,
		buffer_spec=BufferSpec(shape, array.dtype),
		source_window=TomoWindow.from_bounds(shape.to(ProjOrder), dtype=array.dtype),
	)


def test_gridrec_is_registered_and_uses_center_artifact(monkeypatch):
	config = StageConfig(
		kind="recon",
		algorithm="gridrec",
		params={"circle_mask": True, "ncore": 2},
	)
	working = np.ones((2, 3, 5), dtype=np.float32)
	calls = {}

	def recon(data, angles, **kwargs):
		calls["recon"] = (data, angles, kwargs)
		kwargs["init_recon"].fill(3)
		return kwargs["init_recon"]

	def circ_mask(data, **kwargs):
		calls["mask"] = kwargs
		data += 1
		return data

	monkeypatch.setattr("tomopy.recon", recon)
	monkeypatch.setattr("tomopy.circ_mask", circ_mask)
	stage = create_stage(config)
	result = Pipeline([stage]).run(_state(
		working, config, {"center": 2.5}, layout=SinoOrder,
	))

	assert STAGE_REGISTRY["recon.gridrec"] is GridrecStage
	assert stage.contract.accepted_layouts == (ProjOrder, SinoOrder)
	assert stage.contract.produces is ReconOrder
	assert calls["recon"][2]["algorithm"] == "gridrec"
	assert calls["recon"][2]["sinogram_order"] is True
	assert calls["recon"][2]["center"] == pytest.approx(2.5)
	np.testing.assert_allclose(calls["recon"][1], np.deg2rad([10, 70, 130]))
	assert calls["mask"] == {"axis": 0, "ratio": 1.0, "val": 0.0, "ncore": 2}
	assert result.working_array.dtype == np.float32
	np.testing.assert_array_equal(result.working_array, 4)
	assert stage.params.filter_name == "shepp"
	assert stage.params.circle_mask_ratio == 1.0


def test_gridrec_projection_order_accepts_center_param_and_crops(monkeypatch):
	config = StageConfig(
		kind="recon",
		algorithm="gridrec",
		params={
			"center": [2.0, 2.1],
			"circle_mask": False,
			"edge_buffer_left": 1,
			"edge_buffer_right": 1,
			"edge_buffer_input": True,
			"output_width": 2,
			"filter_name": "butterworth",
			"filter_parameters": [0.2, 2],
		},
	)
	working = np.ones((3, 2, 5), dtype=np.float32)
	calls = {}

	def recon(data, _angles, **kwargs):
		calls["data_shape"] = data.shape
		calls.update(kwargs)
		kwargs["init_recon"][...] = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
		return kwargs["init_recon"]

	monkeypatch.setattr("tomopy.recon", recon)
	stage = create_stage(config)
	result = Pipeline([stage]).run(_state(working, config))

	assert ProjOrder in stage.contract.accepted_layouts
	assert stage.contract.consumes_artifacts == ()
	assert calls["sinogram_order"] is False
	assert calls["data_shape"] == (3, 2, 3)
	np.testing.assert_allclose(calls["center"], [2.0, 2.1])
	assert calls["filter_name"] == "butterworth"
	assert calls["filter_par"] == [0.2, 2]
	assert result.working_array.shape == (2, 3, 2)


def test_gridrec_uses_local_core_rows_and_keeps_global_window(monkeypatch):
	config = StageConfig(
		kind="recon", algorithm="gridrec",
		params={"circle_mask": False,
			"edge_buffer_left": 1, "edge_buffer_right": 1},
	)
	working = np.arange(3 * 4 * 7, dtype=np.float32).reshape(3, 4, 7)
	calls = {}

	def recon(data, _angles, **kwargs):
		calls["data"] = data.copy()
		calls["center"] = kwargs["center"]
		kwargs["init_recon"].fill(1)
		return kwargs["init_recon"]

	monkeypatch.setattr("tomopy.recon", recon)
	state = _state(working, config)
	window = TomoWindow(
		ProjOrder(3, 2, 5), ProjOrder(0, 1, 1), ProjOrder(0, 10, 1),
		ProjOrder(1, 1, 1), np.float32, np.float32,
		pad_after=ProjOrder(0, 1, 1),
	)
	state.source_window = window
	state.artifacts["center"] = np.array([2.0, 2.1, 2.2, 2.3])
	result = create_stage(config).run(state)

	np.testing.assert_array_equal(calls["data"], working[:, 1:3, 1:6])
	np.testing.assert_allclose(calls["center"], [2.1, 2.2])
	assert result.working_array.shape == (2, 5, 3)
	assert result.source_window is window


def test_gridrec_rejects_missing_or_wrong_length_center(monkeypatch):
	config = StageConfig(kind="recon", algorithm="gridrec")
	state = _state(np.ones((3, 2, 5), dtype=np.float32), config)
	stage = create_stage(config)

	with pytest.raises(ValueError, match="center parameter or center artifact"):
		stage.run(state)

	state.artifacts["center"] = [2.0]
	with pytest.raises(ValueError, match="one value for each of 2 slices"):
		stage.run(state)
