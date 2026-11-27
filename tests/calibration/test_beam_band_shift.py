"""Scientific model and source-coordinate tests for the band-shift normalizer."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import affine_transform, gaussian_filter1d

from lftomo.calibration import (
	BeamBandShiftModel, CalibrationBuildSession, CalibrationBundle, CalibrationManifest, stream_sha256,
)
from lftomo.config import Recon, Scan, StageConfig
from lftomo.pipeline import LocalArray, PipelineRunContext, PipelineState
from lftomo.pipeline.domains import DataDomain
from lftomo.pipeline.stages import BeamBandShiftNormalizeStage, create_stage
from lftomo.util.shapes import ProjOrder, SinoOrder
from tests.sample_data import attach_source_geometry


def _fixture(tmp_path, *, stretch=True, residual=True, acquisition_indices=None, height=64):
	count, width = 7, 48
	y, x = np.indices((height, width))
	dark = (8 + 0.02 * x).astype(np.float32)
	white = (dark + 1000 * (1 + 0.15 * np.sin(y / 3) + 0.025 * np.cos(x / 5))).astype(np.float32)
	schedule = np.column_stack((np.linspace(-3, 9, count), np.linspace(1, 0.96, count) if stretch else np.ones(count)))
	change = (0.025 * np.sin(y / 9) * np.cos(x / 31)).astype(np.float32)
	model = BeamBandShiftModel(yc=32, dy_end=9, sigma_x=3, residual_component="residual" if residual else None)
	base = gaussian_filter1d(np.log(np.maximum(white - dark, 1)), model.sigma_x, axis=1)
	transmission = (0.7 + 0.08 * np.sin(y / 7) * np.cos(x / 4)).astype(np.float32)
	projections = []
	for dy, magnification in schedule:
		shifted = affine_transform(base, ((1 / magnification, 0), (0, 1)),
			offset=(model.yc + (-model.yc - dy) / magnification, 0), order=3, mode="nearest")
		ratio = np.exp(shifted - base)
		if residual:
			ratio *= np.exp(np.clip(dy / model.dy_end, 0, model.s_max) * change)
		projections.append(transmission * np.maximum(white - dark, 1) * ratio + dark)
	build = CalibrationBuildSession(
		tmp_path / "bundle", calibration_id="band-shift", producer={"package": "test", "version": "1"},
		compatibility={"detector_id": "test", "region": {"height": height, "width": width}},
	)
	for name, values, kind in (("pre_gain", white, "detector_static"), ("pre_dark", dark, "dark"), ("residual", change, "beam_residual")):
		build.write_array_component(name, [values], shape=values.shape, dtype="float32", axes=("Y", "X"),
			kind=kind, coordinate_frame="beam" if kind == "beam_residual" else "detector")
	build.write_array_component("beam_schedule", [schedule], shape=schedule.shape, dtype="float64", axes=("Theta", "Parameter"),
		kind="beam_schedule", coordinate_frame="beam",
		time_axis={"kind": "projection_index", "values": acquisition_indices}, band_shift=model)
	ref = build.finalize()
	scan = Scan.model_validate({
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": count, "degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": height, "columns": width, "name": "test"},
		"acquisition": {"n_projections": count, "rotation_end_deg": 180},
		"projections": {"format": "tiff_stack", "path": ".", "pattern": "*.tif"},
	})
	return np.asarray(projections, dtype=np.float32), transmission, scan, ref


def _state(data, scan, ref, layout=ProjOrder, **geometry):
	values = data.copy() if layout is ProjOrder else data.transpose(1, 0, 2).copy()
	return attach_source_geometry(PipelineState(
		LocalArray(values, layout(*values.shape)), scan, Recon(pipeline=[]),
		artifacts={"calibration_bundle": ref}, domain=DataDomain.RAW,
	), physical_y=scan.detector.rows, **geometry)


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
@pytest.mark.parametrize("stretch,residual", [(True, True), (True, False), (False, False)])
def test_shift_stretch_and_residual_recover_object_transmission(tmp_path, layout, stretch, residual):
	data, transmission, scan, ref = _fixture(tmp_path, stretch=stretch, residual=residual)
	state = _state(data, scan, ref, layout)
	result = BeamBandShiftNormalizeStage().run(state)
	expected = np.broadcast_to(transmission, data.shape)
	if layout is SinoOrder:
		expected = expected.transpose(1, 0, 2)
	np.testing.assert_allclose(np.asarray(result.working_array), expected, rtol=5e-7, atol=1e-7)
	assert set(result.artifacts) == {"calibration_bundle"}


def test_row_roi_and_strided_theta_use_full_calibration_context(tmp_path):
	data, _transmission, scan, ref = _fixture(tmp_path)
	full = BeamBandShiftNormalizeStage().run(_state(data, scan, ref))
	selected = BeamBandShiftNormalizeStage().run(_state(data[1:7:2, 19:27], scan, ref,
		theta_start=1, theta_stride=2, y_start=19))
	np.testing.assert_array_equal(np.asarray(selected.working_array), np.asarray(full.working_array)[1:7:2, 19:27])


def test_hdf5_maps_are_loaded_in_bounded_rows_and_preserve_normalization(tmp_path, monkeypatch):
	import h5py

	data, transmission, scan, ref = _fixture(tmp_path, height=145)
	bundle = CalibrationBundle.open(ref.manifest_path)
	path = ref.root / "maps.h5"
	names = ("pre_gain", "pre_dark", "residual")
	with h5py.File(path, "w") as handle:
		for name in names:
			handle.create_dataset(name, data=bundle.read_component(name))
	document = bundle.manifest.model_dump(mode="json")
	for name in names:
		document["components"][name]["payload"].update(
			path=path.name, format="hdf5", dataset=name, sha256=stream_sha256(path),
		)
	ref.manifest_path.write_text(CalibrationManifest.model_validate(document).to_yaml())
	reads = []
	read = h5py.Dataset.__getitem__

	def bounded_read(dataset, selection, *args, **kwargs):
		if dataset.name.removeprefix("/") in names:
			rows = selection[0]
			reads.append(rows.stop - rows.start)
		return read(dataset, selection, *args, **kwargs)

	monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded_read)
	result = BeamBandShiftNormalizeStage().run(_state(data, scan, ref))
	np.testing.assert_allclose(np.asarray(result.working_array),
		np.broadcast_to(transmission, data.shape), rtol=5e-7, atol=1e-7)
	assert reads == [64, 64, 17] * len(names)


def test_band_shift_model_roundtrips_as_component_authority(tmp_path):
	_data, _expected, _scan, ref = _fixture(tmp_path)
	bundle = CalibrationBundle.open(ref.manifest_path)
	model = bundle.component("beam_schedule").band_shift
	assert model.yc == 32
	assert model.residual_component == "residual"
	stage = create_stage(StageConfig(kind="normalize", algorithm="beam_band_shift"))
	assert stage.params.model_dump() == {"schedule_component": "beam_schedule"}
	assert stage.contract.accepts_domain == (DataDomain.RAW,)
	assert stage.contract.produces_domain is DataDomain.TRANSMISSION
	with pytest.raises(ValueError, match="nonzero dy_end"):
		BeamBandShiftModel(yc=32, residual_component="residual")


def test_schedule_queries_acquisition_slots_after_dropped_and_selected_frames(tmp_path):
	acquisition_indices = [0, 2, 3, 6, 7, 9, 12]
	data, transmission, scan, ref = _fixture(tmp_path, acquisition_indices=acquisition_indices)
	state = _state(data[1:7:2, 19:27], scan, ref, theta_start=1, theta_stride=2, y_start=19)
	state.dataset = replace(state.dataset, acquisition_indices=np.asarray(acquisition_indices))
	result = BeamBandShiftNormalizeStage().run(state)
	np.testing.assert_allclose(np.asarray(result.working_array),
		np.broadcast_to(transmission[19:27], data[1:7:2, 19:27].shape), rtol=5e-7, atol=1e-7)


def test_process_dispatch_matches_serial(tmp_path):
	data, _transmission, scan, ref = _fixture(tmp_path)
	expected = np.asarray(BeamBandShiftNormalizeStage().run(_state(data, scan, ref)).working_array)
	with PipelineRunContext(max_workers=2) as context:
		buffer = context.storage.allocate(ProjOrder(*data.shape), np.float32, role="raw_staging")
		buffer.array()[...] = data
		state = _state(data, scan, ref)
		state.working_array, state.storage, state.executor = buffer, context.storage, context.executor
		result = context.executor.run_stage(state, BeamBandShiftNormalizeStage())
		np.testing.assert_array_equal(np.asarray(result.working_array), expected)
		assert not any(key.startswith("_beam_band_shift") for key in state.artifacts)


def test_invalid_magnification_fails_with_projection_identity(tmp_path):
	data, _transmission, scan, ref = _fixture(tmp_path)
	bundle = CalibrationBundle.open(ref.manifest_path)
	schedule_path = ref.root / bundle.component("beam_schedule").payload.path
	schedule = np.load(schedule_path)
	schedule[3, 1] = 0
	np.save(schedule_path, schedule)
	state = _state(data, scan, ref)
	with pytest.raises(ValueError, match="projection 3"):
		BeamBandShiftNormalizeStage().run(state)
	assert set(state.artifacts) == {"calibration_bundle"}
