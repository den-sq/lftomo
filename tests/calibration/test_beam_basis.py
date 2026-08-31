from pathlib import Path

import h5py
import numpy as np
import pytest

from lftomo.calibration import (
	CalibrationBuildSession, CalibrationBundle, ConcatenatedFrameSource,
	HDF5RandomFrameSource, build_beam_basis, fit_projection_beam,
)
from lftomo.config import Recon, Scan, StageConfig
from lftomo.pipeline import BufferSpec, NumpyBackend, PipelineState, validate_pipeline
from lftomo.pipeline.domains import DataDomain
from lftomo.pipeline.stages import (
	BeamBasisFitNormalizeStage,
	STAGE_REGISTRY,
	create_stage,
)
from lftomo.util.shapes import ProjOrder, SinoOrder
from tests.sample_data import attach_source_geometry


class ArrayFrameSource:
	def __init__(self, frames):
		self.frames = np.asarray(frames)
		self.frame_count, self.rows, self.columns = self.frames.shape
		self.dtype = self.frames.dtype
		self.windows = []

	def read(self, index, *, y_start=0, y_stop=None, x_start=0, x_stop=None):
		self.windows.append((index, y_start, y_stop, x_start, x_stop))
		return self.frames[index, y_start:y_stop, x_start:x_stop]


def _beam_frames(rows=40, columns=52, count=9):
	y, x = np.indices((rows, columns))
	mean = np.log(1200 * (1 + 0.08 * np.cos(y / 7)))
	vertical = np.sin((y - rows / 2) / 5) / 7
	horizontal = np.cos((x - columns / 2) / 9) / 8
	coefficients = np.column_stack([
		np.linspace(-0.8, 0.8, count), np.sin(np.linspace(0, 2 * np.pi, count)),
	])
	frames = np.stack([
		np.exp(mean + a * vertical + b * horizontal)
		for a, b in coefficients
	]).astype(np.float32)
	return frames, mean, vertical, horizontal


def _build(tmp_path, frames, *, crop=None, name="bundle"):
	source = ArrayFrameSource(frames)
	if crop is None:
		crop = (0, frames.shape[1], 0, frames.shape[2])
	row0, row1, column0, column1 = crop
	build = CalibrationBuildSession(
		tmp_path / name, calibration_id=name,
		producer={"package": "lftomo", "version": "test"},
		compatibility={
			"detector_id": "camera",
			"region": {"height": row1 - row0, "width": column1 - column0,
				"y_start": row0, "x_start": column0},
			"camera_segment": "segment-a",
		},
	)
	result = build_beam_basis(
		build, source, n_components=2, y_chunk=7, crop=crop,
		camera_segment="segment-a", training_coverage=["scan-a", "scan-b"],
	)
	return source, result, build.finalize()


def _scan(rows, columns, count):
	return Scan.model_validate({
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": count, "degrees_per_fov": 180,
		"detector": {"name": "camera", "pixel_size_um": 1,
			"rows": rows, "columns": columns},
		"acquisition": {"n_projections": count, "rotation_end_deg": 180},
		"projections": {"format": "tiff_stack", "path": ".", "pattern": "*.tif"},
	})


def test_builder_uses_bounded_two_pass_gram_pca_and_persists_coverage(tmp_path):
	frames, *_ = _beam_frames()
	source, result, ref = _build(tmp_path, frames, crop=(3, 37, 4, 48))
	bundle = CalibrationBundle.open(ref.manifest_path, verify_hashes=True)
	basis = bundle.read_component("beam_basis", y_start=3, y_stop=37)
	descriptor = bundle.component("beam_basis")

	assert basis.shape == (3, 34, 44)
	assert result.training_residual < 1e-5
	assert sum(window[1] == 3 for window in source.windows) == 2 * len(frames)
	assert max(window[2] - window[1] for window in source.windows) <= 7
	assert descriptor.compatibility.training_coverage == ["scan-a", "scan-b"]
	assert descriptor.metadata["training_frame_count"] == len(frames)
	assert len(descriptor.metadata["training_coefficient_min"]) == 2


def test_robust_fit_rejects_sample_false_positives_and_fails_without_air(tmp_path):
	frames, mean, vertical, horizontal = _beam_frames()
	_, _, ref = _build(tmp_path, frames)
	basis = CalibrationBundle.open(ref.manifest_path).read_component("beam_basis")
	beam = np.exp(mean + 0.17 * vertical - 0.23 * horizontal)
	y, x = np.indices(beam.shape)
	sample = (y - 20) ** 2 + (x - 26) ** 2 < 9 ** 2
	transmission = np.ones_like(beam); transmission[sample] = 0.58
	projection = beam * transmission
	supplied = np.ones_like(sample)

	fit = fit_projection_beam(
		projection, basis, air_mask=supplied, minimum_air_pixels=200,
		trim_sigma=3, erode=1,
	)
	assert np.std(fit.transmission[~sample] - 1) < 2e-3
	assert fit.qc.rejected_fraction > 0.05
	assert fit.qc.basis_rank == 2
	assert fit.qc.out_of_basis < 0.05

	with pytest.raises(ValueError, match="insufficient trustworthy air"):
		fit_projection_beam(
			beam * 0.5, basis, minimum_air_pixels=200, minimum_air_fraction=0.1,
		)


def test_stage_preserves_both_layouts_and_reports_coefficients_and_qc(tmp_path):
	frames, mean, vertical, horizontal = _beam_frames()
	_, _, ref = _build(tmp_path, frames)
	y, x = np.indices(mean.shape)
	projections = []
	for index, coefficient in enumerate((-0.4, 0.0, 0.4)):
		beam = np.exp(mean + coefficient * vertical + 0.2 * horizontal)
		centre = 16 + 5 * index
		transmission = np.ones_like(beam)
		transmission[(y - 20) ** 2 + (x - centre) ** 2 < 7 ** 2] = 0.65
		projections.append(beam * transmission)
	projections = np.asarray(projections, dtype=np.float32)

	for layout, values in (
		(ProjOrder, projections.copy()),
		(SinoOrder, projections.transpose(1, 0, 2).copy()),
	):
		state = attach_source_geometry(PipelineState(
			values, _scan(40, 52, 3), Recon.model_validate({"pipeline": []}),
			artifacts={"calibration_bundle": ref}, backend=NumpyBackend(),
			buffer_spec=BufferSpec(layout(*values.shape), values.dtype),
			domain=DataDomain.RAW,
		))
		BeamBasisFitNormalizeStage(maximum_out_of_basis=1.1).run(state)
		corrected = values if layout is ProjOrder else values.transpose(1, 0, 2)
		assert np.quantile(np.abs(corrected - 1), 0.6) < 0.01
		assert len(state.artifacts["beam_basis_qc"]["coefficient_std"]) == 2
		assert state.artifacts["beam_basis_qc"]["violating_projection_count"] == 0

	assert STAGE_REGISTRY["normalize.beam_basis_fit"] is BeamBasisFitNormalizeStage
	assert BeamBasisFitNormalizeStage.contract.accepts_domain == (DataDomain.RAW,)
	assert BeamBasisFitNormalizeStage.contract.produces_domain is DataDomain.TRANSMISSION
	assert BeamBasisFitNormalizeStage.contract.prefers is ProjOrder


def test_gross_mask_contamination_is_caught_by_stage_qc(tmp_path):
	frames, mean, vertical, horizontal = _beam_frames()
	_, _, ref = _build(tmp_path, frames)
	beam = np.exp(mean + 0.17 * vertical - 0.23 * horizontal)
	y, x = np.indices(beam.shape)
	transmission = np.ones_like(beam)
	transmission[(y - 20) ** 2 + (x - 26) ** 2 < 22 ** 2] = 0.45
	projection = (beam * transmission).astype(np.float32)[None]
	artifacts = {
		"calibration_bundle": ref,
		"air_mask": np.ones_like(beam, dtype=bool),
	}
	state = attach_source_geometry(PipelineState(
		projection.copy(), _scan(40, 52, 1), Recon.model_validate({"pipeline": []}),
		artifacts=artifacts.copy(), backend=NumpyBackend(),
		buffer_spec=BufferSpec(ProjOrder(*projection.shape), projection.dtype),
		domain=DataDomain.RAW,
	))
	with pytest.raises(ValueError, match="failed QC"):
		BeamBasisFitNormalizeStage().run(state)
	diagnostic = attach_source_geometry(PipelineState(
		projection.copy(), _scan(40, 52, 1), Recon.model_validate({"pipeline": []}),
		artifacts=artifacts.copy(), backend=NumpyBackend(),
		buffer_spec=BufferSpec(ProjOrder(*projection.shape), projection.dtype),
		domain=DataDomain.RAW,
	))
	BeamBasisFitNormalizeStage(fail_on_qc=False).run(diagnostic)
	qc = diagnostic.artifacts["beam_basis_qc"]
	assert qc["violating_projection_count"] == 1
	assert qc["projections"][0]["violations"]
	assert "candidate-mask air pixels are the majority" in qc["mask_assumption"]


def test_beam_basis_is_the_sole_normalization_in_a_complete_pipeline():
	configs = [
		StageConfig(kind="read", algorithm="tiff_stack", params={"order": "projection"}),
		StageConfig(kind="normalize", algorithm="beam_basis_fit"),
		StageConfig(kind="phase", algorithm="paganin"),
		StageConfig(kind="recon", algorithm="gridrec", params={"center": 26}),
	]
	plan = validate_pipeline(
		[create_stage(config) for config in configs], _scan(40, 52, 3),
		available_artifacts={"calibration_bundle": object()}, mode="complete",
	)
	normalize = next(step for step in plan.steps if step.stage_name == "normalize.beam_basis_fit")
	assert normalize.input_domain is DataDomain.RAW
	assert normalize.output_domain is DataDomain.TRANSMISSION


def test_beam_basis_mask_quantiles_are_configurable_and_validated():
	stage = BeamBasisFitNormalizeStage(exposure_quantile=0.1, edge_quantile=0.9)
	assert stage.exposure_quantile == 0.1
	assert stage.edge_quantile == 0.9
	with pytest.raises(ValueError, match="quantiles"):
		BeamBasisFitNormalizeStage(edge_quantile=1.1)


@pytest.mark.integration
def test_ra00808_cross_scan_basis_improves_heldout_flat_and_real_air(tmp_path, request):
	if not request.config.getoption("--run-integration"):
		pytest.skip("requires --run-integration and the mounted RA-00808 data")
	root = Path("/mnt/z/gpfs/phenome/Berkeley/RA-00808_ang")
	training_names = (
		"20260610_201754_251K86sec3.h5", "20260610_210107_phantom.h5",
		"20260610_213558_236ksec1.h5",
	)
	heldout_path = root / "20260610_215400_236ksec2.h5"
	if not heldout_path.is_file() or any(not (root / name).is_file() for name in training_names):
		pytest.skip("the compatible RA-00808 training group is unavailable")
	sources = [
		HDF5RandomFrameSource(root / name, frame_stop=5) for name in training_names
	]
	source = ConcatenatedFrameSource(sources)
	crop = (1000, 1128, 0, 2560)
	build = CalibrationBuildSession(
		tmp_path / "ra-basis", calibration_id="ra-basis",
		producer={"package": "lftomo", "version": "test"},
		compatibility={
			"detector_id": "als832", "beamline": "als832", "energy_keV": 20,
			"region": {"y_start": 1000, "height": 128, "width": 2560},
			"camera_segment": "i0x-10_i0y0_ssd150.427215",
		},
	)
	build_beam_basis(
		build, source, n_components=3, y_chunk=32, crop=crop,
		camera_segment="i0x-10_i0y0_ssd150.427215",
		training_coverage=list(training_names),
	)
	ref = build.finalize()
	basis = CalibrationBundle.open(ref.manifest_path).read_component(
		"beam_basis", y_start=1000, y_stop=1128,
	)
	with h5py.File(heldout_path, "r") as handle:
		heldout = np.asarray(handle["exchange/data_white"][0, 1000:1128, :])
		projection = np.asarray(handle["exchange/data"][200, 1000:1128, :])
	clean_fit = fit_projection_beam(
		heldout, basis, air_mask=np.ones_like(heldout, bool), minimum_air_pixels=5000,
	)
	mean_error = float(np.sqrt(np.mean((heldout - np.exp(basis[0])) ** 2)) / np.mean(heldout))
	basis_error = float(np.sqrt(np.mean((heldout - clean_fit.beam) ** 2)) / np.mean(heldout))
	assert basis_error < mean_error

	real_fit = fit_projection_beam(
		projection, basis, minimum_air_pixels=5000, minimum_air_fraction=0.02,
	)
	mean_air_residual = float(np.sqrt(np.mean(
		(np.log(np.clip(projection, 1e-6, None)) - basis[0])[real_fit.air_mask] ** 2
	)))
	assert real_fit.qc.fit_residual < mean_air_residual
	assert real_fit.qc.air_fraction > 0.02
	assert real_fit.qc.basis_rank == 3
