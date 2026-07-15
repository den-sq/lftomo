import numpy as np
import pytest
from pathlib import Path
from natsort import natsorted
import tifffile as tf

from lftomo.calibration import BeamSchedule, calibrate_schedule, optimal_transport_interp
from lftomo.config import Recon, Scan
from lftomo.pipeline import NumpyBackend, PipelineState
from lftomo.pipeline.domains import DataDomain
from lftomo.pipeline.stages import BeamTrackingNormalizeStage, create_stage
from lftomo.util.shapes import ProjOrder, SinoOrder


def _scan(flats, count=9):
	return Scan.model_validate({
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": count, "degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": flats.shape[1], "columns": flats.shape[2]},
		"acquisition": {"n_projections": count, "rotation_end_deg": 180},
		"projections": {"format": "tiff_stack", "path": ".", "pattern": "*.tif"},
		"metadata": {"flat_fields": flats},
	})


def _fixture(count=9):
	y, x = np.indices((24, 32))
	detector = np.ones((24, 32)); detector[8:10, 13:15] = 0.7
	pre = detector * 1000 * (1 + 0.08 * np.sin(y / 4) + 0.03 * np.cos(x / 11))
	post = detector * 1050 * (1 + 0.08 * np.sin((y - 3) / 4) + 0.03 * np.cos((x - 2) / 11))
	dark0, dark1 = np.full_like(pre, 8), np.full_like(pre, 10)
	schedule = BeamSchedule(alpha=0.2, beta=0.65)
	gain = optimal_transport_interp(pre, post, transport_sigma=3, schedule=schedule)
	projections = np.stack([
		gain(index / (count - 1))
		for index in range(count)
	]).astype(np.float32)
	return projections, np.stack([pre, post, dark0, dark1]).astype(np.float32), schedule


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
def test_beam_tracking_preserves_layout_and_removes_nonlinear_drift(layout):
	projections, flats, schedule = _fixture()
	data = projections.copy() if layout is ProjOrder else np.transpose(projections, (1, 0, 2)).copy()
	recon = Recon.model_validate({"pipeline": [
		{"kind": "normalize", "algorithm": "beam_tracking", "params": {
			"transport_sigma": 3, "schedule": schedule.__dict__,
		}},
	]})
	state = PipelineState(data, _scan(flats), recon, backend=NumpyBackend(), layout=layout, domain=DataDomain.RAW)
	result = BeamTrackingNormalizeStage(
		transport_sigma=3, schedule=schedule.__dict__,
	).run(state)
	assert result.layout is layout
	assert np.std(np.asarray(result.working_array) - 1) < 2e-6
	assert result.artifacts["beam_tracking_qc"]["projection_count"] == 9


def test_tracking_beats_naive_arithmetic_endpoint_interpolation():
	projections, flats, schedule = _fixture()
	pre, post, dark0, dark1 = flats
	tracked = projections.copy()
	state = PipelineState(
		tracked, _scan(flats), Recon.model_validate({"pipeline": []}),
		backend=NumpyBackend(), layout=ProjOrder, domain=DataDomain.RAW,
	)
	BeamTrackingNormalizeStage(transport_sigma=3, schedule=schedule.__dict__).run(state)
	naive = []
	for index, projection in enumerate(projections):
		t = index / (len(projections) - 1)
		gain = (1 - t) * pre + t * post
		dark = (1 - t) * dark0 + t * dark1
		naive.append((projection - dark) / (gain - dark))
	assert np.std(tracked - 1) < np.std(np.asarray(naive) - 1) / 20


def test_schedule_fit_recovers_nonzero_initial_beam_time():
	_, flats, schedule = _fixture()
	log_pre, log_post = np.log(flats[0]), np.log(flats[1])
	projections = np.stack([
		np.exp((1 - schedule(fraction)) * log_pre + schedule(fraction) * log_post)
		for fraction in (0, 0.5, 1)
	])
	fit = calibrate_schedule(projections, [0, 0.5, 1], flats[0], flats[1], mask=np.ones(flats.shape[1:], bool))
	assert fit.schedule.alpha == pytest.approx(schedule.alpha, abs=0.03)
	assert fit.schedule.beta == pytest.approx(schedule.beta, abs=0.03)
	assert fit.residual_std < 0.01


def test_stage_registry_and_domain_contract():
	stage = create_stage(Recon.model_validate({"pipeline": [
		{"kind": "normalize", "algorithm": "beam_tracking"},
	]}).pipeline[0])
	assert stage.contract.accepts == (ProjOrder, SinoOrder)
	assert stage.contract.prefers is ProjOrder
	assert stage.contract.accepts_domain == (DataDomain.RAW,)
	assert stage.contract.produces_domain is DataDomain.TRANSMISSION


@pytest.mark.integration
def test_aaa392_timestamped_air_residual_improves_over_weighted(request):
	if not request.config.getoption("--run-integration"):
		pytest.skip("requires --run-integration and local AAA392 data")
	root = Path("/mnt/wsl/PHYSICALDRIVE2p1/flatfield_advanced/AAA392_20keV")
	if not root.is_dir():
		pytest.skip("AAA392 copy is unavailable")
	def median(pattern):
		directory = next(root.glob(pattern))
		return np.median(np.stack([
			tf.memmap(path, mode="r")[:192, :1024]
			for path in natsorted(directory.glob("*.tif"))
		]), axis=0)
	pre, post = median("*rescan5_pregains_tiff"), median("*rescan5_postgains_tiff")
	dark0, dark1 = median("*rescan5_predarks_tiff"), median("*rescan5_postdarks_tiff")
	projection_dir = next(root.glob("*rescan5_projections_tiff/projections"))
	timestamps = np.array([162723463462000, 162832427675000, 162911873666000,
		162915736571000, 163012286547000])
	projections = np.stack([
		tf.memmap(projection_dir / f"proj_{timestamp}.tif", mode="r")[:192, :1024]
		for timestamp in timestamps
	])
	fractions = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
	from lftomo.calibration import auto_air_mask
	mask = auto_air_mask(projections[0], pre)
	fit = calibrate_schedule(projections, fractions, pre, post, mask=mask)
	tracked_gain = optimal_transport_interp(pre, post, schedule=fit.schedule)
	def residual(gain_at):
		values = []
		for projection, fraction in zip(projections, fractions, strict=True):
			dark = (1 - fraction) * dark0 + fraction * dark1
			corrected = (projection - dark) / np.clip(gain_at(fraction) - dark, 1e-6, None)
			values.append(np.std(corrected[mask] - np.median(corrected[mask])))
		return float(np.mean(values))
	weighted = residual(lambda fraction: (1 - fraction) * pre + fraction * post)
	tracked = residual(tracked_gain)
	assert fit.air_pixel_count > 1000
	assert fit.schedule.alpha > 0.4
	assert tracked < weighted
