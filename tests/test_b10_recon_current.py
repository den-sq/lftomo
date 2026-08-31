import sys
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import tifffile as tf
import yaml

from examples.b10_recon_current import B10ReconConfig, run
from lftomo.pipeline import WindowConfig
from lftomo.runner import run_reconstruction


def _write_source(path: Path, projections: np.ndarray) -> None:
	with h5py.File(path, "w") as handle:
		exchange = handle.create_group("exchange")
		exchange.create_dataset("data", data=projections)


def _write_flats(path: Path, size: int) -> None:
	path.mkdir()
	for name, value in (
		("pregain", 2.0),
		("postgain", 2.0),
		("predark", 0.0),
		("postdark", 0.0),
	):
		tf.imwrite(path / f"{name}_median.tif", np.full((size, size), value, dtype=np.float32))


def test_current_b10_driver_runs_end_to_end_without_touching_snapshot(tmp_path):
	size = 8
	projections = np.full((6, size, size), 1.0, dtype=np.float32)
	source = tmp_path / "scan.h5"
	flats = tmp_path / "gains"
	output = tmp_path / "recon"
	_write_source(source, projections)
	_write_flats(flats, size)

	def reconstruct(data, _angles, _center, _processes):
		assert data.shape == (6, size, size)
		return np.repeat(data.mean(axis=0)[None, :, :], size, axis=0)

	written = run(B10ReconConfig(
		source=source,
		flats=flats,
		output=output,
		center=size / 2,
		window_size=size,
		apply_bac=False,
	), reconstructor=reconstruct)

	assert len(written) == size
	assert all(path.is_file() for path in written)
	assert tf.imread(written[0]).shape == (size, size)


def test_current_b10_driver_runs_a_global_row_selection(tmp_path):
	size = 8
	projections = np.broadcast_to(
		np.arange(size, dtype=np.float32)[None, :, None], (6, size, size),
	).copy()
	source = tmp_path / "scan.h5"
	flats = tmp_path / "gains"
	output = tmp_path / "recon"
	_write_source(source, projections)
	_write_flats(flats, size)

	def reconstruct(data, _angles, _center, _processes):
		return np.repeat(data.mean(axis=0)[:, None, :], data.shape[2], axis=1)

	written = run(B10ReconConfig(
		source=source,
		flats=flats,
		output=output,
		center=size / 2,
		window_size=2,
		y_start=2,
		y_stop=6,
		apply_bac=False,
	), reconstructor=reconstruct)

	assert [path.name for path in written] == ["00002.tif", "00003.tif", "00004.tif", "00005.tif"]
	assert [float(tf.imread(path)[0, 0]) for path in written] == [1.0, 1.5, 2.0, 2.5]


def test_streaming_bac_matches_current_driver_at_fixed_window_extent(tmp_path, monkeypatch):
	"""The FFT oracle is the legacy driver at the same window and zero pad."""

	size = 8
	theta = 6
	window_rows = 4
	y, t, x = np.indices((size, theta, size))
	projections = np.transpose(0.7 + 0.01 * y + 0.005 * t + 0.002 * x, (1, 0, 2))
	projections = projections.astype(np.float32)
	source = tmp_path / "scan.h5"
	flats = tmp_path / "gains"
	legacy_output = tmp_path / "legacy"
	streaming_output = tmp_path / "streaming"
	_write_source(source, projections)
	with h5py.File(source, "a") as handle:
		handle["exchange"].create_dataset(
			"data_white", data=np.full((2, size, size), 2, dtype=np.float32),
		)
		handle["exchange"].create_dataset(
			"data_dark", data=np.zeros((2, size, size), dtype=np.float32),
		)
	_write_flats(flats, size)

	class IndependentBAC:
		def __init__(self, shape, alpha, gamma, **_kwargs):
			qy = 2 * np.pi * np.fft.fftfreq(shape[0])
			qx = 2 * np.pi * np.fft.fftfreq(shape[1])
			w2 = np.add.outer(np.square(qy), np.square(qx))
			self.kernel = w2 / (w2 + alpha)
			self.gamma = gamma

		def __call__(self, image):
			correction = -np.real(
				np.fft.ifft2(np.fft.fft2(image - 1) * self.kernel)
			)
			return -np.log(image / (1 - self.gamma * correction))

	fake_hotopy = ModuleType("hotopy")
	fake_holo = ModuleType("hotopy.holo")
	fake_holo.BronnikovAidedCorrection = IndependentBAC
	fake_hotopy.holo = fake_holo
	monkeypatch.setitem(sys.modules, "hotopy", fake_hotopy)
	monkeypatch.setitem(sys.modules, "hotopy.holo", fake_holo)

	def reconstruct(data, _angles, _center, _processes):
		return np.repeat(data.mean(axis=0)[:, None, :], data.shape[2], axis=1)

	run(B10ReconConfig(
		source=source, flats=flats, output=legacy_output, center=size / 2,
		window_size=window_rows, alpha=1.6, gamma=1.2, apply_bac=True,
	), reconstructor=reconstruct)

	scan_path = tmp_path / "scan.yaml"
	scan_path.write_text(yaml.safe_dump({
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": theta, "degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": size, "columns": size},
		"acquisition": {"n_projections": theta, "rotation_end_deg": 180},
		"projections": {
			"format": "hdf5_dxchange", "path": str(source),
			"data_path": "exchange/data", "white_path": "exchange/data_white",
			"dark_path": "exchange/data_dark",
		},
	}, sort_keys=False), encoding="utf-8")
	pipeline_path = tmp_path / "pipeline.yaml"
	pipeline_path.write_text(yaml.safe_dump({"pipeline": [
		{"kind": "read", "algorithm": "hdf5_dxchange", "params": {
			"order": "projection", "calibration_policy": "required",
		}},
		{"kind": "normalize", "algorithm": "weighted", "params": {"pre_sub": True}},
		{"kind": "phase", "algorithm": "bac_correction", "params": {
			"alpha": 1.6, "gamma": 1.2,
		}},
		{"kind": "phase", "algorithm": "negative_log", "params": {"epsilon": None}},
		{"kind": "recon", "algorithm": "gridrec", "params": {
			"center": size / 2, "circle_mask": False,
		}},
		{"kind": "write", "algorithm": "tiff_slices", "params": {
			"naming_template": "{slice:05d}.tif",
		}},
	]}, sort_keys=False), encoding="utf-8")

	import tomopy

	def tomopy_reconstruct(data, *args, **kwargs):
		kwargs["init_recon"][...] = reconstruct(data, None, None, None)
		return kwargs["init_recon"]

	monkeypatch.setattr(tomopy, "recon", tomopy_reconstruct)
	result = run_reconstruction(
		pipeline_path, scan_path, streaming_output,
		window_config=WindowConfig(rows=window_rows),
	)[0]

	assert "phase.bac_correction" in result.artifacts["window_plan"]["warnings"][0]
	for row in range(size):
		np.testing.assert_allclose(
			tf.imread(streaming_output / f"{row:05d}.tif"),
			tf.imread(legacy_output / f"{row:05d}.tif"),
			rtol=2e-6, atol=2e-7,
		)
