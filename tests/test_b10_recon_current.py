from pathlib import Path

import h5py
import numpy as np
import tifffile as tf

from examples.b10_recon_current import B10ReconConfig, run


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
