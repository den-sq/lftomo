import pytest

np = pytest.importorskip("numpy")
tf = pytest.importorskip("tifffile")

from tests.fixtures.tier1 import (  # noqa: E402
	flat_dark_frames,
	projection_stack,
	shepp_logan,
	striped_sinogram,
	write_flat_dark_stack,
	write_projection_stack,
)


def test_shepp_logan_is_parameterized():
	phantom = shepp_logan(size=32)

	assert phantom.shape == (32, 32)
	assert phantom.dtype == np.float32
	assert 0.0 <= phantom.min() <= phantom.max() <= 1.0


def test_projection_stack_and_misalignment_controls():
	aligned = projection_stack(size=16, n_projections=5, misalignment=0)
	misaligned = projection_stack(size=16, n_projections=5, misalignment=2)

	assert aligned.shape == (5, 16, 16)
	assert not np.allclose(aligned, misaligned)


def test_striped_sinogram_changes_columns():
	base = projection_stack(size=32, n_projections=4)
	striped = striped_sinogram(size=32, n_projections=4, stripe_strength=0.5)

	assert striped.shape == base.shape
	assert not np.allclose(base[:, :, 0], striped[:, :, 0])


def test_flat_dark_frames_are_parameterized():
	flats, darks = flat_dark_frames(size=12, n_flats=3, gain=1.5, dark_level=0.1)

	assert flats.shape == (3, 12, 12)
	assert darks.shape == (3, 12, 12)
	assert np.allclose(darks, 0.1)


def test_write_projection_stack_to_tmp_path(tmp_path):
	fixture = write_projection_stack(tmp_path / "scan", size=10, n_projections=3, striped=True)

	assert len(fixture.projection_paths) == 3
	assert all(path.exists() for path in fixture.projection_paths)
	assert tf.imread(fixture.projection_paths[0]).shape == (10, 10)


def test_write_flat_dark_stack_to_tmp_path(tmp_path):
	fixture = write_flat_dark_stack(tmp_path / "scan", size=10, n_flats=2)

	assert len(fixture.flat_paths) == 2
	assert len(fixture.dark_paths) == 2
	assert tf.imread(fixture.flat_paths[0]).shape == (10, 10)
