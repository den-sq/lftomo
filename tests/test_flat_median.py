from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile as tf
import yaml

from lftomo.calibration import DetectorRegion
from lftomo.prep.flat_median import (
	FOUR_MAP_ORDER,
	FlatMedianSourceError,
	FourMapSources,
	HDF5FrameSeries,
	TIFFFrameSeries,
	generate_four_map_medians,
	hdf5_four_map_sources,
)


def _tiff_stack(tmp_path: Path, name: str, values: np.ndarray) -> TIFFFrameSeries:
	path = tmp_path / f"{name}.tif"
	tf.imwrite(path, values, photometric="minisblack")
	return TIFFFrameSeries(path)


def test_odd_even_integer_medians_chunk_boundaries_and_order(tmp_path):
	shape = (5, 5, 7)
	odd = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
	even = np.stack([np.zeros((5, 7)), np.ones((5, 7)) * 3]).astype(np.uint16)
	sources = FourMapSources.from_series(
		_tiff_stack(tmp_path, "pre_gain_source", odd),
		_tiff_stack(tmp_path, "post_gain_source", even),
		_tiff_stack(tmp_path, "pre_dark_source", odd + 2),
		_tiff_stack(tmp_path, "post_dark_source", even + 4),
	)
	result = generate_four_map_medians(sources, tmp_path / "maps", y_chunk=2)

	assert tuple(path.name for path in result.flat_fields) == tuple(
		f"{name}.tif" for name in FOUR_MAP_ORDER
	)
	assert result.scan_metadata()["flat_fields"] == [str(path) for path in result.flat_fields]
	for path in result.flat_fields:
		assert tf.imread(path).dtype == np.float32
	np.testing.assert_array_equal(tf.imread(result.pre_gain), np.median(odd, axis=0).astype(np.float32))
	np.testing.assert_array_equal(tf.imread(result.post_gain), np.full((5, 7), 1.5, np.float32))
	record = yaml.safe_load(result.metadata_path.read_text())
	assert record["flat_fields"] == [f"{name}.tif" for name in FOUR_MAP_ORDER]
	assert record["y_chunk"] == 2


def test_hdf5_sources_crop_orientation_and_atomic_output(tmp_path):
	path = tmp_path / "scan.h5"
	white = np.arange(4 * 4 * 6, dtype=np.uint16).reshape(4, 4, 6)
	dark = white[:3] // 10
	with h5py.File(path, "w") as handle:
		handle.create_dataset("exchange/data_white", data=white, chunks=(1, 2, 6))
		handle.create_dataset("exchange/data_dark", data=dark, chunks=(1, 2, 6))
	region = DetectorRegion(
		y_start=1, x_start=1, height=3, width=2, orientation="transpose",
		binning_y=2, binning_x=2,
	)
	result = generate_four_map_medians(
		hdf5_four_map_sources(path), tmp_path / "maps",
		y_chunk=2, orientation="transpose", region=region,
		build_parameters={"purpose": "test"},
	)
	expected = np.median(np.transpose(white, (0, 2, 1)), axis=0)[1:4, 1:3]
	np.testing.assert_array_equal(tf.imread(result.pre_gain), expected.astype(np.float32))
	assert sorted(path.name for path in result.output_dir.iterdir()) == [
		"flat_medians.yaml", "post_dark.tif", "post_gain.tif", "pre_dark.tif", "pre_gain.tif",
	]


def test_mismatched_and_corrupt_sources_never_publish_partial_set(tmp_path):
	good = _tiff_stack(tmp_path, "good", np.ones((3, 4, 5), dtype=np.uint16))
	bad_shape = _tiff_stack(tmp_path, "bad", np.ones((3, 5, 5), dtype=np.uint16))
	with pytest.raises(FlatMedianSourceError, match="do not match"):
		generate_four_map_medians(
			FourMapSources.from_series(good, good, good, bad_shape), tmp_path / "maps",
		)
	assert not (tmp_path / "maps").exists()

	corrupt = tmp_path / "corrupt.tif"
	corrupt.write_bytes(b"not a tiff")
	with pytest.raises(FlatMedianSourceError, match="cannot open TIFF"):
		TIFFFrameSeries(corrupt)


def test_hdf5_frame_ranges_support_explicit_pre_and_post_series(tmp_path):
	path = tmp_path / "scan.h5"
	values = np.arange(6 * 3 * 4, dtype=np.uint16).reshape(6, 3, 4)
	with h5py.File(path, "w") as handle:
		handle.create_dataset("white", data=values)
	pre = HDF5FrameSeries(path, "white", frame_start=0, frame_stop=3)
	post = HDF5FrameSeries(path, "white", frame_start=3, frame_stop=6)
	result = generate_four_map_medians(
		FourMapSources.from_series(pre, post, pre, post), tmp_path / "maps", y_chunk=1,
	)
	np.testing.assert_array_equal(tf.imread(result.pre_gain), np.median(values[:3], axis=0))
	np.testing.assert_array_equal(tf.imread(result.post_gain), np.median(values[3:], axis=0))


@pytest.mark.parametrize("orientation,crop", [
	("identity", None),
	("flip_x", DetectorRegion(
		y_start=0, x_start=1, height=3, width=3, orientation="flip_x",
	)),
])
def test_multipage_tiff_source_is_never_modified(tmp_path, orientation, crop):
	values = np.stack([
		np.full((3, 5), value, dtype=np.float32) for value in (5, 1, 9)
	])
	path = tmp_path / "source.tif"
	tf.imwrite(path, values, photometric="minisblack")
	before = tf.imread(path).copy()
	source = TIFFFrameSeries(path)

	generate_four_map_medians(
		FourMapSources.from_series(source, source, source, source),
		tmp_path / f"maps-{orientation}", y_chunk=2,
		orientation=orientation, region=crop,
	)

	np.testing.assert_array_equal(tf.imread(path), before)


def test_tiff_memmap_format_error_uses_decoder_fallback(tmp_path, monkeypatch):
	values = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
	path = tmp_path / "source.tif"
	tf.imwrite(path, values, photometric="minisblack")
	source = TIFFFrameSeries(path)

	def unsupported_memmap(*args, **kwargs):
		raise ValueError("image data are not memory-mappable")

	monkeypatch.setattr(tf, "memmap", unsupported_memmap)
	actual = source.read_oriented_window(1, 3, "identity")
	np.testing.assert_array_equal(actual, values[:, 1:3])


@pytest.mark.parametrize("multipage", [True, False], ids=["multipage", "frame-series"])
def test_tiff_memmap_index_error_is_not_swallowed(tmp_path, monkeypatch, multipage):
	values = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
	if multipage:
		path = tmp_path / "source.tif"
		tf.imwrite(path, values, photometric="minisblack")
		source = TIFFFrameSeries(path)
	else:
		paths = []
		for index, frame in enumerate(values):
			path = tmp_path / f"source-{index}.tif"
			tf.imwrite(path, frame)
			paths.append(path)
		source = TIFFFrameSeries(paths)

	def broken_memmap(*args, **kwargs):
		raise IndexError("unexpected memmap indexing failure")

	monkeypatch.setattr(tf, "memmap", broken_memmap)
	with pytest.raises(IndexError, match="unexpected memmap indexing failure"):
		source.read_oriented_window(0, 2, "identity")


def test_nonfinite_values_are_counted_and_ignored_but_empty_pixels_fail(tmp_path):
	path = tmp_path / "nonfinite.h5"
	values = np.array([
		[[1, np.nan], [1, 1]],
		[[3, 5], [np.inf, 3]],
		[[7, 9], [5, 5]],
	], dtype=np.float32)
	with h5py.File(path, "w") as handle:
		handle.create_dataset("white", data=values)
	source = HDF5FrameSeries(path, "white")
	result = generate_four_map_medians(
		FourMapSources.from_series(source, source, source, source),
		tmp_path / "finite-maps", y_chunk=1,
	)
	record = yaml.safe_load(result.metadata_path.read_text())
	assert record["nonfinite_counts"] == {name: 2 for name in FOUR_MAP_ORDER}
	np.testing.assert_array_equal(
		tf.imread(result.pre_gain), np.array([[3, 7], [3, 3]], dtype=np.float32),
	)

	values[:, 0, 0] = np.nan
	with h5py.File(path, "w") as handle:
		handle.create_dataset("white", data=values)
	bad = HDF5FrameSeries(path, "white")
	with pytest.raises(FlatMedianSourceError, match="no finite frame value"):
		generate_four_map_medians(
			FourMapSources.from_series(bad, bad, bad, bad),
			tmp_path / "invalid-maps", y_chunk=1,
		)
	assert not (tmp_path / "invalid-maps").exists()


def test_generated_maps_run_through_legacy_weighted_pipeline(tmp_path, monkeypatch):
	pytest.importorskip("tomopy")
	from lftomo.runner import run_reconstruction

	sources = FourMapSources.from_series(*[
		_tiff_stack(tmp_path, name, np.full((3, 2, 4), value, dtype=np.uint16))
		for name, value in zip(FOUR_MAP_ORDER, (10, 12, 1, 2), strict=True)
	])
	result = generate_four_map_medians(sources, tmp_path / "maps", y_chunk=1)

	h5_path = tmp_path / "scan.h5"
	with h5py.File(h5_path, "w") as handle:
		handle.create_dataset("exchange/data", data=np.full((3, 2, 4), 8, dtype=np.uint16))
	# Explicit legacy paths must override absent embedded whites/darks.
	scan_path = tmp_path / "scan.yaml"
	scan_path.write_text(yaml.safe_dump({
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": 3, "degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": 2, "columns": 4},
		"acquisition": {"n_projections": 3, "rotation_end_deg": 180},
		"projections": {"format": "hdf5_dxchange", "path": str(h5_path),
			"data_path": "exchange/data"},
		"metadata": result.scan_metadata(),
	}, sort_keys=False), encoding="utf-8")
	pipeline_path = tmp_path / "pipeline.yaml"
	pipeline_path.write_text(yaml.safe_dump({"pipeline": [
		{"kind": "read", "algorithm": "hdf5_dxchange", "params": {"order": "projection"}},
		{"kind": "normalize", "algorithm": "weighted"},
		{"kind": "phase", "algorithm": "paganin"},
		{"kind": "center", "algorithm": "manual", "params": {"value_px": 2}},
		{"kind": "recon", "algorithm": "gridrec", "params": {"circle_mask": False}},
		{"kind": "write", "algorithm": "tiff_slices"},
	]}, sort_keys=False), encoding="utf-8")
	import tomopy
	monkeypatch.setattr(tomopy, "recon", lambda data, *args, **kwargs:
		np.repeat(np.asarray(data).mean(axis=0)[:, None, :], data.shape[2], axis=1))
	runs = run_reconstruction(pipeline_path, scan_path, tmp_path / "recon")
	assert runs[0].artifacts["calibration_resolution"]["source"] == "legacy_flat_fields"
	assert runs[0].artifacts["output_files"]


@pytest.mark.integration
def test_ra00808_toothpick_matches_b11_reference(tmp_path, request):
	if not request.config.getoption("--run-integration"):
		pytest.skip("requires --run-integration and local B11/RA-00808 data")
	source = Path("/mnt/z/gpfs/phenome/Berkeley/RA-00808_ang/20260610_101717_toothpick.h5")
	references = Path("/mnt/wsl/PHYSICALDRIVE2p1/B11/b11_median_flats")
	white_reference = references / "20260610_101717_toothpick__white_median.tif"
	dark_reference = references / "darks/(688, 2560)/20260610_101717_toothpick__dark_median.tif"
	if not all(path.is_file() for path in (source, white_reference, dark_reference)):
		pytest.skip("local toothpick source/reference pair is incomplete")
	result = generate_four_map_medians(
		hdf5_four_map_sources(source), tmp_path / "maps", y_chunk=128,
	)
	for actual_path, expected_path in (
		(result.pre_gain, white_reference), (result.pre_dark, dark_reference),
	):
		actual = tf.memmap(actual_path, mode="r")
		expected = tf.memmap(expected_path, mode="r")
		for y_start in range(0, actual.shape[0], 32):
			np.testing.assert_allclose(
				actual[y_start:y_start + 32], expected[y_start:y_start + 32],
				rtol=0, atol=0,
			)
