import hashlib

import numpy as np
import pytest
import tifffile as tf

from lftomo.reference import NumericalTolerance, checksum_manifest, compare_slice, sha256_file


def test_checksum_manifest_and_streaming_hash(tmp_path):
	data = tmp_path / "slice.tif"
	data.write_bytes(b"reference bytes")
	digest = hashlib.sha256(b"reference bytes").hexdigest()
	manifest = tmp_path / "recon.sha256"
	manifest.write_text(f"{digest}  recon/slice.tif\n", encoding="utf-8")

	assert checksum_manifest(manifest) == {"recon/slice.tif": digest}
	assert sha256_file(data, chunk_bytes=3) == digest


def test_bounded_memory_numerical_comparison_passes_small_perturbations(tmp_path):
	reference = np.linspace(-2, 4, 120, dtype=np.float32).reshape(10, 12)
	path = tmp_path / "reference.tif"
	tf.imwrite(path, reference)
	generated = reference + np.float32(0.01)

	result = compare_slice(
		generated,
		path,
		NumericalTolerance(max_nrmse=0.01, min_correlation=0.999, max_normalized_bias=0.01),
		tile_rows=2,
	)

	assert result.passed
	assert result.nrmse == pytest.approx(0.01 / 6)
	assert result.correlation == pytest.approx(1)


def test_numerical_comparison_rejects_unrelated_output(tmp_path):
	reference = np.arange(64, dtype=np.float32).reshape(8, 8)
	path = tmp_path / "reference.tif"
	tf.imwrite(path, reference)

	result = compare_slice(reference[::-1], path, NumericalTolerance(), tile_rows=3)

	assert not result.passed
	assert result.correlation < NumericalTolerance().min_correlation
