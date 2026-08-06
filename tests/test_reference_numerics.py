import hashlib

import numpy as np
import pytest
import tifffile as tf

import lftomo.reference as reference_module
from lftomo.reference import (
	NumericalTolerance,
	ReferenceNotReproducible,
	RuntimeEnvironmentError,
	_validate_tomopy_numpy_runtime,
	aligned_producer_window,
	checksum_manifest,
	compare_slice,
	run_reference_archive,
	sha256_file,
)


@pytest.mark.parametrize(("numpy_version", "tomopy_version", "compatible"), (
	("1.26.4", "1.14.4", True),
	("2.0.0", "1.15.2", False),
	("2.0.0", "1.15.3", True),
))
def test_reference_runtime_rejects_only_incompatible_tomopy_numpy_pair(
	monkeypatch, numpy_version, tomopy_version, compatible,
):
	monkeypatch.setattr(reference_module.np, "__version__", numpy_version)
	monkeypatch.setattr(reference_module, "package_version", lambda _name: tomopy_version)

	if compatible:
		_validate_tomopy_numpy_runtime()
	else:
		with pytest.raises(RuntimeEnvironmentError, match="requires TomoPy >=1.15.3"):
			_validate_tomopy_numpy_runtime()


def test_phase_contrast_rows_use_aligned_inclusive_producer_chunk():
	window, core = aligned_producer_window(
		list(range(2000, 2010)), row_count=4440, chunk_size=140, overlap=30,
	)

	assert window == (1920, 2060)
	assert core == (1950, 2030)


def test_reference_with_post_run_unversioned_producer_is_not_reproducible(tmp_path):
	archive = tmp_path / "reference" / "r254"
	archive.mkdir(parents=True)
	(archive / "metadata.yaml").write_text(
		"""source:
  surviving_copy_modified_at: '2025-01-17T17:27:13-05:00'
  provenance_note: The surviving script copy was modified after this output was produced.
""",
		encoding="utf-8",
	)

	with pytest.raises(ReferenceNotReproducible, match="modified after"):
		run_reference_archive(tmp_path, "r254")


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
	assert type(result.passed) is bool
	assert result.nrmse == pytest.approx(0.01 / 6)
	assert result.correlation == pytest.approx(1)


def test_numerical_comparison_opens_frozen_reference_read_only(tmp_path, monkeypatch):
	reference = np.arange(12, dtype=np.float32).reshape(3, 4)
	path = tmp_path / "reference.tif"
	tf.imwrite(path, reference)
	modes = []
	real_memmap = tf.memmap

	def recording_memmap(filename, *args, **kwargs):
		modes.append(kwargs.get("mode"))
		return real_memmap(filename, *args, **kwargs)

	monkeypatch.setattr(tf, "memmap", recording_memmap)
	compare_slice(reference, path, NumericalTolerance())

	assert modes == ["r"]


def test_numerical_comparison_rejects_unrelated_output(tmp_path):
	reference = np.arange(64, dtype=np.float32).reshape(8, 8)
	path = tmp_path / "reference.tif"
	tf.imwrite(path, reference)

	result = compare_slice(reference[::-1], path, NumericalTolerance(), tile_rows=3)

	assert not result.passed
	assert result.correlation < NumericalTolerance().min_correlation
