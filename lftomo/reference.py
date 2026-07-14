"""Tier-2 numerical reproduction and bounded-memory comparison helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import tifffile as tf
import yaml

from lftomo.pipeline import NumpyBackend, PipelineState, validate_pipeline
from lftomo.pipeline.stages import create_stage
from lftomo.runner import _make_executable, _resolve_flat_fields, load_recon, load_scan


class ReferenceNotReproducible(RuntimeError):
	"""The archived metadata records that the exact producer input is absent."""


class RuntimeEnvironmentError(RuntimeError):
	"""The numerical runtime violates the reconstruction environment contract."""


@dataclass(frozen=True)
class NumericalTolerance:
	"""Scale-independent acceptance thresholds for frozen historical outputs."""

	max_nrmse: float = 0.15
	min_correlation: float = 0.95
	max_normalized_bias: float = 0.05


@dataclass(frozen=True)
class SliceComparison:
	slice: int
	file: str
	sha256: str
	nrmse: float
	correlation: float
	normalized_bias: float
	passed: bool


@dataclass(frozen=True)
class ReferenceReport:
	dataset: str
	producer: str
	pipeline: str
	window: tuple[int, int]
	tolerance: NumericalTolerance
	comparisons: tuple[SliceComparison, ...]

	@property
	def passed(self) -> bool:
		return all(item.passed for item in self.comparisons)

	def as_dict(self) -> dict[str, Any]:
		result = asdict(self)
		result["passed"] = self.passed
		return result


_TEMPLATES = {
	"b10_recon": "b10-recon-canonical",
	"r254": "r254-canonical",
	"phase_contrast": "phase-contrast-canonical",
	"recon_minimal": "recon-minimal-canonical",
}


def _yaml(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as stream:
		value = yaml.safe_load(stream)
	if not isinstance(value, dict):
		raise ValueError(f"expected YAML mapping in {path}")
	return value


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
	"""Hash a potentially multi-gigabyte reference without loading it at once."""
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		while chunk := stream.read(chunk_bytes):
			digest.update(chunk)
	return digest.hexdigest()


def checksum_manifest(path: Path) -> dict[str, str]:
	"""Parse a sha256sum-compatible reference manifest."""
	entries = {}
	for line in path.read_text(encoding="utf-8").splitlines():
		if not line.strip():
			continue
		digest, relative = line.split(maxsplit=1)
		if len(digest) != 64:
			raise ValueError(f"invalid SHA-256 entry in {path}: {line!r}")
		entries[relative.lstrip("* ")] = digest
	return entries


def compare_slice(generated: np.ndarray, reference_path: Path,
			tolerance: NumericalTolerance, tile_rows: int = 256) -> SliceComparison:
	"""Compare one generated slice to a TIFF using bounded-memory row tiles."""
	reference = tf.memmap(reference_path)
	generated = np.asarray(generated)
	if generated.shape != reference.shape:
		raise ValueError(
			f"shape mismatch for {reference_path.name}: {generated.shape} != {reference.shape}"
		)
	count = generated.size
	sse = sum_ref = sum_gen = sum_ref2 = sum_gen2 = sum_cross = 0.0
	minimum = np.inf
	maximum = -np.inf
	for start in range(0, generated.shape[0], tile_rows):
		stop = min(start + tile_rows, generated.shape[0])
		actual = np.asarray(generated[start:stop], dtype=np.float64)
		expected = np.asarray(reference[start:stop], dtype=np.float64)
		difference = actual - expected
		sse += float(np.sum(difference * difference))
		sum_ref += float(np.sum(expected))
		sum_gen += float(np.sum(actual))
		sum_ref2 += float(np.sum(expected * expected))
		sum_gen2 += float(np.sum(actual * actual))
		sum_cross += float(np.sum(expected * actual))
		minimum = min(minimum, float(np.min(expected)))
		maximum = max(maximum, float(np.max(expected)))
	dynamic_range = maximum - minimum
	if not np.isfinite(dynamic_range) or dynamic_range <= 0:
		raise ValueError(f"reference has no finite dynamic range: {reference_path}")
	nrmse = np.sqrt(sse / count) / dynamic_range
	bias = abs(sum_gen - sum_ref) / count / dynamic_range
	ref_variance = sum_ref2 - sum_ref * sum_ref / count
	gen_variance = sum_gen2 - sum_gen * sum_gen / count
	denominator = np.sqrt(max(ref_variance, 0) * max(gen_variance, 0))
	correlation = (sum_cross - sum_ref * sum_gen / count) / denominator if denominator else 0.0
	passed = (
		nrmse <= tolerance.max_nrmse
		and correlation >= tolerance.min_correlation
		and bias <= tolerance.max_normalized_bias
	)
	return SliceComparison(
		slice=-1,
		file=reference_path.name,
		sha256="",
		nrmse=float(nrmse),
		correlation=float(correlation),
		normalized_bias=float(bias),
		passed=passed,
	)


def _reference_file(archive: Path, slice_index: int) -> Path:
	matches = [path for path in (archive / "recon").glob("*.tif*") if str(slice_index) in path.stem]
	if len(matches) != 1:
		raise ValueError(f"expected one reference TIFF for slice {slice_index}, found {matches}")
	return matches[0]


def _processing_halo(recon) -> int:
	values = [0]
	for stage in recon.pipeline:
		for name in ("window_size", "chunk_overlap"):
			value = stage.params.get(name)
			if isinstance(value, int):
				values.append(value)
	if any(stage.algorithm in {"bronnikov_aided", "bac"} for stage in recon.pipeline):
		values.append(12)
	return max(values)


def _cap_parallelism(recon, workers: int = 2) -> None:
	"""Avoid multiplying large row windows across producer-era worker counts."""
	for stage in recon.pipeline:
		if isinstance(stage.params.get("processes"), int):
			stage.params["processes"] = min(stage.params["processes"], workers)
		if isinstance(stage.params.get("ncore"), int):
			stage.params["ncore"] = min(stage.params["ncore"], workers)


def run_reference_archive(root: str | Path, producer: str,
			tolerance: NumericalTolerance | None = None,
			slices: list[int] | tuple[int, ...] | None = None) -> ReferenceReport:
	"""Regenerate selected rows with their canonical producer and compare them."""
	root = Path(root)
	archive = root / "reference" / producer
	metadata = _yaml(archive / "metadata.yaml")
	input_metadata = metadata.get("input") or {}
	if input_metadata.get("recovered_input_note"):
		raise ReferenceNotReproducible(str(input_metadata["recovered_input_note"]))
	if np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion("2.0.0"):
		raise RuntimeEnvironmentError(
			f"TomoPy numerical reproduction requires NumPy <2; found {np.__version__}. "
			"Recreate the pinned environment.yml runtime."
		)
	selected = list(slices or metadata["reference_selection"]["slices"])
	if not selected:
		raise ValueError("reference selection must not be empty")
	if selected != list(range(min(selected), max(selected) + 1)):
		raise ValueError("Tier-2 reference selection must be a contiguous row window")
	try:
		template = _TEMPLATES[producer]
	except KeyError as error:
		raise ValueError(f"no canonical template registered for producer {producer!r}") from error
	recon, pipeline_path = load_recon(template)
	recon = _make_executable(recon)
	_cap_parallelism(recon)
	scan, _ = load_scan(root / "scan.yaml")
	halo = _processing_halo(recon)
	window_start = max(0, min(selected) - halo)
	window_stop = min(scan.detector.rows or max(selected) + halo + 1, max(selected) + halo + 1)
	for config in recon.pipeline:
		if config.kind == "read":
			config.params.update({"y_start": window_start, "y_stop": window_stop, "order": "sinogram"})
		elif config.kind == "recon":
			config.params["sinogram_order"] = True
	_resolve_flat_fields(scan, recon)
	configs = [config for config in recon.pipeline if config.kind != "write"]
	stages = [create_stage(config) for config in configs]
	validate_pipeline(stages, scan, NumpyBackend())
	state = PipelineState(
		working_array=np.empty((0,), dtype=np.float32),
		scan=scan,
		recon=recon,
		backend=NumpyBackend(),
	)
	cropped = False
	for config, stage in zip(configs, stages, strict=True):
		if not cropped and config.kind in {"center", "recon"}:
			start = min(selected) - window_start
			stop = max(selected) - window_start + 1
			state = state.with_array(np.asarray(state.working_array)[start:stop].copy())
			cropped = True
		state = stage.run(state)
	if not cropped:
		raise ValueError("canonical pipeline has no reconstruction boundary")
	generated = np.asarray(state.working_array)
	if generated.shape[0] != len(selected):
		raise ValueError(f"reconstruction returned {generated.shape[0]} rows for {len(selected)} selected rows")

	tolerance = tolerance or NumericalTolerance()
	manifest = checksum_manifest(archive / "recon.sha256")
	comparisons = []
	for local_index, slice_index in enumerate(selected):
		reference_path = _reference_file(archive, slice_index)
		relative = str(reference_path.relative_to(archive))
		expected_digest = manifest.get(relative)
		if expected_digest is None:
			raise ValueError(f"reference checksum missing for {relative}")
		digest = sha256_file(reference_path)
		if digest != expected_digest:
			raise ValueError(f"reference checksum mismatch for {relative}")
		comparison = compare_slice(generated[local_index], reference_path, tolerance)
		comparisons.append(SliceComparison(
			slice=slice_index,
			file=comparison.file,
			sha256=digest,
			nrmse=comparison.nrmse,
			correlation=comparison.correlation,
			normalized_bias=comparison.normalized_bias,
			passed=comparison.passed,
		))
	return ReferenceReport(
		dataset=str(metadata["dataset"]),
		producer=producer,
		pipeline=str(pipeline_path),
		window=(window_start, window_stop),
		tolerance=tolerance,
		comparisons=tuple(comparisons),
	)
