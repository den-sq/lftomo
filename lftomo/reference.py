"""Tier-2 numerical reproduction and bounded-memory comparison helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any

import numpy as np
import tifffile as tf
import yaml

from lftomo.pipeline import (
	BufferSpec,
	DataDomain,
	NumpyBackend,
	PipelineRunContext,
	PipelineState,
	resolve_streaming_plan,
	validate_pipeline,
)
from lftomo.pipeline.runtime import tomopy_numpy_incompatible
from lftomo.pipeline.stages import create_stage
from lftomo.runner import _resolve_flat_fields, load_recon, load_scan
from lftomo.util.converter import CONVERTER
from lftomo.util.shapes import ChunkGrid, Interval, ProjOrder, Y
from lftomo.util.window import TomoDataSet, TomoWindow


class ReferenceNotReproducible(RuntimeError):
	"""The archived metadata records that the exact producer or input is absent."""


class RuntimeEnvironmentError(RuntimeError):
	"""The numerical runtime violates the reconstruction environment contract."""


def _validate_tomopy_numpy_runtime() -> None:
	"""Reject only the TomoPy/NumPy pairing known to be incompatible."""

	try:
		tomopy_version = package_version("tomopy")
	except PackageNotFoundError as error:
		raise RuntimeEnvironmentError(
			"TomoPy is required for Tier-2 numerical reproduction. "
			"Recreate the environment.yml runtime."
		) from error
	if tomopy_numpy_incompatible(np.__version__, tomopy_version):
		raise RuntimeEnvironmentError(
			f"NumPy {np.__version__} requires TomoPy >=1.15.3; "
			f"found TomoPy {tomopy_version}. Recreate the environment.yml runtime."
		)


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
	window: Interval
	core_window: Interval
	layout: str
	theta_batch_size: int | None
	tolerance: NumericalTolerance
	comparisons: tuple[SliceComparison, ...]

	@property
	def passed(self) -> bool:
		return all(item.passed for item in self.comparisons)

	def as_dict(self) -> dict[str, Any]:
		result = asdict(self)
		result["window"] = CONVERTER.unstructure(self.window)
		result["core_window"] = CONVERTER.unstructure(self.core_window)
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
	reference = tf.memmap(reference_path, mode="r")
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
	passed = bool(
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


def aligned_producer_window(selected: list[int] | tuple[int, ...], row_count: int,
			chunk_size: int, overlap: int, start: int = 0) -> tuple[Interval, Interval]:
	"""Return the producer chunk and retained core containing selected detector rows."""
	if not selected:
		raise ValueError("selected detector rows cannot be empty")
	first, last = min(selected), max(selected)
	grid = ChunkGrid(
		size=chunk_size,
		stride=chunk_size - 2 * overlap,
		length=row_count - start,
		tail="clamp",
		halo=overlap,
	)
	index = grid.index_of(first - start)
	window, core = grid[index]
	window_start, window_stop = start + window.start, start + window.stop
	core_start, core_stop = start + core.start, start + core.stop
	if first < core_start or last >= core_stop:
		raise ValueError(
			f"selected rows [{first}, {last + 1}) do not fit one producer core "
			f"[{core_start}, {core_stop})"
		)
	return Y.interval(window_start, window_stop), Y.interval(core_start, core_stop)


def _run_phase_contrast_projection_order(
		scan, recon, dataset: TomoDataSet,
		selected: list[int], theta_batch_size: int,
) -> np.ndarray:
	"""Execute AAA523 in bounded Theta batches backed by a projection-order memmap."""
	if theta_batch_size < 1:
		raise ValueError("Theta batch size must be positive")
	configs = [config for config in recon.pipeline if config.kind != "write"]
	stages = [create_stage(config) for config in configs]
	source_window = dataset.full_window()
	validate_pipeline(
		stages, scan, NumpyBackend(),
		dataset=dataset, source_window=source_window,
	)
	stripe_index = next(
		(index for index, config in enumerate(configs) if config.kind == "stripe")
	)
	read_stage = stages[0]
	if configs[0].kind != "read":
		raise ValueError("phase-contrast pipeline must begin with a read Stage")
	pre_stages = stages[1:stripe_index]
	stripe_stage = stages[stripe_index]
	post_stages = stages[stripe_index + 1:]
	projection_count = scan.n_projections
	y = source_window.axis_slice("Y")

	scratch_root = Path(os.environ.get("LFTOMO_TIER2_SCRATCH", gettempdir()))
	scratch_root.mkdir(parents=True, exist_ok=True)
	with TemporaryDirectory(prefix=".lftomo-reference-", dir=scratch_root) as temporary:
		volume = None
		for theta_start in range(0, projection_count, theta_batch_size):
			theta_stop = min(theta_start + theta_batch_size, projection_count)
			window = TomoWindow.from_bounds(
				dataset.physical_shape,
				y=(y.start, y.stop), theta=(theta_start, theta_stop),
				dtype=dataset.dtype, work_dtype=np.float32,
			)
			with PipelineRunContext() as run_context:
				state = PipelineState(
					working_array=np.empty((0,), dtype=np.float32), scan=scan, recon=recon,
					backend=NumpyBackend(), storage=run_context.storage,
					dataset=dataset, source_window=window,
				)
				state = read_stage.run(state)
				for stage in pre_stages:
					state = stage.run(state)
				batch = np.asarray(state.working_array).astype(np.float32, copy=True)
			if volume is None:
				shape = (projection_count, batch.shape[1], batch.shape[2])
				volume = np.memmap(
					Path(temporary) / "projection-order.float32", mode="w+",
					dtype=np.float32, shape=shape,
				)
			volume[theta_start:theta_stop] = batch
			volume.flush()

		if volume is None:
			raise ValueError("phase-contrast input contains no projections")
		state = PipelineState(
			working_array=volume, scan=scan, recon=recon, backend=NumpyBackend(),
			buffer_spec=BufferSpec(ProjOrder(*volume.shape), volume.dtype),
			domain=DataDomain.ATTENUATION,
			dataset=dataset,
			source_window=source_window,
		)
		state = stripe_stage.run(state)
		volume.flush()
		local_start = min(selected) - y.start
		local_stop = max(selected) - y.start + 1
		cropped = volume[:, local_start:local_stop, :]
		state = state.with_array(
			cropped, buffer_spec=BufferSpec(ProjOrder(*cropped.shape), cropped.dtype),
		)
		for stage in post_stages:
			state = stage.run(state)
		return np.asarray(state.working_array).astype(np.float32, copy=True)


def _cap_parallelism(recon, workers: int = 2) -> None:
	"""Avoid multiplying large row windows across producer-era worker counts."""
	for stage in recon.pipeline:
		if isinstance(stage.params.get("processes"), int):
			stage.params["processes"] = min(stage.params["processes"], workers)
		if isinstance(stage.params.get("ncore"), int):
			stage.params["ncore"] = min(stage.params["ncore"], workers)


def run_reference_archive(root: str | Path, producer: str,
			tolerance: NumericalTolerance | None = None,
			slices: list[int] | tuple[int, ...] | None = None,
			theta_batch_size: int | None = None) -> ReferenceReport:
	"""Regenerate selected rows with their canonical producer and compare them."""
	root = Path(root)
	archive = root / "reference" / producer
	metadata = _yaml(archive / "metadata.yaml")
	input_metadata = metadata.get("input") or {}
	if input_metadata.get("recovered_input_note"):
		raise ReferenceNotReproducible(str(input_metadata["recovered_input_note"]))
	source_metadata = metadata.get("source") or {}
	provenance_note = str(source_metadata.get("provenance_note", ""))
	exact_producer_unavailable = source_metadata.get("exact_producer_available") is False
	legacy_unversioned_copy = (
		"surviving_copy_modified_at" in source_metadata
		and "modified after" in provenance_note.lower()
	)
	if exact_producer_unavailable or legacy_unversioned_copy:
		reason = source_metadata.get("reproduction_note") or provenance_note
		raise ReferenceNotReproducible(str(reason))
	_validate_tomopy_numpy_runtime()
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
	scan, _ = load_scan(root / "scan.yaml")
	_cap_parallelism(recon)
	if producer == "phase_contrast":
		read = next(config for config in recon.pipeline if config.kind == "read")
		chunk_size = int(read.params["sinogram_chunk_size"])
		overlap = int(read.params.get("chunk_overlap", 0))
		window, core_window = aligned_producer_window(
			selected, scan.detector.projection_shape().Y, chunk_size, overlap,
		)
		for config in recon.pipeline:
			if config.kind == "geometry" and config.algorithm == "affine_skew":
				config.params["center_y_px"] = -window.start.Y
	else:
		halo = _processing_halo(recon)
		window = Y.interval(
			max(0, min(selected) - halo),
			min(scan.detector.projection_shape().Y, max(selected) + halo + 1),
		)
		core_window = Y.interval(min(selected), max(selected) + 1)
		for config in recon.pipeline:
			if config.kind == "read":
				config.params["order"] = "sinogram"

	planning_stages = [
		create_stage(config) for config in recon.pipeline if config.kind != "write"
	]
	dataset = resolve_streaming_plan(planning_stages, scan)
	window_length = window.length.Y
	dataset = dataset.rescope(
		pos=Y(window.start.Y), shape=Y(window_length),
	)
	source_window = dataset.full_window()
	_resolve_flat_fields(scan, recon, source_window)

	if producer == "phase_contrast":
		batch_size = theta_batch_size or int(os.environ.get("LFTOMO_REFERENCE_THETA_BATCH", "4"))
		generated = _run_phase_contrast_projection_order(
			scan, recon, dataset, selected, batch_size,
		)
		layout = "projection"
	else:
		configs = [config for config in recon.pipeline if config.kind != "write"]
		stages = [create_stage(config) for config in configs]
		validate_pipeline(
			stages, scan, NumpyBackend(),
			dataset=dataset, source_window=source_window,
		)
		with PipelineRunContext() as run_context:
			state = PipelineState(
				working_array=np.empty((0,), dtype=np.float32),
				scan=scan,
				recon=recon,
				backend=NumpyBackend(),
				storage=run_context.storage,
				dataset=dataset,
				source_window=source_window,
			)
			cropped = False
			for config, stage in zip(configs, stages, strict=True):
				if not cropped and config.kind in {"center", "recon"}:
					start = min(selected) - window.start.Y
					stop = max(selected) - window.start.Y + 1
					values = np.asarray(state.working_array)[start:stop].copy()
					shape = type(state.buffer_spec.shape)(**{
						**state.buffer_spec.shape.as_dict(), "Y": stop - start,
					})
					state = state.with_array(
						values, buffer_spec=BufferSpec(shape, values.dtype),
					)
					cropped = True
				state = stage.run(state)
			if not cropped:
				raise ValueError("canonical pipeline has no reconstruction boundary")
			generated = np.asarray(state.working_array).copy()
		layout = "sinogram"
		batch_size = None
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
		window=window,
		core_window=core_window,
		layout=layout,
		theta_batch_size=batch_size,
		tolerance=tolerance,
		comparisons=tuple(comparisons),
	)
