"""Load canonical YAML configurations and execute reconstruction pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Callable

import click
import h5py
import numpy as np
import tifffile as tf
import yaml

from lftomo.calibration import (
	CalibrationBundle,
	CalibrationBundleRef,
	CalibrationTarget,
	CompatibilityReport,
	DetectorRegion,
	check_compatibility,
	load_manifest,
	validate_bundle,
)
from lftomo.config import (
	CALIBRATION_RESOLUTION_FIELDS,
	HDF5DxchangeProjections,
	HardwareTimestampAngleSource,
	Recon,
	Scan,
	StageConfig,
)
from lftomo.pipeline import (
	Backend,
	CandidateLimitError,
	NumpyBackend,
	Pipeline,
	PipelineRunContext,
	PipelineState,
	ExecutionSpanObserver,
	Stage,
	StageObserver,
	WindowConfig,
	expand_recon_sweeps,
	resolve_streaming_plan,
	validate_pipeline,
)
from lftomo.pipeline.stages import create_stage
from lftomo.pipeline.domains import DataDomain
from lftomo.pipeline.planning import PipelinePlan
from lftomo.storage import ManagedBuffer, buffer_view
from lftomo.util.converter import CONVERTER
from lftomo.util.mem import SharedNPManager
from lftomo.util.shapes import Y
from lftomo.util.window import TomoDataSet, TomoWindow


PipelineStateExecutor = Callable[
	[PipelineState, tuple[Stage, ...], tuple[StageConfig, ...]],
	PipelineState,
]


@dataclass(frozen=True)
class RunResult:
	"""Summary of one completed concrete reconstruction run."""

	index: int
	parameters: dict[str, Any]
	output_path: Path | None
	artifacts: dict[str, Any]
	candidate_key: str = ""
	resolved_recon: dict[str, Any] | None = None


@dataclass(frozen=True)
class CalibrationResolution:
	"""Resolved calibration source and provenance seeded into a pipeline run."""

	source: str
	bundle_ref: CalibrationBundleRef | None = None
	compatibility: CompatibilityReport | None = None
	legacy_paths: tuple[Path, ...] = ()

	def provenance(self) -> dict[str, Any]:
		return {
			"source": self.source,
			"bundle_ref": None if self.bundle_ref is None else self.bundle_ref.model_dump(mode="json"),
			"compatibility": (
				None if self.compatibility is None else self.compatibility.model_dump(mode="json")
			),
			"legacy_paths": [str(path) for path in self.legacy_paths],
		}


def _load_yaml(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as stream:
		value = yaml.safe_load(stream)
	if not isinstance(value, dict):
		raise ValueError(f"configuration must contain a YAML mapping: {path}")
	return value


def resolve_pipeline_path(value: str | Path) -> Path:
	"""Resolve an explicit path or a bundled canonical template name."""
	path = Path(value)
	if path.is_file():
		return path.resolve()
	name = path.name if path.suffix else f"{path.name}.yaml"
	if not name.endswith("-canonical.yaml") and not path.suffix:
		name = f"{path.name}-canonical.yaml"
	candidate = Path(__file__).with_name("pipelines") / name
	if candidate.is_file():
		return candidate
	available = sorted(item.stem for item in candidate.parent.glob("*.yaml"))
	raise FileNotFoundError(f"unknown pipeline {value!s}; bundled templates: {available}")


def load_recon(value: str | Path) -> tuple[Recon, Path]:
	"""Load a Recon from an explicit YAML path or bundled template name."""
	path = resolve_pipeline_path(value)
	data = _load_yaml(path)
	# Provenance envelopes used by frozen references may add these fields around
	# the canonical Recon document; retain them as metadata instead of rejecting.
	metadata = dict(data.get("metadata") or {})
	for key in ("schema_version", "resolved_from"):
		if key in data:
			metadata[key] = data.pop(key)
	data["metadata"] = metadata
	return Recon.model_validate(data), path


def load_scan(path: str | Path) -> tuple[Scan, Path]:
	"""Load Scan YAML and resolve its projection source relative to that YAML."""
	path = Path(path).resolve()
	scan = Scan.model_validate(_load_yaml(path))
	projection_path = Path(scan.projections.path)
	if not projection_path.is_absolute():
		scan.projections.path = (path.parent / projection_path).resolve()
	angle_source = scan.acquisition.angle_source
	if isinstance(angle_source, HardwareTimestampAngleSource) and not angle_source.path.is_absolute():
		angle_source.path = (path.parent / angle_source.path).resolve()
	return scan, path


@dataclass(frozen=True)
class ResolvedScan:
	"""One parsed Scan and its source path for relative calibration provenance."""

	scan: Scan
	path: Path


def resolve_scan(value: str | Path | ResolvedScan) -> ResolvedScan:
	"""Convert a file input once, retaining an already resolved Scan by identity."""

	if isinstance(value, ResolvedScan):
		return value
	return ResolvedScan(*load_scan(value))


def _read_tiff_map(path: Path, y_start: int, y_stop: int | None) -> np.ndarray:
	try:
		values = tf.memmap(path)
	except (OSError, ValueError):
		values = np.asarray(tf.imread(path))
	if values.ndim == 3:
		values = values[:, y_start:y_stop].mean(axis=0)
		return np.asarray(values, dtype=np.float32)
	return np.asarray(values[y_start:y_stop], dtype=np.float32)


_LEGACY_COMPONENTS = ("pre_gain", "post_gain", "pre_dark", "post_dark")


def _normalization_stage(recon: Recon):
	return next((stage for stage in recon.pipeline if stage.kind == "normalize"), None)


def _disable_embedded_calibration(recon: Recon) -> None:
	read = next((stage for stage in recon.pipeline if stage.kind == "read"), None)
	if read is not None and read.algorithm == "hdf5_dxchange":
		read.params["calibration_policy"] = "ignore"


def _resolve_path(value: str | Path, root: Path) -> Path:
	path = Path(value).expanduser()
	return path.resolve() if path.is_absolute() else (root / path).resolve()


def _detector_shape(scan: Scan) -> tuple[int, int]:
	detector = scan.detector
	if detector.rows is None or detector.columns is None:
		if not isinstance(scan.projections, HDF5DxchangeProjections):
			return detector.projection_shape().as_tuple()
		with h5py.File(scan.projections.path, "r") as handle:
			shape = handle[scan.projections.data_path].shape
		detector = detector.model_copy(update={
			"rows": int(shape[1]), "columns": int(shape[2]),
		})
	return detector.projection_shape().as_tuple()


def calibration_target(scan: Scan, source_window: TomoWindow) -> CalibrationTarget:
	"""Build compatibility facts from the canonical source selection."""

	columns = _detector_shape(scan)[1]
	selection = source_window.interval("Y")
	metadata = scan.metadata
	return CalibrationTarget(
		detector_id=metadata.get("detector_id") or scan.detector.name,
		beamline=metadata.get("beamline") or metadata.get("location"),
		optics_id=metadata.get("optics_id"),
		energy_keV=scan.energy_keV,
		exposure_ms=scan.acquisition.exposure_ms,
		temperature_c=metadata.get("temperature_c"),
		region=DetectorRegion(
			y_start=selection.start.Y, x_start=0,
			height=selection.length.Y, width=columns,
			binning_y=int(metadata.get("binning_y", 1)),
			binning_x=int(metadata.get("binning_x", 1)),
			orientation=metadata.get("orientation", "identity"),
		),
		static_artifact_epoch=metadata.get("static_artifact_epoch"),
		camera_segment=metadata.get("camera_segment"),
		beam_state=metadata.get("beam_state"),
		motor_axes=metadata.get("motor_axes", {}),
		acquisition_time=metadata.get("acquisition_time"),
	)


def _manifest_selection(recon: Recon, pipeline_root: Path) -> tuple[Path | None, dict[str, Any]]:
	selections = []
	options: dict[str, Any] = {}
	for stage in recon.pipeline:
		if stage.kind != "normalize":
			continue
		if stage.params.get("calibration_manifest") is not None:
			selections.append(_resolve_path(stage.params["calibration_manifest"], pipeline_root))
		for name in CALIBRATION_RESOLUTION_FIELDS - {"calibration_manifest"}:
			if name in stage.params:
				options[name] = stage.params[name]
	if selections and any(path != selections[0] for path in selections[1:]):
		raise ValueError("pipeline selects conflicting calibration manifests")
	return (selections[0] if selections else None), options


def _strip_resolution_params(recon: Recon) -> None:
	for stage in recon.pipeline:
		for name in CALIBRATION_RESOLUTION_FIELDS:
			stage.params.pop(name, None)


def _bundle_ref_path(ref: CalibrationBundleRef) -> Path:
	return Path(ref.manifest_path).expanduser().resolve()


def _selected_components(bundle: CalibrationBundle, normalize, options: dict[str, Any]) -> list[str]:
	configured = options.get("calibration_components")
	if configured is not None:
		return [configured] if isinstance(configured, str) else list(configured)
	if normalize is not None and normalize.algorithm in ("weighted", "theta_switch", "taper_blended"):
		if all(name in bundle.manifest.components for name in _LEGACY_COMPONENTS):
			return list(_LEGACY_COMPONENTS)
	return list(bundle.manifest.components)


def _bundle_flat_fields(
	bundle: CalibrationBundle, names: list[str], y_start: int, y_stop: int | None,
) -> np.ndarray | None:
	if not all(name in names for name in _LEGACY_COMPONENTS):
		return None
	return np.stack([
		bundle.read_component(name, y_start=y_start, y_stop=y_stop) for name in _LEGACY_COMPONENTS
	]).astype(np.float32)


def resolve_calibration(
	scan: Scan,
	recon: Recon,
	*,
	source_window: TomoWindow,
	pipeline_root: str | Path,
	scan_root: str | Path,
	seeded_bundle: CalibrationBundleRef | None = None,
	verify_hashes: bool = True,
) -> CalibrationResolution:
	"""Resolve calibration precedence and mutate Scan/Recon to the selected source."""

	pipeline_root = Path(pipeline_root).resolve()
	scan_root = Path(scan_root).resolve()
	selection = source_window.interval("Y")
	y_start, y_stop = selection.start.Y, selection.stop.Y
	normalize = _normalization_stage(recon)
	manifest_path, options = _manifest_selection(recon, pipeline_root)
	if manifest_path is not None and seeded_bundle is not None:
		if manifest_path != _bundle_ref_path(seeded_bundle):
			raise ValueError("pipeline manifest conflicts with caller-seeded calibration bundle")
	selected_path = manifest_path or (
		None if seeded_bundle is None else _bundle_ref_path(seeded_bundle)
	)
	if selected_path is not None:
		manifest, ref = load_manifest(selected_path)
		if seeded_bundle is not None and seeded_bundle.calibration_id != manifest.calibration_id:
			raise ValueError("caller-seeded calibration ID does not match its manifest")
		validate_bundle(ref.manifest_path, verify_hashes=verify_hashes, raise_on_error=True)
		bundle = CalibrationBundle(manifest, ref)
		names = _selected_components(bundle, normalize, options)
		policy = options.get("compatibility_policy", "error")
		if policy not in ("error", "override"):
			raise ValueError("runner compatibility_policy must be 'error' or 'override'")
		report = check_compatibility(
			bundle,
			calibration_target(scan, source_window),
			components=names,
			policy=policy,
			override_reason=options.get("compatibility_override_reason"),
		)
		fields = _bundle_flat_fields(bundle, names, y_start, y_stop)
		if normalize is not None and normalize.algorithm in ("weighted", "theta_switch", "taper_blended"):
			if fields is None:
				raise ValueError(
					"selected external bundle must provide pre_gain, post_gain, pre_dark, and post_dark "
					"for legacy normalization"
				)
			scan.metadata["flat_fields"] = fields
			_disable_embedded_calibration(recon)
		# Legacy normalizers consume four materialized Y-window maps. Dynamic
		# normalizers consume the typed bundle reference from PipelineState directly;
		# never fall through and let a lower-precedence legacy/HDF5 source replace it.
		_strip_resolution_params(recon)
		return CalibrationResolution(
			source="external_bundle", bundle_ref=ref, compatibility=report,
		)

	_strip_resolution_params(recon)
	if normalize is None:
		return CalibrationResolution(source="none")
	if "flat_fields" in scan.metadata:
		fields = scan.metadata["flat_fields"]
		paths: tuple[Path, ...] = ()
		if isinstance(fields, (list, tuple)) and all(isinstance(item, (str, Path)) for item in fields):
			paths = tuple(_resolve_path(item, scan_root) for item in fields)
			scan.metadata["flat_fields"] = np.stack([
				_read_tiff_map(path, y_start, y_stop) for path in paths
			])
		else:
			values = np.asarray(fields)
			rows, columns = _detector_shape(scan)
			if values.shape == (4, rows, columns):
				scan.metadata["flat_fields"] = values[:, y_start:y_stop]
		_disable_embedded_calibration(recon)
		return CalibrationResolution(source="legacy_flat_fields", legacy_paths=paths)

	path_params = [normalize.params.get(name) for name in _LEGACY_COMPONENTS]
	if all(value is not None for value in path_params):
		paths = tuple(_resolve_path(value, pipeline_root) for value in path_params)
		scan.metadata["flat_fields"] = np.stack([
			_read_tiff_map(path, y_start, y_stop) for path in paths
		])
		_disable_embedded_calibration(recon)
		return CalibrationResolution(source="legacy_stage_paths", legacy_paths=paths)

	if isinstance(scan.projections, HDF5DxchangeProjections):
		with h5py.File(scan.projections.path, "r") as handle:
			missing = [
				path for path in (scan.projections.white_path, scan.projections.dark_path)
				if path not in handle
			]
			if missing:
				raise ValueError("embedded HDF5 calibration dataset(s) missing: " + ", ".join(missing))
			white = handle[scan.projections.white_path]
			dark = handle[scan.projections.dark_path]
			scan.metadata["flat_fields"] = np.stack([
				white[0, y_start:y_stop], white[-1, y_start:y_stop],
				dark[0, y_start:y_stop], dark[-1, y_start:y_stop],
			]).astype(np.float32)
		# The four bounded maps above are the resolved calibration source. Do not
		# have the read Stage materialize every embedded flat/dark frame again.
		_disable_embedded_calibration(recon)
		return CalibrationResolution(source="embedded_hdf5")

	raise ValueError(
		"normalization requires an explicit calibration bundle, legacy four-map fields, "
		"or embedded HDF5 whites/darks"
	)


def _resolve_flat_fields(scan: Scan, recon: Recon, source_window: TomoWindow) -> None:
	"""Retain the private reference-generation compatibility path.

	Reference archives predate separate scan/pipeline roots and keep legacy map
	paths beside their projection source. New callers should use
	:func:`resolve_calibration` and pass both roots explicitly.
	"""

	root = Path(scan.projections.path).parent
	resolve_calibration(
		scan, recon, source_window=source_window,
		pipeline_root=root, scan_root=root,
	)


def _estimated_output_bytes(recon: Recon, dataset: TomoDataSet) -> int:
	"""Estimate float32 reconstructed-volume output without reading projection data."""

	if recon.output_path is None:
		return 0
	return dataset.full_window().recon_shape().nbytes(np.float32)


class _ReplayCenterArtifactStage(Stage):
	"""Main-pass placeholder for a center produced by the replay pre-pass."""

	def __init__(self, source: Stage):
		self.name = f"{source.name}.prepass_artifact"
		self.contract = source.contract
		self.occurrence_id = getattr(source, "occurrence_id", None)

	def run(self, state: PipelineState) -> PipelineState:
		if "center" not in state.artifacts:
			raise RuntimeError("center replay pre-pass did not produce a center artifact")
		return state


def _detached_artifact(value: Any) -> Any:
	if isinstance(value, ManagedBuffer):
		return buffer_view(value).copy()
	if isinstance(value, dict):
		return {key: _detached_artifact(item) for key, item in value.items()}
	if isinstance(value, list):
		return [_detached_artifact(item) for item in value]
	if isinstance(value, tuple):
		return tuple(_detached_artifact(item) for item in value)
	if isinstance(value, np.ndarray):
		return value.copy()
	return deepcopy(value)


def _retire_window_state(state: PipelineState) -> None:
	if state.storage is None:
		return
	handles = [state.working_array]
	handles.extend(state.artifacts.values())
	seen = set()
	for value in handles:
		if not isinstance(value, ManagedBuffer) or value.name in seen:
			continue
		seen.add(value.name)
		state.storage.retire(value, role=getattr(value, "role", None))


def _planned_pipeline(
	stages: list[Stage], scan: Scan, backend: Backend,
	available_artifacts: dict[str, Any], observer: StageObserver | None, recon: Recon,
	*, execution_observer: ExecutionSpanObserver | None = None,
	stage_index_offset: int = 0, mode: str = "complete",
	initial_layout=None, initial_domain=None,
	dataset=None, source_window=None,
) -> Pipeline:
	plan = validate_pipeline(
		stages, scan, backend, available_artifacts=available_artifacts,
		mode=mode, initial_layout=initial_layout, initial_domain=initial_domain,
		enable_aliasing=recon.buffer_aliasing,
		enable_engine_aliasing=recon.tomopy_input_aliasing,
		dataset=dataset, source_window=source_window,
	)
	pipeline = Pipeline(
		stages, backend=backend, validate=False, observer=observer,
		execution_observer=execution_observer,
		stage_indices=tuple(range(stage_index_offset, stage_index_offset + len(stages))),
	)
	pipeline.plan = plan
	pipeline._validated = True
	return pipeline


def _center_artifacts(state: PipelineState) -> dict[str, Any]:
	return {
		key: _detached_artifact(value)
		for key, value in state.artifacts.items()
		if key == "center" or key.startswith("center_") or key == "output_files"
	}


def _run_streaming_pipeline(
	*, stages: list[Stage], concrete: Recon, run_scan: Scan,
	initial_artifacts: dict[str, Any], backend: Backend, run_context,
	dataset: TomoDataSet, buffer_plan: PipelinePlan,
	stage_observer: StageObserver | None,
	execution_observer: ExecutionSpanObserver | None,
) -> dict[str, Any]:
	"""Execute one whole-volume or multi-window run through the same driver."""

	aggregated = dict(initial_artifacts)
	aggregated["streaming_dataset"] = CONVERTER.unstructure(dataset)
	window_records = []
	output_files = []
	reserve = getattr(run_context.storage, "reserve", None)
	if reserve is not None:
		for event in buffer_plan.allocation_schedule:
			reserve(event.role, event.bytes)
	for window in dataset:
		window_stages = list(stages)
		window_initial = dict(initial_artifacts)
		center_artifacts = {}
		auto_centers = [
			index for index, stage in enumerate(window_stages)
			if stage.name.startswith("center.") and stage.name != "center.manual"
		]
		if auto_centers:
			last_center = max(auto_centers)
			prepass_stages = window_stages[:last_center + 1]
			prepass_state = PipelineState(
				np.empty((0,), dtype=np.float32), run_scan, concrete,
				artifacts=dict(window_initial), backend=backend,
				storage=run_context.storage, executor=run_context.executor,
				dataset=dataset, source_window=window,
			)
			prepass = _planned_pipeline(
				prepass_stages, run_scan, backend, window_initial, stage_observer,
				concrete, execution_observer=execution_observer,
				dataset=dataset, source_window=window,
			)
			prepass_state = prepass.run(prepass_state)
			center_artifacts = _center_artifacts(prepass_state)
			_retire_window_state(prepass_state)
			window_initial.update(center_artifacts)
			window_stages = [
				_ReplayCenterArtifactStage(stage)
				if stage.name.startswith("center.") and stage.name != "center.manual"
				else stage
				for stage in window_stages
			]
		writer_start = next((
			index for index, stage in enumerate(window_stages)
			if stage.name.startswith("write.")
		), len(window_stages))
		compute_stages = window_stages[:writer_start]
		writer_stages = window_stages[writer_start:]
		state = PipelineState(
			np.empty((0,), dtype=np.float32), run_scan, concrete,
			artifacts=dict(window_initial), backend=backend,
			storage=run_context.storage, executor=run_context.executor,
			dataset=dataset, source_window=window,
		)
		reconstruction_start = next((
			stage_index for stage_index, stage in enumerate(compute_stages)
			if stage.name.startswith("recon.")
		), None)
		if reconstruction_start is None:
			compute = _planned_pipeline(
				compute_stages, run_scan, backend, window_initial, stage_observer,
				concrete, execution_observer=execution_observer,
				dataset=dataset, source_window=window,
			)
			state = compute.run(state)
		else:
			prefix = _planned_pipeline(
				compute_stages[:reconstruction_start], run_scan, backend,
				window_initial, stage_observer, concrete,
				execution_observer=execution_observer,
				dataset=dataset, source_window=window,
			)
			state = prefix.run(state)
			reconstruction = _planned_pipeline(
				compute_stages[reconstruction_start:], run_scan, backend,
				state.artifacts, stage_observer, concrete,
				execution_observer=execution_observer,
				stage_index_offset=reconstruction_start,
				mode="fragment", initial_layout=state.layout, initial_domain=state.domain,
				dataset=dataset, source_window=window,
			)
			state = reconstruction.run(state)
		if writer_stages and state.domain is not DataDomain.RECON:
			raise ValueError("windowed writers require a reconstruction-producing Stage")
		if writer_stages:
			state = Pipeline(
				writer_stages, backend=backend, validate=False, observer=stage_observer,
				execution_observer=execution_observer,
				stage_indices=tuple(range(writer_start, writer_start + len(writer_stages))),
			).run(state)
		window_artifacts = {
			key: _detached_artifact(value) for key, value in state.artifacts.items()
			if not isinstance(value, ManagedBuffer)
		}
		files = window_artifacts.get("output_files", [])
		output_files.extend(path for path in files if path not in output_files)
		window_records.append({
			"window": CONVERTER.unstructure(window),
			"center_artifacts": center_artifacts,
			"artifacts": window_artifacts,
		})
		aggregated.update(window_artifacts)
		_retire_window_state(state)
	aggregated["window_results"] = window_records
	if output_files:
		aggregated["output_files"] = output_files
	allocation_events = getattr(run_context.storage, "allocation_events", ())
	aggregated["allocation_events"] = [dict(event) for event in allocation_events]
	physical = {
		event["name"]: int(event["bytes"])
		for event in allocation_events if event["action"] == "allocate"
	}
	aggregated["measured_shared_memory_bytes"] = sum(physical.values())
	return aggregated


def run_reconstruction(pipeline_path: str | Path, scan_path: str | Path | ResolvedScan,
			output_path: str | Path | None = None, backend: Backend | None = None, *,
			calibration_bundle: CalibrationBundleRef | None = None,
			verify_calibration_hashes: bool = True,
			state_executor: PipelineStateExecutor | None = None,
			stage_observer: StageObserver | None = None,
			execution_observer: ExecutionSpanObserver | None = None,
			window_config: WindowConfig | None = None) -> list[RunResult]:
	"""Expand and execute YAML reconstruction runs with a file or resolved Scan."""
	recon, resolved_pipeline_path = load_recon(pipeline_path)
	resolved_scan = resolve_scan(scan_path)
	scan = resolved_scan.scan
	if output_path is not None:
		recon.output_path = Path(output_path).resolve()
	elif recon.output_path is not None and not recon.output_path.is_absolute():
		recon.output_path = (Path.cwd() / recon.output_path).resolve()
	backend = backend or NumpyBackend()
	runs = expand_recon_sweeps(recon)
	# Construct every resolved Stage before touching calibration or projection
	# payloads. This rejects invalid structures and algorithm-specific parameters
	# for the entire candidate set as one preflight operation.
	preflight_stages = []
	for run in runs:
		preflight = run.recon.model_copy(deep=True)
		_strip_resolution_params(preflight)
		preflight_stages.append([create_stage(config) for config in preflight.pipeline])
	planned = []
	for run, planning_stages in zip(runs, preflight_stages, strict=True):
		concrete = run.recon.model_copy(deep=True)
		if len(runs) > 1 and concrete.output_path is not None:
			concrete.output_path = concrete.output_path / run.output_key
		# Capture the rerunnable candidate before calibration resolution removes
		# source-selection directives from the executable copy.
		resolved_document = concrete.model_dump(mode="json")
		run_scan = scan.model_copy(deep=False)
		# Candidates may add metadata keys but must treat shared detector-sized
		# arrays as immutable. Copy only the mapping.
		run_scan.metadata = dict(scan.metadata)
		dataset = resolve_streaming_plan(planning_stages, run_scan, window_config)
		planned.append((
			run, concrete, resolved_document, run_scan, dataset,
		))
	estimates = [
		_estimated_output_bytes(concrete, dataset)
		for _run, concrete, _document, _scan, dataset in planned
	]
	total_output = sum(estimates)
	limit = recon.candidate_limits.max_estimated_output_bytes
	if total_output > limit:
		raise CandidateLimitError(
			f"candidate outputs are estimated at {total_output} bytes, exceeding "
			f"max_estimated_output_bytes={limit}; narrow the space/output window "
			"or raise the configured limit explicitly"
		)
	prepared = []
	for run, concrete, resolved_document, run_scan, dataset in planned:
		resolution = resolve_calibration(
			run_scan,
			concrete,
			source_window=dataset.full_window(),
			pipeline_root=resolved_pipeline_path.parent,
			scan_root=resolved_scan.path.parent,
			seeded_bundle=calibration_bundle,
			verify_hashes=verify_calibration_hashes,
		)
		initial_artifacts: dict[str, Any] = {
			"calibration_resolution": resolution.provenance(),
		}
		if resolution.bundle_ref is not None:
			initial_artifacts["calibration_bundle"] = resolution.bundle_ref
		if resolution.compatibility is not None:
			initial_artifacts["calibration_compatibility"] = resolution.compatibility
		stages = [create_stage(config) for config in concrete.pipeline]
		# Validate every selected complete structure before the first projection
		# Stage runs. Candidate N cannot fail contract validation after candidate
		# N-1 has already produced output.
		largest = max(dataset, key=lambda item: item.with_pad(["Y"]).shape.Y)
		buffer_plan = validate_pipeline(
			stages,
			run_scan,
			backend,
			available_artifacts=initial_artifacts,
			mode="complete",
			enable_aliasing=concrete.buffer_aliasing,
			enable_engine_aliasing=concrete.tomopy_input_aliasing,
			dataset=dataset,
			source_window=largest,
		)
		SharedNPManager.check_capacity(buffer_plan.shared_memory_peak_bytes)
		prepared.append((
			run, concrete, resolved_document, run_scan, initial_artifacts, stages,
			dataset, buffer_plan,
		))
	results = []
	for (
		run, concrete, resolved_document, run_scan, initial_artifacts, stages,
		dataset, buffer_plan,
	) in prepared:
		resolved_metadata = resolved_document.setdefault("metadata", {})
		resolved_metadata["streaming_dataset"] = CONVERTER.unstructure(dataset)
		with PipelineRunContext(
			enable_aliasing=concrete.buffer_aliasing,
			enable_engine_aliasing=concrete.tomopy_input_aliasing,
			max_workers=concrete.process_workers or min(4, os.cpu_count() or 1),
			worker_timeout_seconds=concrete.worker_timeout_seconds,
		) as run_context:
			if state_executor is not None:
				if len(dataset.frange) != 1:
					raise ValueError(
						"custom state executors are whole-volume-only; windowed prefix-cache "
						"semantics remain deferred",
					)
				state = PipelineState(
					working_array=np.empty((0,), dtype=np.float32),
					scan=run_scan,
					recon=concrete,
					artifacts={
						**initial_artifacts,
						"streaming_dataset": CONVERTER.unstructure(dataset),
					},
					backend=backend,
					storage=run_context.storage,
					executor=run_context.executor,
					dataset=dataset,
					source_window=next(iter(dataset)),
				)
				state = state_executor(state, tuple(stages), tuple(concrete.pipeline))
				# Experiment-managed arrays belong to the run context. Do not return
				# handles after its manager has shut down; only durable/ordinary
				# artifacts cross the trial boundary.
				artifacts = {
					key: _detached_artifact(value)
					for key, value in state.artifacts.items()
					if not isinstance(value, ManagedBuffer)
				}
				# Prefix-cache acceptance needs the same run-owned lifecycle ledger
				# returned by the ordinary driver. Capture it before the manager exits;
				# managed handles themselves remain deliberately excluded above.
				allocation_events = getattr(run_context.storage, "allocation_events", ())
				artifacts["allocation_events"] = [dict(event) for event in allocation_events]
				physical = {
					event["name"]: int(event["bytes"])
					for event in allocation_events if event["action"] == "allocate"
				}
				artifacts["measured_shared_memory_bytes"] = sum(physical.values())
			else:
				artifacts = _run_streaming_pipeline(
					stages=stages, concrete=concrete, run_scan=run_scan,
					initial_artifacts=initial_artifacts, backend=backend,
					run_context=run_context, dataset=dataset, buffer_plan=buffer_plan,
					stage_observer=stage_observer,
					execution_observer=execution_observer,
				)
			results.append(RunResult(
				run.index,
				run.parameters,
				concrete.output_path,
				artifacts,
				candidate_key=run.candidate_key,
				resolved_recon=resolved_document,
			))
		if concrete.output_path is not None:
			manifest_root = (
				concrete.output_path.parent
				if concrete.output_path.suffix.lower() in {".h5", ".hdf5"}
				else concrete.output_path
			)
			manifest_root.mkdir(parents=True, exist_ok=True)
			manifest_path = manifest_root / "recon_run.yaml"
			with manifest_path.open("w", encoding="utf-8") as stream:
				yaml.safe_dump({
					"resolved_recon": resolved_document,
					"streaming_dataset": CONVERTER.unstructure(dataset),
				}, stream, sort_keys=False)
			artifacts["recon_run"] = manifest_path
	return results


def _json_value(value: Any) -> Any:
	if hasattr(value, "model_dump"):
		return value.model_dump(mode="json")
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {str(key): _json_value(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_value(item) for item in value]
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	return value


def _json_result(result: RunResult) -> dict[str, Any]:
	return {
		"index": result.index,
		"candidate_key": result.candidate_key,
		"parameters": result.parameters,
		"output_path": None if result.output_path is None else str(result.output_path),
		"resolved_recon": result.resolved_recon,
		"artifacts": _json_value(result.artifacts),
	}


@click.command()
@click.argument("pipeline_path", type=click.Path(path_type=Path))
@click.argument("scan_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), help="Override Recon.output_path.")
@click.option("--gpu", is_flag=True, help="Declare that the selected runtime provides CUDA support.")
@click.option("--json-output", is_flag=True, help="Emit machine-readable run summaries.")
@click.option("--window-rows", type=click.IntRange(min=1), help="Machine-selected core rows per window.")
@click.option("--extra-pad-y", type=click.IntRange(min=0), default=0,
	help="Optional runner-level detector-row padding.")
def main(pipeline_path: Path, scan_path: Path, output_path: Path | None,
			gpu: bool, json_output: bool, window_rows: int | None,
			extra_pad_y: int) -> None:
	"""Execute PIPELINE_PATH (or a bundled template name) for SCAN_PATH."""
	backend = Backend(name="numpy", supports_gpu=True) if gpu else NumpyBackend()
	results = run_reconstruction(
		pipeline_path, scan_path, output_path, backend,
		window_config=WindowConfig(rows=window_rows, extra_pad=Y(extra_pad_y)),
	)
	if json_output:
		click.echo(json.dumps([_json_result(result) for result in results], indent=2))
	else:
		for result in results:
			click.echo(
				f"run {result.index} [{result.candidate_key[:16]}]: "
				f"{result.output_path or '(no output path)'}"
			)


if __name__ == "__main__":
	main()
