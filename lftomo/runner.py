"""Load canonical YAML configurations and execute reconstruction pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

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
from lftomo.config import HDF5DxchangeProjections, Recon, Scan
from lftomo.pipeline import Backend, NumpyBackend, Pipeline, PipelineState, expand_recon_sweeps
from lftomo.pipeline.stages import create_stage


@dataclass(frozen=True)
class RunResult:
	"""Summary of one completed concrete reconstruction run."""

	index: int
	parameters: dict[str, Any]
	output_path: Path | None
	artifacts: dict[str, Any]


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
	return scan, path


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
_RESOLUTION_PARAMS = {
	"calibration_manifest",
	"calibration_components",
	"compatibility_policy",
	"compatibility_override_reason",
}


def _normalization_stage(recon: Recon):
	return next((stage for stage in recon.pipeline if stage.kind == "normalize"), None)


def _read_window(recon: Recon) -> tuple[int, int | None]:
	read = next((stage for stage in recon.pipeline if stage.kind == "read"), None)
	return (
		int(read.params.get("y_start", 0)) if read else 0,
		read.params.get("y_stop") if read else None,
	)


def _disable_embedded_calibration(recon: Recon) -> None:
	read = next((stage for stage in recon.pipeline if stage.kind == "read"), None)
	if read is not None and read.algorithm == "hdf5_dxchange":
		read.params["calibration_policy"] = "ignore"


def _resolve_path(value: str | Path, root: Path) -> Path:
	path = Path(value).expanduser()
	return path.resolve() if path.is_absolute() else (root / path).resolve()


def _detector_shape(scan: Scan) -> tuple[int, int]:
	if scan.detector.rows is not None and scan.detector.columns is not None:
		return scan.detector.rows, scan.detector.columns
	if isinstance(scan.projections, HDF5DxchangeProjections):
		with h5py.File(scan.projections.path, "r") as handle:
			shape = handle[scan.projections.data_path].shape
			return int(shape[1]), int(shape[2])
	raise ValueError("external calibration requires detector rows and columns in scan metadata")


def calibration_target(scan: Scan, recon: Recon) -> CalibrationTarget:
	"""Build component-compatibility facts from one resolved Scan and Recon."""

	rows, columns = _detector_shape(scan)
	y_start, y_stop = _read_window(recon)
	y_stop = rows if y_stop is None else int(y_stop)
	metadata = scan.metadata
	return CalibrationTarget(
		detector_id=metadata.get("detector_id") or scan.detector.name,
		beamline=metadata.get("beamline") or metadata.get("location"),
		optics_id=metadata.get("optics_id"),
		energy_keV=scan.energy_keV,
		exposure_ms=scan.acquisition.exposure_ms,
		temperature_c=metadata.get("temperature_c"),
		region=DetectorRegion(
			y_start=y_start, x_start=0, height=y_stop - y_start, width=columns,
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
		for name in _RESOLUTION_PARAMS - {"calibration_manifest"}:
			if name in stage.params:
				options[name] = stage.params[name]
	if selections and any(path != selections[0] for path in selections[1:]):
		raise ValueError("pipeline selects conflicting calibration manifests")
	return (selections[0] if selections else None), options


def _strip_resolution_params(recon: Recon) -> None:
	for stage in recon.pipeline:
		for name in _RESOLUTION_PARAMS:
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
	pipeline_root: str | Path,
	scan_root: str | Path,
	seeded_bundle: CalibrationBundleRef | None = None,
	verify_hashes: bool = True,
) -> CalibrationResolution:
	"""Resolve calibration precedence and mutate Scan/Recon to the selected source."""

	pipeline_root = Path(pipeline_root).resolve()
	scan_root = Path(scan_root).resolve()
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
			calibration_target(scan, recon),
			components=names,
			policy=policy,
			override_reason=options.get("compatibility_override_reason"),
		)
		y_start, y_stop = _read_window(recon)
		fields = _bundle_flat_fields(bundle, names, y_start, y_stop)
		if normalize is not None and normalize.algorithm in ("weighted", "theta_switch", "taper_blended"):
			if fields is None:
				raise ValueError(
					"selected external bundle must provide pre_gain, post_gain, pre_dark, and post_dark "
					"for legacy normalization"
				)
			scan.metadata["flat_fields"] = fields
		_disable_embedded_calibration(recon)
		_strip_resolution_params(recon)
		return CalibrationResolution(
			source="external_bundle", bundle_ref=ref, compatibility=report,
		)

	_strip_resolution_params(recon)
	if normalize is None:
		return CalibrationResolution(source="none")
	y_start, y_stop = _read_window(recon)
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
			stop = rows if y_stop is None else int(y_stop)
			if values.shape == (4, rows, columns):
				scan.metadata["flat_fields"] = values[:, y_start:stop]
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


def _resolve_flat_fields(scan: Scan, recon: Recon) -> None:
	"""Retain the private reference-generation compatibility path.

	Reference archives predate separate scan/pipeline roots and keep legacy map
	paths beside their projection source. New callers should use
	:func:`resolve_calibration` and pass both roots explicitly.
	"""

	root = Path(scan.projections.path).parent
	resolve_calibration(scan, recon, pipeline_root=root, scan_root=root)


def run_reconstruction(pipeline_path: str | Path, scan_path: str | Path,
			output_path: str | Path | None = None, backend: Backend | None = None, *,
			calibration_bundle: CalibrationBundleRef | None = None,
			verify_calibration_hashes: bool = True) -> list[RunResult]:
	"""Load, expand, and execute every concrete run described by YAML files."""
	recon, resolved_pipeline_path = load_recon(pipeline_path)
	scan, resolved_scan_path = load_scan(scan_path)
	if output_path is not None:
		recon.output_path = Path(output_path).resolve()
	elif recon.output_path is not None and not recon.output_path.is_absolute():
		recon.output_path = (Path.cwd() / recon.output_path).resolve()
	resolution = resolve_calibration(
		scan,
		recon,
		pipeline_root=resolved_pipeline_path.parent,
		scan_root=resolved_scan_path.parent,
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
	backend = backend or NumpyBackend()
	runs = expand_recon_sweeps(recon)
	results = []
	for run in runs:
		concrete = run.recon.model_copy(deep=True)
		if len(runs) > 1 and concrete.output_path is not None:
			concrete.output_path = concrete.output_path / f"sweep-{run.index:04d}"
		stages = [create_stage(config) for config in concrete.pipeline]
		state = PipelineState(
			working_array=np.empty((0,), dtype=np.float32),
			scan=scan.model_copy(deep=False),
			recon=concrete,
			artifacts=dict(initial_artifacts),
			backend=backend,
		)
		state = Pipeline(stages, backend=backend, validation_mode="complete").run(state)
		results.append(RunResult(run.index, run.parameters, concrete.output_path, dict(state.artifacts)))
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
	if isinstance(value, np.generic):
		return value.item()
	return value


def _json_result(result: RunResult) -> dict[str, Any]:
	return {
		"index": result.index,
		"parameters": result.parameters,
		"output_path": None if result.output_path is None else str(result.output_path),
		"artifacts": _json_value(result.artifacts),
	}


@click.command()
@click.argument("pipeline_path", type=click.Path(path_type=Path))
@click.argument("scan_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), help="Override Recon.output_path.")
@click.option("--gpu", is_flag=True, help="Declare that the selected runtime provides CUDA support.")
@click.option("--json-output", is_flag=True, help="Emit machine-readable run summaries.")
def main(pipeline_path: Path, scan_path: Path, output_path: Path | None,
			gpu: bool, json_output: bool) -> None:
	"""Execute PIPELINE_PATH (or a bundled template name) for SCAN_PATH."""
	backend = Backend(name="numpy", supports_gpu=True) if gpu else NumpyBackend()
	results = run_reconstruction(pipeline_path, scan_path, output_path, backend)
	if json_output:
		click.echo(json.dumps([_json_result(result) for result in results], indent=2))
	else:
		for result in results:
			click.echo(f"run {result.index}: {result.output_path or '(no output path)'}")


if __name__ == "__main__":
	main()
