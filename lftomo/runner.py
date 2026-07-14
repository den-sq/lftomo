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

from lftomo.config import HDF5DxchangeProjections, Recon, Scan, TIFFStackProjections
from lftomo.pipeline import Backend, NumpyBackend, Pipeline, PipelineState, SweepRun, expand_recon_sweeps
from lftomo.pipeline.stages import create_stage


@dataclass(frozen=True)
class RunResult:
	"""Summary of one completed concrete reconstruction run."""

	index: int
	parameters: dict[str, Any]
	output_path: Path | None
	artifacts: dict[str, Any]


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
	values = np.asarray(tf.imread(path))
	if values.ndim == 3:
		values = values.mean(axis=0)
	return np.asarray(values[y_start:y_stop], dtype=np.float32)


def _resolve_flat_fields(scan: Scan, recon: Recon) -> None:
	if not any(stage.kind == "normalize" for stage in recon.pipeline):
		return
	if "flat_fields" in scan.metadata:
		fields = scan.metadata["flat_fields"]
		if isinstance(fields, (list, tuple)) and all(isinstance(item, (str, Path)) for item in fields):
			root = Path(scan.projections.path).parent
			scan.metadata["flat_fields"] = np.stack([
				_read_tiff_map(path if Path(path).is_absolute() else root / path, 0, None)
				for path in fields
			])
		return

	read = next((stage for stage in recon.pipeline if stage.kind == "read"), None)
	y_start = int(read.params.get("y_start", 0)) if read else 0
	y_stop = read.params.get("y_stop") if read else None
	if isinstance(scan.projections, HDF5DxchangeProjections):
		with h5py.File(scan.projections.path, "r") as handle:
			white = handle[scan.projections.white_path]
			dark = handle[scan.projections.dark_path]
			scan.metadata["flat_fields"] = np.stack([
				white[0, y_start:y_stop], white[-1, y_start:y_stop],
				dark[0, y_start:y_stop], dark[-1, y_start:y_stop],
			]).astype(np.float32)
		return

	normalize = next(
		(stage for stage in recon.pipeline if stage.kind == "normalize" and
		 any(name in stage.params for name in ("pre_gain", "post_gain", "pre_dark", "post_dark"))),
		None,
	)
	if normalize is None or not isinstance(scan.projections, TIFFStackProjections):
		raise ValueError("normalization requires flat_fields metadata or four calibration paths")
	root = Path(scan.projections.path).parent
	paths = [normalize.params[name] for name in ("pre_gain", "post_gain", "pre_dark", "post_dark")]
	scan.metadata["flat_fields"] = np.stack([
		_read_tiff_map(path if Path(path).is_absolute() else root / path, y_start, y_stop)
		for path in paths
	])


def _make_executable(recon: Recon) -> Recon:
	"""Legacy compatibility rewrite removed by the Phase-B Stage migration."""
	result = recon.model_copy(deep=True)
	uses_sinograms = any(stage.kind in {"normalize", "phase", "stripe", "center", "geometry", "filter"}
		for stage in result.pipeline)
	if uses_sinograms:
		for stage in result.pipeline:
			if stage.kind == "read":
				stage.params["order"] = "sinogram"
			elif stage.kind == "recon":
				stage.params["sinogram_order"] = True
	return result


def run_reconstruction(pipeline_path: str | Path, scan_path: str | Path,
			output_path: str | Path | None = None, backend: Backend | None = None) -> list[RunResult]:
	"""Load, expand, and execute every concrete run described by YAML files."""
	recon, _ = load_recon(pipeline_path)
	scan, _ = load_scan(scan_path)
	recon = _make_executable(recon)
	if output_path is not None:
		recon.output_path = Path(output_path).resolve()
	elif recon.output_path is not None and not recon.output_path.is_absolute():
		recon.output_path = (Path.cwd() / recon.output_path).resolve()
	_resolve_flat_fields(scan, recon)
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
			scan=scan.model_copy(deep=True),
			recon=concrete,
			backend=backend,
		)
		state = Pipeline(
			stages, backend=backend, scan=state.scan, validation_mode="complete",
		).run(state)
		results.append(RunResult(run.index, run.parameters, concrete.output_path, dict(state.artifacts)))
	return results


def _json_result(result: RunResult) -> dict[str, Any]:
	return {
		"index": result.index,
		"parameters": result.parameters,
		"output_path": None if result.output_path is None else str(result.output_path),
		"artifacts": {key: [str(item) for item in value] if isinstance(value, list) else str(value)
			for key, value in result.artifacts.items()},
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
