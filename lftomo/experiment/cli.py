"""Click CLI and complete-pipeline adapter for experiment execution."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import click
import yaml

from lftomo.config import Recon
from lftomo.evaluation import validate_experiment_metric_references
from lftomo.pipeline import NumpyBackend, validate_pipeline
from lftomo.pipeline.stages import create_stage
from lftomo.runner import load_scan, run_reconstruction

from .models import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	OutputRecord,
	canonical_json,
)
from .runner import (
	ExperimentInputs,
	ExperimentRunPolicy,
	IndependentExperimentRunner,
	TrialExecution,
	json_value,
)


def _sha256(value: bytes) -> str:
	return hashlib.sha256(value).hexdigest()


def load_experiment_config(path: str | Path) -> ExperimentConfig:
	"""Load and validate one versioned experiment YAML document."""

	path = Path(path)
	with path.open("r", encoding="utf-8") as stream:
		config = ExperimentConfig.model_validate(yaml.safe_load(stream))
	validate_experiment_metric_references(config)
	return config


def _git(arguments: list[str], root: Path) -> str | None:
	try:
		result = subprocess.run(
			["git", *arguments], cwd=root, check=True, capture_output=True, text=True,
		)
	except (OSError, subprocess.CalledProcessError):
		return None
	return result.stdout.strip()


def build_experiment_inputs(scan_path: str | Path) -> ExperimentInputs:
	"""Derive invocation identity without opening projection payloads."""

	scan, resolved = load_scan(scan_path)
	content_hash = scan.metadata.get("dataset_content_hash")
	if not isinstance(content_hash, str) or len(content_hash) != 64:
		raise ValueError(
			"scan.metadata.dataset_content_hash must provide the immutable 64-character "
			"dataset content hash for experiment identity"
		)
	repository = Path(__file__).resolve().parents[2]
	commit = os.environ.get("LFTOMO_GIT_COMMIT") or _git(["rev-parse", "HEAD"], repository)
	commit = commit or "unknown0"
	dirty = _git(["status", "--porcelain"], repository)
	dependencies = {}
	for package in ("numpy", "scipy", "pydantic"):
		try:
			dependencies[package] = version(package)
		except PackageNotFoundError:
			continue
	calibration_hash = scan.metadata.get("calibration_hash")
	return ExperimentInputs(
		dataset=DatasetIdentity(
			dataset_id=str(scan.metadata.get("dataset_id") or resolved.stem),
			content_hash=content_hash,
			acquisition_id=scan.metadata.get("acquisition_id"),
		),
		configuration=ConfigurationIdentity(
			scan_config_hash=_sha256(canonical_json(scan).encode("utf-8")),
			calibration_hash=calibration_hash,
		),
		code=CodeIdentity(
			version="0.1.0", git_commit=commit, dirty=bool(dirty),
		),
		environment=EnvironmentIdentity(
			python=platform.python_version(),
			platform=platform.platform(),
			backend="numpy",
			dependencies=dependencies,
			hardware={"cpu_count": os.cpu_count() or 1},
		),
		description={"dataset_path": resolved},
	)


class PipelineTrialExecutor:
	"""Run one resolved candidate through the existing complete-pipeline runner."""

	def __init__(self, config: ExperimentConfig, scan_path: str | Path):
		self.config = config
		self.scan_path = Path(scan_path).resolve()

	def validate(self, recon: Recon) -> None:
		scan, _ = load_scan(self.scan_path)
		resolved = recon.model_copy(deep=True)
		for stage in resolved.pipeline:
			for name in (
				"calibration_manifest", "calibration_components",
				"compatibility_policy", "compatibility_override_reason",
			):
				stage.params.pop(name, None)
		stages = [create_stage(config) for config in resolved.pipeline]
		validate_pipeline(stages, scan, NumpyBackend(), mode="complete")

	def __call__(self, identity, output_root: Path) -> TrialExecution:
		recon = identity.resolved_recon.model_copy(deep=True)
		if self.config.output.screening != "configured_writers":
			recon.pipeline = [stage for stage in recon.pipeline if stage.kind != "write"]
			recon.output_path = None
		else:
			recon.output_path = output_root / "output"
		output_root.mkdir(parents=True, exist_ok=True)
		pipeline_path = output_root / "resolved-recon.yaml"
		pipeline_path.write_text(recon.to_yaml(), encoding="utf-8")
		result = run_reconstruction(pipeline_path, self.scan_path)[0]
		artifact = result.artifacts.get("metrics", {})
		aggregates = artifact.get("aggregates", {}) if isinstance(artifact, dict) else {}
		metrics: dict[str, Any] = {}
		for reference in self.config.metrics:
			aggregate = aggregates.get(reference.metric)
			if aggregate is None:
				aggregate = aggregates.get(reference.id)
			if isinstance(aggregate, dict):
				metrics[reference.id] = aggregate.get("value")
			elif aggregate is not None:
				metrics[reference.id] = aggregate
		outputs = []
		for path in result.artifacts.get("output_files", ()):
			path = Path(path)
			if path.is_file():
				outputs.append(OutputRecord(
					role="screening_output", path=path, bytes=path.stat().st_size,
				))
		return TrialExecution(
			metrics=metrics,
			artifacts=json_value(result.artifacts),
			outputs=tuple(outputs),
		)


def _runner(
	experiment_path: Path,
	scan_path: Path,
	root: Path,
	*,
	override_limits: bool,
	failure: str,
) -> IndependentExperimentRunner:
	config = load_experiment_config(experiment_path)
	executor = PipelineTrialExecutor(config, scan_path)
	return IndependentExperimentRunner(
		config,
		build_experiment_inputs(scan_path),
		root,
		executor,
		validator=executor.validate,
		policy=ExperimentRunPolicy(
			override_limits=override_limits,
			failure=failure,
		),
	)


@click.group()
def main() -> None:
	"""Validate, preview, and run deterministic lftomo experiments."""


@main.command("validate")
@click.argument("experiment_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json-output", is_flag=True)
def validate_command(experiment_path: Path, json_output: bool) -> None:
	"""Validate EXPERIMENT_PATH without loading reconstruction data."""

	config = load_experiment_config(experiment_path)
	payload = {
		"valid": True,
		"experiment_id": config.experiment_id,
		"sampler": config.sampler.model_dump(mode="json"),
	}
	click.echo(json.dumps(payload, sort_keys=True) if json_output else (
		f"valid experiment {config.experiment_id}"
	))


def _shared_options(function):
	function = click.option("--failure", type=click.Choice(["continue", "fail_fast"]),
		default="continue", show_default=True)(function)
	function = click.option("--override-limits", is_flag=True)(function)
	function = click.option("--root", type=click.Path(path_type=Path), required=True)(function)
	function = click.argument("scan_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))(function)
	function = click.argument("experiment_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))(function)
	return function


@main.command("dry-run")
@_shared_options
def dry_run_command(
	experiment_path: Path,
	scan_path: Path,
	root: Path,
	override_limits: bool,
	failure: str,
) -> None:
	"""Resolve and validate candidates without data execution."""

	runner = _runner(
		experiment_path, scan_path, root,
		override_limits=override_limits, failure=failure,
	)
	preflight, _ = runner.prepare()
	click.echo(json.dumps(preflight.model_dump(mode="json"), sort_keys=True, indent=2))


@main.command("run")
@_shared_options
def run_command(
	experiment_path: Path,
	scan_path: Path,
	root: Path,
	override_limits: bool,
	failure: str,
) -> None:
	"""Execute independent candidates and emit deterministic results."""

	runner = _runner(
		experiment_path, scan_path, root,
		override_limits=override_limits, failure=failure,
	)
	summary = runner.run()
	click.echo(json.dumps(summary.model_dump(mode="json"), sort_keys=True, indent=2))


if __name__ == "__main__":
	main()
