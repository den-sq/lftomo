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
from .fidelity import FidelityExperimentRunner, aggregate_window_metrics
from .runner import (
	ExperimentInputs,
	ExperimentRunPolicy,
	IndependentExperimentRunner,
	TrialExecution,
	json_value,
)
from .store import (
	PersistedExperimentRunner,
	PersistedExperimentStore,
	ResumePolicy,
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

	def _window_recon(self, identity, window, output_root: Path) -> Recon:
		scan, _ = load_scan(self.scan_path)
		recon = identity.resolved_recon.model_copy(deep=True)
		read = next((stage for stage in recon.pipeline if stage.kind == "read"), None)
		if read is None:
			raise ValueError("fidelity window execution requires a read stage")
		rows = scan.detector.rows
		if rows is not None and (
			window.y_start - window.halo < 0 or window.y_stop + window.halo > rows
		):
			raise ValueError(
				f"window {window.id!r} halo exceeds detector rows [0, {rows})"
			)
		read.params["y_start"] = window.y_start - window.halo
		read.params["y_stop"] = window.y_stop + window.halo
		theta_start = 0 if window.theta_start is None else window.theta_start
		theta_stop = scan.n_projections if window.theta_stop is None else window.theta_stop
		if identity.fidelity.projection_limit is not None:
			theta_stop = min(
				theta_stop,
				theta_start
				+ identity.fidelity.projection_limit * identity.fidelity.projection_stride,
			)
		read.params["theta_start"] = theta_start
		read.params["theta_stop"] = theta_stop
		read.params["theta_stride"] = identity.fidelity.projection_stride
		for stage in recon.pipeline:
			if stage.kind == "evaluate":
				stage.params["windows"] = [{
					"id": window.id,
					"slice_start": window.halo,
					"slice_stop": window.halo + window.y_stop - window.y_start,
				}]
		is_full_finalist = (
			self.config.output.write_full_finalists
			and identity.fidelity.id == self.config.output.final_fidelity_id
		)
		if self.config.output.screening != "configured_writers" and not is_full_finalist:
			recon.pipeline = [stage for stage in recon.pipeline if stage.kind != "write"]
			recon.output_path = None
		else:
			recon.output_path = output_root / "output"
		return recon

	def validate_identity(self, identity) -> None:
		scan, _ = load_scan(self.scan_path)
		for window in identity.fidelity.windows:
			recon = self._window_recon(identity, window, Path("validation") / window.id)
			for stage in recon.pipeline:
				for name in (
					"calibration_manifest", "calibration_components",
					"compatibility_policy", "compatibility_override_reason",
				):
					stage.params.pop(name, None)
			stages = [create_stage(config) for config in recon.pipeline]
			validate_pipeline(stages, scan, NumpyBackend(), mode="complete")

	def __call__(self, identity, output_root: Path) -> TrialExecution:
		output_root.mkdir(parents=True, exist_ok=True)
		window_metrics: dict[str, dict[str, Any]] = {}
		metric_ids = {reference.metric: reference.id for reference in self.config.metrics}
		artifacts = {}
		outputs = []
		for window in identity.fidelity.windows:
			window_root = output_root / "windows" / window.id
			recon = self._window_recon(identity, window, window_root)
			window_root.mkdir(parents=True, exist_ok=True)
			pipeline_path = window_root / "resolved-recon.yaml"
			pipeline_path.write_text(recon.to_yaml(), encoding="utf-8")
			result = run_reconstruction(pipeline_path, self.scan_path)[0]
			artifact = result.artifacts.get("metrics", {})
			aggregates = artifact.get("aggregates", {}) if isinstance(artifact, dict) else {}
			observations = artifact.get("observations", ()) if isinstance(artifact, dict) else ()
			for observation in observations:
				if not isinstance(observation, dict):
					continue
				metric = metric_ids.get(observation.get("metric"), observation.get("metric"))
				if isinstance(metric, str):
					window_metrics.setdefault(window.id, {}).setdefault(metric, []).append(
						observation.get("value") if observation.get("valid") else None
					)
			if not observations:
				for reference in self.config.metrics:
					aggregate = aggregates.get(reference.metric, aggregates.get(reference.id))
					if isinstance(aggregate, dict):
						aggregate = aggregate.get("value")
					window_metrics.setdefault(window.id, {})[reference.id] = [aggregate]
			artifacts[window.id] = json_value(result.artifacts)
			for path in result.artifacts.get("output_files", ()):
				path = Path(path)
				if path.is_file():
					outputs.append(OutputRecord(
						role="finalist_output" if (
							self.config.output.write_full_finalists
							and identity.fidelity.id == self.config.output.final_fidelity_id
						) else "screening_output",
						path=path,
						bytes=path.stat().st_size,
					))
		aggregates = aggregate_window_metrics(window_metrics, self.config)
		return TrialExecution(
			metrics=json_value(aggregates.tuning),
			window_metrics=json_value(window_metrics),
			artifacts={
				"windows": json_value(artifacts),
				"per_window_metrics": json_value(aggregates.per_window),
				"held_out_metrics": json_value(aggregates.held_out),
			},
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
		identity_validator=executor.validate_identity,
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
	if runner.config.promotion is None:
		preflight, _ = runner.prepare()
		payload = preflight.model_dump(mode="json")
	else:
		payload = {"levels": {}}
		for fidelity in runner.config.fidelities:
			level_runner = IndependentExperimentRunner(
				runner.config,
				runner.inputs,
				runner.root,
				runner.executor,
				validator=runner.validator,
				identity_validator=runner.identity_validator,
				policy=runner.policy,
				fidelity_id=fidelity.id,
			)
			preflight, _ = level_runner.prepare()
			payload["levels"][fidelity.id] = preflight.model_dump(mode="json")
	click.echo(json.dumps(payload, sort_keys=True, indent=2))


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
	if runner.config.promotion is not None:
		summary = FidelityExperimentRunner(
			runner.config,
			runner.inputs,
			runner.root,
			runner.executor,
			validator=runner.validator,
			identity_validator=runner.identity_validator,
			run_policy=runner.policy,
		).run()
	else:
		summary = PersistedExperimentRunner(runner).run()
	click.echo(json.dumps(summary.model_dump(mode="json"), sort_keys=True, indent=2))


def _resume_options(function):
	function = click.option(
		"--stale-after", type=click.FloatRange(min=0, min_open=True),
		default=3600.0, show_default=True,
	)(function)
	function = click.option("--recover-stale", is_flag=True)(function)
	function = click.option("--retry-interrupted", is_flag=True)(function)
	function = click.option("--retry-failed", is_flag=True)(function)
	function = click.option(
		"--mismatch", type=click.Choice(["rerun", "skip", "fail"]),
		default="fail", show_default=True,
	)(function)
	return function


def _resume_policy(
	*, mismatch: str, retry_failed: bool, retry_interrupted: bool,
	recover_stale: bool, stale_after: float, force: bool = False,
) -> ResumePolicy:
	return ResumePolicy(
		mismatch=mismatch,
		retry_failed=retry_failed,
		retry_interrupted=retry_interrupted,
		recover_stale=recover_stale,
		stale_after_seconds=stale_after,
		force=force,
	)


@main.command("inspect")
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
def inspect_command(root: Path) -> None:
	"""Inspect ROOT and quarantine malformed partial attempt records."""

	inspection = PersistedExperimentStore(root).inspect()
	click.echo(json.dumps(inspection.model_dump(mode="json"), sort_keys=True, indent=2))


@main.command("resume")
@click.option("--invalidate-stale-promotions", is_flag=True)
@_resume_options
@_shared_options
def resume_command(
	experiment_path: Path,
	scan_path: Path,
	root: Path,
	override_limits: bool,
	failure: str,
	mismatch: str,
	retry_failed: bool,
	retry_interrupted: bool,
	recover_stale: bool,
	stale_after: float,
	invalidate_stale_promotions: bool,
) -> None:
	"""Reuse compatible successes and explicitly retry selected terminal states."""

	runner = _runner(
		experiment_path, scan_path, root,
		override_limits=override_limits, failure=failure,
	)
	resume_policy = _resume_policy(
			mismatch=mismatch,
			retry_failed=retry_failed,
			retry_interrupted=retry_interrupted,
			recover_stale=recover_stale,
			stale_after=stale_after,
		)
	if runner.config.promotion is not None:
		summary = FidelityExperimentRunner(
			runner.config,
			runner.inputs,
			runner.root,
			runner.executor,
			validator=runner.validator,
			identity_validator=runner.identity_validator,
			run_policy=runner.policy,
			resume_policy=resume_policy,
			invalidate_stale_promotions=invalidate_stale_promotions,
		).run()
	else:
		summary = PersistedExperimentRunner(runner, policy=resume_policy).run()
	click.echo(json.dumps(summary.model_dump(mode="json"), sort_keys=True, indent=2))


@main.command("rerun")
@click.option("--trial-id", "trial_ids", multiple=True)
@click.option("--invalidate-stale-promotions", is_flag=True)
@_resume_options
@_shared_options
def rerun_command(
	experiment_path: Path,
	scan_path: Path,
	root: Path,
	override_limits: bool,
	failure: str,
	mismatch: str,
	retry_failed: bool,
	retry_interrupted: bool,
	recover_stale: bool,
	stale_after: float,
	trial_ids: tuple[str, ...],
	invalidate_stale_promotions: bool,
) -> None:
	"""Force a new attempt, optionally limited to one or more resolved trial IDs."""

	runner = _runner(
		experiment_path, scan_path, root,
		override_limits=override_limits, failure=failure,
	)
	resume_policy = _resume_policy(
			mismatch=mismatch,
			retry_failed=retry_failed,
			retry_interrupted=retry_interrupted,
			recover_stale=recover_stale,
			stale_after=stale_after,
			force=True,
		)
	if runner.config.promotion is not None:
		if trial_ids:
			raise click.UsageError(
				"--trial-id filtering is not supported for a multi-fidelity rerun; "
				"use the library boundary API"
			)
		summary = FidelityExperimentRunner(
			runner.config,
			runner.inputs,
			runner.root,
			runner.executor,
			validator=runner.validator,
			identity_validator=runner.identity_validator,
			run_policy=runner.policy,
			resume_policy=resume_policy,
			invalidate_stale_promotions=invalidate_stale_promotions,
		).run()
	else:
		summary = PersistedExperimentRunner(
			runner,
			policy=resume_policy,
			trial_ids=set(trial_ids) if trial_ids else None,
		).run()
	click.echo(json.dumps(summary.model_dump(mode="json"), sort_keys=True, indent=2))


if __name__ == "__main__":
	main()
