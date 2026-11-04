"""Click CLI and complete-pipeline adapter for experiment execution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import click
import numpy as np

from lftomo.config import CALIBRATION_RESOLUTION_FIELDS, Recon
from lftomo.pipeline import NumpyBackend, Pipeline, validate_pipeline
from lftomo.pipeline.stages import create_stage, dispatch_for_stage
from lftomo.review import CandidateReport, ExperimentCandidateResolver
from lftomo.runner import load_scan, run_reconstruction
from lftomo.util.window import TomoWindow

from .models import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	OutputRecord,
	canonical_json,
)
from .fidelity import (
	FidelityRunSummary,
	PreviewBundle,
	aggregate_window_metrics,
	export_blinded_previews,
)
from lftomo.evaluation import PreviewContract
from .protocol import (
	Tier2Bounds,
	Tier2ExperimentPlan,
	audit_tier2_results,
	build_tier2_plan,
	write_tier2_plan,
)
from .prefix_cache import (
	PrefixCacheDecision,
	PrefixCachePlan,
	PrefixCachePolicy,
	PrefixCachingPipelineExecutor,
	PrefixPlanningCandidate,
	plan_common_prefixes,
)
from .runner import (
	ExperimentInputs,
	ExperimentRunPolicy,
	IndependentExperimentRunner,
	TrialExecution,
	json_value,
)
from .runtime import (
	RuntimeCompatibilityReport,
	RuntimeCompatibilityError,
	collect_environment_identity,
	require_runtime_compatibility,
)
from .resources import PathContextRoots, PathResourceRequest
from .events import (
	ProgressInfrastructureError,
	StageProgressMember,
	current_execution_context,
)
from .lifecycle import ExperimentLifecycle, load_experiment, validate_experiment
from .store import ResumePolicy


def _sha256(value: bytes) -> str:
	return hashlib.sha256(value).hexdigest()


def _fidelity_window(scan, identity, window) -> TomoWindow:
	"""Resolve one experiment window through the canonical bounds constructor."""

	theta_start = 0 if window.theta_start is None else window.theta_start
	theta_stop = scan.n_projections if window.theta_stop is None else window.theta_stop
	if identity.fidelity.projection_limit is not None:
		theta_stop = min(
			theta_stop,
			theta_start
			+ identity.fidelity.projection_limit * identity.fidelity.projection_stride,
		)
	return TomoWindow.from_bounds(
		scan.projection_shape(),
		y=(window.y_start, window.y_stop),
		theta=(theta_start, theta_stop, identity.fidelity.projection_stride),
		halo=window.halo,
		dtype=np.dtype(scan.metadata.get("raw_dtype", np.uint16)),
		work_dtype=np.float32,
	)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
	"""Load and validate one versioned experiment YAML document."""

	return load_experiment(path)


def _git(arguments: list[str], root: Path) -> str | None:
	try:
		result = subprocess.run(
			["git", *arguments], cwd=root, check=True, capture_output=True, text=True,
		)
	except (OSError, subprocess.CalledProcessError):
		return None
	return result.stdout.strip()


def build_experiment_inputs(
	scan_path: str | Path,
	*,
	environment: EnvironmentIdentity | None = None,
) -> ExperimentInputs:
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
		environment=environment or collect_environment_identity(),
		description={"dataset_path": resolved},
	)


class PipelineTrialExecutor:
	"""Run one resolved candidate through the existing complete-pipeline runner."""

	def __init__(
		self,
		config: ExperimentConfig,
		scan_path: str | Path,
		prefix_cache_policy: PrefixCachePolicy | None = None,
	):
		self.config = config
		self.scan_path = Path(scan_path).resolve()
		self.prefix_cache_policy = prefix_cache_policy or PrefixCachePolicy()
		self.prefix_executor = PrefixCachingPipelineExecutor(self.prefix_cache_policy)
		self._prefix_decisions: dict[str, PrefixCacheDecision] = {}

	def __getstate__(self) -> dict[str, Any]:
		"""Exclude process-local cache locks from the spawned trial payload."""

		state = self.__dict__.copy()
		state.pop("prefix_executor", None)
		return state

	def __setstate__(self, state: dict[str, Any]) -> None:
		"""Rebuild process-local cache coordination after spawning a trial."""

		self.__dict__.update(state)
		self.prefix_executor = PrefixCachingPipelineExecutor(self.prefix_cache_policy)

	def replay_document(self) -> dict[str, Any]:
		"""Return the normalized non-secret adapter inputs needed for replay."""

		scan, _ = load_scan(self.scan_path)
		return {
			"kind": "pipeline",
			"scan": scan.model_dump(mode="json"),
			"prefix_cache_policy": self.prefix_cache_policy.model_dump(mode="json"),
		}

	@classmethod
	def from_replay(
		cls,
		config: ExperimentConfig,
		scan_path: str | Path,
		document: dict[str, Any],
	) -> "PipelineTrialExecutor":
		policy = PrefixCachePolicy.model_validate(
			document.get("prefix_cache_policy", {}),
		)
		return cls(config, scan_path, policy)

	def validate(self, recon: Recon) -> None:
		scan, _ = load_scan(self.scan_path)
		resolved = recon.model_copy(deep=True)
		for stage in resolved.pipeline:
			for name in CALIBRATION_RESOLUTION_FIELDS:
				stage.params.pop(name, None)
		stages = [create_stage(config) for config in resolved.pipeline]
		validate_pipeline(stages, scan, NumpyBackend(), mode="complete")

	def _window_recon(self, identity, window, output_root: Path) -> Recon:
		scan, _ = load_scan(self.scan_path)
		recon = identity.resolved_recon.model_copy(deep=True)
		read = next((stage for stage in recon.pipeline if stage.kind == "read"), None)
		if read is None:
			raise ValueError("fidelity window execution requires a read stage")
		selected = _fidelity_window(scan, identity, window)
		source = selected.with_pad(["Y"])
		read.params["y_start"] = source.pos.Y
		read.params["y_stop"] = source.pos.Y + source.shape.Y
		read.params["theta_start"] = selected.pos.Theta
		read.params["theta_stop"] = selected.pos.Theta + selected.shape.Theta
		read.params["theta_stride"] = selected.step.Theta
		for stage in recon.pipeline:
			if stage.kind == "evaluate":
				stage.params["windows"] = [{
					"id": window.id,
					"slice_start": selected.pad_before.Y,
					"slice_stop": selected.pad_before.Y + selected.shape.Y,
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
				for name in CALIBRATION_RESOLUTION_FIELDS:
					stage.params.pop(name, None)
			stages = [create_stage(config) for config in recon.pipeline]
			validate_pipeline(stages, scan, NumpyBackend(), mode="complete")

	@staticmethod
	def _prefix_candidate_id(identity, window) -> str:
		return f"{identity.trial_id.value}:{window.id}"

	@staticmethod
	def _prefix_share_identity(identity, window) -> dict[str, Any]:
		return {
			"dataset": identity.dataset,
			"configuration": identity.configuration,
			"code": identity.code,
			"environment": identity.environment,
			"fidelity": identity.fidelity,
			"window": window,
		}

	@staticmethod
	def _prefix_runtime_identity(identity) -> dict[str, Any]:
		"""Return non-geometry identity; resolved state owns runtime geometry."""
		return {
			"dataset": identity.dataset,
			"configuration": identity.configuration,
			"code": identity.code,
			"environment": identity.environment,
		}

	@staticmethod
	def _publish_prefix_cache_progress(active, event, configs, stages) -> None:
		"""Publish cache observability as infrastructure, never candidate science."""

		if active is None:
			return
		try:
			cache_event, cache_bytes, details = event.progress_projection()
			cached_members = ()
			if event.status == "hit":
				cached_members = tuple(StageProgressMember(
					authored_index=index,
					position=index + 1,
					total=len(configs),
					occurrence_id=(
						getattr(config, "id", None)
						or f"{getattr(config, 'kind', stage.name)}[{index}]"
					),
					algorithm=getattr(config, "algorithm", None) or stage.name,
					stage_name=stage.name,
					stage_type=f"{type(stage).__module__}.{type(stage).__qualname__}",
					dispatch_kind=dispatch_for_stage(stage).kind.name.lower(),
					cache_state="cached",
				) for index, (config, stage) in enumerate(
					zip(configs[:event.stage_count], stages[:event.stage_count])
				))
			active.cache_event(
				cache_event,
				cache_bytes=cache_bytes,
				cached_members=cached_members,
				details=details,
			)
			if event.verified:
				active.cache_event(
					"verify", cache_bytes=cache_bytes,
					details={"cache_key": event.cache_key},
				)
			if event.evicted_keys:
				active.cache_event(
					"evict", cache_bytes=cache_bytes,
					details={"evicted_keys": event.evicted_keys},
				)
		except ProgressInfrastructureError:
			raise
		except Exception as error:
			raise ProgressInfrastructureError(
				f"failed to publish prefix cache progress: {error}"
			) from error

	@staticmethod
	def _estimated_snapshot_bytes(scan, identity, window) -> int | None:
		selected = _fidelity_window(scan, identity, window).with_pad(["Y"])
		return selected.frange.length.nbytes(np.float32)

	def plan_prefix_cache(self, trials) -> PrefixCachePlan:
		"""Plan trial/window prefixes without opening projection payloads."""

		scan, _ = load_scan(self.scan_path)
		candidates = []
		for trial in trials:
			for window in trial.identity.fidelity.windows:
				recon = self._window_recon(
					trial.identity, window, Path("prefix-plan") / window.id,
				)
				for config in recon.pipeline:
					for name in CALIBRATION_RESOLUTION_FIELDS:
						config.params.pop(name, None)
				stages = tuple(create_stage(config) for config in recon.pipeline)
				candidates.append(PrefixPlanningCandidate(
					candidate_id=self._prefix_candidate_id(trial.identity, window),
					configs=tuple(recon.pipeline),
					stages=stages,
					share_identity=self._prefix_share_identity(trial.identity, window),
					estimated_snapshot_bytes=self._estimated_snapshot_bytes(
						scan, trial.identity, window,
					),
				))
		plan = plan_common_prefixes(candidates, self.prefix_cache_policy)
		self._prefix_decisions = {item.candidate_id: item for item in plan.decisions}
		return plan

	@staticmethod
	def _execution_observer(configs, *, cache_state="executing"):
		context = current_execution_context()

		return (
			None if context is None else
			lambda event: context.execution_span(
				event, configs, cache_state=cache_state,
			)
		)

	def __call__(self, identity, output_root: Path) -> TrialExecution:
		output_root.mkdir(parents=True, exist_ok=True)
		window_metrics: dict[str, dict[str, Any]] = {}
		metric_ids = {reference.metric: reference.id for reference in self.config.metrics}
		artifacts = {}
		outputs = []
		preview_images = {}
		preview_contract = None
		for window_index, window in enumerate(identity.fidelity.windows):
			context = current_execution_context()
			if context is not None:
				context.window_started(
					window.id, position=window_index + 1,
					total=len(identity.fidelity.windows),
				)
			window_root = output_root / "windows" / window.id
			recon = self._window_recon(identity, window, window_root)
			window_root.mkdir(parents=True, exist_ok=True)
			pipeline_path = window_root / "resolved-recon.yaml"
			pipeline_path.write_text(recon.to_yaml(), encoding="utf-8")
			decision = None
			if self._prefix_decisions:
				decision = self._prefix_decisions.get(
					self._prefix_candidate_id(identity, window)
				)
			def execute_state(state, stages, configs, *, _decision=decision, _window=window):
				observer = self._execution_observer(configs)
				if _decision is not None and _decision.action == "cache":
					result, event = self.prefix_executor.execute(
						_decision,
						state,
						stages,
						configs,
						runtime_identity=self._prefix_runtime_identity(identity),
						execution_observer=observer,
						cache_build_execution_observer=self._execution_observer(
							configs, cache_state="cache_build",
						),
						cache_verify_execution_observer=self._execution_observer(
							configs, cache_state="cache_verify",
						),
						cache_observer=lambda cache_event: self._publish_prefix_cache_progress(
							current_execution_context(), cache_event, configs, stages,
						),
						checkpoint=None if context is None else context.checkpoint,
					)
					result.artifacts["prefix_cache"] = event.model_dump(mode="json")
					return result
				return Pipeline(
					list(stages),
					backend=state.backend,
					validation_mode="complete",
					execution_observer=observer,
					checkpoint=None if context is None else context.checkpoint,
				).run(state)
			if context is None and (decision is None or decision.action != "cache"):
				result = run_reconstruction(pipeline_path, self.scan_path)[0]
			else:
				result = run_reconstruction(
					pipeline_path, self.scan_path, state_executor=execute_state,
				)[0]
			window_artifacts = dict(result.artifacts)
			raw_previews = window_artifacts.pop("preview_images", {})
			artifact = window_artifacts.get("metrics", {})
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
			if isinstance(raw_previews, dict):
				preview_images.update({
					f"{window.id}-{label}": np.asarray(image)
					for label, image in raw_previews.items()
				})
			contract_document = artifact.get("preview") if isinstance(artifact, dict) else None
			if contract_document is not None:
				contract = PreviewContract.model_validate(contract_document)
				if preview_contract is not None and preview_contract != contract:
					raise ValueError("evaluation windows use inconsistent preview contracts")
				preview_contract = contract
			artifacts[window.id] = json_value(window_artifacts)
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
			if context is not None:
				context.window_completed(window.id)
		is_screening = identity.fidelity.id != self.config.output.final_fidelity_id
		if self.config.output.screening == "metrics_and_previews" and is_screening:
			if preview_contract is None or not preview_images:
				raise ValueError(
					"metrics_and_previews screening requires evaluate.metrics preview configuration"
				)
			blinding_key = os.environ.get("LFTOMO_PREVIEW_BLINDING_KEY")
			if not blinding_key:
				raise ValueError(
					"metrics_and_previews screening requires LFTOMO_PREVIEW_BLINDING_KEY"
				)
			bundle = export_blinded_previews(
				{identity.candidate.value: preview_images},
				output_root.parents[1],
				preview_contract,
				blinding_key=blinding_key,
				max_bytes=self.config.output.max_bytes,
			)
			entry = next(
				item for item in bundle.entries if item.candidate_key == identity.candidate.value
			)
			outputs.extend(OutputRecord(
				role="blinded_preview", path=path, bytes=path.stat().st_size,
			) for path in entry.files)
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
	scratch_root: Path | None,
	prefix_cache: bool,
	prefix_cache_root: Path | None,
	prefix_cache_max_disk_bytes: int | None,
	prefix_cache_max_memory_bytes: int | None,
	prefix_cache_max_entries: int,
	prefix_cache_on_error: str,
	prefix_cache_verify_fraction: float,
) -> IndependentExperimentRunner:
	config = load_experiment_config(experiment_path)
	try:
		runtime_report: RuntimeCompatibilityReport = require_runtime_compatibility(config)
	except RuntimeCompatibilityError as error:
		raise click.ClickException(str(error)) from error
	if prefix_cache and (
		prefix_cache_root is None
		or prefix_cache_max_disk_bytes is None
		or prefix_cache_max_memory_bytes is None
	):
		raise click.UsageError(
			"--prefix-cache requires --prefix-cache-root, --prefix-cache-max-disk-bytes, "
			"and --prefix-cache-max-memory-bytes"
		)
	cache_policy = PrefixCachePolicy(
		enabled=prefix_cache,
		scratch=prefix_cache_root,
		max_disk_bytes=prefix_cache_max_disk_bytes,
		max_memory_bytes=prefix_cache_max_memory_bytes,
		max_entries=prefix_cache_max_entries,
		on_error=prefix_cache_on_error,
		verify_fraction=prefix_cache_verify_fraction,
	)
	path_resources = PathResourceRequest(
		result_root=root,
		scratch_root=scratch_root,
		cache_root=prefix_cache_root if prefix_cache else None,
		cache_guard_bytes=prefix_cache_max_disk_bytes if prefix_cache else None,
		contexts=PathContextRoots(
			cwd=Path.cwd(),
			repository=Path(__file__).resolve().parents[2],
			config_directory=experiment_path.resolve().parent,
		),
	)
	executor = PipelineTrialExecutor(config, scan_path, cache_policy)
	return IndependentExperimentRunner(
		config,
		build_experiment_inputs(scan_path, environment=runtime_report.environment),
		root,
		executor,
		validator=executor.validate,
		identity_validator=executor.validate_identity,
		policy=ExperimentRunPolicy(
			override_limits=override_limits,
			failure=failure,
		),
		runtime_report=runtime_report,
		path_resources=path_resources,
	)


@click.group()
def main() -> None:
	"""Validate, preview, and run deterministic lftomo experiments."""


@main.command("validate")
@click.argument("experiment_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json-output", is_flag=True)
def validate_command(experiment_path: Path, json_output: bool) -> None:
	"""Validate EXPERIMENT_PATH without loading reconstruction data."""

	result = validate_experiment(experiment_path)
	payload = result.model_dump(mode="json")
	click.echo(json.dumps(payload, sort_keys=True) if json_output else (
		f"valid experiment {config.experiment_id}"
	))


def _shared_options(function):
	function = click.option(
		"--prefix-cache-verify-fraction",
		type=click.FloatRange(min=0, max=1),
		default=0.05,
		show_default=True,
	)(function)
	function = click.option(
		"--prefix-cache-on-error",
		type=click.Choice(["rebuild", "bypass", "fail"]),
		default="rebuild",
		show_default=True,
	)(function)
	function = click.option(
		"--prefix-cache-max-entries",
		type=click.IntRange(min=1),
		default=64,
		show_default=True,
	)(function)
	function = click.option(
		"--prefix-cache-max-memory-bytes", type=click.IntRange(min=1),
	)(function)
	function = click.option(
		"--prefix-cache-max-disk-bytes", type=click.IntRange(min=1),
	)(function)
	function = click.option(
		"--prefix-cache-root", type=click.Path(file_okay=False, path_type=Path),
	)(function)
	function = click.option("--prefix-cache/--no-prefix-cache", default=False)(function)
	function = click.option("--failure", type=click.Choice(["continue", "fail_fast"]),
		default="continue", show_default=True)(function)
	function = click.option("--override-limits", is_flag=True)(function)
	function = click.option(
		"--scratch-root", type=click.Path(file_okay=False, path_type=Path),
	)(function)
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
	scratch_root: Path | None,
	prefix_cache: bool,
	prefix_cache_root: Path | None,
	prefix_cache_max_disk_bytes: int | None,
	prefix_cache_max_memory_bytes: int | None,
	prefix_cache_max_entries: int,
	prefix_cache_on_error: str,
	prefix_cache_verify_fraction: float,
) -> None:
	"""Resolve and validate candidates without data execution."""

	runner = _runner(
		experiment_path, scan_path, root,
		override_limits=override_limits, failure=failure,
		scratch_root=scratch_root,
		prefix_cache=prefix_cache,
		prefix_cache_root=prefix_cache_root,
		prefix_cache_max_disk_bytes=prefix_cache_max_disk_bytes,
		prefix_cache_max_memory_bytes=prefix_cache_max_memory_bytes,
		prefix_cache_max_entries=prefix_cache_max_entries,
		prefix_cache_on_error=prefix_cache_on_error,
		prefix_cache_verify_fraction=prefix_cache_verify_fraction,
	)
	payload = ExperimentLifecycle(runner).preflight().model_dump(mode="json")
	click.echo(json.dumps(payload, sort_keys=True, indent=2))


@main.command("run")
@_shared_options
def run_command(
	experiment_path: Path,
	scan_path: Path,
	root: Path,
	override_limits: bool,
	failure: str,
	scratch_root: Path | None,
	prefix_cache: bool,
	prefix_cache_root: Path | None,
	prefix_cache_max_disk_bytes: int | None,
	prefix_cache_max_memory_bytes: int | None,
	prefix_cache_max_entries: int,
	prefix_cache_on_error: str,
	prefix_cache_verify_fraction: float,
) -> None:
	"""Execute independent candidates and emit deterministic results."""

	runner = _runner(
		experiment_path, scan_path, root,
		override_limits=override_limits, failure=failure,
		scratch_root=scratch_root,
		prefix_cache=prefix_cache,
		prefix_cache_root=prefix_cache_root,
		prefix_cache_max_disk_bytes=prefix_cache_max_disk_bytes,
		prefix_cache_max_memory_bytes=prefix_cache_max_memory_bytes,
		prefix_cache_max_entries=prefix_cache_max_entries,
		prefix_cache_on_error=prefix_cache_on_error,
		prefix_cache_verify_fraction=prefix_cache_verify_fraction,
	)
	summary = ExperimentLifecycle(runner).run()
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

	inspection = ExperimentLifecycle.inspect_root(root)
	click.echo(json.dumps(inspection.model_dump(mode="json"), sort_keys=True, indent=2))


@main.command("status")
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
def status_command(root: Path) -> None:
	"""Report authoritative current state reconstructed from ROOT alone."""

	status = ExperimentLifecycle.status_root(root)
	click.echo(json.dumps(status.model_dump(mode="json"), sort_keys=True, indent=2))


def _format_candidate(report: CandidateReport) -> str:
	lines = [
		f"candidate: {report.candidate_key}",
		f"blind id: {report.blind_id or '-'}",
		f"matched by: {report.matched_namespace} {report.matched_id}",
		"algorithm chain: " + " -> ".join(
			f"{stage.kind}.{stage.algorithm}" if stage.algorithm else stage.kind
			for stage in report.algorithm_chain
		),
		"varied parameters:",
	]
	if report.variations:
		for variation in report.variations:
			lines.append("  " + json.dumps(
				variation.model_dump(mode="json"), sort_keys=True,
			))
	else:
		lines.append("  {}")
	lines.append("trials:")
	for trial in report.trials:
		lines.append(f"  - {trial.trial_id} ({trial.fidelity_id})")
		lines.append(f"    attempts: {', '.join(trial.attempt_ids) or '-'}")
		attempt = trial.latest_attempt
		if attempt is None:
			lines.append("    latest: no attempt")
			continue
		lines.extend([
			f"    latest: {attempt.attempt_id} [{attempt.status}]",
			f"    rank: {attempt.pareto.rank if attempt.pareto.rank is not None else '-'}",
			f"    elapsed seconds: "
			f"{attempt.elapsed_seconds if attempt.elapsed_seconds is not None else '-'}",
			f"    execution seconds: "
			f"{attempt.execution_elapsed_seconds if attempt.execution_elapsed_seconds is not None else '-'}",
			f"    peak memory bytes: "
			f"{attempt.peak_memory_bytes if attempt.peak_memory_bytes is not None else '-'}",
			"    metrics: " + json.dumps(attempt.metrics, sort_keys=True),
			"    promotion: " + (
				"-" if attempt.promotion is None else json.dumps(
					attempt.promotion.model_dump(mode="json"), sort_keys=True,
				)
			),
		])
		for output in attempt.outputs:
			lines.append(f"    output [{output.role}]: {output.path}")
	if report.preview_paths:
		lines.append("previews:")
		lines.extend(f"  - {path}" for path in report.preview_paths)
	lines.extend(["resolved pipeline:", report.resolved_recon.to_yaml().rstrip()])
	return "\n".join(lines)


@main.command("candidate")
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("identifier")
@click.option("--json", "json_output", is_flag=True, help="Emit structured JSON.")
@click.option("--yaml", "yaml_output", is_flag=True, help="Emit structured YAML.")
def candidate_command(
	root: Path,
	identifier: str,
	json_output: bool,
	yaml_output: bool,
) -> None:
	"""Deliberately unblind one candidate selected by blind/candidate/trial ID."""

	if json_output and yaml_output:
		raise click.UsageError("choose only one of --json or --yaml")
	try:
		report = ExperimentCandidateResolver(root).resolve(identifier)
	except (OSError, ValueError) as error:
		raise click.ClickException(str(error)) from error
	if json_output:
		click.echo(json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2))
	elif yaml_output:
		click.echo(report.to_yaml(), nl=False)
	else:
		click.echo(_format_candidate(report))


@main.command("candidates")
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--json", "json_output", is_flag=True, help="Emit structured JSON.")
def candidates_command(root: Path, json_output: bool) -> None:
	"""Deliberately unblind a compact inventory of every stored candidate."""

	try:
		inventory = ExperimentCandidateResolver(root).inventory()
	except (OSError, ValueError) as error:
		raise click.ClickException(str(error)) from error
	if json_output:
		click.echo(json.dumps([
			item.model_dump(mode="json") for item in inventory
		], sort_keys=True, indent=2))
		return
	click.echo("blind_id\tcandidate_key\tfidelities\tstatuses\talgorithm_chain")
	for item in inventory:
		click.echo("\t".join((
			item.blind_id or "-",
			item.candidate_key,
			",".join(item.fidelity_ids),
			",".join(item.latest_statuses),
			" -> ".join(item.algorithm_chain),
		)))


@main.command("tier2-plan")
@click.argument("experiment_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("scan_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
	"--environment-lock", type=click.Path(exists=True, dir_okay=False, path_type=Path),
	required=True,
)
@click.option(
	"--projection-manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path),
	required=True,
)
@click.option("--blinding-key-id", required=True)
@click.option("--max-candidates", type=click.IntRange(min=1), required=True)
@click.option("--max-screen-window-rows", type=click.IntRange(min=1), required=True)
@click.option("--max-screen-projections", type=click.IntRange(min=1), required=True)
@click.option("--max-output-bytes", type=click.IntRange(min=1), required=True)
@click.option("--max-memory-gb", type=click.FloatRange(min=0, min_open=True), required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def tier2_plan_command(
	experiment_path: Path,
	scan_path: Path,
	environment_lock: Path,
	projection_manifest: Path,
	blinding_key_id: str,
	max_candidates: int,
	max_screen_window_rows: int,
	max_screen_projections: int,
	max_output_bytes: int,
	max_memory_gb: float,
	output: Path,
) -> None:
	"""Write a tamper-evident bounded Tier-2 protocol before loading projections."""

	config = load_experiment_config(experiment_path)
	preview_documents = {
		canonical_json(stage.params["preview"])
		for stage in config.candidate_space.pipeline
		if stage.kind == "evaluate" and stage.params.get("preview") is not None
	}
	if len(preview_documents) != 1:
		raise click.UsageError(
			"Tier-2 planning requires one consistent evaluate.metrics preview contract"
		)
	preview = PreviewContract.model_validate(json.loads(next(iter(preview_documents))))
	plan = build_tier2_plan(
		config,
		build_experiment_inputs(scan_path),
		environment_lock_hash=_sha256(environment_lock.read_bytes()),
		projection_manifest_hash=_sha256(projection_manifest.read_bytes()),
		preview=preview,
		blinding_key_id=blinding_key_id,
		bounds=Tier2Bounds(
			max_candidates=max_candidates,
			max_screen_window_rows=max_screen_window_rows,
			max_screen_projections=max_screen_projections,
			max_output_bytes=max_output_bytes,
			max_memory_gb=max_memory_gb,
		),
	)
	write_tier2_plan(output, plan)
	click.echo(json.dumps(plan.model_dump(mode="json"), sort_keys=True, indent=2))


@main.command("tier2-audit")
@click.argument("plan_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
def tier2_audit_command(plan_path: Path, root: Path) -> None:
	"""Audit a completed real-data bundle against its recorded Tier-2 protocol."""

	plan = Tier2ExperimentPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
	summary = FidelityRunSummary.model_validate_json(
		(root / "fidelity-results.json").read_text(encoding="utf-8")
	)
	previews = PreviewBundle.model_validate_json(
		(root / "private" / "preview-mapping" / "mapping.json").read_text(
			encoding="utf-8"
		)
	)
	audit = audit_tier2_results(plan, summary, previews)
	click.echo(json.dumps(audit.model_dump(mode="json"), sort_keys=True, indent=2))
	if not audit.passed:
		raise click.ClickException("Tier-2 audit failed")


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
	scratch_root: Path | None,
	mismatch: str,
	retry_failed: bool,
	retry_interrupted: bool,
	recover_stale: bool,
	stale_after: float,
	invalidate_stale_promotions: bool,
	prefix_cache: bool,
	prefix_cache_root: Path | None,
	prefix_cache_max_disk_bytes: int | None,
	prefix_cache_max_memory_bytes: int | None,
	prefix_cache_max_entries: int,
	prefix_cache_on_error: str,
	prefix_cache_verify_fraction: float,
) -> None:
	"""Reuse compatible successes and explicitly retry selected terminal states."""

	runner = _runner(
		experiment_path, scan_path, root,
		override_limits=override_limits, failure=failure,
		scratch_root=scratch_root,
		prefix_cache=prefix_cache,
		prefix_cache_root=prefix_cache_root,
		prefix_cache_max_disk_bytes=prefix_cache_max_disk_bytes,
		prefix_cache_max_memory_bytes=prefix_cache_max_memory_bytes,
		prefix_cache_max_entries=prefix_cache_max_entries,
		prefix_cache_on_error=prefix_cache_on_error,
		prefix_cache_verify_fraction=prefix_cache_verify_fraction,
	)
	resume_policy = _resume_policy(
			mismatch=mismatch,
			retry_failed=retry_failed,
			retry_interrupted=retry_interrupted,
			recover_stale=recover_stale,
			stale_after=stale_after,
		)
	summary = ExperimentLifecycle(runner).resume(
		resume_policy,
		invalidate_stale_promotions=invalidate_stale_promotions,
	)
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
	scratch_root: Path | None,
	mismatch: str,
	retry_failed: bool,
	retry_interrupted: bool,
	recover_stale: bool,
	stale_after: float,
	trial_ids: tuple[str, ...],
	invalidate_stale_promotions: bool,
	prefix_cache: bool,
	prefix_cache_root: Path | None,
	prefix_cache_max_disk_bytes: int | None,
	prefix_cache_max_memory_bytes: int | None,
	prefix_cache_max_entries: int,
	prefix_cache_on_error: str,
	prefix_cache_verify_fraction: float,
) -> None:
	"""Force a new attempt, optionally limited to one or more resolved trial IDs."""

	runner = _runner(
		experiment_path, scan_path, root,
		override_limits=override_limits, failure=failure,
		scratch_root=scratch_root,
		prefix_cache=prefix_cache,
		prefix_cache_root=prefix_cache_root,
		prefix_cache_max_disk_bytes=prefix_cache_max_disk_bytes,
		prefix_cache_max_memory_bytes=prefix_cache_max_memory_bytes,
		prefix_cache_max_entries=prefix_cache_max_entries,
		prefix_cache_on_error=prefix_cache_on_error,
		prefix_cache_verify_fraction=prefix_cache_verify_fraction,
	)
	resume_policy = _resume_policy(
			mismatch=mismatch,
			retry_failed=retry_failed,
			retry_interrupted=retry_interrupted,
			recover_stale=recover_stale,
			stale_after=stale_after,
			force=True,
		)
	if runner.config.promotion is not None and trial_ids:
		raise click.UsageError(
			"--trial-id filtering is not supported for a multi-fidelity rerun"
		)
	summary = ExperimentLifecycle(runner).rerun(
		resume_policy,
		trial_ids=set(trial_ids) if trial_ids else None,
		invalidate_stale_promotions=invalidate_stale_promotions,
	)
	click.echo(json.dumps(summary.model_dump(mode="json"), sort_keys=True, indent=2))


if __name__ == "__main__":
	main()
