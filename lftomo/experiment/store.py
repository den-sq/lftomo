"""Versioned filesystem persistence, locking, resume, and rerun semantics."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
from typing import Callable, Iterator, Literal, Mapping
from uuid import uuid4

from pydantic import Field

from lftomo.config import ConfigModel

from .models import (
	CompatibilityDiagnostic,
	ExperimentConfig,
	ExperimentManifest,
	FailureRecord,
	TrialAttemptRecord,
	TrialIdentity,
	compare_trial_identities,
)
from .resources import (
	PathRelocationCompatibility,
	PathResourceReport,
	compare_path_resource_reports,
)
from .events import (
	AttemptProgress,
	CancellationDetails,
	CancellationToken,
	ExperimentStatus,
	FailureContext,
	ProgressEventWriter,
	secret_material_paths,
)
from .runner import (
	ExperimentRunSummary,
	IndependentExperimentRunner,
	PreparedTrial,
	atomic_write_model,
	json_value,
)


STORE_SCHEMA_VERSION = 1
MismatchPolicy = Literal["rerun", "skip", "fail"]
ResumeAction = Literal["execute", "reuse", "skip", "incompatible"]


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


class StoreMetadata(ConfigModel):
	"""Root marker that prevents accidental reuse of an incompatible layout."""

	schema_version: Literal[1] = STORE_SCHEMA_VERSION
	experiment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
	created_at: datetime


class ResumePolicy(ConfigModel):
	"""Explicit policies for compatibility, retry, and abandoned claims."""

	mismatch: MismatchPolicy = "fail"
	retry_failed: bool = False
	retry_interrupted: bool = False
	force: bool = False
	recover_stale: bool = False
	stale_after_seconds: float = Field(default=3600.0, gt=0)


class ClaimRecord(ConfigModel):
	owner: str
	pid: int = Field(gt=0)
	host: str
	claimed_at: datetime


class ResumeDecision(ConfigModel):
	action: ResumeAction
	reason: str
	diagnostics: tuple[CompatibilityDiagnostic, ...] = ()
	prior_attempt: TrialAttemptRecord | None = None


class TrialHistory(ConfigModel):
	identity: TrialIdentity
	attempts: tuple[TrialAttemptRecord, ...]

	@property
	def latest(self) -> TrialAttemptRecord | None:
		return self.attempts[-1] if self.attempts else None


class StoreInspection(ConfigModel):
	metadata: StoreMetadata
	trials: tuple[TrialHistory, ...]
	quarantined: tuple[Path, ...] = ()


class PersistedExperimentStore:
	"""Atomic, inspectable storage for append-safe trial attempt history."""

	def __init__(self, root: str | Path):
		self.root = Path(root)
		self.metadata_path = self.root / "store.json"

	def initialize(
		self,
		config: ExperimentConfig,
		*,
		manifest_metadata: Mapping[str, object] | None = None,
	) -> StoreMetadata:
		unsafe = secret_material_paths(config)
		if unsafe:
			raise ValueError(
				"secret-bearing fields cannot be persisted in an experiment config: "
				+ ", ".join(unsafe)
			)
		self.root.mkdir(parents=True, exist_ok=True)
		if self.metadata_path.exists():
			metadata = self._load_metadata()
			if metadata.experiment_id != config.experiment_id:
				raise ValueError(
					"experiment store belongs to a different experiment config: "
					f"{metadata.experiment_id} != {config.experiment_id}"
				)
			manifest_path = self.root / "manifest.json"
			if manifest_path.exists() and manifest_metadata is not None:
				manifest = ExperimentManifest.model_validate_json(
					manifest_path.read_text(encoding="utf-8")
				)
				merged = dict(manifest.metadata)
				merged.update(json_value(manifest_metadata))
				atomic_write_model(manifest_path, manifest.model_copy(update={
					"metadata": merged,
				}))
			return metadata
		metadata = StoreMetadata(experiment_id=config.experiment_id, created_at=_utcnow())
		atomic_write_model(self.metadata_path, metadata)
		atomic_write_model(self.root / "manifest.json", ExperimentManifest(
			experiment_id=config.experiment_id,
			config=config,
			created_at=metadata.created_at,
			metadata={} if manifest_metadata is None else json_value(manifest_metadata),
		))
		return metadata

	def _load_metadata(self) -> StoreMetadata:
		try:
			document = json.loads(self.metadata_path.read_text(encoding="utf-8"))
		except (OSError, json.JSONDecodeError) as error:
			raise ValueError(f"invalid experiment store metadata: {error}") from error
		if document.get("schema_version") != STORE_SCHEMA_VERSION:
			raise ValueError(
				f"unsupported experiment store schema {document.get('schema_version')!r}; "
				f"expected {STORE_SCHEMA_VERSION}"
			)
		return StoreMetadata.model_validate(document)

	def validate_path_relocation(
		self,
		current: PathResourceReport,
	) -> PathRelocationCompatibility:
		"""Compare operational roots with an existing manifest before claims."""

		manifest_path = self.root / "manifest.json"
		if not manifest_path.exists():
			return PathRelocationCompatibility(compatible=True, cache_reusable=True)
		manifest = ExperimentManifest.model_validate_json(
			manifest_path.read_text(encoding="utf-8")
		)
		prior_document = manifest.metadata.get("path_resource_preflight")
		if prior_document is None:
			return PathRelocationCompatibility(compatible=True, cache_reusable=False)
		prior = PathResourceReport.model_validate(prior_document)
		comparison = compare_path_resource_reports(prior, current)
		if not comparison.compatible:
			reasons = "; ".join(
				item.message for item in comparison.diagnostics if item.severity == "error"
			)
			raise ValueError(f"incompatible experiment resource relocation: {reasons}")
		return comparison

	def _trial_root(self, identity: TrialIdentity) -> Path:
		return self.root / "trials" / identity.trial_id.value

	def register_trial(self, identity: TrialIdentity) -> Path:
		unsafe = secret_material_paths(identity)
		if unsafe:
			raise ValueError(
				"secret-bearing fields cannot be persisted in a trial identity: "
				+ ", ".join(unsafe)
			)
		trial_root = self._trial_root(identity)
		path = trial_root / "identity.json"
		if path.exists():
			stored = TrialIdentity.model_validate_json(path.read_text(encoding="utf-8"))
			comparison = compare_trial_identities(identity, stored)
			if not comparison.compatible:
				raise ValueError("trial directory contains a conflicting identity")
		else:
			atomic_write_model(path, identity)
		return trial_root

	def _quarantine(self, path: Path, reason: str) -> Path:
		relative = path.relative_to(self.root)
		destination = (
			self.root / "quarantine" / relative.parent
			/ f"{relative.name}.{reason}-{uuid4().hex}"
		)
		destination.parent.mkdir(parents=True, exist_ok=True)
		os.replace(path, destination)
		return destination

	def quarantine_partial_outputs(self, identity: TrialIdentity) -> tuple[Path, ...]:
		trial_root = self._trial_root(identity)
		paths = []
		partial = trial_root / "partial"
		if partial.exists():
			paths.append(self._quarantine(partial, "partial"))
		attempts = trial_root / "attempts"
		if attempts.exists():
			for path in sorted(attempts.iterdir()):
				if path.name.startswith(".") or path.suffix != ".json":
					paths.append(self._quarantine(path, "partial"))
		return tuple(paths)

	def load_attempts(
		self,
		identity: TrialIdentity,
		*,
		quarantine_invalid: bool = False,
	) -> tuple[TrialAttemptRecord, ...]:
		attempt_root = self._trial_root(identity) / "attempts"
		if not attempt_root.exists():
			return ()
		attempts = []
		for path in sorted(attempt_root.glob("*.json")):
			try:
				attempt = TrialAttemptRecord.model_validate_json(path.read_text(encoding="utf-8"))
				if attempt.trial_id != identity.trial_id:
					raise ValueError("attempt trial_id differs from its directory identity")
			except (OSError, ValueError) as error:
				if quarantine_invalid:
					self._quarantine(path, "invalid")
					continue
				raise ValueError(f"invalid attempt record {path}: {error}") from error
			attempts.append(attempt)
		return tuple(sorted(attempts, key=lambda item: (item.sequence, item.created_at)))

	def histories(self, *, quarantine_invalid: bool = False) -> tuple[TrialHistory, ...]:
		trials_root = self.root / "trials"
		if not trials_root.exists():
			return ()
		histories = []
		for path in sorted(trials_root.glob("*/identity.json")):
			identity = TrialIdentity.model_validate_json(path.read_text(encoding="utf-8"))
			histories.append(TrialHistory(
				identity=identity,
				attempts=self.load_attempts(
					identity, quarantine_invalid=quarantine_invalid,
				),
			))
		return tuple(histories)

	def inspect(self, *, quarantine_invalid: bool = True) -> StoreInspection:
		metadata = self._load_metadata()
		before = set((self.root / "quarantine").rglob("*")) if (
			self.root / "quarantine"
		).exists() else set()
		trials = self.histories(quarantine_invalid=quarantine_invalid)
		after = set((self.root / "quarantine").rglob("*")) if (
			self.root / "quarantine"
		).exists() else set()
		return StoreInspection(
			metadata=metadata,
			trials=trials,
			quarantined=tuple(sorted(path for path in after - before if path.is_file())),
		)

	def _progress_from_attempt(self, attempt: TrialAttemptRecord) -> AttemptProgress:
		state = {
			"pending": "planned",
			"running": "started",
			"succeeded": "succeeded",
			"failed": "failed",
			"interrupted": "interrupted",
			"skipped": "skipped",
			"incompatible": "skipped",
		}[attempt.status]
		failure = None
		if attempt.failure is not None:
			failure = FailureContext(
				type=attempt.failure.type,
				message=attempt.failure.message,
				traceback=attempt.failure.traceback,
				window_id=attempt.failure.window_id,
				stage_occurrence_id=attempt.failure.stage_id,
				stage_algorithm=attempt.failure.stage_algorithm,
			)
		return AttemptProgress(
			experiment_id=self._load_metadata().experiment_id,
			fidelity_id=attempt.identity.fidelity.id,
			candidate_key=attempt.identity.candidate.value,
			trial_id=attempt.trial_id.value,
			attempt_id=attempt.attempt_id,
			structural_variant=attempt.identity.sampler.explicit_case,
			state=state,
			updated_at=attempt.finished_at or attempt.started_at or attempt.created_at,
			started_at=attempt.started_at,
			finished_at=attempt.finished_at,
			elapsed_seconds=attempt.elapsed_seconds,
			resource_bytes=attempt.peak_memory_bytes,
			failure=failure,
		)

	def write_progress(self, progress: AttemptProgress) -> None:
		atomic_write_model(
			self.root / "trials" / progress.trial_id / "progress"
			/ f"{progress.attempt_id}.json",
			progress,
		)

	def record_cancellation(self, details: CancellationDetails) -> None:
		atomic_write_model(self.root / "cancellation.json", details)

	def status(self) -> ExperimentStatus:
		"""Reconstruct authoritative current state without consulting JSONL events."""

		metadata = self._load_metadata()
		attempts = []
		for history in self.histories():
			latest = history.latest
			if latest is None:
				continue
			path = (
				self.root / "trials" / latest.trial_id.value / "progress"
				/ f"{latest.attempt_id}.json"
			)
			try:
				progress = AttemptProgress.model_validate_json(path.read_text(encoding="utf-8"))
			except (OSError, ValueError):
				progress = self._progress_from_attempt(latest)
			attempts.append(progress)
		attempts.sort(key=lambda item: (item.fidelity_id, item.trial_id, item.attempt_id))
		counts: dict[str, int] = {}
		for progress in attempts:
			counts[progress.state] = counts.get(progress.state, 0) + 1
		states = set(counts)
		if not states:
			state = "planned"
		elif states <= {"succeeded", "reused", "skipped"}:
			state = "succeeded"
		elif "started" in states or "planned" in states:
			state = "running"
		elif states == {"failed"}:
			state = "failed"
		elif "interrupted" in states:
			state = "interrupted" if states <= {"interrupted", "succeeded"} else "mixed"
		else:
			state = "mixed"
		cancellation = None
		cancellation_path = self.root / "cancellation.json"
		if cancellation_path.exists():
			cancellation = CancellationDetails.model_validate_json(
				cancellation_path.read_text(encoding="utf-8")
			)
			if not cancellation.requested:
				cancellation = None
		updated = max((item.updated_at for item in attempts), default=metadata.created_at)
		return ExperimentStatus(
			experiment_id=metadata.experiment_id,
			state=state,
			updated_at=updated,
			attempts=tuple(attempts),
			counts=counts,
			cancellation=cancellation,
		)

	def _comparable_history(self, identity: TrialIdentity) -> TrialHistory | None:
		candidates = [
			history for history in self.histories()
			if history.identity.candidate == identity.candidate
			and history.identity.fidelity.id == identity.fidelity.id
		]
		if not candidates:
			return None
		return max(candidates, key=lambda history: (
			history.latest.created_at if history.latest else datetime.min.replace(tzinfo=timezone.utc),
			history.identity.trial_id.value,
		))

	def decide(self, identity: TrialIdentity, policy: ResumePolicy) -> ResumeDecision:
		attempts = self.load_attempts(identity)
		if policy.force:
			return ResumeDecision(action="execute", reason="forced rerun")
		if attempts and attempts[-1].status == "incompatible":
			if policy.mismatch == "rerun":
				return ResumeDecision(
					action="execute", reason="rerun previously incompatible trial",
					prior_attempt=attempts[-1],
				)
			if policy.mismatch == "fail":
				raise ValueError("trial was previously skipped as identity-incompatible")
			return ResumeDecision(
				action="reuse", reason="reuse recorded incompatibility",
				prior_attempt=attempts[-1],
			)
		for attempt in reversed(attempts):
			if attempt.status == "succeeded":
				return ResumeDecision(
					action="reuse", reason="compatible successful result", prior_attempt=attempt,
				)
		for attempt in reversed(attempts):
			if attempt.status == "failed":
				return ResumeDecision(
					action="execute" if policy.retry_failed else "skip",
					reason="retry failed attempt" if policy.retry_failed else (
						"failed attempt requires retry_failed=True"
					),
					prior_attempt=attempt,
				)
			if attempt.status in ("interrupted", "running", "pending"):
				return ResumeDecision(
					action="execute" if policy.retry_interrupted else "skip",
					reason="retry interrupted attempt" if policy.retry_interrupted else (
						"interrupted attempt requires retry_interrupted=True"
					),
					prior_attempt=attempt,
				)
		comparable = self._comparable_history(identity)
		if comparable is not None and comparable.identity.trial_id != identity.trial_id:
			comparison = compare_trial_identities(identity, comparable.identity)
			fields = ", ".join(item.field for item in comparison.diagnostics)
			if policy.mismatch == "fail":
				raise ValueError(f"incompatible prior trial identity: {fields}")
			return ResumeDecision(
				action="execute" if policy.mismatch == "rerun" else "incompatible",
				reason=(
					"rerun under changed identity" if policy.mismatch == "rerun"
					else "skip incompatible prior identity"
				),
				diagnostics=comparison.diagnostics,
				prior_attempt=comparable.latest,
			)
		return ResumeDecision(action="execute", reason="no prior attempt")

	def _latest_attempt(self, identity: TrialIdentity) -> TrialAttemptRecord | None:
		attempts = self.load_attempts(identity)
		return attempts[-1] if attempts else None

	def allocate(self, trial: PreparedTrial) -> PreparedTrial:
		prior = self._latest_attempt(trial.identity)
		sequence = 0 if prior is None else prior.sequence + 1
		return PreparedTrial(
			selected=trial.selected,
			identity=trial.identity,
			output_root=trial.output_root,
			attempt_id=f"attempt-{trial.identity.trial_id.value[:12]}-{sequence:04d}",
			sequence=sequence,
			previous_attempt_id=None if prior is None else prior.attempt_id,
		)

	def record_nonexecution(
		self,
		trial: PreparedTrial,
		decision: ResumeDecision,
	) -> TrialAttemptRecord:
		allocated = self.allocate(trial)
		finished = _utcnow()
		attempt = TrialAttemptRecord(
			attempt_id=allocated.attempt_id,
			sequence=allocated.sequence,
			previous_attempt_id=allocated.previous_attempt_id,
			trial_id=trial.identity.trial_id,
			identity=trial.identity,
			status="incompatible" if decision.action == "incompatible" else "skipped",
			created_at=finished,
			finished_at=finished,
			failure=FailureRecord(
				type="IdentityMismatch" if decision.action == "incompatible" else "RetryNotRequested",
				message=decision.reason,
				details={
					"diagnostics": [item.model_dump(mode="json") for item in decision.diagnostics],
				},
			),
		)
		atomic_write_model(
			trial.output_root / "attempts" / f"{attempt.attempt_id}.json", attempt,
		)
		self.write_progress(self._progress_from_attempt(attempt))
		return attempt

	def record_cancellation_before_claim(
		self,
		trial: PreparedTrial,
		reason: str,
	) -> TrialAttemptRecord:
		"""Record resumable cancellation without claiming or starting an executor."""

		self.register_trial(trial.identity)
		allocated = self.allocate(trial)
		finished = _utcnow()
		attempt = TrialAttemptRecord(
			attempt_id=allocated.attempt_id,
			sequence=allocated.sequence,
			previous_attempt_id=allocated.previous_attempt_id,
			trial_id=trial.identity.trial_id,
			identity=trial.identity,
			status="interrupted",
			created_at=finished,
			finished_at=finished,
			failure=FailureRecord(type="CancellationRequested", message=reason),
		)
		atomic_write_model(
			trial.output_root / "attempts" / f"{attempt.attempt_id}.json", attempt,
		)
		self.write_progress(self._progress_from_attempt(attempt))
		return attempt

	def _claim_path(self, identity: TrialIdentity) -> Path:
		return self._trial_root(identity) / "claim.json"

	def _recover_abandoned(self, identity: TrialIdentity, policy: ResumePolicy) -> None:
		claim_path = self._claim_path(identity)
		claim = ClaimRecord.model_validate_json(claim_path.read_text(encoding="utf-8"))
		age = (_utcnow() - claim.claimed_at).total_seconds()
		if age <= policy.stale_after_seconds:
			raise RuntimeError(
				f"trial is claimed by {claim.host}:{claim.pid} ({age:.1f}s old)"
			)
		if not policy.recover_stale:
			raise RuntimeError(
				"trial claim is stale; rerun with explicit recover_stale=True"
			)
		self._quarantine(claim_path, "stale-claim")
		latest = self._latest_attempt(identity)
		if latest is not None and latest.status == "running":
			document = latest.model_dump(mode="json")
			document.update({
				"status": "interrupted",
				"finished_at": _utcnow(),
				"failure": FailureRecord(
					type="AbandonedClaim",
					message=f"recovered stale claim from {claim.host}:{claim.pid}",
				).model_dump(mode="json"),
			})
			recovered = TrialAttemptRecord.model_validate(document)
			atomic_write_model(
				self._trial_root(identity) / "attempts" / f"{latest.attempt_id}.json",
				recovered,
			)
		self.quarantine_partial_outputs(identity)

	@contextmanager
	def claim(self, identity: TrialIdentity, policy: ResumePolicy) -> Iterator[None]:
		self.register_trial(identity)
		path = self._claim_path(identity)
		path.parent.mkdir(parents=True, exist_ok=True)
		if path.exists():
			self._recover_abandoned(identity, policy)
		owner = uuid4().hex
		record = ClaimRecord(
			owner=owner, pid=os.getpid(), host=socket.gethostname(), claimed_at=_utcnow(),
		)
		flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
		try:
			descriptor = os.open(path, flags, 0o644)
		except FileExistsError as error:
			raise RuntimeError("trial was claimed concurrently") from error
		with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
			json.dump(record.model_dump(mode="json"), stream, sort_keys=True)
			stream.write("\n")
			stream.flush()
			os.fsync(stream.fileno())
		try:
			yield
		finally:
			try:
				current = ClaimRecord.model_validate_json(path.read_text(encoding="utf-8"))
			except (OSError, ValueError):
				current = None
			if current is not None and current.owner == owner:
				path.unlink()

	def refresh_manifest(
		self,
		config: ExperimentConfig,
		*,
		manifest_metadata: Mapping[str, object] | None = None,
	) -> ExperimentManifest:
		metadata = self._load_metadata()
		attempts = [
			attempt for history in self.histories() for attempt in history.attempts
		]
		attempts.sort(key=lambda item: (item.created_at, item.attempt_id))
		manifest_path = self.root / "manifest.json"
		prior_metadata = {}
		if manifest_path.exists():
			prior_metadata = dict(ExperimentManifest.model_validate_json(
				manifest_path.read_text(encoding="utf-8")
			).metadata)
		if manifest_metadata is not None:
			prior_metadata.update(json_value(manifest_metadata))
		manifest = ExperimentManifest(
			experiment_id=config.experiment_id,
			config=config,
			created_at=metadata.created_at,
			attempt_ids=tuple(item.attempt_id for item in attempts),
			metadata=prior_metadata,
		)
		atomic_write_model(self.root / "manifest.json", manifest)
		return manifest


class PersistedExperimentRunner:
	"""Apply store decisions while retaining the independent runner's execution model."""

	def __init__(
		self,
		runner: IndependentExperimentRunner,
		*,
		policy: ResumePolicy | None = None,
		trial_ids: set[str] | None = None,
		event_writer: ProgressEventWriter | None = None,
		cancellation: CancellationToken | None = None,
		before_execute: Callable[[object], None] | None = None,
	):
		self.runner = runner
		self.store = PersistedExperimentStore(runner.root)
		self.policy = policy or ResumePolicy()
		self.trial_ids = trial_ids
		self.event_writer = event_writer or runner.event_writer
		self.cancellation = cancellation or runner.cancellation
		self.runner.event_writer = self.event_writer
		self.runner.cancellation = self.cancellation
		self.runner.progress_sink = self.store.write_progress
		self.before_execute = before_execute

	def _one(self, trial: PreparedTrial) -> TrialAttemptRecord:
		if self.cancellation.requested:
			attempt = self.store.record_cancellation_before_claim(
				trial, self.cancellation.reason or "cancellation requested",
			)
			if self.event_writer is not None:
				self.event_writer.emit(
					"trial.finalized", "trial interrupted before claim",
					severity="warning", fidelity_id=trial.identity.fidelity.id,
					candidate_key=trial.identity.candidate.value,
					trial_id=trial.identity.trial_id.value,
					attempt_id=attempt.attempt_id, lifecycle_state="interrupted",
				)
			return attempt
		if self.event_writer is not None:
			self.event_writer.emit(
				"trial.claim", "claiming trial",
				fidelity_id=trial.identity.fidelity.id,
				candidate_key=trial.identity.candidate.value,
				trial_id=trial.identity.trial_id.value,
				structural_variant=trial.identity.sampler.explicit_case,
				sampler_index=trial.selected.index,
				lifecycle_state="planned",
			)
		with self.store.claim(trial.identity, self.policy):
			decision = self.store.decide(trial.identity, self.policy)
			if decision.action == "reuse":
				assert decision.prior_attempt is not None
				if self.event_writer is not None:
					self.event_writer.emit(
						"trial.reused", "reused compatible successful trial",
						fidelity_id=trial.identity.fidelity.id,
						candidate_key=trial.identity.candidate.value,
						trial_id=trial.identity.trial_id.value,
						attempt_id=decision.prior_attempt.attempt_id,
						lifecycle_state="reused",
					)
				return decision.prior_attempt
			if decision.action in ("skip", "incompatible"):
				attempt = self.store.record_nonexecution(trial, decision)
				if self.event_writer is not None:
					self.event_writer.emit(
						"trial.skipped", decision.reason,
						severity="warning",
						fidelity_id=trial.identity.fidelity.id,
						candidate_key=trial.identity.candidate.value,
						trial_id=trial.identity.trial_id.value,
						attempt_id=attempt.attempt_id,
						lifecycle_state="skipped",
					)
				return attempt
			return self.runner._execute(self.store.allocate(trial))

	def run(self) -> ExperimentRunSummary:
		preflight, trials = self.runner.prepare()
		if self.before_execute is not None:
			self.before_execute(preflight)
		if self.event_writer is not None:
			self.event_writer.activate()
		relocation = self.store.validate_path_relocation(preflight.path_resources)
		manifest_metadata = {
			"runtime_compatibility": preflight.runtime,
			"path_resource_preflight": preflight.path_resources,
			"path_resource_relocation": relocation,
		}
		self.store.initialize(
			self.runner.config, manifest_metadata=manifest_metadata,
		)
		if preflight.validation_failures:
			raise ValueError(
				"candidate preflight failed: "
				+ "; ".join(item.message for item in preflight.validation_failures)
			)
		if self.trial_ids is not None:
			available = {trial.identity.trial_id.value for trial in trials}
			unknown = sorted(self.trial_ids - available)
			if unknown:
				raise ValueError(f"requested trial IDs are not in the resolved plan: {unknown}")
			trials = tuple(
				trial for trial in trials if trial.identity.trial_id.value in self.trial_ids
			)
		attempts = []
		unstarted = 0
		with ThreadPoolExecutor(max_workers=preflight.max_concurrency) as pool:
			remaining = iter(trials)
			futures: dict[Future, PreparedTrial] = {}
			for _ in range(min(preflight.max_concurrency, len(trials))):
				trial = next(remaining)
				futures[pool.submit(self._one, trial)] = trial
			stop = False
			while futures:
				done, _ = wait(futures, return_when=FIRST_COMPLETED)
				for future in done:
					futures.pop(future)
					attempt = future.result()
					attempts.append(attempt)
					if self.runner.policy.failure == "fail_fast" and attempt.status not in (
						"succeeded", "skipped",
					):
						stop = True
				if self.cancellation.requested:
					stop = True
				while not stop and len(futures) < preflight.max_concurrency:
					try:
						trial = next(remaining)
					except StopIteration:
						break
					futures[pool.submit(self._one, trial)] = trial
			if stop:
				unstarted = sum(1 for _ in remaining)
		attempts.sort(key=lambda item: item.trial_id.value)
		attempts, leaderboard = self.runner._rank(trials, attempts)
		self.store.refresh_manifest(
			self.runner.config, manifest_metadata=manifest_metadata,
		)
		interruption = None
		if self.cancellation.requested:
			interruption = CancellationDetails(
				requested=True,
				reason=self.cancellation.reason,
				requested_at=self.cancellation.requested_at,
				interrupted_attempt_ids=tuple(
					item.attempt_id for item in attempts if item.status == "interrupted"
				),
				unstarted_candidates=unstarted,
			)
			self.store.record_cancellation(interruption)
			if self.event_writer is not None:
				self.event_writer.emit(
					"cancellation.finalized",
					"cancellation finalized after in-flight attempts reached durable states",
					severity="warning", lifecycle_state="interrupted",
					details=interruption.model_dump(mode="json"),
				)
		elif (self.store.root / "cancellation.json").exists():
			self.store.record_cancellation(CancellationDetails())
		summary = ExperimentRunSummary(
			experiment_id=self.runner.config.experiment_id,
			preflight=preflight,
			attempts=tuple(attempts),
			leaderboard=leaderboard,
			interruption=interruption,
		)
		self.runner._write_outputs(summary)
		return summary
