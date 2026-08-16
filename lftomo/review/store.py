"""Separate append-oriented persistence for experiment-review state."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from pathlib import Path
from typing import Callable, Iterator

from lftomo.experiment.events import active_secret_values, secret_material_paths
from lftomo.experiment.models import _sha256
from lftomo.experiment.runner import atomic_write_model, atomic_write_text

from .models import (
	PairwiseOutcome,
	PairwiseVote,
	RankingConfig,
	RankingRecord,
	ReasonTag,
	ReviewExport,
	ReviewProtocol,
	ReviewSession,
	TriageJudgment,
	TriageVerdict,
	UnblindEvent,
)
from .protocol import current_review_protocol
from .ranking import derive_bradley_terry
from .resolution import ExperimentCandidateResolver


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


def results_root_fingerprint(results_root: str | Path, experiment_id: str) -> str:
	"""Bind an external review store to one resolved experiment-store location."""

	return _sha256({
		"experiment_id": experiment_id,
		"results_root": str(Path(results_root).resolve()),
	})


class ReviewStore:
	"""Atomic raw review observations kept outside immutable experiment records."""

	def __init__(self, root: str | Path):
		self.root = Path(root)
		self.session_path = self.root / "session.json"
		self._resolver_cache: ExperimentCandidateResolver | None = None

	@classmethod
	def initialize(
		cls,
		results_root: str | Path,
		*,
		session_id: str,
		reviewer_id: str,
		reviewer_label: str | None = None,
		root: str | Path | None = None,
		protocol: ReviewProtocol | None = None,
		created_at: datetime | None = None,
	) -> "ReviewStore":
		"""Create or reopen a session without writing to experiment-owned paths."""

		resolver = ExperimentCandidateResolver(results_root)
		if resolver.preview_bundle is None:
			raise ValueError(
				"review sessions require the recorded private preview mapping contract"
			)
		resolved_results = Path(results_root).resolve()
		resolved_review = Path(
			root or resolved_results / "review" / session_id
		).resolve()
		if (
			resolved_review.is_relative_to(resolved_results)
			and not resolved_review.is_relative_to(resolved_results / "review")
		):
			raise ValueError(
				"review state inside RESULTS_ROOT must live under its review directory"
			)
		store = cls(resolved_review)
		protocol = protocol or current_review_protocol()
		now = created_at or _utcnow()
		session = ReviewSession(
			session_id=session_id,
			experiment_id=resolver.experiment_id,
			results_root=resolved_results,
			results_root_fingerprint=results_root_fingerprint(
				resolved_results, resolver.experiment_id,
			),
			preview_contract=resolver.preview_bundle.contract,
			reviewer_id=reviewer_id,
			reviewer_label=reviewer_label,
			protocol=protocol,
			created_at=now,
			updated_at=now,
		)
		store.root.mkdir(parents=True, exist_ok=True)
		if store.session_path.exists():
			stored = store.load_session()
			for field in (
				"session_id",
				"experiment_id",
				"results_root_fingerprint",
				"preview_contract",
				"reviewer_id",
				"reviewer_label",
				"protocol",
			):
				if getattr(stored, field) != getattr(session, field):
					raise ValueError(f"review session differs in {field}")
			return store
		store._safe_write(store.session_path, session)
		return store

	def load_session(self) -> ReviewSession:
		try:
			return ReviewSession.model_validate_json(
				self.session_path.read_text(encoding="utf-8")
			)
		except (OSError, ValueError) as error:
			raise ValueError(f"invalid review session {self.session_path}: {error}") from error

	def resolver(self) -> ExperimentCandidateResolver:
		"""Open the session's read-only experiment candidate resolver."""

		if self._resolver_cache is not None:
			return self._resolver_cache
		session = self.load_session()
		fingerprint = results_root_fingerprint(
			session.results_root, session.experiment_id,
		)
		if fingerprint != session.results_root_fingerprint:
			raise ValueError("review session results-root fingerprint is invalid")
		resolver = ExperimentCandidateResolver(session.results_root)
		if resolver.experiment_id != session.experiment_id:
			raise ValueError("review session points to a different experiment")
		if (
			resolver.preview_bundle is None
			or resolver.preview_bundle.contract != session.preview_contract
		):
			raise ValueError("recorded preview contract has drifted since review creation")
		self._resolver_cache = resolver
		return self._resolver_cache

	def _safe_write(self, path: Path, value) -> None:
		unsafe = secret_material_paths(value, secrets=active_secret_values())
		if unsafe:
			raise ValueError(
				"secret-bearing fields cannot be persisted in review records: "
				+ ", ".join(unsafe)
			)
		atomic_write_model(path, value)

	def _touch_session(self) -> None:
		session = self.load_session()
		self._safe_write(self.session_path, session.model_copy(update={
			"updated_at": max(session.updated_at, _utcnow()),
		}))

	@contextmanager
	def _locked(self) -> Iterator[None]:
		self.root.mkdir(parents=True, exist_ok=True)
		with (self.root / ".lock").open("a+", encoding="utf-8") as lock:
			fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
			try:
				yield
			finally:
				fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

	def _records(self, directory: str, model):
		root = self.root / directory
		if not root.exists():
			return ()
		records = []
		for path in sorted(root.glob("*.json")):
			try:
				records.append(model.model_validate_json(path.read_text(encoding="utf-8")))
			except (OSError, ValueError) as error:
				raise ValueError(f"invalid review record {path}: {error}") from error
		return tuple(sorted(records, key=lambda item: item.sequence))

	def triage_judgments(self) -> tuple[TriageJudgment, ...]:
		return self._records("triage", TriageJudgment)

	def pairwise_votes(self) -> tuple[PairwiseVote, ...]:
		return self._records("pairwise", PairwiseVote)

	def unblind_events(self) -> tuple[UnblindEvent, ...]:
		return self._records("unblind", UnblindEvent)

	def rankings(self) -> tuple[RankingRecord, ...]:
		root = self.root / "rankings"
		if not root.exists():
			return ()
		records = tuple(
			RankingRecord.model_validate_json(path.read_text(encoding="utf-8"))
			for path in sorted(root.glob("*.json"))
		)
		return records

	def latest_triage(self, pass_number: int) -> dict[str, TriageJudgment]:
		"""Reconstruct resumable current state without rewriting raw judgments."""

		state = {}
		for judgment in self.triage_judgments():
			if judgment.pass_number == pass_number:
				state[judgment.blind_id] = judgment
		return state

	def _next_sequence(self) -> int:
		sequences = [
			item.sequence
			for items in (
				self.triage_judgments(), self.pairwise_votes(), self.unblind_events(),
			)
			for item in items
		]
		return max(sequences, default=-1) + 1

	def _known_blind_ids(self) -> set[str]:
		return {
			item.blind_id for item in self.resolver().inventory()
			if item.blind_id is not None
		}

	def _validate_blind_ids(self, blind_ids: tuple[str, ...]) -> None:
		unknown = sorted(set(blind_ids) - self._known_blind_ids())
		if unknown:
			raise ValueError(f"review records reference unknown blind IDs: {unknown}")

	def _append(
		self,
		directory: str,
		kind: str,
		build: Callable[[int, ReviewSession], object],
	):
		with self._locked():
			session = self.load_session()
			sequence = self._next_sequence()
			record = build(sequence, session)
			path = self.root / directory / f"{sequence:012d}-{kind}.json"
			self._safe_write(path, record)
			self._touch_session()
			return record

	def record_triage(
		self,
		blind_id: str,
		*,
		pass_number: int,
		verdict: TriageVerdict,
		reason_tags: tuple[ReasonTag, ...] = (),
		timestamp: datetime | None = None,
	) -> TriageJudgment:
		"""Atomically append one pass-1 or pass-2 judgment."""

		self._validate_blind_ids((blind_id,))
		return self._append("triage", "triage", lambda sequence, session: TriageJudgment(
			protocol_version=session.protocol.protocol_version,
			pass_number=pass_number,
			blind_id=blind_id,
			verdict=verdict,
			reason_tags=reason_tags,
			timestamp=timestamp or _utcnow(),
			sequence=sequence,
		))

	def record_pairwise(
		self,
		left_blind_id: str,
		right_blind_id: str,
		*,
		outcome: PairwiseOutcome,
		timestamp: datetime | None = None,
	) -> PairwiseVote:
		"""Atomically append one comparison in its displayed left/right order."""

		self._validate_blind_ids((left_blind_id, right_blind_id))
		return self._append("pairwise", "vote", lambda sequence, session: PairwiseVote(
			protocol_version=session.protocol.protocol_version,
			left_blind_id=left_blind_id,
			right_blind_id=right_blind_id,
			outcome=outcome,
			timestamp=timestamp or _utcnow(),
			sequence=sequence,
		))

	def record_unblind(
		self,
		*,
		scope: str,
		blind_ids: tuple[str, ...] = (),
		reviewer_id: str | None = None,
		timestamp: datetime | None = None,
	) -> UnblindEvent:
		"""Record the explicit boundary crossing before exposing candidate data."""

		self._validate_blind_ids(blind_ids)
		return self._append("unblind", "unblind", lambda sequence, session: UnblindEvent(
			protocol_version=session.protocol.protocol_version,
			reviewer_id=reviewer_id or session.reviewer_id,
			scope=scope,
			blind_ids=blind_ids,
			timestamp=timestamp or _utcnow(),
			sequence=sequence,
		))

	def require_unblinded(self, blind_ids: tuple[str, ...]) -> None:
		"""Reject an unblinded export without a covering explicit audit event."""

		events = self.unblind_events()
		if any(event.scope == "session" for event in events):
			return
		authorized = {
			blind_id
			for event in events
			if event.scope in ("candidate", "export")
			for blind_id in event.blind_ids
		}
		missing = sorted(set(blind_ids) - authorized)
		if missing:
			raise ValueError(
				"explicit unblind event required for blind IDs: " + ", ".join(missing)
			)

	def derive_ranking(
		self,
		candidates: tuple[str, ...],
		config: RankingConfig | None = None,
		*,
		derived_at: datetime | None = None,
	) -> RankingRecord:
		"""Derive and append a ranking record without changing any raw vote."""

		self._validate_blind_ids(candidates)
		session = self.load_session()
		record = derive_bradley_terry(
			self.pairwise_votes(),
			candidates,
			config,
			protocol=session.protocol,
			derived_at=derived_at,
		)
		with self._locked():
			index = len(self.rankings())
			self._safe_write(
				self.root / "rankings"
				/ f"ranking-{index:06d}-{record.ranking_id[:12]}.json",
				record,
			)
			self._touch_session()
		return record

	def persist_export(
		self,
		record: ReviewExport,
		report: str,
	) -> tuple[Path, Path]:
		"""Append one machine-readable export and its concise Markdown report."""

		with self._locked():
			root = self.root / "exports"
			index = len(tuple(root.glob("export-*.json"))) if root.exists() else 0
			json_path = root / f"export-{index:06d}.json"
			report_path = root / f"export-{index:06d}.md"
			unsafe = secret_material_paths(report, secrets=active_secret_values())
			if unsafe:
				raise ValueError("secret material cannot be persisted in review reports")
			self._safe_write(json_path, record)
			atomic_write_text(report_path, report.rstrip() + "\n")
			self._touch_session()
		return json_path, report_path
