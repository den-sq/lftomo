"""Concurrency-safe runtime accounting and atomic experiment output publication."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Iterator, Literal
from uuid import uuid4

from pydantic import Field

from lftomo.config import ConfigModel

from .models import (
	ExperimentConfig,
	OutputAccounting,
	OutputLimits,
	OutputRecord,
	PendingOutput,
	TrialAttemptRecord,
)


COUNTED_OUTPUT_ROLES = frozenset({
	"finalist_output",
	"screening_output",
	"blinded_preview",
})


def output_role_is_counted(role: str) -> bool:
	"""Return whether an artifact consumes experiment output budgets."""

	return role in COUNTED_OUTPUT_ROLES


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _atomic_write(path: Path, value: ConfigModel) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
	try:
		with temporary.open("x", encoding="utf-8") as stream:
			json.dump(value.model_dump(mode="json"), stream, sort_keys=True, indent=2)
			stream.write("\n")
			stream.flush()
			os.fsync(stream.fileno())
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


class OutputLimitExceeded(ValueError):
	"""A serialized publication would exceed one configured output budget."""

	def __init__(
		self,
		*,
		scope: Literal["experiment", "fidelity", "policy"],
		limit_bytes: int,
		committed_bytes: int,
		requested_bytes: int,
		artifact_label: str = "attempt",
	):
		self.scope = scope
		self.limit_bytes = limit_bytes
		self.committed_bytes = committed_bytes
		self.requested_bytes = requested_bytes
		super().__init__(
			f"{artifact_label} output would exceed configured {scope} limit "
			f"{limit_bytes} bytes: "
			f"{committed_bytes} committed + {requested_bytes} requested"
		)

	def failure_details(self) -> dict[str, int | str]:
		return {
			"scope": self.scope,
			"limit_bytes": self.limit_bytes,
			"committed_bytes": self.committed_bytes,
			"requested_bytes": self.requested_bytes,
		}


class OutputLedgerIntegrityError(RuntimeError):
	"""Durable output files no longer match their authoritative accounting."""


class OutputLedgerEntry(ConfigModel):
	attempt_id: str
	trial_id: str
	fidelity_id: str
	role: str
	path: Path
	bytes: int = Field(ge=0)
	sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OutputLedgerState(ConfigModel):
	schema_version: Literal[1] = 1
	updated_at: datetime
	entries: tuple[OutputLedgerEntry, ...] = ()


class OutputPublicationTransaction(ConfigModel):
	schema_version: Literal[1] = 1
	transaction_id: str
	attempt_id: str
	trial_id: str
	fidelity_id: str
	created_at: datetime
	published: bool = False
	new_entries: tuple[OutputLedgerEntry, ...]
	prior_entries: tuple[OutputLedgerEntry, ...] = ()
	backup_paths: dict[str, Path] = Field(default_factory=dict)


class OutputBudgetLedger:
	"""Serialize output replacement, budget decisions, and durable accounting."""

	def __init__(
		self,
		root: str | Path,
		config: ExperimentConfig,
		fidelity_id: str,
		*,
		override_limits: bool = False,
	):
		self.root = Path(root).resolve(strict=False)
		self.config = config
		self.fidelity_id = fidelity_id
		self.override_limits = override_limits
		self.path = self.root / "output-ledger.json"
		self.lock_path = self.root / ".output-ledger.lock"
		self.transactions = self.root / ".output-transactions"
		self._thread_lock = threading.RLock()

	def limits(self) -> OutputLimits:
		fidelity = next(
			item for item in self.config.fidelities if item.id == self.fidelity_id
		)
		return OutputLimits(
			experiment_bytes=self.config.resources.max_output_bytes,
			fidelity_bytes=fidelity.max_output_bytes,
			policy_bytes=self.config.output.max_bytes,
			override_limits=self.override_limits,
		)

	@contextmanager
	def _locked(self) -> Iterator[None]:
		self.root.mkdir(parents=True, exist_ok=True)
		with self._thread_lock, self.lock_path.open("a+", encoding="utf-8") as stream:
			fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
			try:
				yield
			finally:
				fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

	def _empty(self) -> OutputLedgerState:
		return OutputLedgerState(updated_at=_utcnow())

	def _load(self) -> OutputLedgerState:
		if not self.path.exists():
			return self._empty()
		try:
			return OutputLedgerState.model_validate_json(
				self.path.read_text(encoding="utf-8")
			)
		except (OSError, ValueError) as error:
			raise OutputLedgerIntegrityError(f"invalid output ledger: {error}") from error

	def _write(self, entries: list[OutputLedgerEntry]) -> OutputLedgerState:
		state = OutputLedgerState(
			updated_at=_utcnow(),
			entries=tuple(sorted(entries, key=lambda item: str(item.path))),
		)
		_atomic_write(self.path, state)
		return state

	def _relative(self, path: Path) -> Path:
		resolved = path.resolve(strict=False)
		try:
			return resolved.relative_to(self.root)
		except ValueError as error:
			raise ValueError(f"output path must remain inside result root: {path}") from error

	def _absolute(self, path: Path) -> Path:
		return self.root / path

	@staticmethod
	def _usage(
		entries: list[OutputLedgerEntry] | tuple[OutputLedgerEntry, ...],
		fidelity_id: str,
	) -> tuple[int, int, int]:
		counted = [item for item in entries if output_role_is_counted(item.role)]
		experiment = sum(item.bytes for item in counted)
		fidelity = sum(item.bytes for item in counted if item.fidelity_id == fidelity_id)
		# output.max_bytes covers the same configured output roles across the invocation.
		policy = experiment
		return experiment, fidelity, policy

	def _accounting(
		self,
		entries: list[OutputLedgerEntry] | tuple[OutputLedgerEntry, ...],
		*,
		attempt_bytes: int,
	) -> OutputAccounting:
		experiment, fidelity, policy = self._usage(entries, self.fidelity_id)
		return OutputAccounting(
			limits=self.limits(),
			experiment_committed_bytes=experiment,
			fidelity_committed_bytes=fidelity,
			policy_committed_bytes=policy,
			attempt_committed_bytes=attempt_bytes,
		)

	def _enforce(
		self,
		prior: list[OutputLedgerEntry],
		proposed: list[OutputLedgerEntry],
	) -> None:
		limits = self.limits()
		if limits.override_limits:
			return
		before = self._usage(prior, self.fidelity_id)
		after = self._usage(proposed, self.fidelity_id)
		for index, (scope, limit) in enumerate((
			("experiment", limits.experiment_bytes),
			("fidelity", limits.fidelity_bytes),
			("policy", limits.policy_bytes),
		)):
			if limit is not None and after[index] > limit:
					raise OutputLimitExceeded(
					scope=scope,
					limit_bytes=limit,
					committed_bytes=before[index],
					requested_bytes=max(0, after[index] - before[index]),
				)

	def _transaction_path(self, transaction_id: str) -> Path:
		return self.transactions / transaction_id / "transaction.json"

	def _read_transaction(self, path: Path) -> OutputPublicationTransaction:
		try:
			return OutputPublicationTransaction.model_validate_json(
				path.read_text(encoding="utf-8")
			)
		except (OSError, ValueError) as error:
			raise OutputLedgerIntegrityError(
				f"invalid output publication transaction {path}: {error}"
			) from error

	def _restore_transaction(
		self,
		transaction: OutputPublicationTransaction,
		state: OutputLedgerState,
	) -> None:
		new_paths = {item.path for item in transaction.new_entries}
		entries = [item for item in state.entries if item.path not in new_paths]
		entries.extend(transaction.prior_entries)
		for entry in transaction.new_entries:
			target = self._absolute(entry.path)
			backup = transaction.backup_paths.get(str(entry.path))
			backup_path = None if backup is None else self._absolute(backup)
			if backup_path is not None and backup_path.exists():
				if target.exists():
					target.unlink()
				target.parent.mkdir(parents=True, exist_ok=True)
				os.replace(backup_path, target)
			elif target.exists() and _sha256(target) == entry.sha256:
				target.unlink()
		self._write(entries)
		shutil.rmtree(self._transaction_path(transaction.transaction_id).parent, ignore_errors=True)

	def _finalize_transaction(self, transaction: OutputPublicationTransaction) -> None:
		shutil.rmtree(self._transaction_path(transaction.transaction_id).parent, ignore_errors=True)

	def publish(
		self,
		pending: tuple[PendingOutput, ...],
		*,
		attempt_id: str,
		trial_id: str,
	) -> tuple[tuple[OutputRecord, ...], OutputAccounting]:
		"""Measure and publish one attempt's complete output set under one lock."""

		if not pending:
			with self._locked():
				state = self._load()
				return (), self._accounting(state.entries, attempt_bytes=0)
		with self._locked():
			state = self._load()
			prepared: list[tuple[PendingOutput, Path, int, str]] = []
			seen: set[Path] = set()
			for item in pending:
				if not item.temporary_path.is_file():
					raise FileNotFoundError(
						f"pending output does not exist: {item.temporary_path}"
					)
				target = self.root / item.relative_path
				relative = self._relative(target)
				if relative in seen:
					raise ValueError(f"duplicate pending output target: {relative}")
				seen.add(relative)
				prepared.append((
					item, relative, item.temporary_path.stat().st_size,
					_sha256(item.temporary_path),
				))

			prior_entries = [item for item in state.entries if item.path in seen]
			proposed = [item for item in state.entries if item.path not in seen]
			new_entries = [OutputLedgerEntry(
				attempt_id=attempt_id,
				trial_id=trial_id,
				fidelity_id=self.fidelity_id,
				role=item.role,
				path=relative,
				bytes=size,
				sha256=digest,
			) for item, relative, size, digest in prepared]
			proposed.extend(new_entries)
			attempt_bytes = sum(
				item.bytes for item in new_entries if output_role_is_counted(item.role)
			)
			self._enforce(list(state.entries), proposed)

			transaction_id = f"{attempt_id}-{uuid4().hex}"
			transaction_root = self.transactions / transaction_id
			new_root = transaction_root / "new"
			backup_root = transaction_root / "backup"
			backup_paths: dict[str, Path] = {}
			new_paths: dict[Path, Path] = {}
			for index, (item, relative, _size, _digest) in enumerate(prepared):
				copy = new_root / f"{index:08d}"
				copy.parent.mkdir(parents=True, exist_ok=True)
				with item.temporary_path.open("rb") as source, copy.open("xb") as destination:
					shutil.copyfileobj(source, destination)
					destination.flush()
					os.fsync(destination.fileno())
				new_paths[relative] = copy
				target = self._absolute(relative)
				if target.exists():
					backup = backup_root / f"{index:08d}"
					backup_paths[str(relative)] = self._relative(backup)
			transaction = OutputPublicationTransaction(
				transaction_id=transaction_id,
				attempt_id=attempt_id,
				trial_id=trial_id,
				fidelity_id=self.fidelity_id,
				created_at=_utcnow(),
				new_entries=tuple(new_entries),
				prior_entries=tuple(prior_entries),
				backup_paths=backup_paths,
			)
			_atomic_write(self._transaction_path(transaction_id), transaction)
			try:
				for entry in new_entries:
					target = self._absolute(entry.path)
					target.parent.mkdir(parents=True, exist_ok=True)
					backup = transaction.backup_paths.get(str(entry.path))
					if backup is not None:
						backup_path = self._absolute(backup)
						backup_path.parent.mkdir(parents=True, exist_ok=True)
						os.replace(target, backup_path)
					os.replace(new_paths[entry.path], target)
				self._write(proposed)
				transaction = transaction.model_copy(update={"published": True})
				_atomic_write(self._transaction_path(transaction_id), transaction)
			except BaseException:
				self._restore_transaction(transaction, self._load())
				raise

			records = tuple(OutputRecord(
				role=item.role,
				path=self._absolute(item.path),
				sha256=item.sha256,
				bytes=item.bytes,
			) for item in new_entries)
			return records, self._accounting(proposed, attempt_bytes=attempt_bytes)

	def finalize_attempt(self, attempt_id: str) -> None:
		"""Remove replacement backups after the successful attempt record is durable."""

		with self._locked():
			if not self.transactions.exists():
				return
			for path in self.transactions.glob("*/transaction.json"):
				transaction = self._read_transaction(path)
				if transaction.attempt_id == attempt_id:
					self._finalize_transaction(transaction)

	def rollback_attempt(self, attempt_id: str) -> None:
		"""Remove this attempt's publications and restore any replaced artifacts."""

		with self._locked():
			if not self.transactions.exists():
				return
			for path in tuple(self.transactions.glob("*/transaction.json")):
				transaction = self._read_transaction(path)
				if transaction.attempt_id == attempt_id:
					self._restore_transaction(transaction, self._load())

	def _successful_entries(self) -> list[OutputLedgerEntry]:
		latest: dict[str, TrialAttemptRecord] = {}
		for path in self.root.glob("trials/*/attempts/*.json"):
			try:
				attempt = TrialAttemptRecord.model_validate_json(path.read_text(encoding="utf-8"))
			except (OSError, ValueError):
				continue
			if attempt.status != "succeeded" or attempt.result is None:
				continue
			prior = latest.get(attempt.trial_id.value)
			if prior is None or attempt.sequence > prior.sequence:
				latest[attempt.trial_id.value] = attempt
		entries = []
		for attempt in latest.values():
			assert attempt.result is not None
			for output in attempt.result.outputs:
				try:
					relative = self._relative(output.path)
				except ValueError:
					continue
				if not output.path.is_file():
					raise OutputLedgerIntegrityError(
						f"recorded output is missing: {output.path}"
					)
				size = output.path.stat().st_size
				digest = _sha256(output.path)
				if output.bytes is not None and output.bytes != size:
					raise OutputLedgerIntegrityError(
						f"recorded output size changed: {output.path}"
					)
				if output.sha256 is not None and output.sha256 != digest:
					raise OutputLedgerIntegrityError(
						f"recorded output content changed: {output.path}"
					)
				entries.append(OutputLedgerEntry(
					attempt_id=attempt.attempt_id,
					trial_id=attempt.trial_id.value,
					fidelity_id=attempt.identity.fidelity.id,
					role=output.role,
					path=relative,
					bytes=size,
					sha256=digest,
				))
		return entries

	def reconcile(self) -> OutputAccounting:
		"""Recover interrupted transactions and rebuild accounting from successes."""

		with self._locked():
			state = self._load()
			if self.transactions.exists():
				for path in tuple(self.transactions.glob("*/transaction.json")):
					transaction = self._read_transaction(path)
					attempt_path = (
						self.root / "trials" / transaction.trial_id / "attempts"
						/ f"{transaction.attempt_id}.json"
					)
					try:
						attempt = TrialAttemptRecord.model_validate_json(
							attempt_path.read_text(encoding="utf-8")
						)
					except (OSError, ValueError):
						attempt = None
					if attempt is not None and attempt.status == "succeeded":
						self._finalize_transaction(transaction)
					else:
						self._restore_transaction(transaction, self._load())
				state = self._load()

			entries = self._successful_entries()
			recorded = {item.path for item in entries}
			published = self.root / "published"
			if published.exists():
				for path in sorted(item for item in published.rglob("*") if item.is_file()):
					relative = self._relative(path)
					if relative in recorded:
						continue
					quarantine = (
						self.root / "quarantine" / "output-ledger"
						/ f"unrecorded-{uuid4().hex}" / relative
					)
					quarantine.parent.mkdir(parents=True, exist_ok=True)
					os.replace(path, quarantine)
			state = self._write(entries)
			return self._accounting(state.entries, attempt_bytes=0)

	def snapshot(self) -> OutputAccounting:
		with self._locked():
			return self._accounting(self._load().entries, attempt_bytes=0)

	def accounting_for_attempt(self, attempt_id: str) -> OutputAccounting:
		"""Return current totals with this attempt's counted contribution."""

		with self._locked():
			state = self._load()
			attempt_bytes = sum(
				item.bytes for item in state.entries
				if item.attempt_id == attempt_id and output_role_is_counted(item.role)
			)
			return self._accounting(state.entries, attempt_bytes=attempt_bytes)
