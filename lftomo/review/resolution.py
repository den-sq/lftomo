"""Read-only resolution of blinded preview, candidate, and trial identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

from lftomo.experiment.fidelity import PreviewBundle, PreviewMappingEntry
from lftomo.experiment.models import ExperimentManifest, TrialAttemptRecord, canonical_json
from lftomo.experiment.selection import select_candidates
from lftomo.experiment.store import PersistedExperimentStore, TrialHistory

from .models import (
	CandidateInventoryEntry,
	CandidateReport,
	CandidateVariation,
	ResolvedAttempt,
	ResolvedTrial,
	StageChainEntry,
)


class CandidateResolutionError(ValueError):
	"""An identifier was missing or ambiguous across public lookup namespaces."""


@dataclass(frozen=True)
class _IdentifierMatch:
	namespace: str
	value: str
	candidate_key: str


def _blind_id(blinding_key: str, candidate_key: str) -> str:
	return hashlib.sha256(
		f"{blinding_key}:{candidate_key}".encode("utf-8")
	).hexdigest()[:16]


def _algorithm_chain(history: TrialHistory) -> tuple[StageChainEntry, ...]:
	return tuple(
		StageChainEntry(
			position=position,
			stage_id=stage.id,
			kind=stage.kind,
			algorithm=stage.algorithm,
		)
		for position, stage in enumerate(history.identity.resolved_recon.pipeline, start=1)
	)


def _resolved_attempt(attempt: TrialAttemptRecord) -> ResolvedAttempt:
	result = attempt.result
	return ResolvedAttempt(
		attempt_id=attempt.attempt_id,
		sequence=attempt.sequence,
		status=attempt.status,
		created_at=attempt.created_at,
		started_at=attempt.started_at,
		finished_at=attempt.finished_at,
		elapsed_seconds=attempt.elapsed_seconds,
		execution_elapsed_seconds=attempt.execution_elapsed_seconds,
		peak_memory_bytes=attempt.peak_memory_bytes,
		metrics={} if result is None else result.metrics,
		observations=attempt.observations,
		pareto=attempt.pareto if result is None else result.pareto,
		promotion=None if result is None else result.promotion,
		outputs=() if result is None else result.outputs,
		failure=attempt.failure,
	)


class ExperimentCandidateResolver:
	"""Resolve candidate reports without mutating the experiment output store."""

	def __init__(self, results_root: str | Path):
		self.results_root = Path(results_root).resolve()
		inspection = PersistedExperimentStore(self.results_root).inspect(
			quarantine_invalid=False,
		)
		manifest_path = self.results_root / "manifest.json"
		self.manifest = ExperimentManifest.model_validate_json(
			manifest_path.read_text(encoding="utf-8")
		)
		if self.manifest.experiment_id != inspection.metadata.experiment_id:
			raise ValueError("manifest experiment_id differs from store metadata")
		self._histories = inspection.trials
		self._by_candidate: dict[str, list[TrialHistory]] = {}
		for history in self._histories:
			self._by_candidate.setdefault(
				history.identity.candidate.value, [],
			).append(history)
		self._selected = {
			candidate.candidate_key: candidate
			for candidate in select_candidates(self.manifest.config)
		}
		self.preview_bundle, self._previews = self._load_previews()

	def _load_previews(
		self,
	) -> tuple[PreviewBundle | None, dict[str, PreviewMappingEntry]]:
		mapping_path = self.results_root / "private" / "preview-mapping" / "mapping.json"
		entries: dict[str, PreviewMappingEntry] = {}
		bundle = None
		try:
			bundle = PreviewBundle.model_validate_json(
				mapping_path.read_text(encoding="utf-8")
			)
		except FileNotFoundError:
			pass
		except PermissionError:
			pass
		if bundle is not None:
			entries.update({entry.candidate_key: entry for entry in bundle.entries})

		key = os.environ.get("LFTOMO_PREVIEW_BLINDING_KEY")
		if key:
			for candidate_key in self._by_candidate:
				if candidate_key in entries:
					continue
				blind_id = _blind_id(key, candidate_key)
				files = tuple(sorted(
					path for path in (self.results_root / "previews" / blind_id).glob("*")
					if path.is_file()
				))
				entries[candidate_key] = PreviewMappingEntry(
					blind_id=blind_id,
					candidate_key=candidate_key,
					files=files,
				)
		return bundle, entries

	@property
	def experiment_id(self) -> str:
		return self.manifest.experiment_id

	def _identifier_matches(self) -> tuple[_IdentifierMatch, ...]:
		matches = []
		for candidate_key, histories in self._by_candidate.items():
			preview = self._previews.get(candidate_key)
			if preview is not None:
				matches.append(_IdentifierMatch(
					"blind_id", preview.blind_id, candidate_key,
				))
			matches.append(_IdentifierMatch(
				"candidate_key", candidate_key, candidate_key,
			))
			matches.extend(
				_IdentifierMatch(
					"trial_id", history.identity.trial_id.value, candidate_key,
				)
				for history in histories
			)
		return tuple(matches)

	def _match(self, identifier: str) -> _IdentifierMatch:
		all_matches = self._identifier_matches()
		exact = [item for item in all_matches if item.value == identifier]
		matches = exact or [item for item in all_matches if item.value.startswith(identifier)]
		unique = {
			(item.namespace, item.value, item.candidate_key): item
			for item in matches
		}
		matches = sorted(unique.values(), key=lambda item: (
			item.namespace, item.value, item.candidate_key,
		))
		if not matches:
			raise CandidateResolutionError(f"no candidate matches identifier {identifier!r}")
		if len(matches) != 1:
			details = ", ".join(
				f"{item.namespace}:{item.value} -> {item.candidate_key}"
				for item in matches
			)
			raise CandidateResolutionError(
				f"ambiguous identifier {identifier!r}; matches: {details}"
			)
		return matches[0]

	def _variations(self, candidate_key: str) -> tuple[CandidateVariation, ...]:
		selected = self._selected.get(candidate_key)
		parameters = {} if selected is None else selected.parameters
		variations = {}
		for history in self._by_candidate[candidate_key]:
			sampler = history.identity.sampler
			variation = CandidateVariation(
				sampler=sampler.name,
				seed=sampler.seed,
				parameters=parameters or sampler.coordinate,
				explicit_case=sampler.explicit_case,
			)
			variations[canonical_json(variation)] = variation
		return tuple(variations[key] for key in sorted(variations))

	def resolve(self, identifier: str) -> CandidateReport:
		"""Resolve an exact or unambiguous prefix into a complete candidate report."""

		match = self._match(identifier)
		histories = sorted(
			self._by_candidate[match.candidate_key],
			key=lambda item: (item.identity.fidelity.id, item.identity.trial_id.value),
		)
		first = histories[0]
		trials = tuple(
			ResolvedTrial(
				trial_id=history.identity.trial_id.value,
				fidelity_id=history.identity.fidelity.id,
				attempt_ids=tuple(attempt.attempt_id for attempt in history.attempts),
				latest_attempt=(
					None if history.latest is None else _resolved_attempt(history.latest)
				),
			)
			for history in histories
		)
		preview = self._previews.get(match.candidate_key)
		return CandidateReport(
			matched_namespace=match.namespace,
			matched_id=match.value,
			blind_id=None if preview is None else preview.blind_id,
			candidate_key=match.candidate_key,
			algorithm_chain=_algorithm_chain(first),
			variations=self._variations(match.candidate_key),
			resolved_recon=first.identity.resolved_recon,
			trials=trials,
			preview_paths=() if preview is None else preview.files,
		)

	def inventory(self) -> tuple[CandidateInventoryEntry, ...]:
		"""Return a compact deterministic inventory of all experiment candidates."""

		items = []
		for candidate_key in sorted(self._by_candidate):
			histories = sorted(
				self._by_candidate[candidate_key],
				key=lambda item: (item.identity.fidelity.id, item.identity.trial_id.value),
			)
			latest = [history.latest for history in histories if history.latest is not None]
			ranks = [
				(attempt.result.pareto.rank if attempt.result else attempt.pareto.rank)
				for attempt in latest
			]
			ranks = [rank for rank in ranks if rank is not None]
			preview = self._previews.get(candidate_key)
			chain = _algorithm_chain(histories[0])
			items.append(CandidateInventoryEntry(
				blind_id=None if preview is None else preview.blind_id,
				candidate_key=candidate_key,
				trial_ids=tuple(history.identity.trial_id.value for history in histories),
				fidelity_ids=tuple(history.identity.fidelity.id for history in histories),
				latest_statuses=tuple(attempt.status for attempt in latest),
				best_rank=min(ranks) if ranks else None,
				promoted=any(
					attempt.result is not None and attempt.result.promotion is not None
					for attempt in latest
				),
				algorithm_chain=tuple(
					f"{item.kind}.{item.algorithm}" if item.algorithm else item.kind
					for item in chain
				),
			))
		return tuple(items)
