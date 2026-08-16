"""Anchor, negative-control, and confirmation-shortlist review exports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .models import (
	ConfirmationCandidate,
	ReviewExport,
	ReviewExportSelection,
	ReviewedCandidate,
)
from .store import ReviewStore


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


def _selected_blind_ids(selection: ReviewExportSelection) -> tuple[str, ...]:
	return tuple(dict.fromkeys((
		*selection.positive_anchors,
		*(
			blind_id
			for kind in sorted(selection.negative_controls)
			for blind_id in selection.negative_controls[kind]
		),
		*selection.confirmation_shortlist,
	)))


def build_review_export(
	store: ReviewStore,
	selection: ReviewExportSelection,
	*,
	exported_at: datetime | None = None,
) -> ReviewExport:
	"""Build an unblinded export after verifying the explicit audit boundary."""

	blind_ids = _selected_blind_ids(selection)
	store.require_unblinded(blind_ids)
	resolver = store.resolver()
	triage = store.triage_judgments()
	votes = store.pairwise_votes()
	rankings = store.rankings()
	ranking = rankings[-1] if rankings else None
	ranks = {} if ranking is None else {
		entry.blind_id: entry.rank for entry in ranking.entries
	}
	reports = {blind_id: resolver.resolve(blind_id) for blind_id in blind_ids}

	def reviewed(blind_id: str) -> ReviewedCandidate:
		report = reports[blind_id]
		return ReviewedCandidate(
			blind_id=blind_id,
			candidate_key=report.candidate_key,
			trial_ids=tuple(trial.trial_id for trial in report.trials),
			triage_judgments=tuple(
				item for item in triage if item.blind_id == blind_id
			),
			pairwise_votes=tuple(
				item for item in votes
				if blind_id in (item.left_blind_id, item.right_blind_id)
			),
			ranking_position=ranks.get(blind_id),
			resolved_recon=report.resolved_recon,
		)

	positive = tuple(reviewed(blind_id) for blind_id in selection.positive_anchors)
	negative = {
		kind: tuple(reviewed(blind_id) for blind_id in selection.negative_controls[kind])
		for kind in sorted(selection.negative_controls)
	}
	shortlist = []
	seen_candidates = set()
	for blind_id in selection.confirmation_shortlist:
		report = reports[blind_id]
		if report.candidate_key in seen_candidates:
			continue
		seen_candidates.add(report.candidate_key)
		shortlist.append(ConfirmationCandidate(
			candidate_key=report.candidate_key,
			resolved_recon=report.resolved_recon,
		))
	session = store.load_session()
	return ReviewExport(
		session_id=session.session_id,
		experiment_id=session.experiment_id,
		exported_at=exported_at or _utcnow(),
		ranking_id=None if ranking is None else ranking.ranking_id,
		positive_anchors=positive,
		negative_controls=negative,
		confirmation_shortlist=tuple(shortlist),
	)


def format_review_report(record: ReviewExport) -> str:
	"""Render a concise human-readable companion to a review JSON artifact."""

	lines = [
		"# Experiment review export",
		"",
		f"- Session: `{record.session_id}`",
		f"- Experiment: `{record.experiment_id}`",
		f"- Ranking: `{record.ranking_id or 'not derived'}`",
		"",
		"## Positive anchors",
		"",
	]
	if record.positive_anchors:
		for item in record.positive_anchors:
			lines.append(
				f"- rank {item.ranking_position or 'unranked'}: `{item.blind_id}` -> "
				f"`{item.candidate_key}`"
			)
	else:
		lines.append("- None selected")
	lines.extend(["", "## Negative controls", ""])
	if record.negative_controls:
		for kind, items in record.negative_controls.items():
			lines.append(f"### {kind.replace('_', ' ')}")
			lines.append("")
			for item in items:
				lines.append(
					f"- rank {item.ranking_position or 'unranked'}: `{item.blind_id}` -> "
					f"`{item.candidate_key}`"
				)
			lines.append("")
	else:
		lines.extend(["- None selected", ""])
	lines.extend(["## Confirmation shortlist", ""])
	if record.confirmation_shortlist:
		for item in record.confirmation_shortlist:
			lines.append(f"- `{item.candidate_key}`")
	else:
		lines.append("- None selected")
	return "\n".join(lines)


def export_review_selection(
	store: ReviewStore,
	selection: ReviewExportSelection,
	*,
	exported_at: datetime | None = None,
) -> tuple[ReviewExport, tuple[Path, Path]]:
	"""Build and persist versioned JSON plus a human-readable report."""

	record = build_review_export(store, selection, exported_at=exported_at)
	paths = store.persist_export(record, format_review_report(record))
	return record, paths
