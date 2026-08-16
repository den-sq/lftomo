from __future__ import annotations

from datetime import datetime, timezone
import hashlib

import pytest

from lftomo.review import (
	ReviewExportSelection,
	ReviewStore,
	export_review_selection,
)

from .conftest import BLINDING_KEY


NOW = datetime(2026, 8, 16, 12, tzinfo=timezone.utc)


def _source_hashes(root):
	owned = [root / "manifest.json", root / "store.json"]
	for name in ("trials", "previews", "private"):
		owned.extend(path for path in (root / name).rglob("*") if path.is_file())
	return {
		str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
		for path in sorted(owned)
	}


def test_review_fragments_resume_with_one_global_monotonic_sequence(
	experiment_root, tmp_path,
):
	store = ReviewStore.initialize(
		experiment_root,
		session_id="reviewer-a",
		reviewer_id="reviewer-1",
		reviewer_label="first pass",
		root=tmp_path / "external-review",
		created_at=NOW,
	)
	blind_ids = tuple(item.blind_id for item in store.resolver().inventory())
	first = store.record_triage(
		blind_ids[0], pass_number=1, verdict="retain", reason_tags=("rings",),
		timestamp=NOW,
	)
	second = store.record_pairwise(
		blind_ids[0], blind_ids[1], outcome="left", timestamp=NOW,
	)
	partial = store.root / "triage" / ".000000000002-triage.json.tmp-interrupted"
	partial.write_text("{", encoding="utf-8")

	reopened = ReviewStore.initialize(
		experiment_root,
		session_id="reviewer-a",
		reviewer_id="reviewer-1",
		reviewer_label="first pass",
		root=store.root,
	)
	third = reopened.record_triage(
		blind_ids[1], pass_number=2, verdict="shortlist", timestamp=NOW,
	)
	assert (first.sequence, second.sequence, third.sequence) == (0, 1, 2)
	assert reopened.latest_triage(1)[blind_ids[0]] == first
	assert reopened.latest_triage(2)[blind_ids[1]] == third
	assert len(reopened.triage_judgments()) == 2
	assert partial.is_file()
	with pytest.raises(ValueError, match="must live under its review directory"):
		ReviewStore.initialize(
			experiment_root,
			session_id="invalid-location",
			reviewer_id="reviewer-1",
			root=experiment_root / "trials" / "invalid-location",
		)


def test_blinding_boundary_ranking_export_and_source_immutability(
	experiment_root, tmp_path, monkeypatch,
):
	monkeypatch.setenv("LFTOMO_PREVIEW_BLINDING_KEY", BLINDING_KEY)
	before = _source_hashes(experiment_root)
	store = ReviewStore.initialize(
		experiment_root,
		session_id="anchor-review",
		reviewer_id="reviewer-2",
		root=tmp_path / "anchor-review",
		created_at=NOW,
	)
	inventory = store.resolver().inventory()
	blind_ids = tuple(item.blind_id for item in inventory)
	assert all(blind_ids)
	store.record_triage(
		blind_ids[0], pass_number=1, verdict="retain", timestamp=NOW,
	)
	store.record_triage(
		blind_ids[1], pass_number=1, verdict="discard",
		reason_tags=("noise",), timestamp=NOW,
	)
	store.record_pairwise(
		blind_ids[0], blind_ids[2], outcome="cannot_judge", timestamp=NOW,
	)
	raw_votes_before = {
		path.name: path.read_bytes() for path in (store.root / "pairwise").glob("*.json")
	}
	ranking = store.derive_ranking(blind_ids, derived_at=NOW)
	assert ranking.incomplete
	assert ranking.excluded_cannot_judge == 1
	repeated_ranking = store.derive_ranking(blind_ids, derived_at=NOW)
	assert repeated_ranking == ranking
	assert len(store.rankings()) == 2

	blinded_payload = b"\n".join(
		path.read_bytes()
		for path in sorted(store.root.rglob("*"))
		if path.is_file()
	)
	assert BLINDING_KEY.encode() not in blinded_payload
	assert b"paganin" not in blinded_payload
	assert b"alpha" not in blinded_payload
	for item in inventory:
		assert item.candidate_key.encode() not in blinded_payload

	selection = ReviewExportSelection(
		positive_anchors=(blind_ids[0],),
		negative_controls={"low_ring_high_noise": (blind_ids[1],)},
		confirmation_shortlist=(blind_ids[0], blind_ids[2]),
	)
	with pytest.raises(ValueError, match="explicit unblind event required"):
		export_review_selection(store, selection, exported_at=NOW)
	store.record_unblind(scope="export", blind_ids=blind_ids, timestamp=NOW)
	record, (json_path, report_path) = export_review_selection(
		store, selection, exported_at=NOW,
	)
	assert json_path.is_file()
	assert report_path.is_file()
	assert record.ranking_id == ranking.ranking_id
	assert record.positive_anchors[0].triage_judgments[0].verdict == "retain"
	assert record.negative_controls["low_ring_high_noise"][0].candidate_key
	assert len(record.confirmation_shortlist) == 2
	assert "Positive anchors" in report_path.read_text(encoding="utf-8")
	assert {
		path.name: path.read_bytes() for path in (store.root / "pairwise").glob("*.json")
	} == raw_votes_before
	assert _source_hashes(experiment_root) == before
	assert BLINDING_KEY not in json_path.read_text(encoding="utf-8")
