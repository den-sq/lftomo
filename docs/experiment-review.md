# Experiment review contracts

`lftomo.review` supplies the headless, versioned records used by a visual
review client. It does not render images or own an interactive workflow. The
`lfreview` consumer can present the protocol while lftomo remains responsible
for experiment provenance, review persistence, candidate resolution, ranking,
and export formats.

## Ownership boundary

An experiment results root remains read-only input. `manifest.json`,
`store.json`, `trials/`, `previews/`, and `private/preview-mapping/` are never
rewritten by review operations. The default review location is:

```text
RESULTS_ROOT/
  review/
    SESSION_ID/
      session.json
      .lock
      triage/*.json
      pairwise/*.json
      unblind/*.json
      rankings/*.json
      exports/*.json
      exports/*.md
```

Pass an external `root` to `ReviewStore.initialize` when review state should
live outside the experiment tree. `session.json` records both the experiment
identity and a fingerprint of the resolved results-root location, and copies
the `PreviewContract` so scaling/crop drift is detected when the session is
reopened.

Opening or reopening a review session requires read access to
`private/preview-mapping/`. The session copies its `PreviewContract` from that
mapping, and the resolver keeps the blind-ID-to-candidate join in process
memory while blinded observations are recorded. Blinding is therefore a
durable-record boundary, enforced by the secret-leak checks, rather than a
filesystem or process-isolation boundary: the reviewer process may read the
mapping, but the append-only blinded records must not disclose it. The
`LFTOMO_PREVIEW_BLINDING_KEY` fallback can recover blind IDs for lookup, but it
does not supply the missing preview contract needed to open a session. A
review station that must not be able to read the private mapping needs a
separately published public preview contract; that is outside this contract.

Each triage judgment, pairwise vote, and unblind event is a separate immutable
JSON fragment. A session lock assigns one global monotonic sequence, and the
existing atomic-write helper publishes each fragment with write-temp-then-
rename semantics. Re-reviewing a candidate appends a new observation; it does
not rewrite the earlier one. `latest_triage(pass_number)` reconstructs the
resumable current view.

```python
from lftomo.review import ReviewStore

store = ReviewStore.initialize(
    results_root,
    session_id="review-2026-08-16",
    reviewer_id="reviewer-a",
    reviewer_label="B10 broad review",
)
store.record_triage(
    blind_id,
    pass_number=1,
    verdict="retain",
    reason_tags=("rings",),
)
store.record_pairwise(left_id, right_id, outcome="left")
```

## Versioned protocol

`ReviewProtocol` declares the vocabularies a client renders:

- Pass 1: `discard`, `uncertain`, `retain`;
- Pass 2: `reject`, `acceptable`, `shortlist`;
- pairwise: `left`, `right`, `tie`, `cannot_judge`;
- reason tags: noise, rings, blur/detail loss, contrast loss, structural
  artifact, and display/scaling problem;
- tie convention: `half_credit`.

The protocol version is stamped onto the session and every raw observation.
Vocabulary changes require a protocol-version change rather than a UI-local
interpretation.

## Pair scheduling

`derive_pair_schedule(votes, candidates, config)` is the I/O-free scheduling
contract consumed by review clients. `PairScheduleConfig` records policy
version, the round-robin threshold (default `12`), the bounded Swiss-round
extension (default `2`), and the seed. Candidate ordering and every final
tiebreak use canonical SHA-256 keys over that configuration; scheduling does
not depend on input order, wall-clock state, or a language-runtime RNG.

At or below the threshold, the seed-keyed circle method emits every unordered
pair exactly once. It owns the explicit left/right display assignment and
balances each candidate's complete schedule to within one appearance per
side. `remaining` contains the full unconsumed round-robin tail. Above the
threshold, the schedule uses
`ceil(log2(candidate_count)) + swiss_extra_rounds` rounds, with one comparison
per candidate per round except for an odd-set bye. Custom Swiss configurations
that request more rounds than candidates are rejected so no candidate receives
a second bye.

Swiss round one pairs the seed-keyed order adjacently. Later rounds sort by
standings from fully completed scheduled rounds: wins contribute one point,
ties contribute one half-point to each candidate under the review protocol,
and `cannot_judge` contributes no score. Adjacent rematches are repaired by
splitting the ordered standings into alternating sides and matching each side
to its nearest available standing through deterministic augmenting paths. A
rematch is accepted only when that fixed partition has no complete
rematch-free matching. An odd-set bye goes to the lowest-standing candidate
without a prior bye, using the same hash order as the final tiebreak. Only the
current Swiss round is exposed in `remaining`; later rounds are derived after
it is consumed.

Any stored vote consumes its unordered pair, including `cannot_judge`. The
earliest vote by monotonic sequence is the scheduling observation; later
re-reviews remain available to ranking but neither change standings nor advance
progress again. A vote on a pair not yet emitted has no immediate scheduling
effect, but consumes that pair if it appears in a later round. A last-resort
Swiss rematch is therefore already consumed by the prior vote and is not
rescored. Votes referencing candidates outside the locked set and votes with a
different protocol version are errors.

Resume requires no schedule record: replaying the same candidates, raw votes,
and config reconstructs the same completed rounds, current remainder, and next
left/right assignment. `pairs_consumed`, `pairs_total`, `complete`, and
`next_pair` provide client progress. Fewer than two candidates yield an empty,
complete round-robin schedule. Pair scheduling treats `cannot_judge` as a
completed presentation; the downstream `RankingRecord.missing_pairs` and
`incomplete` fields still report that the pair supplied no judgeable ranking
observation.

## Candidate lookup and deliberate unblinding

The read-only `ExperimentCandidateResolver` joins the existing
`ExperimentManifest`, `TrialAttemptRecord` histories, and private
`PreviewBundle`. It does not create a second candidate index. If the private
mapping cannot be read, blind IDs can be recomputed from
`LFTOMO_PREVIEW_BLINDING_KEY`; the key is never included in returned models,
records, logs, or errors.

Exact identifiers take precedence. Otherwise, prefixes are matched across the
blind-ID, candidate-key, and trial-ID namespaces and ambiguity is reported with
the matching full identities.

The CLI wrappers are explicit unblinding operations:

```console
lftomo-experiment candidate RESULTS_ROOT ID
lftomo-experiment candidate RESULTS_ROOT ID --json
lftomo-experiment candidate RESULTS_ROOT ID --yaml
lftomo-experiment candidates RESULTS_ROOT
lftomo-experiment candidates RESULTS_ROOT --json
```

The single-candidate report includes the ordered Stage/algorithm chain,
varied parameters recovered through the canonical candidate selector, complete
resolved `Recon`, every trial and attempt identity, current status, fidelity,
Pareto rank, metrics, timings, peak memory, promotion lineage, previews, and
retained outputs. The plural command emits a compact inventory for filtering
and joins.

## Pairwise ranking

`derive_bradley_terry` is an I/O-free numerical derivation over stored votes.
It applies the fixed protocol as follows:

- `cannot_judge` is retained in the consumed-vote hash but excluded from the
  fit;
- a tie contributes one half-win to each candidate;
- every possible pair receives a symmetric pseudo-count with total weight
  `pseudo_count_tau` (default `1.0`), preventing infinite strengths under
  complete separation;
- stable candidate ordering, a fixed tolerance/iteration cap, and blind-ID
  lexical fallback produce a deterministic total order.

The ranking record includes the exact order-independent vote-set hash,
candidate-set hash, numerical configuration, strengths, missing comparison
pairs, excluded-vote count, convergence state, and incomplete-data flag.
`ReviewStore.derive_ranking` appends the versioned record under `rankings/` and
never changes raw votes. Ranking is strict over the complete candidate set
referenced by the recorded pairwise votes; re-shortlisting after voting is not
part of this contract. Derive and persist the full voted-set ranking, then
select any smaller presentation or confirmation subset downstream.

## Anchors, controls, and confirmation reruns

Candidate keys and resolved pipelines may enter review artifacts only after an
explicit `UnblindEvent` covers the requested candidates or the whole session.
`export_review_selection` enforces that boundary, then writes versioned JSON
and a concise Markdown companion. The JSON preserves judgments and ranking
position for:

- a set of positive anchors;
- typed negative controls (`low_ring_high_noise`, `oversmoothed`,
  `prominent_ring`, `contrast_detail_loss`, or `structural_artifact`);
- a confirmation shortlist containing candidate keys and complete resolved
  `Recon` configurations.

```python
from lftomo.review import ReviewExportSelection, export_review_selection

selection = ReviewExportSelection(
    positive_anchors=(best_blind_id,),
    negative_controls={"low_ring_high_noise": (noisy_blind_id,)},
    confirmation_shortlist=(best_blind_id, alternate_blind_id),
)
store.record_unblind(
    scope="export",
    blind_ids=(best_blind_id, noisy_blind_id, alternate_blind_id),
)
record, (json_path, report_path) = export_review_selection(store, selection)
```

The confirmation entries are directly validated `Recon` models. A downstream
experiment builder can place them into a new experiment configuration without
reverse-engineering pipeline parameters from filenames. No TIFF is copied to
produce the review store or export.
