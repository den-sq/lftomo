# Canonical data-model migration — wave overview

Migration of lftomo to the canonical data model — `TomoDataSet` / `TomoWindow` /
`DataRef` (geometry, indexing, padding), `DataShape` (semantic axes),
`DataRange` (+ iterators) — and removal of alternate sources of truth,
duplicate data, and non-semantic variable storage. Line references throughout
this directory are exact against `8fff4b7`.

Derived from the 2026-08 design-review arc (canonical-migration plan, per-gap
consumer map, issue breakout; sources live in the design-review vault — the
implementation-relevant content is distilled into the per-issue briefings
under `briefings/`).

## Structure: vertical waves

Each wave pairs canonical-class extensions with the consumers that delete code
through them, instead of an additive extension layer followed by adoption
layers. This is the primary anti-overbuild mechanism. B-numbers are stable
identifiers from the breakout; they do not imply order — the waves do.

- **Wave 0** — independent bug fixes: B1, B2, B3
- **Wave A** — typed shapes & binding (spine B2 → B4 → B5 → B11 → B12)
- **Wave B** — window-driven pipeline: B6, B10 → B13 → B14 → {B15, B16, B17}
  → B18
- **Track C-early** — records & grids (concurrent with Wave B): B7 → B9 →
  B25; B8
- **Wave C-late** — driver authority: B19, B20, B21 → B22
- **Wave D** — experiment subsystem: B23 → B24
- **Wave E** — optional tail: B26, B27

Waves 0 and A are filed; later waves are filed as their predecessors near
completion. One GitHub milestone per wave.

Filed so far (2026-08-31):

| B-id | issue | milestone |
|---|---|---|
| B1 | #240 | Wave 0 |
| B2 | #241 | Wave 0 |
| B3 | #242 | Wave 0 |
| B4 | #243 | Wave A |
| B5 | #244 | Wave A |
| B11 | #245 | Wave A |
| B12 | #246 | Wave A |

```
B1,B2,B3 (parallel, alongside Wave A)
Wave A:        B2 → B4 → B5 → B11 → B12
Wave B:        B5 → B6;  B4 → B10;  {B6,B10} → B13 → B14 → {B15,B17} → B18;  B13 → B16
Track C-early: B4 → B7 → B9 → B25;  B8 anytime          (concurrent with Wave B)
Wave C-late:   {B18,B9} → B19;  {B8,B6} → B20;  {B12,B13,B7} → B21 → B22
Wave D:        {B9,B10} → B23 → B24
Wave E:        B26, B27
```

Critical path: **B4 → B5 → B6 → B13 → B14 → B18 → B19.**

Decided 2026-08-31: the §3.2 lossy-fingerprint bug rides B9 (one coordinated
prefix-cache version bump); the late landing is accepted — no early B7→B9
pull, Track C-early runs as scheduled.

## Guardrails against overbuild

The point of the unification is *net code reduction*.

1. **Deletion ledger per extension issue.** Every extension issue lists the
   workaround sites it exists to delete. Acceptance is two-sided: the API as
   built suffices for every ledger entry, **and no public surface exists that
   no ledger entry demands.** The briefings' "minimal extension" scopes are
   ceilings, not floors.
2. **No consumer, no freeze.** An extension's API is not settled until its
   first in-wave consumer has merged against it — B4/B5 settle under B11/B12,
   B6/B10 under B13, B7 under its in-PR streaming ports + B9, B8 under
   B20/B22. Late-arriving API changes driven by the consumer are expected,
   not churn.
3. **Net-LOC per wave.** Track cumulative non-test net-LOC; the migration
   overall must land negative, and each wave should individually land
   at-or-below zero — a wave landing positive without a stated one-line
   reason is a stop-and-reassess signal. (Track C-early is measured at the
   end of Wave C-late.)
4. **S-issues end when the ledger clears.** Done means the listed
   deletions/fixes landed; nothing else rides along.

Every stage PR additionally carries a **provenance test** — an assertion that
the canonical seam was exercised (e.g. no `projection_start` key in
artifacts; layout.py imports gone), not just output equivalence — plus the
existing numerical tests unchanged.

## Data-copy ledger

The duplicate-data-copies tail is distinct from truth duplication and easy to
drop silently. Every item, with its owning issue:

| copy item | issue |
|---|---|
| per-row `sinogram_scratch` gather/scatter (stripe + shared execution machinery) | B6/B15 |
| `sinogram_scratch` gather/scatter (center) | B17 |
| full-volume transpose round-trip: stripe `AllOfTheStripe:342-352` | B15 |
| full-volume transpose round-trip: super_resolve `_store_result` | B20 |
| full-volume transpose round-trip: write.py per-block | B19 |
| phase per-projection `.copy()` (:544-547) | B16 |
| filter `.astype` parallel buffers bypassing `work_dtype` | B16 |
| recon defensive `np.shares_memory` copy-backs | B18 |
| pad-by-copy `np.pad` in geometry | B14 |
| pad-by-copy `np.pad` in prep-phase | B26 |

## Consumer constraints (what the migration may NOT break)

Neither hatmehyt nor lfreview imports the geometry internals — the geometry
migration is consumer-safe. Must stay stable: `lftomo.pipeline.Y` identical
object to `util.shapes.Y`; `WindowConfig(rows=, extra_pad=)`;
`runner.run_reconstruction(..., window_config=)` signature + RunResult attrs;
`STAGE_REGISTRY` keys / `create_stage` / `StageConfig`;
`config.models.Scan/Recon` schemas; the `lftomo.experiment` / `.review` /
`.calibration` / `prep.flat_median` public surfaces; the
`LFTOMO_PREVIEW_BLINDING_KEY` env contract; results-root disk layout; the
lfreview-matched error substrings (`"private preview mapping"`, `"review
session differs"`). `experiment.WindowReference`'s *serialized field names*
are external contract (lfreview builds config dicts with `windows:` entries);
`WindowIdentity` is pure-internal and free to delete. Keep the literal class
names `SharedNPArray`/`TomoDataSet` (hatmehyt's B10 importer greps driver
source text for them).
