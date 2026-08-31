# B9 — Document contract + cache identity (G3, coordinated)

**Track C-early · size M/L · issue #278 · depends on B4 (#243, merged), B7 (#277) · in-wave consumer B25 (#279)**

Line references exact against `f0e4e89` (post-Wave-B master). This is the
migration's **only** prefix-cache version bump.

## Context

`prefix_cache._fingerprint` (prefix_cache.py:67) handles pydantic
(`model_dump`) and dataclasses (`asdict`), then
**`raise TypeError("... lacks a cache identity contract")`**
(prefix_cache.py:108) for everything else. `DataRef`/`TomoWindow`/
`TomoDataSet` are plain objects with private attrs → they hit the raise. That
is *why* the pydantic twins are injected into the identity dict — the canonical
object cannot participate in its own cache key.

Three more idioms exist for the same job: `state.pkl` pickles the geometry by
name-mangled attribute graph (rename-invalidates every cache);
`streaming.record()` (streaming.py:82) hand-writes a document with **no reader**
(a recorded plan cannot round-trip); `localct_patterns` has its own JSON idiom
(B25). One contract must serve all four, or a twin stays alive — and the twins
are the regrowth mechanism.

Post-Wave-B note: `record()` already uses `axis_slice("Y")` (Wave B ported the
positional `.index[1]`), so its remaining gap is only the reader counterpart,
not the axis port.

## Work items — one PR, separate commits, ONE cache-version bump (decided)

1. **Commit 1 — `to_document()`/`from_document()`** on DataShape, DataRange,
   DataRef, TomoWindow, TomoDataSet. Type-tagged (class name + ordered
   fields), plain dicts of ints/strings — **not** an object graph. Pairs with
   B7's interval as a documentable value.
2. **Commit 2 — `_fingerprint` fix** (plan §3.2): the qualname/lossy-fingerprint
   bug, plus a `hasattr(value, "to_document")` branch so geometry objects
   fingerprint through the contract instead of the raise.
3. **Commit 3 — `prefix_cache` `state.pkl` → documents**: store documents
   instead of the attribute-graph pickle (rename-proof, versionable). One
   cache-version bump covers all three commits.

Plus:
4. **`streaming.record()` reader** — give it a real `from_document`
   counterpart so a recorded plan round-trips; drop the write-only hand-roll.
5. **Stop deep-copying immutable geometry holders** (plan Phase 5.3):
   prefix_cache deepcopies `dataset`/`source_window`/`buffer_spec` at
   prefix_cache.py:505-507 on every snapshot/rehydrate (and scan/recon/artifacts
   at :693/:709/:712). The geometry holders are immutable-by-convention —
   `TomoDataSet.angles`/`acquisition_indices` are `setflags(write=False)`, all
   DataShape ops return fresh instances — so serialize via `to_document` /
   reference-share instead of cloning full angle vectors per snapshot.

## Guardrail (the session's live lesson)

**No document field without a consumer that reads it back.** B13's
`angle_provenance` was a `to_document`-shaped write with no `from_document`
reader — dead weight. Every field `to_document` emits must be consumed by a
`from_document` or the fingerprint; if a field has no reader, it does not go in
the document.

## Deletion ledger

- The `_fingerprint` raise-path for geometry objects (prefix_cache.py:108) →
  the `to_document` branch; the injected pydantic identity twins.
- `state.pkl` attribute-graph pickle → documents.
- `record()`'s write-only hand-roll → thin wrapper + reader.
- prefix_cache.py:505-507 geometry-holder deepcopies (and the redundant
  siblings once documents replace them).

## Acceptance

- A `DataRef`/`TomoWindow`/`TomoDataSet` fingerprints via `to_document`, no
  raise; a recorded plan round-trips through `from_document`; cache identity is
  stable across an attribute rename (the property the pickle lacked).
- One prefix-cache version bump; a pre-bump snapshot is invalidated/rebuilt
  cleanly (state the `on_error` behavior, per the Wave A snapshot-rebuild
  precedent).
- Provenance test: no geometry object hits the `_fingerprint` raise; no
  `to_document` field lacks a reader.
- Cross-check the §3.2 fix with a fingerprint-collision regression (the bug it
  fixes).
