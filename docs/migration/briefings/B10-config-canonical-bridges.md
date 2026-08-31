# B10 — Config→canonical bridge constructors (G9)

**Wave B · size M · issue #261 · depends on B4 (#243, landed) · in-wave consumer B13 (#262)**

Line references exact against `670eca8` (post-Wave-A master).

## Context

The workaround for the missing bridge is "every consumer plumbs scalars
by hand" — 28 sites cataloged in the gap map. The structural failures:

- `planning._read_spec` re-implements the *entire* read-stage selection
  algebra (theta/y defaulting, strided count) that the read stages also
  implement — two derivations that must agree or memory planning lies.
- The `theta_stop` default + `projection_limit·stride` clamp block is
  copy-pasted verbatim between experiment/cli.py sites.
- `detector.rows/columns: int | None` has **four divergent
  None-policies**: streaming.py:118-121 raises `WindowPlanError`;
  planning.py:111-113 returns None; cli/runner-estimate return None;
  reference falls back to a default.
- `read._bind_source_state` assembles the main TomoWindow from six loose
  ints, with an in-source comment explaining exactly the shape-vs-step
  subtlety a factory would encapsulate.
- **No `TomoWindow.from_bounds` exists.** `WindowConfig(rows=…,
  extra_pad=Y(…))` is the one place an int is already lifted into a
  DataShape at the boundary — the pattern to generalize.

Both factories are **pure constructors over existing types** — they
delete duplicated derivations rather than adding a layer.

## Work items

1. **`Scan/Detector → ProjOrder/Proj` bridge constructors** in
   config/models.py. The `rows is None` policy, decided **once**
   (2026-08-30): **raise**, with the streaming wording — *except*
   pre-TomoDataSet contexts, where the HDF5-probe path remains the
   legitimate pre-dataset resolution (probe fills rows, then the bridge
   runs). The four divergent policies above migrate onto the bridge.
2. **`TomoWindow.from_bounds(y=…, theta=…, halo=…)`** — accepts interval
   bounds + optional theta stride + halo; halo policy decided
   (2026-08-30): **clamp+record**, matching `_resolved_padding`
   semantics; the experiment preflight keeps its *warning* (the old
   cli halo-raise becomes that warning when B24 lands). Realized-vs-
   requested pads must be readable off the result — Wave D's identity
   hashing (B23) needs realized pads, so record them, don't discard.
3. **Port the verbatim-duplicated clamp block** (the two cli.py sites)
   and `planning._read_spec`'s selection algebra onto the factories —
   planning and the read stages then *share* one derivation by
   construction. (The read-stage side completes in B13; here the
   factories land with at least the planning + cli consumers so the API
   has in-PR exercise.)

## Deletion ledger

- The four `detector.rows is None` policy variants (streaming.py:118-121
  stays as the *wording source*; the check moves into the bridge).
- The cli.py duplicated theta-clamp block (both copies).
- `planning._read_spec`'s hand-rolled selection algebra (shared
  derivation; final deletion of the read-stage twin lands with B13).
- Six-loose-ints window assembly in `_bind_source_state` (consumed by
  B13; the factory must be sufficient for it — that is the two-sided
  acceptance check).

## Acceptance

- Tests: bridge raises on `rows is None` with the streaming wording;
  HDF5-probe context resolves then bridges; `from_bounds` clamps at both
  detector edges and records realized pads (full-detector scan with halo
  must *not* raise — that case is the reason clamp won); strided theta
  bounds produce the same window `_bind_source_state` builds today
  (fixture comparison, not shared code, until B13 swaps the call).
- No consumer-visible API change (`WindowConfig`, runner signatures
  stay; see README consumer constraints).
- API not frozen until B13 (#262) merges against it. Acceptance is
  two-sided against this ledger.
