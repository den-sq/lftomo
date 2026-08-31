# B12 — PipelineState / BufferSpec unification

**Wave A · size M · depends on B5, B11 · closes Wave A**

## Context

`PipelineState` carries three parallel truth-carriers: `working_array`
(bare ndarray) + `layout` (class tag) + `buffer_spec` (untyped
`shape: Any`) — the two-copies problem is institutionalized as
`sync_buffer_spec()` (core.py:329-333), which exists purely to re-reconcile
`state.layout` against `buffer_spec.layout` after in-place stages. This
issue collapses them: one shaped working buffer, layout *derived*, spec
unified.

## Work items

1. **`buffer_spec` becomes DataRef-based** (typed shape + dtype). `layout`
   stays available as a derived property (`type(spec.shape)`) for the #97
   stage-contract vocabulary — the contract is a planner vocabulary, not a
   shape store; it keeps its accepts/produces surface unchanged.
2. **`with_array` binds shape at handoff** — array and DataRef enter the
   state together; no post-hoc reconciliation. Delete `sync_buffer_spec()`.
   `PipelineState` does not accept a `layout` compatibility input: typed
   managed buffers provide their spec directly, non-empty plain arrays require
   an explicit `BufferSpec`, and only the empty pre-read state may have no spec.
3. **`BufferSpec` in planning unifies** with the same frozen DataRef-spec —
   removes the `shape: Any` and the `_shape_tuple` decompose helpers
   (planning.py:104, mem.py:29, core.py:270; the fourth copy at core.py:349
   dies with B5's drop-site fix — verify it's gone, don't double-count).
4. **Single implementation for the drifted derived-geometry pairs:**
   - Helical output geometry: one function consumed by both
     execution.py:579-592 and planning.py:173-188 (today: two independent
     implementations that can drift).
   - Layout-convert: `transposed_view`/`to` replaces the hand-permuted
     tuples at execution.py:518 and planning.py:165-169.

## Deletion ledger

- `sync_buffer_spec()` (core.py:329-333).
- `_shape_tuple` ×3 (planning.py:104, mem.py:29, core.py:270).
- `shape: Any` on BufferSpec.
- One of the two helical output-geometry implementations.
- Hand-permuted layout-convert tuples (execution.py:518,
  planning.py:165-169).

## Acceptance

- Provenance test: `state.layout` is derived (no independent attribute to
  drift); `sync_buffer_spec` absent.
- Helical geometry: single-source pin — a test that computes the output
  geometry through the one function and asserts planning and execution
  agree by construction (call-identity, not just value-equality).
- Existing numerical tests unchanged.
- Wave A net-LOC checkpoint: with B4/B5/B11/B12 all landed, the wave's
  cumulative non-test net-LOC should be at-or-below zero (guardrail 3);
  B4's additions must be paid for by B11/B12's deletions.
- This merge freezes the B4/B5 APIs (no-consumer-no-freeze satisfied).
