# B15 — Stripe: per-row sub-windows from units()

**Wave B · size M · issue #264 · depends on B14 (#263)**

Line references exact against `670eca8` (post-Wave-A master).

## Context

`SINOGRAM_ROW_ARTIFACT` (stripe.py:30, read at :45; written by the
bounded-segment machinery at execution.py:135-137 and popped at
:245-247) is — per the gap-map finding — a **buffer-local
scratch-indexing channel mislabeled as source mapping**: it carries the
buffer-local row so Vo scratch planes can be addressed per row. Rows
currently travel *twice* (fan-out tuple slot and artifact).

The per-row gather/scatter is the data-copy cost: ProjOrder inputs get
`np.ascontiguousarray` per row in `_run_sinogram_stage` (stripe.py) and
in `run_sinogram_chain_kernel` / `run_bounded_sinogram_segment`
(execution.py), then a write-back — a full row copy in each direction
per row per stage chain.

`AllOfTheStripe` owns the other ledger item: the full-volume transpose
round-trip into tomopy (post-Wave-A this is a zero-copy *view* on our
side — `transposed_view` — but tomopy materializes, and the write-back
copies the whole volume again).

## Work items

1. **Bounded dispatch passes per-row sub-windows** from `units()` (B6):
   the row a kernel works on arrives as a unit ref (memory index +
   global pos), not as an artifact + int arithmetic.
2. **Delete `SINOGRAM_ROW_ARTIFACT`** — Vo scratch planes address by
   unit pos (stripe.py:45's consumer). This is the deletion B6
   explicitly deferred here.
3. **Gather/scatter through unit-shaped views** — the
   `ascontiguousarray` row copies remain only where the kernel demands
   contiguity (Vo's shared scratch reuse); where they exist merely to
   simplify indexing, the unit's `.index` view replaces them. Each
   surviving copy must name the kernel constraint that demands it
   (the robustness-needs-a-real-failure rule applied to copies).
4. **`AllOfTheStripe` round-trip** — re-judge the write-back copy once
   rows travel as units; if tomopy's materialization makes the full
   round-trip unavoidable, state that in the PR and close the ledger
   item as "irreducible at the library boundary" rather than silently
   keeping it.

## Deletion ledger

- `SINOGRAM_ROW_ARTIFACT` constant, its write sites
  (execution.py:135-137, :245-247), its read (stripe.py:45), and the
  duplicate row slot in the sinogram-chain fan-out tuples.
- The indexing-convenience subset of the per-row `ascontiguousarray`
  copies (kernel-demanded contiguity copies survive, each with its
  stated constraint).

## Acceptance

- **Provenance test:** no `_sinogram_row_index` key in artifacts during
  a bounded-segment run — asserted in-run, not grepped.
- Vo scratch addressing regression: multi-row bounded run with reused
  scratch planes produces identical output when rows are addressed by
  unit pos (the scratch-reuse seam is where a silent off-by-one would
  hide).
- Numerical suites unchanged (stripe-chain regression suite is the
  gate).
