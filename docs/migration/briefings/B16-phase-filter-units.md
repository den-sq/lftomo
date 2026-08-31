# B16 — Phase + filter: serial twins to units; copy ledger

**Wave B · size M · issue #265 · depends on B13 (#262) · in-wave consumer B18 (#267)**

Line references exact against `670eca8` (post-Wave-A master).

## Context

The phase family (paganin_filter, negative_log, beta_scale, paganin,
paganin_phase_absorption, bac_correction, bac) and filter.bilateral run
the serial `for i in range(count)` twin of the parallel fan-out — same
per-projection loop, inline. Post-Wave-A these loops already speak named
axes (`layout.point_index(Theta=i)`), so the remaining migration is the
*iteration source*, not the indexing: the loop bound and view come from
the window/units, which is what makes the B14 provenance guarantee hold
on the serial paths through these stages.

Two data-copy ledger items live here (the reason this issue was resized
S/M → M):

- **Phase per-projection `.copy()`** — the BAC stages' `image()` closures
  (phase.py:516, :606) copy every projection before correction because
  the correction generator consumes lazily while the loop writes back
  into the same buffer. The copy is a real aliasing guard *today*; the
  fix is restructuring the consume/write-back order (or unit-scoped
  scratch), not deleting the guard while the hazard remains.
- **Filter `.astype` parallel buffer** (filter.py:70) — allocates a
  float32 copy per projection when the working dtype is float64.
  **Numerical-behavior constraint:** the in-source comment declares the
  float32 conversion part of the stage's public numerical behavior —
  the conversion *stays*. The fix routes dtype through
  `TomoWindow.work_dtype` (window.py:167-173) so the working buffer is
  already float32 where this stage runs, killing the per-projection
  allocation without changing a single output bit. If the pipeline
  cannot guarantee float32 at this stage, the allocation survives with
  that stated reason.

## Work items

1. **Serial loops → `slice_iter`/units** across the seven phase stages
   and filter.bilateral: loop bound and per-projection view sourced from
   the window/unit, not `data.shape[axis_of("Theta")]` + `range`.
   (Aug-18 SliceIter answer: these sites take a 2-D view and don't care
   how it was produced.)
2. **BAC copy restructure** — consume-then-write or unit scratch;
   equivalence pinned by the existing BAC numerical tests.
3. **work_dtype routing for filter.bilateral** — per the constraint
   above; output-bit equivalence explicitly asserted.

## Deletion ledger

- The `for i in range(count)` + `point_index` loop headers in the seven
  phase stages and filter.bilateral (replaced by window/unit iteration).
- phase.py:516/:606 per-projection `.copy()` (or a stated surviving
  hazard).
- filter.py:70 per-projection `.astype` allocation (or the stated
  dtype-guarantee gap).

## Acceptance

- Numerical: phase-chain and bilateral suites byte-identical — the
  filter conversion comment makes this a *public-behavior* gate, not a
  tolerance gate.
- Provenance: no bare `range(count)` projection loops remain in the two
  modules (grep pin); serial paths yield correct global positions under
  a strided theta window (inherits B14's stride regression, applied to
  the serial twin).
- Each ledger copy either deleted or annotated with the concrete
  constraint that keeps it.
