# B6 — Unit dispatch + FileBackedNPArray DataShape (G4)

**Wave B · size L · issue #260 · depends on B5 (#244, landed) · in-wave consumer B13 (#262)**

Line references exact against `670eca8` (post-Wave-A master).

## Context

Eight fan-outs ship hand-built positional tuples through
`submit_all(function, tuples)`: execution.py:497 (projection dispatch),
:533 (layout fill), :547 (fused projection), :563 (affine fill), :595
(helical fill), :672 (sinogram chains), write.py:124 (tiff slices),
super_resolve.py:118 (layout fill reuse). Per-unit context is
`_WorkerState` (execution.py:34-42) — pickled *identically for every
unit*, carrying the **whole-stage** `source_window` while the buffer
handle is per-unit: geometry and data disagree about granularity, which
is the seam itself.

Workers reconstruct identity by arithmetic: `_projection_handle`
(execution.py:74-80) re-derives the memory slice from `(layout, index)`;
global position = `artifacts["projection_start"] + index`; stripe rows
travel **twice** (tuple slot and `SINOGRAM_ROW_ARTIFACT`).
dispatch.py carries the in-repo TODO waiting for this: *"shape-changing
fill remains parent-orchestrated until its unit adapter is selected."*

Wave A moved the seam without closing it: `_WorkerState` now carries
`buffer_spec`, and `run_projection_kernel` hand-builds a `Theta=1` spec
per unit (execution.py:94-96) — per-unit *shape* without per-unit
*position*. `units()` replaces that with a real per-unit window.

The union of what a unit needs is already fully covered by existing
`DataRef` fields: memory index → `.index` (generic tuple-of-slices —
`buffer[window.index]` is legal today); global position → `.pos`
(source_window pos is already global); extent → parent `.shape`; stride
→ `.step`; dtype → `.dtype`/`internal`; layout → `type(shape)`. Pickling
is already satisfied — dataclasses + a type + ints. No API *yields*
those per-unit refs; that is the entire gap.

Both halves of this issue land as **one PR** because both touch the
worker-pickle contract (decided 2026-08-30). This is one of the two
contained flag-days in the migration — worker pickles are transient;
workers die with the run.

## Work items

1. **`units(axis=…)` on DataRef/TomoWindow** — a generator yielding
   lightweight per-unit `DataRef`s (memory-coordinate index + global
   `pos`/`step`, pickle-cheap). Implementation is iteration over
   `rescope` arithmetic — no new types beyond the generator, no executor
   redesign. `submit_all` keeps its shape; tuples carry `(handle, unit)`
   instead of `(handle, layout, index, count, …)`.
2. **`_WorkerState.source_window` becomes the unit's window** — the
   worker-state constructor takes the per-unit ref; `_worker_state`
   derives its `BufferSpec` from it (replacing the hand-built `Theta=1`
   spec at execution.py:94-96).
3. **Migrate the shared fan-out machinery** (the 6 execution.py sites;
   write.py:124 and super_resolve.py:118 follow the same mechanical
   shape). Per-stage serial twins are explicitly **not** here — they
   migrate with their stages (B13–B18).
4. **`FileBackedNPArray` optional `shape: DataShape` constructor param**
   (storage.py:54), validated rank-against-NPY-header, tuple fallback
   stays as explicit legacy for shape-less callers; `__reduce__`
   (storage.py:125) gains the field — the accepted pickle-shape change.
5. **Payload measurement before any sharing machinery.** `_WorkerState`
   — including the whole artifacts dict — is pickled per *unit*, not per
   pool. Slimming the tuples is in scope; if pickle cost still looks
   material, *measure it and report the number* in the PR rather than
   building pool-level sharing here (overbuild risk, decided watch-item).

## Deletion ledger

- `_projection_handle` (execution.py:74-80) + the air-mask twin call
  (execution.py:110).
- The `base + index` global-position rebase in `run_projection_kernel`
  and the hand-built `Theta=1` spec (execution.py:94-96).
- The `count`/`layout` slots in all 8 fan-out tuple shapes.
- The per-fan-out `range(count)` index arithmetic at the 8 sites.
- `SINOGRAM_ROW_ARTIFACT` is **consumed** by B15, not deleted here —
  its write sites (execution.py:135-137, :245-247) keep working against
  units until B15 lands.

## Acceptance

- Tests: `units()` yields refs whose `.pos`/`.index`/`.step` match the
  hand-computed values for ProjOrder and SinoOrder parents, including
  strided windows; a worker round-trip test pickles `(handle, unit)` and
  asserts the worker-side state sees the unit's window and spec;
  FileBackedNPArray with DataShape survives its own `__reduce__`
  round-trip and rejects rank mismatches.
- Provenance: no `_projection_handle` symbol; fan-out tuples carry no
  bare layout/count slots.
- Numerical suites unchanged; process-dispatch suite green (the
  fork-in-threads seam this machinery lives on has burned us before —
  run it, don't infer it).
- API not frozen until B13 (#262) merges against it. Acceptance is
  two-sided against this ledger.
