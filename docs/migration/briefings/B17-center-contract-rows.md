# B17 — Center: declared-contract row math + unit sampling

**Wave B · size S/M · issue #266 · depends on B14 (#263) · in-wave consumer B18 (#267)**

Line references exact against `670eca8` (post-Wave-A master).

## Context

Center is the best-behaved source-mapping consumer: it already reads
`state.source_window` for global row math — `pos.Y` at center.py:124
(entropy_local) and :222 (its diagnostic twin), `pos.Y`/`step.Y` at
:451-453 (Vo tilt fit) — and the Vo tilt fit is the one production site
that already consumes `step.Y` correctly. B5 declared `source_window` a
stage contract; this issue makes the declared contract the *only*
channel (the `0 if state.source_window is None else …` fallbacks become
contract violations rather than silent defaults on the paths that
require correct global rows).

The data-copy item: center's per-row `sinogram_scratch` gather/scatter —
three `np.ascontiguousarray` row copies (entropy_local's sampled row,
Vo `_tilt_center`'s sampled rows, Vo `run`'s single row).

**Meaning constraint:** the global row math is **unchanged in meaning**.
This issue changes where the numbers come from, never what they are —
any center-value diff is a migration bug.

## Work items

1. **Row math sourced from the declared contract** — the three read
   sites keep their arithmetic; the None-fallbacks on paths where a
   window is guaranteed (post-B13, main pipeline always has one) tighten
   to loud errors; fragment/test paths construct windows instead of
   relying on the 0-default.
2. **Vo sampling via units** — `_tilt_center`'s sampled rows and the
   single-row paths iterate unit refs; the scratch copies follow the
   B15 rule: survive only where the finder kernel demands contiguity,
   with the constraint stated.

## Deletion ledger

- The `source_window is None → 0` silent defaults on guaranteed-window
  paths (center.py:124/:222/:451 region).
- The indexing-convenience subset of the three `ascontiguousarray` row
  copies.

## Acceptance

- Center-value equivalence on the existing center suites (tilt fit,
  entropy expansion, quantization) — byte-identical, per the meaning
  constraint.
- Global-row regression under a strided/offset Y window: sampled source
  rows match hand-computed `pos.Y + i·step.Y` (pins the one consumer
  that historically got `step.Y` right — keep it right).
- S/M sizing discipline: done when this ledger clears; nothing else
  rides along.
