# B14 — Normalize + geometry + helical: window/unit position, stride-aware

**Wave B · size M · issue #263 · depends on B13 (#262) · in-wave consumers B15 (#264), B17 (#266)**

Line references exact against `670eca8` (post-Wave-A master).

## Context

The `projection_start`/`projection_count` artifact pair is the main
source-mapping side-channel: written by the worker rebase and read by
seven normalize blocks (weighted flats, beam tracking/keyframes/
theta-switch/basis fractions), geometry's drift fraction, and
reference's per-theta-batch seed. Two defects make it replaceable rather
than refactorable:

1. **Stride-blind.** The channel computes global index as
   `base + index`; the correct form under strided theta windows is
   `pos.Theta + local·step.Theta`.
2. **Unseeded on the main pipeline path.** Nothing seeds the artifacts
   on the ordinary path, so consumers default `projection_start` to 0
   under theta windows — silently wrong positions, defended today only
   by those windows being rare.

Helical maintains a *parallel unsynchronized channel* for the same fact:
a `projection_start` constructor attribute (helical.py:29/:50, consumed
at :75 and mirrored in planning's `output_geometry` inputs) — stage
config carrying what is really source geometry.

`_calibration_y_window` (normalize.py:166) greps the *config* for
y_start to recover the calibration row window — config re-derivation of
what `state.source_window` already carries.

## Work items

1. **Global theta position from the window/unit** — parallel paths get
   it from the unit's `pos`/`step` (B6); serial twins in these stages
   get it from `state.source_window` with the same stride-aware form
   `pos.Theta + local·step.Theta`. Extents from `window.internal.shape`.
   The artifact pair is deleted, not reinterpreted — a consumer that
   cannot reach a window is a bug to fix, not a reason to keep the
   channel.
2. **Delete helical's stage-attr channel** — `projection_start` leaves
   the constructor; `output_geometry` takes its indices from the window.
   (Config compatibility: the param is dropped from `from_config`
   consumption — stale configs fail loudly at construction, the
   Wave A precedent for voided inputs.)
3. **`_calibration_y_window` → source_window arithmetic** —
   `source_window.pos.Y` + `with_pad`; the config grep dies.
4. **Ledger add (plan §2 data copies):** geometry's pad-by-copy
   `np.pad` (geometry.py:68) → DataRef padding (`with_pad` + the
   `_resolved_padding` seam).
5. **reference.py's per-theta-batch seed** migrates to the same window
   form (it is a writer of the artifact channel; it cannot outlive it).

## Deletion ledger

- `projection_start`/`projection_count` artifact writes and all reads
  across execution, normalize (×7 blocks), geometry, reference.
- Helical's `projection_start` constructor attribute + planning mirror.
- `_calibration_y_window` (normalize.py:166) and its config grep.
- geometry.py:68 `np.pad` copy.

## Acceptance

- **Provenance test (the wave's signature):** no `projection_start` /
  `projection_count` key present in artifacts on the migrated paths —
  asserted, not grepped.
- Stride regression: a strided theta window produces correct global
  indices (the case the old channel got wrong — pin it with a test that
  fails on `base + index`).
- Unseeded-path regression: main-path run under a theta window gets
  non-zero global positions.
- Numerical suites unchanged.
