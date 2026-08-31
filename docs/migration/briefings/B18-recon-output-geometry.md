# B18 — Recon: typed output geometry + crop sugar

**Wave B · size M · issue #267 · depends on B16 (#265), B17 (#266) · closes Wave B**

Line references exact against `670eca8` (post-Wave-A master).

## Context

Recon terminates source mapping (the gap map's two-truths finding: the
Theta axis dies here; Y is republished as output placement) — so it is
where output geometry must become typed rather than arithmetic:

- The edge-buffer crop family — `_prepare_input` (recon.py:116),
  `_crop` (:126), `_crop_buffer` (:141) — hand-rolls symmetric and
  asymmetric X-slicing three times with three duplicated
  `left >= right` guards.
- `_allocate_reconstruction` hand-builds
  `ReconOrder(Y=slice_count, Z=width, X=width)` — `ReconOrder.of`'s
  Y·Z·X relation written arithmetically — and the post-crop commit
  re-derives `ReconOrder(*np.asarray(cropped).shape)` from the array.
- `_crop_buffer` duck-types the binding per call
  (`getattr(reconstruction.shape, "X", None)` with positional fallback)
  — the B5 ledger deferred its deletion here, where typed buffers make
  it consumable properly.
- The defensive `np.shares_memory` copy-backs (:180, :190) exist
  because the tomopy-output/target aliasing relationship is untyped.

**Scope correction vs the original breakout:** dropping
`_state_sinogram_order` was listed here; Wave A already deleted it
(#253). Out of scope.

## Work items

1. **Crop sugar** — the asymmetric edge-buffer crop becomes canonical
   window arithmetic (`rescope`-based X-interval sugar per the Phase-1
   asymmetric-crop item); the three guard duplicates collapse to one.
2. **Output shape via `recon_shape()`/`ReconOrder.of`** —
   `_allocate_reconstruction` consumes the B5 seam
   (`window.recon_shape()` where a window is in scope; `ReconOrder.of`
   on the working shape otherwise); the post-crop spec derives from the
   crop arithmetic instead of re-reading the array.
3. **Delete `_crop_buffer` duck-typing** — buffers are typed and bound
   (Wave A); the getattr/positional fallback is dead weight.
4. **Ledger re-judge: `np.shares_memory` copy-backs** — with typed,
   bound buffers the aliasing relationship is decidable at allocation
   (`can_alias_input` already computes it). Delete each copy-back the
   binding makes provably dead; any survivor must name the concrete
   aliasing case it defends against (robustness-needs-a-real-failure).

## Deletion ledger

- recon.py:116-141 hand-rolled crop arithmetic ×3 + duplicated guards.
- `_allocate_reconstruction`'s hand-built ReconOrder + post-crop
  array re-derivation.
- `_crop_buffer`'s `getattr(...shape, "X", None)` duck-typing (B5
  ledger carry-over).
- recon.py:180/:190 `shares_memory` copy-backs (or stated surviving
  failure cases).

## Acceptance

- Numerical: gridrec/astra suites unchanged, including the edge-buffer
  crop-by-default test and output_width paths.
- Alias regression: the `can_alias_input` reuse path produces identical
  output with copy-backs removed (this is the test that proves "provably
  dead" — if it can't be written, the copy-back stays).
- **Wave close-out:** Wave B cumulative non-test net-LOC measured at
  this merge — at-or-below zero, or a stated one-line reason
  (stop-and-reassess signal per the README guardrails).
