# B5 — Typed storage & declared source_window (G2)

**Wave A · size M · depends on B4 · in-wave consumers B11/B12**

## Context

The array↔shape binding half-exists: `ManagedBuffer` is already an ABC
(storage.py:13-35) with abstract `shape`/`dtype`/`array()`/`__getitem__`,
exactly two implementations, and 20 `isinstance` dispatch sites. Typed
shapes already flow into `storage.allocate` at read.py:284-288/453-458 and
recon.py:178/181. The gap is not a missing abstraction — it's that the
binding is **dropped at exactly four sites**, plus an untyped abstract
contract (`shape: Any`). No new wrapper class (decided; the seam type not
only exists, it's already abstract). `FileBackedNPArray` is deliberately
**not** in scope here — its DataShape constructor param touches the
worker-pickle contract and rides B6 with the unit-dispatch change.

## Work items

1. **Tighten the ABC contract:** `ManagedBuffer.shape: Any → DataShape`.
   `SharedNPArray` side: it already keeps its DataShape when a real manager
   is present — drop the reverse-engineering path (mem.py:399-410) that
   re-wraps the numpy tuple with views open, and the rank-reduced branch
   that *guesses* trailing field names and synthesizes an anonymous class no
   `state.layout` can ever equal.
2. **Type `BufferSpec.shape`** and derive `layout` as `type(spec.shape)`
   (kept as a property for the #97 stage-contract vocabulary).
3. **Delete the `WorkingArray` Protocol** (core.py:245-250) — it declares
   `shape -> tuple[int, ...]`, directly contradicting the ABC for the same
   field; used once, never isinstance-checked. Don't grow a second interface
   beside the ABC.
4. **Fix the four shape drop-sites:**
   | # | site | today |
   |---|---|---|
   | 1 | `allocate_output` no-storage fallback (core.py:348-350) | typed shape → `astuple` → bare `np.empty`; caller gets ndarray as buffer *and* view. Pinned by test_geometry_stages.py:128-130 — that test's expectation changes with this fix |
   | 2 | normalize.py:43-48 | biggest hot-path allocation reads `np.asarray(state.working_array).shape` (numpy tuple) while `state.layout` sits unconsulted |
   | 3 | artifact allocations (read.py:341-346, phase.py:38, normalize.py:72, execution.py:183-196) | hand-built tuples straight into `storage.allocate` |
   | 4 | `SharedNPArray.__getitem__` view path (mem.py:368-375 → 404-410) | original DataShape carried but shape re-derived per access; correct by coincidence of rank |
5. **Promote `source_window` to a declared stage contract.** In-repo
   precedent: `AffineSkewMovementStage.output_window` (geometry.py:297-300).
   This also removes the need for `sync_buffer_spec()`-style re-reconciling
   (the method itself is deleted in B12 with the state unification).
6. **`TomoWindow.recon_shape()`** = `ReconOrder.of(self.internal.shape)` —
   one line whose entire value is choosing padded-vs-core **once**. Decided:
   **core** (`internal`) semantics — the real pipeline crops to
   `window.internal` before recon (runner.py:481-489), so core is what recon
   sees; the two existing hand-rolled versions disagree (examples/b10:132
   and io.py:353 use padded `shape.Y` — wrong side of the decision, they
   migrate to this method). Honest limits, documented on the method: it
   yields the square *physical* shape only (published/cropped
   `Z=base_width, X=output_width` stays the recon stage's policy), and it's
   meaningless for windows never reconstructed.
   **No Z-clobber guard on `ReconOrder.of`** (decided): a guard is
   underinclusive as a safety net and unreliable — X/Y-only inputs are
   legitimately convertible in some contexts and not others, which only the
   call site can know. `of` stays permissive; correctness is the caller's.
7. **Geometry-copy seam hardening** (from the deepcopy investigation):
   `rescope`/`with_pad` open with `deepcopy(self)` (window.py:90, 110). The
   load-bearing job is *subclass preservation* (polymorphic copy-constructor
   idiom), not aliasing protection — the re-init overwrites every
   DataRef-level field, and `rescope` already shares `step`/`pad_*`
   DataShape instances between original and copy, so the code is already
   betting on immutability-by-convention. On TomoWindow/DataRef, deepcopy ≡
   shallow; on TomoDataSet it would clone the full angles array per call
   (never rescoped in production today — by convention only). Change:
   `deepcopy` → `copy.copy()` + re-init, with the two conventions it relies
   on **stated at the seam**: (a) DataShape instances are immutable (never
   field-assigned; all operators return fresh instances), (b)
   `TomoDataSet.angles`/`acquisition_indices` are read-only. Cheap
   enforcement for (b): `angles.setflags(write=False)` (and
   `acquisition_indices` likewise) at construction — turns the convention
   into an enforced invariant for the arrays.

## Deletion ledger

- mem.py:399-410 reverse-engineering + anonymous-class synthesis.
- `WorkingArray` Protocol (core.py:245-250).
- The four drop-sites' hand-built tuples.
- recon.py:155-171 `_crop_buffer`'s
  `getattr(reconstruction.shape, "X", None)` duck-typing (consumed properly
  once buffers are typed; final deletion may land with B18).
- examples/b10:132 and io.py:353 hand-rolled recon-shape math → `recon_shape()`.
- window.py:90/110 `deepcopy` depth.

## Acceptance

- Tests: ABC contract (a ManagedBuffer without DataShape shape is a type
  error at the seam); drop-site regressions (buffer_spec.shape stays typed
  through allocate_output fallback — flips the test_geometry_stages.py
  expectation); `recon_shape()` pinned core-semantics (padded window in,
  core Y out); rescope-preserves-subclass pin (TomoWindow in → TomoWindow
  out) surviving the copy.copy switch; `angles` write attempt raises.
- No consumer-visible API change (geometry internals are consumer-safe).
- API not frozen until B11/B12 merge against it.
