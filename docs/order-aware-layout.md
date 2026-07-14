# Order-aware layout & domain contracts

Design spec for the pipeline layout/ordering contract tracked in **#97**.

This is the consolidated, implementable specification. The rationale, measurements,
and back-and-forth that produced it live in the #97 comment thread; this document
is the version-controlled reference the implementing PRs (A–D) update as reality
diverges from the sketch. Verified against master `53ea83d`.

## 1. Problem

The pipeline must not hardcode a universal `ProjOrder → SinoOrder` transition. The
frozen producers prove the efficient layout is producer-specific:

- **recon_minimal** starts in `SinoOrder (Y, Theta, X)` (sliding-Y retention +
  the three Vo stripe passes are sinogram-local) and converts to
  `ProjOrder (Theta, Y, X)` **fused into** the skew/rotation/padding rewrite.
- **phase_contrast / AAA523** creates `ProjOrder` and **keeps it** — normalize,
  BAC, bilateral, skew, and movement are all projection-local. A forced transpose
  there needlessly moves ~5.15 GiB and risks a numerically different partition
  being mistaken for an algorithm failure.

On master today, **every non-recon stage declares `SinoOrder → SinoOrder`**
(normalize, phase incl. BAC, stripe, center, geometry, helical); only
`recon.gridrec`/ASTRA is order-flexible. That is exactly the universal-`SinoOrder`
assumption this work removes.

## 2. Two contract axes

Each Stage declares two orthogonal properties, both backward-compatible with the
existing `expected_input`/`produced_output` fields (defaults derive from them, so
current contracts keep their meaning unchanged).

### 2.1 Layout axis

Storage layout of the working array: `ProjOrder`, `SinoOrder`, `ReconOrder`.

```python
PRESERVE = object()  # sentinel: produced layout/domain == the consumed one

@dataclass(frozen=True)
class StageContract:
    # ... existing fields unchanged ...
    accepts: tuple[type, ...] | None = None   # consumable layouts; None -> (expected_input,)
    prefers: type | None = None               # planner tiebreak; None -> accepts[0]
    produces: type | object | None = None     # shape class | PRESERVE | None(=produced_output)
    scratch_from: dict[type, type] | None = None   # {ProjOrder: SinoOrder}: bounded per-sinogram scratch
    fused_transition: bool = False            # rewrite already touches every voxel; layout change rides free
```

### 2.2 Domain axis (ordering)

Data domain progresses one way only:

```
raw counts → transmission (flat/dark) → attenuation (log / phase-retrieved) → recon
```

```python
    accepts_domain: tuple[type, ...] | None = None
    produces_domain: type | object | None = None   # domain | PRESERVE
```

The two axes share the `PRESERVE` sentinel, the validator plumbing, and the test
structure — implement them together (PR A).

### 2.3 Execution state

```python
@dataclass
class PipelineState:
    ...
    layout: type = SinoOrder   # set by the read stage from its `order` param
    domain: type = RAW         # advanced by domain-producing stages
```

Stages consult `state.layout`/`state.domain` for axis mapping instead of hardcoding
`(Y, Theta, X)`. A small `lftomo/pipeline/stages/_layout.py` provides the helpers
(`projection_view`, `sinogram_view`, `sinogram_scratch`, `n_projections`,
`n_rows`) so per-stage migrations are mechanical.

## 3. The tomopy recon boundary (verified)

`tomopy.recon`'s internal working layout is **SinoOrder**. From
`tomopy/recon/algorithm.py::init_tomo`:

```python
# "Convert data to sinogram order"
if not sinogram_order:
    tomo = np.swapaxes(tomo, 0, 1)      # free view
tomo = np.require(tomo, requirements="AC")  # materializes a full contiguous copy
```

- `sinogram_order=True` (SinoOrder in) → **zero-copy**.
- `sinogram_order=False` (ProjOrder in) → the swap is free but `np.require("AC")`
  **copies the whole volume** (the swapped view isn't C-contiguous).
- **ASTRA is identical** — `AstraCudaStage` drives the same `init_tomo` path;
  `wrappers.py::astra_rec` consumes `tomo[i]` per sinogram.

So recon is `accepts=(ProjOrder, SinoOrder)`, **`prefers=SinoOrder`** — flexible
but asymmetric, not free both ways. For the retained-ProjOrder lineage the one
unavoidable conversion happens **once, inside tomopy**; an extra explicit transpose
earlier would make it two. The planner should surface this hidden boundary copy so
"conversions are explicit" holds even though the copy lives in tomopy.

## 4. Ordering invariants

Encoded via the **domain axis**, not a stage-order table. The planner **never**
reorders algorithms to reduce conversions.

**Hard (validator-enforced):**

1. `read` first — sets initial domain + layout.
2. `normalize.*` is the only raw-consumer → first after read, **exactly once**
   (the `beam_*` variants #93–#96 belong here too, authored order-aware).
3. **Transmission → attenuation exactly once.** `phase.paganin` embeds its own
   `-log`, so Paganin / BAC / a future standalone `minus_log` are *alternatives at
   the same boundary*, not composable steps. The exactly-once rule catches both
   "forgot to normalize" and "double log".
4. `center.*` before recon — already enforced by the `center` artifact dependency;
   no new rule.
5. `recon` → recon domain; `super_resolve.*`, recon-space filters, and `write.*`
   can only follow.
6. Geometry stages exclude `raw` from `accepts_domain` (flats are detector-aligned;
   never resample before normalization). Helical index-remap must precede any warp
   (it re-indexes rows; resampling first destroys row identity).

**Preferences (planner *warnings*, never rules — the frozen lineages disagree):**

- Stripe / dead-pixel cleanup prefers original detector alignment (before warp):
  recon_minimal conforms; the BAC lineages recorded geometry→stripe deliberately.
- Phase retrieval prefers unresampled data: BAC lineages conform; recon_minimal's
  Paganin-after-warp is safe only because its recorded tilt is `0.0`.

## 5. Skew / geometry and the fused transition

The recon_minimal boundary bundles three separable functions with different
necessity:

| function | necessity |
|---|---|
| **skew/rotation** | **data-conditional.** Every pinned lineage records tilt `0.0` (`SkewCorrectStage` early-returns at 0°), so today it is a pass-through everywhere. When nonzero it is `cv2.warpAffine` cubic — an **irreversible resample**; never run for convenience. For small tilts the resample-free path is `center.vo(tilt_fit=polynomial)` → per-row center array → gridrec per-slice center. |
| **padding** | required by gridrec window handling. |
| **transpose** | pure performance, never correctness. |

**The fusion is the legitimate convenience:** any one of the three needs a
full-volume rewrite anyway, so one pass beats three. The fused stage is
**direction-parametric** — the warp is projection-local, so it mirrors:

```
recon_minimal: read sino[:, p, :] → rotate/pad → write rotated[p, :, :]   (Sino → Proj)
mirror:        read proj[p]       → rotate/pad → write sino[:, p, :]       (Proj → Sino)
```

`accepts` either layout, `produces` the other, `fused_transition=True`, and it
**must special-case tilt=0 as transpose+pad with no interpolation** (as the producer
does). This yields the recommended ProjOrder shape (stripe on detector-aligned data,
one volume rewrite, zero-copy recon):

```
ProjOrder:  read → normalize → stripe.vo_* (scratch_from, detector-aligned)
         →  fused skew+transpose (Proj → Sino)     ← the one full-volume rewrite
SinoOrder:  → center → recon (zero-copy into tomopy)
```

### 5.1 Warp/retrieval non-commutativity (measured)

Cubic `warpAffine` + Paganin transfer/`-log`, A = warp-then-retrieve vs
B = retrieve-then-warp, on a synthetic sharp-edge + fringe image:

| tilt | global NRMS(A−B) | max local |
|---|---|---|
| 0.056° | 0.8% | ~17% of signal std |
| 0.5°–3° | ~1.6–1.7% | ~23% of signal std |

The cost **saturates at any nonzero angle** (it's the resample event, not the
angle). It's algorithmic, not implementation: cubic kernels overshoot at edges
(high-band energy rose to ~123% after a 1° warp — ringing injected, not just blur),
and Paganin's `-log` nonlinearity doesn't commute with interpolation. In the
continuum rotation commutes with the isotropic kernel; the divergence is purely
discrete. Neither order is "correct" — the recorded lineage order is truth. ~1.6%
global is inside the Tier-2 NRMSE≤0.15 tolerance, but ~20% edge-local differences
are a real reproducibility hazard at exactly the features people measure.

## 6. `scratch_from` semantics + the open benchmark

`scratch_from={ProjOrder: SinoOrder}` **never runs the algorithm on ProjOrder
data.** Per detector row `y`: gather `data[:, y, :]` into contiguous `(Theta, X)`
scratch → run the whole sinogram-local **chain** (`vo_dead → vo_large → vo_sort`)
on the buffer, one gather/scatter per row (not per stage) → scatter back. Rows are
independent → bounded-concurrency parallelism. tomopy's `remove_all_stripe` works
this way internally.

In theory one shared scratch pass ≈ one transpose in total traffic, with peak
memory of one sinogram (~MBs) vs a second volume (~GiBs). **Open empirical
question:** the recon_minimal straight-to-sino rewrite was a *significant measured*
speedup, so "gather/scatter is a small fraction of Vo runtime" is a **hypothesis,
not a fact** (cache/TLB behavior at scale, mmap, tomopy sharedmem fan-out could all
favor the rewrite). Therefore `scratch_from` is a **capability declaration, not a
planner default**, decided by the AC-7 benchmark with three arms:

1. shared per-row scratch (chain-grouped, bounded concurrency),
2. fused/explicit transition → contiguous SinoOrder processing (recon_minimal),
3. direct strided in-place processing (baseline to avoid),

measured at the ~5 GiB AAA523 window and a full 1100-row stack (wall-clock + peak
RSS). If (2) wins decisively at scale, planner guidance flips to "prefer the fused
transition whenever a sinogram-local segment exists," and scratch stays as the
bounded-memory fallback.

## 7. Stage migration list (verified iteration patterns on `53ea83d`)

Effort: **A** = contract + helper swap (mechanical); **B** = A plus a
scratch/writeback path; **C** = behavioral redesign.

| Stage(s) | locality | target contract | class |
|---|---|---|---|
| `normalize.weighted/theta_switch/taper_blended` | projection-local (`data[:, p, :]`) | accepts (Proj, Sino), prefers **Proj**, PRESERVE | A |
| `phase.paganin` / `paganin_phase_absorption` | projection-local | accepts (Proj, Sino), prefers **Proj**, PRESERVE | A |
| **`phase.bac`** | projection-local | accepts (Proj, Sino), prefers **Proj**, PRESERVE; one `(Y,X)` projection at a time, bounded concurrency, in-place writeback | A |
| `filter.bilateral` | projection-local (already has `order` param) | accepts (Proj, Sino), prefers **Proj**, PRESERVE | A |
| `geometry.skew` / `movement` | projection-local | accepts (Proj, Sino), prefers **Proj**, PRESERVE | A |
| **`geometry.affine_skew_movement`** | the boundary rewrite | accepts either, **produces the other, `fused_transition=True`** | C |
| `helical.index_remap` | per-projection row shifts | accepts (Proj, Sino), prefers Proj, PRESERVE | A |
| `stripe.vo_dead/large/sort/family`, `stripe.fw` | **sinogram-local** (`data[y]`) | accepts Sino, prefers Sino, PRESERVE, **`scratch_from={Proj: Sino}`** | B |
| `stripe.all_of_the_stripe` | batched tomopy (transposes internally) | accepts (Proj, Sino); derive from `state.layout` | A |
| `center.manual` | no data access | accepts (Proj, Sino), PRESERVE | A |
| `center.vo/pc/entropy_local` | sinogram/projection reads | accepts (Proj, Sino) via per-view extraction; PRESERVE | B |
| `recon.gridrec` + ASTRA family | both (tomopy native) | accepts (Proj, Sino), **prefers Sino**; flag from `state.layout` | A |
| `read.tiff_stack` / `read.hdf5_dxchange` | — | set `state.layout` from `order`; stop forcing sinogram at read | A |
| `write.*`, `super_resolve.*` | recon-domain | unaffected | — |
| `normalize.beam_*` (#93–#96) | projection-local | **author order-aware from the start** | — |

Net: ~10 class-A (mechanical — and the projection-local ones get *faster* in
ProjOrder: strided `data[:, p, :]` → contiguous slices), 2 class-B, 1 class-C.

## 8. Phasing → sub-issues

- **A — mechanism:** contract fields (both axes) + `PipelineState.layout/domain` +
  `_layout.py` helpers + validator rules (layout composition, domain monotonicity,
  exactly-once normalize + transmission→attenuation) + `layout.convert` stage +
  gridrec layout derivation. Backward-compatible; **gates B/C/D**. Also lands this
  doc.
- **B — projection-local migrations:** the class-A stages above.
  **Independently unblocks the AAA523 ProjOrder producer path**, with the corrected
  140-row BAC window (80 core + 2×30 overlap).
- **C — sinogram-local scratch + AC-7 benchmark:** the `scratch_from` path and the
  three-arm benchmark that settles scratch-vs-rewrite.
- **D — fused boundary + lineage tests:** `geometry.affine_skew_movement` → fused
  direction-parametric Sino↔Proj rewrite; lineage tests pinning recon_minimal's
  `Sino→Proj` and phase_contrast's retained `Proj`.

## 9. Open questions for the maintainer

1. Is `scratch_from` acceptable as a contract field, or is it a pure
   execution-planner concern?
2. Should `layout.convert` refuse to run unless the YAML carries an explicit
   `reason:` param, to keep full-volume conversions deliberate?
3. `RAW`/`TRANSMISSION`/`ATTENUATION`/`RECON` as marker classes in
   `lftomo/util/shapes.py` alongside the layout classes, or a separate `domains`
   module?
