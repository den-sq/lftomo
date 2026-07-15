# Order-aware layout & domain contracts

Design spec for the pipeline layout/ordering contract tracked in **#97**.

This is the consolidated, implementable specification. The rationale, measurements,
and back-and-forth that produced it live in the #97 comment thread; this document
is the version-controlled reference the implementing PRs (A–D) update as reality
diverges from the sketch. Phase A implementation started from master `f08dba6`,
after #103 was merged.

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

Master no longer has one uniform behavior. #91 added working, per-instance
`order`/`array_order` branches for theta-switch/taper normalization, BAC,
bilateral, skew, movement, helical remap, Fourier-wavelet stripe removal,
manual center, and both readers. The Tier-2 AAA523 runner uses those branches
successfully in ProjOrder. Weighted normalization, both Paganin variants,
Vo-family stripe removal, `all_of_the_stripe`, and the automatic center stages
remain SinoOrder-only. Gridrec/ASTRA can be configured for either layout.

Those ad-hoc flags solve individual algorithms but not the architectural
problem: layout is not carried in `PipelineState`, contracts describe one
constructor-selected layout rather than capabilities, the runner still rewrites
producer configurations through `_make_executable`, and no plan records explicit
or hidden conversions. This work replaces the flags as the source of truth while
preserving them as compatibility inputs during migration.

## 2. Two contract axes

Each Stage declares two orthogonal properties, both backward-compatible with the
existing `expected_input`/`produced_output` fields (defaults derive from them, so
current contracts keep their meaning unchanged).

### 2.1 Layout axis

Storage layout of the working array: `ProjOrder`, `SinoOrder`, `ReconOrder`.

```python
from enum import Enum, auto

class ContractEffect(Enum):
    PRESERVE = auto()

PRESERVE = ContractEffect.PRESERVE

class Locality(Enum):
    PROJECTION = auto()
    SINOGRAM = auto()
    VOLUME = auto()
    METADATA = auto()

@dataclass(frozen=True)
class StageContract:
    # ... existing fields unchanged ...
    accepts: tuple[type, ...] | None = None   # consumable layouts; None -> (expected_input,)
    prefers: type | None = None               # planner tiebreak; None -> accepts[0]
    produces: type | ContractEffect | None = None  # shape class | PRESERVE | None(=produced_output)
    locality: Locality | None = None
    bounded_views_from: tuple[type, ...] = ()      # full-volume layouts supporting bounded locality views
    fused_transition: bool = False            # rewrite already touches every voxel; layout change rides free
    resamples_detector: bool = False           # irreversible detector-space interpolation
    requires_row_identity: bool = False        # invalid after detector resampling
```

Use the enum singleton rather than a bare `object()` so identity remains clear
through imports, test diagnostics, and future plan serialization. A frozen
contract must not contain a mutable `dict`.

### 2.2 Domain axis (ordering)

Data domain progresses one way only:

```
raw counts → transmission (flat/dark) → attenuation (log / phase-retrieved) → recon
```

Domains live in a separate `lftomo/pipeline/domains.py`; they are processing
state, not array shapes:

```python
class DataDomain(Enum):
    RAW = auto()
    TRANSMISSION = auto()
    ATTENUATION = auto()
    RECON = auto()

# StageContract fields
accepts_domain: tuple[DataDomain, ...] | None = None
produces_domain: DataDomain | ContractEffect | None = None
```

The two axes share the `PRESERVE` sentinel, the validator plumbing, and the test
structure — implement them together (PR A).

### 2.3 Execution state

```python
@dataclass
class PipelineState:
    ...
    layout: type | None = None                # set by read or supplied by a fragment caller
    domain: DataDomain | None = None           # advanced by domain-producing stages
```

Stages consult `state.layout`/`state.domain` for axis mapping instead of hardcoding
`(Y, Theta, X)`. `PipelineState.with_array()` must preserve both fields unless a
stage explicitly changes them. `None` is retained only as a legacy-fragment
compatibility state; complete pipelines establish both axes at their read Stage,
and a fragment requesting axis validation supplies them explicitly. A small
`lftomo/pipeline/stages/_layout.py` provides the helpers
(`projection_view`, `sinogram_view`, `sinogram_scratch`, `n_projections`,
`n_rows`) so per-stage migrations are mechanical.

Constructor `order` flags remain accepted during migration but are checked
against `state.layout`; they do not override it silently. Canonical YAML should
ultimately omit redundant per-stage order declarations.

## 3. The tomopy recon boundary (verified)

`tomopy.recon`'s internal working layout is **SinoOrder**. From
`tomopy/recon/algorithm.py::init_tomo`:

```python
# "Convert data to sinogram order"
if not sinogram_order:
    tomo = np.swapaxes(tomo, 0, 1)      # free view
tomo = np.require(tomo, requirements="AC")  # materializes a full contiguous copy
```

- `sinogram_order=True` with contiguous SinoOrder input → **zero-copy**.
- `sinogram_order=False` with contiguous ProjOrder input → the swap is free but
  `np.require("AC")` **copies the whole volume** (the swapped view is not
  C-contiguous).
- **ASTRA is identical** — `AstraCudaStage` drives the same `init_tomo` path;
  `wrappers.py::astra_rec` consumes `tomo[i]` per sinogram.

So recon is `accepts=(ProjOrder, SinoOrder)`, **`prefers=SinoOrder`** — flexible
but asymmetric, not free both ways. Copy accounting must distinguish a view from
a materialized transition:

| input to recon | lftomo copy | TomoPy copy | total full-volume copies |
|---|---:|---:|---:|
| contiguous ProjOrder, `sinogram_order=False` | 0 | 1 | 1 |
| non-contiguous SinoOrder transpose view, `sinogram_order=True` | 0 | 1 | 1 |
| contiguous SinoOrder produced by explicit copy, `sinogram_order=True` | 1 | 0 | 1 |
| contiguous SinoOrder produced during an already-required fused rewrite | 0 additional | 0 | 0 layout-only |

An explicit, contiguous Proj→Sino conversion therefore **replaces** TomoPy's
copy; it does not inherently add a second one. Two copies occur only if lftomo
materializes a conversion and then supplies the wrong layout flag, or otherwise
hands TomoPy another non-contiguous representation. The planner must surface the
hidden TomoPy copy and choose where the one required materialization occurs based
on lifetime, scratch storage, fusion opportunities, and downstream locality.

## 4. Ordering invariants

Most are encoded via the **domain axis**. Row-identity constraints that cannot be
expressed by the four domains use explicit contract capabilities. The planner
**never** reorders algorithms to reduce conversions.

Validation has two modes:

- **complete pipeline** (runner/YAML): enforces read-first and exactly-once
  boundary rules;
- **fragment** (Stage unit tests, reusable sub-pipelines, callers supplying an
  existing array): validates composition from the caller-supplied initial
  layout/domain but does not require read, normalize, attenuation, recon, or
  write to be present.

This distinction is required for backward compatibility with existing Stage
tests and legitimate pipeline fragments.

**Hard (validator-enforced):**

1. In complete mode, `read` first — sets initial domain + layout.
2. In complete mode, `normalize.*` is the only raw-consumer → first after read, **exactly once**
   (the `beam_*` variants #93–#96 belong here too, authored order-aware).
3. In complete reconstruction mode, **Transmission → attenuation exactly once.** `phase.paganin` embeds its own
   `-log`, so Paganin / BAC / a future standalone `minus_log` are *alternatives at
   the same boundary*, not composable steps. The exactly-once rule catches both
   "forgot to normalize" and "double log".
4. `center.*` before recon — already enforced by the `center` artifact dependency;
   no new rule.
5. `recon` → recon domain; `super_resolve.*`, recon-space filters, and `write.*`
   can only follow.
6. Geometry stages exclude `raw` from `accepts_domain` (flats are detector-aligned;
   never resample before normalization). Helical index-remap declares
   `requires_row_identity=True`; the validator rejects it after any stage with
   `resamples_detector=True` because resampling destroys row identity. The
   four-value domain axis alone cannot encode this ordering.

Contracts are parameter-dependent. A disabled Stage that returns without
changing data declares `produces_domain=PRESERVE`, does not claim a fused layout
transition, and does not satisfy an exactly-once boundary. The same applies to
zero-effect configuration branches generally; validation describes what the
concrete Stage instance will execute, not what its class could execute.

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
| **skew/rotation** | **data-conditional.** Recon-minimal and the frozen B10 compatibility lineages record zero skew at their relevant boundary, but AAA523 phase contrast records and numerically reproduces a nonzero **0.149°** affine skew after BAC and before Fourier-wavelet stripe removal. When nonzero it is `cv2.warpAffine` cubic — an **irreversible resample**; never run for layout convenience. For lineages that permit it, the resample-free small-tilt path is `center.vo(tilt_fit=polynomial)` → per-row center array → gridrec per-slice center. It is not a substitute when reproducing AAA523's recorded producer order. |
| **padding** | required by gridrec window handling. |
| **transpose** | pure performance, never correctness. |

**The fusion is the legitimate convenience:** any one of the three needs a
full-volume rewrite anyway, so one pass beats three. The fused stage is
**direction-parametric** — the warp is projection-local, so it mirrors:

```
recon_minimal: read sino[:, p, :] → rotate/pad → write rotated[p, :, :]   (Sino → Proj)
mirror:        read proj[p]       → rotate/pad → write sino[:, p, :]       (Proj → Sino)
```

For configurations that request the boundary transition, the fused Stage
`accepts` either layout, `produces` the other, `fused_transition=True`, and it
**must special-case tilt=0 as transpose+pad with no interpolation** (as the
recon-minimal producer does). It must not force every ordinary affine-skew Stage
to flip layout: AAA523's separate nonzero affine skew preserves ProjOrder. This
yields the recommended ProjOrder shape for a lineage choosing the fused boundary
(stripe on detector-aligned data, one volume rewrite, zero-copy recon):

```
ProjOrder:  read → normalize → stripe.vo_* (bounded views, detector-aligned)
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

**This hazard does not actually bite any pinned lineage** — but note *why*, so
AAA523's nonzero skew is not mistaken for tripping it. AAA523 applies its 0.149°
skew **after** BAC, i.e. retrieve-then-warp (the numerically-safe order); the
warp-then-retrieve case only arises in recon_minimal, whose recorded skew is
`0.0`. So the non-commutativity is realized in no pinned producer today; it
becomes live only if a future lineage warps *before* phase retrieval with a
nonzero skew.

## 6. Bounded-view semantics, segment execution, and the open benchmark

`locality=SINOGRAM, bounded_views_from=(ProjOrder,)` declares a capability; it
does not select an execution strategy and does not claim that a sinogram-local
kernel can consume the full ProjOrder array directly. Per detector row `y`, the
bounded-view executor may gather `data[:, y, :]` into contiguous `(Theta, X)`
scratch → run the whole sinogram-local **segment**
(`vo_dead → vo_large → vo_sort`) on the buffer, one gather/scatter per row (not
per Stage) → scatter back. Rows are independent → bounded-concurrency
parallelism. TomoPy's `remove_all_stripe` works this way internally.

`StageContract` declares locality and bounded-view capability. `PipelinePlan`
groups adjacent compatible Stages into execution segments and selects bounded
views versus a materialized conversion. `Pipeline.run` (or a dedicated segment
executor) executes that plan. Merely adding a mapping to each Stage contract is
insufficient: the current sequential `Pipeline.run` would otherwise gather and
scatter once per Stage, defeating the chain-sharing requirement.

In theory one shared scratch pass ≈ one transpose in total traffic, with peak
memory of one sinogram (~MBs) vs a second volume (~GiBs). **Open empirical
question:** the recon_minimal straight-to-sino rewrite was a *significant measured*
speedup, so "gather/scatter is a small fraction of Vo runtime" is a **hypothesis,
not a fact** (cache/TLB behavior at scale, mmap, tomopy sharedmem fan-out could all
favor the rewrite). Therefore bounded-view support is a **capability declaration,
not a planner default**, decided by the AC-7 benchmark with three arms:

1. shared per-row scratch (chain-grouped, bounded concurrency),
2. fused/explicit transition → contiguous SinoOrder processing (recon_minimal),
3. direct strided in-place processing (baseline to avoid),

measured at the ~5 GiB AAA523 window and a full 1100-row stack (wall-clock + peak
RSS). If (2) wins decisively at scale, planner guidance flips to "prefer the fused
transition whenever a sinogram-local segment exists," and bounded views remain
the bounded-memory fallback.

Arm (3) already has a **live, numerically-validated anchor**: post-#91,
`stripe.fw` in the AAA523 phase-contrast path iterates ProjOrder strided
sinograms directly (exactly the direct-strided-in-place case), and #91's Tier-2
run reproduced it within tolerance. The benchmark should measure against that
recorded configuration rather than only a synthetic arm-3, so its baseline is
grounded in a real producer.

## 7. Stage migration list (verified on `c775826`)

Effort: **A** = contract + helper swap (mechanical); **B** = A plus a
bounded-view/writeback path; **C** = behavioral redesign. “Order branch present”
means #91 already supplied working Proj/Sino iteration but the Stage still needs
to adopt `state.layout`, capability contracts, and domain declarations.

| Stage(s) | locality | master status | target contract | class |
|---|---|---|---|---|
| `normalize.weighted/theta_switch/taper_blended` | projection-local | theta/taper order branch present; weighted Sino-only | accepts (Proj, Sino), prefers **Proj**, PRESERVE | A |
| `phase.paganin` / `paganin_phase_absorption` | projection-local | Sino-only | accepts (Proj, Sino), prefers **Proj**, PRESERVE | A |
| **`phase.bac`** | projection-local | order branch present and AAA523-validated | accepts (Proj, Sino), prefers **Proj**; one `(Y,X)` projection at a time, bounded concurrency, in-place writeback | A |
| `filter.bilateral` | projection-local | order branch present | accepts (Proj, Sino), prefers **Proj**, PRESERVE | A |
| `geometry.skew` / `movement` | projection-local | order branches present | accepts (Proj, Sino), prefers **Proj**, PRESERVE; skew sets `resamples_detector` when enabled/nonzero | A |
| **`geometry.affine_skew_movement`** | optional boundary rewrite | Sino-only compatibility wrapper | accepts either; when transition configured, **produces the other, `fused_transition=True`** | C |
| `helical.index_remap` | per-projection row shifts | order branch present | accepts (Proj, Sino), prefers Proj, PRESERVE; `requires_row_identity=True` | A |
| `stripe.vo_dead/large/sort/family` | **sinogram-local** (`data[y]`) | Sino-only | accepts Sino, prefers Sino, PRESERVE; `locality=SINOGRAM`, bounded views from Proj | B |
| `stripe.fw` | sinogram-local | order branch directly iterates Proj strided views | same bounded-view segment contract as other sinogram-local kernels | B |
| `stripe.all_of_the_stripe` | batched TomoPy | Sino-only | accepts (Proj, Sino); derive TomoPy flag from `state.layout` | A |
| `center.manual` | no data access | order branch present | accepts (Proj, Sino), PRESERVE | A |
| `center.vo/pc/entropy_local` | sinogram/projection reads | Sino-only | accepts (Proj, Sino) through bounded view extraction; PRESERVE | B |
| `recon.gridrec` + ASTRA family | both (TomoPy native) | constructor-selected layout | accepts (Proj, Sino), **prefers Sino**; flag from `state.layout` | A |
| `read.tiff_stack` / `read.hdf5_dxchange` | — | already emit either layout | set `state.layout` from `order`; remove runner's universal `_make_executable` rewrite | A |
| `write.*`, `super_resolve.*` | recon-domain | unaffected | recon domain only | — |
| `normalize.beam_*` (#93–#96) | projection-local | not present on master | author directly against the state-driven order-aware contract | A |

The algorithmic ProjOrder work already merged in #91 reduces Phase B risk, but
does not remove it: the migration must delete special reference-runner layout
injection and prove ordinary YAML execution selects the same layout.

## 8. Phasing → sub-issues

- **A — mechanism:** contract fields (both axes, locality, row-identity flags) +
  `PipelineState.layout/domain` + `_layout.py` helpers + complete/fragment
  validator modes (layout composition, domain monotonicity, exactly-once
  normalize + transmission→attenuation in complete mode) + `PipelinePlan` records
  and warnings + `layout.convert` stage + gridrec layout derivation. Every
  YAML-authored `layout.convert` requires a non-empty `reason:`. Backward-compatible;
  **gates B/C/D**. Also updates this doc as implementation details settle.
- **B — projection-local migrations:** the class-A stages above.
  #91 already proved the AAA523 ProjOrder algorithms with the corrected 140-row
  BAC window (80 core + 2×30 overlap); B moves that behavior from ad-hoc order
  flags/reference-runner injection into the common state/contract mechanism and
  covers the remaining projection-local Stages.
- **C — sinogram-local bounded views + segment executor + AC-7 benchmark:** add
  locality kernels, plan adjacent compatible Stages as one gather/scatter segment,
  implement bounded concurrency, and run the three-arm benchmark that settles
  bounded-view-vs-rewrite guidance.
- **D — fused boundary + lineage tests:** `geometry.affine_skew_movement` → fused
  direction-parametric Sino↔Proj rewrite when configured as a transition; add the
  missing fused Stage to the recon-minimal canonical YAML; lineage tests pin
  recon-minimal's `Sino→Proj`, AAA523's nonzero 0.149° Proj-preserving skew and
  retained `Proj`, and the absence of an extra lftomo layout conversion before
  AAA523 recon (TomoPy's recorded boundary copy remains visible in the plan).

## 9. Resolved design decisions

1. Contracts declare locality and `bounded_views_from`; the planner chooses and
   the segment executor implements gather/scatter. Scratch strategy is not a
   per-Stage mutable mapping.
2. A YAML-authored `layout.convert` refuses to run without a non-empty `reason:`.
   Planner-generated conversions carry a generated rationale in `PipelinePlan`.
3. Domains use `DataDomain` in a separate `lftomo/pipeline/domains.py` module.
4. `PRESERVE` is an enum singleton, not a bare object.
5. Complete-pipeline invariants are opt-in to complete validation; fragment
   validation composes from caller-supplied layout/domain without requiring a
   read/normalize/recon skeleton.
