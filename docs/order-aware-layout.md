# Order-aware layout & domain contracts

Design spec for the pipeline layout/ordering contract tracked in **#97**.

This is the consolidated, implementable specification. The rationale, measurements,
and back-and-forth that produced it live in the #97 comment thread; this document
is the version-controlled reference the implementing PRs (A–D) update as reality
diverges from the sketch. Phase A implementation started from master `f08dba6`,
after #103 was merged.

**Status:** all four phases are merged — A (#104), B (#105), C (#106), D (#107).
This document now reflects the landed implementation rather than a forward plan;
§6 records the completed AC-7 benchmark result and the planner guidance it settled.

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

Processing stages no longer accept per-instance `order`, `array_order`, or
`sinogram_order` inputs. Their contracts declare capabilities, and their
runtime axes come exclusively from the shaped working buffer in
`PipelineState`. Readers still declare the layout they produce, while explicit
layout-transition stages declare the layout they create.

## 2. Two contract axes

Each Stage declares two orthogonal properties.

### 2.1 Layout axis

Storage layout of the working array: `ProjOrder`, `SinoOrder`, `ReconOrder`.

```python
from enum import Enum, auto

class ContractEffect(Enum):
    PRESERVE = auto()

PRESERVE = ContractEffect.PRESERVE

@dataclass(frozen=True)
class StageContract:
    # ... existing fields unchanged ...
    accepts: tuple[type, ...] | None = None   # consumable layouts; None -> unconstrained
    prefers: type | None = None               # planner tiebreak; None -> accepts[0]
    produces: type | ContractEffect | None = None  # shape class | PRESERVE | None(unknown)
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
and a fragment requesting axis validation supplies them explicitly. Stages use
the named-axis operations on their canonical `DataShape` directly; the former
compatibility helper module has been retired.

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
3. In complete reconstruction mode, **Transmission → attenuation exactly once.**
   Legacy `phase.paganin` and `phase.bac` embed `-log` and remain terminal
   compatibility alternatives. Composable `phase.paganin_filter` and
   `phase.bac_correction` preserve transmission and may be repeated before one
   explicit `phase.negative_log`. The exactly-once rule catches both a missing
   terminal conversion and a double log.
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
concrete Stage instance will execute, not what its class could execute. Each
Stage owns a nested immutable `Params(StageParams)` model; the base validates
configuration once and calls `contract_for(cls, params)` to derive those facts
from the class contract. Kernels consume `stage.params`, and planners consume
the resulting instance contract. Parameter schemas are exposed through
`StageClass.Params.model_json_schema()`.

Same-domain operations intentionally have no implied adjacency rule. A Stage
that accepts transmission and preserves transmission can follow another such
Stage; non-commutativity alone is not invalid. The phase audit found that the
scientifically invalid phase orders all cross the terminal domain boundary, so
the existing domain contract is sufficient and no generic semantic-effect
fields are added. A future effect mechanism requires an audited precondition
that cannot be represented by input/output domain, not an algorithm-name
allowlist.

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
| **skew/rotation** | **data-conditional.** The recon-minimal Click-default canonical path and frozen B10 compatibility lineages use zero skew at this boundary, but the AAA592 recon-minimal reference manifest records `tilt_radians=-0.003134796238244514` (a corrective affine angle of about **+0.1796°**) before Paganin. AAA523 phase contrast separately records and numerically reproduces a nonzero **0.149°** affine skew after BAC and before Fourier-wavelet stripe removal. When nonzero these are cubic `cv2.warpAffine` resamples — **irreversible** and never run for layout convenience. For lineages that permit it, the resample-free small-tilt path is `center.vo(tilt_fit=polynomial)` → per-row center array → gridrec per-slice center. It is not a substitute when reproducing a recorded producer order. |
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

AAA523 does **not** trip this hazard: it applies its 0.149° skew **after** BAC,
i.e. retrieve-then-warp. The AAA592 recon-minimal reference does: its recorded
nonzero `projection_rotation` precedes Paganin, so reproducing that reference
must preserve warp-then-retrieve exactly. The Click-default recon-minimal
canonical template remains the zero-skew special case. Tests therefore pin both
the default fused transpose-without-interpolation and the nonzero AAA592 lineage
ordering instead of treating all recon-minimal provenance as zero skew.

## 6. Bounded-view semantics, segment execution, and the AC-7 benchmark

`SinogramStage` with `bounded_views_from=(ProjOrder,)` declares a capability; it
does not select an execution strategy and does not claim that a sinogram-local
kernel can consume the full ProjOrder array directly. Per detector row `y`, the
bounded-view executor may gather `data[:, y, :]` into contiguous `(Theta, X)`
scratch → run the whole sinogram-local **segment**
(`vo_dead → vo_large → vo_sort`) on the buffer, one gather/scatter per row (not
per Stage) → scatter back. Rows are independent → bounded-concurrency
parallelism. TomoPy's `remove_all_stripe` works this way internally.

Each invocation retains the full working-buffer handle and held source window.
Its `PipelineState.work_range` assigns packed buffer ordinals on the fan-out
axis. Numerical views and matching scratch planes derive from that assignment;
source positions derive through `TomoWindow.source_index`, including source
offsets, sampling stride, and halo. The row view passed to `run_sinogram` is a
numerical boundary, not a separate source window. This contract also applies
to serial execution and projection-local kernels.

`SinogramStage` supplies the row kernel adapter; `StageContract` declares
bounded-view capability. `PipelinePlan`
groups adjacent compatible Stages into execution segments and selects bounded
views versus a materialized conversion. `Pipeline.run` (or a dedicated segment
executor) executes that plan. Merely adding a mapping to each Stage contract is
insufficient: the current sequential `Pipeline.run` would otherwise gather and
scatter once per Stage, defeating the chain-sharing requirement.

In theory one shared scratch pass ≈ one transpose in total traffic, with peak
memory of one sinogram (~MBs) vs a second volume (~GiBs). The empirical question
AC-7 was built to answer: the recon_minimal straight-to-sino rewrite was a
*significant measured* speedup, so "gather/scatter is a small fraction of Vo
runtime" was a **hypothesis, not a fact** (cache/TLB behavior at scale, mmap,
tomopy sharedmem fan-out could all favor the rewrite). Bounded-view support is
therefore a **capability declaration, not a planner default**, settled by the
AC-7 benchmark with three arms:

1. shared per-row scratch (chain-grouped, bounded concurrency),
2. fused/explicit transition → contiguous SinoOrder processing (recon_minimal),
3. direct strided in-place processing (baseline to avoid),

measured at the ~5 GiB AAA523 window and a full 1100-row stack (wall-clock + peak
RSS). **AC-7 result (recorded below):** no arm won decisively under the 10%
threshold and arm (2) never won — it added up to 23.4 GiB peak RSS at full scale.
Planner guidance therefore did **not** flip to the fused transition: chain-grouped
bounded execution is the general contiguous-input default, direct strided stays a
validated kernel-specific optimization, and a full transition is reserved for
cases where a downstream consumer or an already-required fused boundary amortizes
it. Bounded views remain the capability-gated default, with the safe fallback of
one row worker.

Arm (3) already has a **live, numerically-validated anchor**: post-#91,
`stripe.fw` in the AAA523 phase-contrast path iterates ProjOrder strided
sinograms directly (exactly the direct-strided-in-place case), and #91's Tier-2
run reproduced it within tolerance. The benchmark should measure against that
recorded configuration rather than only a synthetic arm-3, so its baseline is
grounded in a real producer.

The Phase-C implementation provides `benchmarks/order_layout.py`. It runs one
arm per process, reports wall time and peak RSS, and requires `--scratch` for
arrays larger than 1 GiB so the large cases use memmap storage rather than
risk another WSL OOM. All arms apply the requested row concurrency. The
transition arm includes the ProjOrder-to-SinoOrder materialization in its timer
and releases the obsolete source mapping before processing. Run all three arms
for each required shape:

```text
python -m benchmarks.order_layout --arm bounded    --shape 1969,140,5008 --workers 4 --scratch /fast/ac7
python -m benchmarks.order_layout --arm transition --shape 1969,140,5008 --workers 4 --scratch /fast/ac7
python -m benchmarks.order_layout --arm strided    --shape 1969,140,5008 --workers 4 --scratch /fast/ac7

python -m benchmarks.order_layout --arm bounded    --shape 1969,1100,5008 --workers 4 --scratch /fast/ac7
python -m benchmarks.order_layout --arm transition --shape 1969,1100,5008 --workers 4 --scratch /fast/ac7
python -m benchmarks.order_layout --arm strided    --shape 1969,1100,5008 --workers 4 --scratch /fast/ac7
```

### AC-7 result

The six isolated WSL2 runs used four row workers, NumPy `float32` memmaps on
the WSL ext4 filesystem, a 72 GiB WSL limit, and one thread each for OpenMP,
OpenBLAS, and MKL. The timer begins after synthetic projection initialization;
the absolute RSS includes the resident projection mapping, while RSS delta
shows additional memory used during the measured strategy. All three arms
produced identical fixed-coordinate fingerprints at each shape.

The `bounded` arm drives the real chain-grouped segment executor via
`Pipeline.run`, while the `strided` and `transition` arms use the direct
per-row baseline loop. The `strided` arm is therefore a simplified
direct-strided loop over the Vo stages, not `stripe.fw`'s production
Fourier-wavelet kernel; its timing isolates the layout/execution path rather
than that specific kernel's cost. The asymmetry is deliberately conservative —
the bounded arm runs the "real" executor yet still does not win decisively.

| Shape `(Theta,Y,X)` | Arm | Time | vs fastest | Peak RSS | RSS delta |
| --- | --- | ---: | ---: | ---: | ---: |
| `1969,140,5008` | bounded | 376.21 s | fastest | 6.34 GiB | 1.03 GiB |
| `1969,140,5008` | transition | 400.84 s | +6.55% | 11.34 GiB | 6.02 GiB |
| `1969,140,5008` | strided | 394.68 s | +4.91% | 6.20 GiB | 0.88 GiB |
| `1969,1100,5008` | bounded | 3042.07 s | +4.38% | 41.61 GiB | 1.03 GiB |
| `1969,1100,5008` | transition | 3194.29 s | +9.61% | 65.00 GiB | 24.42 GiB |
| `1969,1100,5008` | strided | 2914.31 s | fastest | 41.46 GiB | 0.89 GiB |

No arm wins decisively under the project's 10% threshold: bounded is fastest
for the 140-row window, while direct strided is fastest for 1100 rows. The
explicit transition never wins and costs 23.4 GiB more peak RSS than bounded
at full scale. Therefore a sinogram-local segment alone does **not** justify a
full-volume layout transition. Use the chain-grouped bounded executor as the
general default because it supplies contiguous kernel input while remaining
within 4.38% of the full-scale fastest arm. Direct strided execution remains a
valid kernel-specific optimization when that kernel is numerically validated
on strided input. Select a full transition only when a downstream consumer or
an already-required fused resampling boundary can amortize it across more work.

The large empirical result remains outside unit CI; CI covers equal-concurrency
dispatch, cross-arm numerical fingerprints, and release of the obsolete source
mapping after a transition. The safe default uses one bounded row worker unless
the caller supplies a higher resource limit.

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
| `phase.paganin_filter` / `bac_correction` / `negative_log` / `beta_scale` | projection-local | split phase-chain operations | accepts (Proj, Sino), prefers **Proj**, PRESERVE layout; corrections preserve transmission, negative-log produces attenuation, beta scale preserves attenuation | A |
| `filter.bilateral` | projection-local | order branch present | accepts (Proj, Sino), prefers **Proj**, PRESERVE | A |
| `geometry.skew` / `movement` | projection-local | order branches present | accepts (Proj, Sino), prefers **Proj**, PRESERVE; skew sets `resamples_detector` when enabled/nonzero | A |
| **`geometry.affine_skew_movement`** | optional boundary rewrite | Sino-only compatibility wrapper | accepts either; when transition configured, **produces the other, `fused_transition=True`** | C |
| `helical.index_remap` | per-projection row shifts | order branch present | accepts (Proj, Sino), prefers Proj, PRESERVE; `requires_row_identity=True` | A |
| `stripe.vo_dead/large/sort/family` | **sinogram-local** (`data[y]`) | Sino-only | accepts Sino, prefers Sino, PRESERVE; `SinogramStage`, bounded views from Proj | B |
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
	`PipelineState.layout/domain` + named-axis `DataShape` operations + complete/fragment
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
  recon-minimal's `Sino→Proj`, AAA592's nonzero warp-before-Paganin reference,
  AAA523's nonzero 0.149° Proj-preserving skew and
  retained `Proj`, and the absence of an extra lftomo layout conversion before
  AAA523 recon (TomoPy's recorded boundary copy remains visible in the plan).

## 9. Resolved design decisions

1. `SinogramStage` and `ProjectionStage` iterate assigned buffer samples;
   contracts declare
   `bounded_views_from`. The planner chooses and the segment executor implements
   gather/scatter. Scratch strategy is not a
   per-Stage mutable mapping.
2. A YAML-authored `layout.convert` refuses to run without a non-empty `reason:`.
   Planner-generated conversions carry a generated rationale in `PipelinePlan`.
3. Domains use `DataDomain` in a separate `lftomo/pipeline/domains.py` module.
4. `PRESERVE` is an enum singleton, not a bare object.
5. Complete-pipeline invariants are opt-in to complete validation; fragment
   validation composes from caller-supplied layout/domain without requiring a
   read/normalize/recon skeleton.
