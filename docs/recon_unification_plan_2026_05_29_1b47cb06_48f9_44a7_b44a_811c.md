# Reconstruction-Pipeline Unification — Initial Plan

**Date:** 2026-05-29
**Author:** Tava (assisted analysis for Sorotassu)
**Companion document:** `recon-survey-2026-05-29.md` (per-source findings)
**Status:** Initial proposal, opinionated. Revised twice after Sorotassu feedback (2026-05-29), then amended after a local code review of the checked-out repos. Not yet vetted against the lab's actual constraints, ops calendar, or ChengLab consent for `localCT` super-res lifts.

This plan is the synthesis half of the survey. The survey describes what exists; this describes what to do with it.

---

## TL;DR

1. **Three-repo structure.** `lftomo` stays a **comprehensive library** and grows in place. The **CLI is `hatmehyt` repurposed** — its scope narrows from phenotyping-pipeline to recon CLI, stub modules get stripped, and the useful CLI concepts are retained after audit/fix rather than copied wholesale. **Capture moves to sibling `lfcapture`** (repo already exists; it needs the package skeleton and migration README). lftomo can also be imported into `localCT` (or any other downstream consumer) without dragging the CLI along. Inside lftomo, invent a small but explicit `Stage` + `Pipeline` layer to fill the missing abstraction.
2. **Algebra engine:** TomoPy stays as the recon backend. ASTRA optional for CUDA. Don't try to replace either.
3. **Capture stays separate, in `lfcapture`.** Acquisition code from `dan_scripts/capture.py`, `localCT/hardware/`, `synchrotron/ALS_B8/`, and the `capture/` subtree being stripped out of hatmehyt lands there. `lfcapture` shares the `scan.yaml` schema with lftomo+hatmehyt but not the codebase. Conflating capture and recon is what produced the 3,000-line monoliths.
4. **On-disk format: TIFF stack and HDF5 dxchange are both first-class.** Neither is "the" internal format; the canonical pivot is `scan.yaml`. TIFF stack stays first-class because the lab uses it extensively and it has real performance advantages on large datasets in practice. HDF5 dxchange (`exchange/data` + `data_white` + `data_dark`) is first-class because synchrotron data lands in that form and TomoPy ingests it natively. Conversion between the two is a utility (`recon convert`), not a required ingest step.
5. **Algorithm consolidation is within-algorithm, not across algorithms.** Different algorithms (FW stripe vs `tomopy.remove_all_stripe` vs Vo stripe; Paganin vs BAC phase retrieval; gridrec vs SIRT_CUDA) stay as **separate algorithm options under one Stage class** — they do different things mathematically, datasets vary in which one works, and treating them as interchangeable would catastrophically fail regression tests. Consolidation only happens *within* a single algorithm where multiple Python implementations of the same algorithm exist (e.g., multiple Paganin impls; or `recon_minimal.py`'s wrappers around TomoPy algorithms that fix bugs in TomoPy's versions). Benchmark + ≤10% regression gate applies to within-algorithm impl consolidation only.
6. **Dask is a design constraint, not a Phase-1 commitment.** The `Stage` ABC should be backend-agnostic so a `state` object can hold either NumPy arrays (lftomo's current `SharedNPArray` pattern) or `dask.array.Array`. Actual Dask wiring lands in Phase 2 or 3 once Stages exist.
7. **Generalized sweep abstraction** for parameter exploration. Lives at the `Pipeline` builder, not per-Stage; supports `list` / `grid` / `linspace` / `zip` shapes; forks the pipeline at the swept Stage so upstream work is shared and downstream Stages replay per combo. Subsumes R2.54's `hoto_tomo_algo_list` pattern and generalizes it to COR, Paganin α, recon-algorithm choice, slice-range sub-runs, anything else parameterized. Detailed below.
8. **Phase 0/1 now gates on boring-but-critical stabilization.** Current `lftomo` has dependency and API drift: B10's `b10_recon.py` and the lftomo notebooks still import old symbols (`DataWindowSet`, `SharedNP`, `distribute_read`) while current lftomo exposes `TomoDataSet`, `TomoWindow`, and `SharedNPArray`; the prep modules also import old names. Decide compatibility shim vs porting before building Stages.
9. **Package management decision:** standardize each repo on `pyproject.toml` + `hatchling` for package metadata/builds, but keep **conda/mamba environment files as the runtime contract** for recon and capture deployments. TomoPy/ASTRA/hotopy/CUDA-heavy packages come from conda first; local packages and PyPI-only extras can be installed into those envs with `uv pip` (or plain pip). Do not make uv-only virtualenvs the supported deployment path.
10. **Migration:** four phases, with each lineage getting a deprecation README and a config-import shim before its scripts are retired.

The single strongest signal driving the recommendation: **B10 already contains a `b10_recon.py` (≈150 lines, lftomo-based) sitting next to its 3,240-line R2.54 monolith — and Sorotassu wrote it.** That's an in-flight migration onto lftomo, not a recommendation from outside. The unified tool's job is to finish what that file started.

---

## Why lftomo as the base

| Candidate | For | Against |
|---|---|---|
| **`lftomo`** | Most recent design; shared-memory architecture; typed shape dataclasses; tested I/O; already has B10 migration prototypes; aligns with where TomoPy ecosystems are landing. | No CLI; no `Stage`/`Pipeline` abstraction; prep functions untested and currently stale against the present utility API; packaging/dependencies are underdeclared. |
| `recon_minimal.py` | Closest to production-shape; working CLI; YAML-triplet config is sane. | Inline windowed loop tightly couples I/O+normalize+phase+recon — every new phase-retrieval flavor would mean editing the central loop. Hard to absorb other people's algorithms. |
| `localCT` (ReconChengLab) | Cleanest packaging; has CLI; modular `recon/pipeline.py`; useful request/result/summary and scan-folder inspection patterns; hermetic tiny-TIFF tests worth copying. | Lab-CT-shaped (hardware abstraction is half the package); recon parts thinner than lftomo's; entangled with super-res research code. |
| `phase_contrast/recon_v1.py` | Working BAC holotomography; INI configs. | INI semantics are stricter than YAML; no Stage abstraction; would still need a CLI layer; written around one beamline's geometry. |
| `original/recon_dan_v1.py` and B10 R2.54 | Deepest functionality; HPC-aware. | Monolithic; per-VOI hand-edited forks; no CLI; the *load* the unified tool has to absorb, not the base. |
| `hatmehyt` | Best CLI ideas by a clear margin: config-overlay, command grouping, custom Click param types, metadata I/O. | Zero algorithms; no package metadata; several implementation defects in current CLI/window code. Treat as a pattern source, not a drop-in chassis. |

The recommendation is **lftomo (library) + hatmehyt repurposed as the CLI + lfcapture as the capture sibling + an invented `Stage` ABC inside lftomo**. Three repos, all already-named; the `Stage`/`Pipeline` abstraction layer is the only new thing being written.

### Three-repo structure in detail

```
lftomo (library)                          # grows in place; gains Stage/Pipeline ABCs
   └── importable into:
         ├── hatmehyt (repurposed)         # CLI: `hatmehyt recon …`
         │                                  # AND `hatmehyt capture …` if [capture] extras installed
         │                                  # (wraps lfcapture as an optional dep)
         ├── localCT                        # already a python package; can drop lftomo in
         ├── notebooks                      # current usage; stays supported
         └── any other downstream consumer

lfcapture (new sibling, library + own CLI)  # acquisition
   ├── own CLI for capture-only installs: `lfcapture …`
   └── importable into hatmehyt via `pip install hatmehyt[capture]`
   shares scan.yaml schema with lftomo; no other code dep
```

- **`lftomo` (library):** absorbs algorithm Stages, owns the `Stage`/`Pipeline` ABC, owns the shared-memory + windowing infrastructure that already exists, owns I/O for TIFF and HDF5. **Does not own the CLI.** A user with their own driver (notebook, localCT, a SLURM submit script) imports lftomo and assembles a `Pipeline` directly.
- **`hatmehyt` (CLI):** repurposed from phenotyping-pipeline to **multi-purpose recon-and-optionally-capture CLI**. Keeps the useful CLI ideas — `@hht_step` (`cli/cli.py:163`), `Window`+`Slider` concepts (`window.py:13`), `GroupWithConfigsWrapper` (`cli/cli.py:52`), custom param types `Range`/`Crop`/`Frange`/`KVList`/`EnumParameter`, and `handle/meta.py`'s polyglot scan-metadata I/O — but the current implementation needs audit/fix before reuse. Stub modules (`recon/`, `prep/phase.py`, `prep/stripe.py`, `segment/`, `analyze/`) get stripped — their replacements either re-export from lftomo or get deleted. `capture/` moves to `lfcapture`. **The top-level command surface has two subcommand groups: `recon` (always) and `capture` (only if optional extras installed).** See "CLI architecture" subsection below.
- **`lfcapture` (sibling for acquisition):** repo exists now, but is effectively an empty initial commit. It receives Sorotassu's `dan_scripts/capture.py` (Euresys Coaxlink + ESP301), the camera/motion abstractions from `localCT/hardware/`, the `synchrotron/ALS_B8/R3_Berkeley8_v0.20_gui.py` GUI, and the `capture/general.py` + `capture/camera.py` pieces being stripped out of hatmehyt. **Ships its own CLI** (`lfcapture` console script) for capture-only deployments. Shares the `scan.yaml` schema with lftomo+hatmehyt but no other code dependency. The boundary makes both sides finishable.
- **`localCT` consumes lftomo directly.** Because lftomo doesn't require hatmehyt, `localCT` can adopt the unified algorithms without taking on the CLI's dependency footprint. Same applies to any other consumer that already has its own orchestration.

### CLI architecture: hatmehyt + lfcapture

Confirmed with Sorotassu on 2026-05-29. The shape:

- **`lfcapture` ships its own CLI** for capture-only installs (acquisition workstations that don't need recon).
- **`hatmehyt` adds a `capture` subcommand group** that's a thin wrapper around `lfcapture`'s library API. `lfcapture` is an **optional dependency** declared as `pip install hatmehyt[capture]`. If the extras aren't installed, `hatmehyt capture …` exits with a helpful "install the `capture` extras" error.
- Both expose the same underlying lfcapture library API; hatmehyt just provides convenience surface for combined deployments.

Install matrix:

| Deployment | Install | Available commands |
|---|---|---|
| Acquisition workstation | `pip install lfcapture` | `lfcapture …` |
| Analysis workstation / HPC head node | `pip install hatmehyt` | `hatmehyt recon …` |
| Lab workstation (does both) | `pip install hatmehyt[capture]` | `hatmehyt recon …` and `hatmehyt capture …` |

Subcommand structure:

```
hatmehyt
├── recon
│   ├── ingest
│   ├── prep flats
│   ├── run
│   ├── convert
│   └── export
└── capture                    # only if [capture] extras installed
    ├── plan
    ├── run
    ├── status
    └── ...                    # whatever shape lfcapture's surface lands at
```

Rationale: the dep sets for capture (harvesters / GenICam / EPICS / pipython / ACS / etc.) and recon (TomoPy / ASTRA / hotopy / Dask / pyfftw / etc.) don't naturally mix, and most users only need one. Extras keep installs lean. The single-CLI surface for users who *do* need both means they don't have to learn two tools.

**Out of scope for Phase 1–3:** live-recon-during-capture (the 360°-grab-180° helical-throughput case). It would require capture and recon Stages to coordinate via shared memory or event hooks. Flag it as a *check* on the Stage ABC ("does this preclude streaming inputs?") but do not build for it. Possible future "live workflows" pattern once both libraries are stable.

### What lftomo needs added

- **Stabilized package/API surface** → do first (lives inside lftomo). Move package metadata to `pyproject.toml` + `hatchling`, keep conda/mamba env files for real recon deployments, and mine `dan_scripts/conda_env/recon.yml` for the conda-first dependency set. `uv pip install -e ".[dev]"` is useful inside the activated conda env, but uv-only envs should not be the supported path for TomoPy/ASTRA/hotopy-heavy work. Decide whether old lftomo names (`DataWindowSet`, `SharedNP`, `distribute_read`, etc.) get a compatibility shim or whether B10/notebooks/prep are ported directly to `TomoDataSet` / `TomoWindow` / `SharedNPArray`. This is a Phase-0/1 gate, not cleanup.
- **`Stage` + `Pipeline` ABC** → invent (lives inside lftomo). Recommended shape (~120 LOC after contracts are explicit):
  - `class Stage(ABC)` with `run(state: PipelineState) -> PipelineState`, plus `expected_input: type[DataShape]` and `produced_output: type[DataShape]` (using lftomo's existing `ProjOrder` / `SinoOrder` / `ReconOrder`).
  - Each Stage also declares `mutates_input`, `window_axes`, `required_metadata`, `artifacts_produced`, and rough resource hints (`memory_multiplier`, `preferred_ncore`, optional GPU requirement). This matters because existing algorithms mostly mutate shared memory in place, and sweeps need to know whether a fork is a copy or a shared prefix.
  - `class PipelineState` holds the working data plus the metadata `Scan`/`Recon` configs and an artifact registry. **Backend-agnostic by design** — the array field is typed as a union of NumPy `SharedNPArray` and `dask.array.Array`, so future Dask wiring is a Stage-implementation change, not an ABC change.
  - `class Pipeline` is just an ordered sequence of `Stage`s with a shape-compatibility check at build time.
  - No DAG, no scheduler. Stages run sequentially. Concurrency stays inside individual Stages (matching how lftomo already does it via `SharedNPArray` + `multiprocessing.Pool`).
- **Unified config schema** → write (lives inside lftomo, importable by the CLI). **Pydantic** preferred over attrs/dataclasses because runtime validation matters when the configs span machine/session/scan/recon layers. Importers built for: `recon_minimal`'s YAML triplet, `phase_contrast/init_*.ini`, and a one-time conversion of R2.54's top-300-line hardcoded blocks. The R2.54 / b10_recon migration importer is the most valuable one because that's the actively-in-flight migration.

---

## Phase plan

### Phase 0 — alignment (1 week)

Decisions to lock before any code:

- **Confirmed (2026-05-29):** the **3-repo structure** — lftomo (library) + hatmehyt (CLI, repurposed) + lfcapture (capture, new). lftomo also drops into localCT.
- **Confirmed (2026-05-29): TomoPy as the algebra engine.** All five surveyed lineages already delegate the actual reconstruction math (projection / back-projection / FBP / iterative) to TomoPy. ASTRA `*_CUDA` is reached *through* TomoPy's bridge in `recon_minimal.py` ll.1202–1209 and B10 R2.54 ll.3119–3131. TomoPy's known quirks are wrappable (next-moves #8 catalogs them); switching to direct ASTRA / SCICO / TIGRE / a custom layer would invalidate every reference recon and gain nothing the wrappable-fixes pattern doesn't already give us. Reopens only if a future Stage requires an algorithm TomoPy fundamentally can't wrap.
- **Confirmed (2026-05-29): Python (not a CUDA-native rewrite) for the orchestration layer.** Algorithm hot paths already live in compiled libraries (TomoPy's C extensions, ASTRA's CUDA kernels via TomoPy, pyfftw's FFTW, numpy BLAS, scipy's compiled routines). The orchestration layer — Stage / Pipeline / Sweep / config — never touches the inner loop, so Python overhead is bounded. The lab and the surveyed code are entirely Python-shaped; going non-Python would split future contributors and require rebuilding wrappers the ecosystem already provides. GPU acceleration is per-Stage (ASTRA `*_CUDA`, potential cupy-backed Stages) — done inside the Stage, not by replacing the orchestration. Reopens only if persistent profiling shows the Python orchestration *itself* is the bottleneck (implausible for minutes-to-hours recon jobs).
- Decide the **lftomo API-drift path**: either carry a thin compatibility shim for old `DataWindowSet` / `SharedNP` / `distribute_read` consumers long enough to migrate B10 and the notebooks, or port those consumers immediately to current `TomoDataSet` / `TomoWindow` / `SharedNPArray`. Do not start Stage work until this is settled.
- **Confirmed: package management strategy.** Use `pyproject.toml` + `hatchling` as the package/build metadata standard across lftomo, hatmehyt, and lfcapture. Keep conda/mamba `environment.yml` files as the deployment/runtime contract for recon and capture, with conda-first installs for TomoPy, ASTRA, hotopy, CUDA-adjacent packages, OpenCV, dxchange, and other compiled scientific dependencies. Use `uv pip` only as fast pip-compatible glue inside an activated conda env for editable local packages and PyPI-only extras. Recreate envs instead of mutating them repeatedly after pip/uv installs.
- Confirm YAML + pydantic as the config format.
- Confirm **TIFF stack and HDF5 dxchange both first-class** as internal data formats; benchmark before promoting one over the other on any given Stage.
- Confirm **Dask as a design constraint** (the `Stage`/`PipelineState` ABC must not preclude Dask-backed state) without committing to Dask wiring in Phase 1.
- Get ChengLab consent for the `localCT` lifts only — per Sorotassu's note, localCT is the sole source needing explicit consent; `phase_contrast/`, `Neuroglancer/`, and `original/` are all "do whatever with". Concretely, consent is needed for:
  - **DRHT / LDM super-res** from `localCT/recon/super_recon/`.
  - Anything else specifically pulled from the `localCT/` subtree (e.g. its modular `recon/pipeline.py` patterns, hardware abstractions if `lfcapture` wants them).
- (No consent needed for **BAC** — it lives in `phase_contrast/funcs.py` not `localCT/`, which is in the "do whatever with" bucket. Same for `Neuroglancer/ct_to_ng.py`.)
- Decide on hatmehyt's repo name. **Recommendation: keep the name** for lineage; the scope is narrowed but the lab fennec-naming convention is preserved. Renaming to `lftomo-cli` is cheap on GitHub but breaks any pip-install name continuity if one was planned.
- Decide which `localCT` patterns are safe/valuable to lift without adopting the full localCT architecture: especially `ReconstructionRequest`/result dataclasses, JSON run summaries, folder inspection, TomoPy runtime bootstrap, and tiny generated TIFF tests.

### Phase 1 — foundation (4–6 weeks of focused work)

**Inside lftomo:**

- Real repo housekeeping: make imports pass, migrate package metadata to `pyproject.toml` + `hatchling`, update conda/mamba env documentation, decide core vs optional extras, mkdocs, CI on the existing tests. This is a gate, because the current source imports dependencies the package does not declare.
- Resolve the old-vs-current lftomo API drift. Either add a temporary compatibility module for old `DataWindowSet` / `SharedNP` / `distribute_read` consumers, or port B10/notebooks/prep to the current `TomoDataSet` / `TomoWindow` / `SharedNPArray` API.
- Define **`Stage` + `Pipeline` + `PipelineState` ABCs** with explicit contracts. `PipelineState` must hold an array field typed as `Union[SharedNPArray, dask.array.Array, np.ndarray]` so Dask backend wiring later is additive, but the Stage contract also needs mutability, windowing, artifact, and resource metadata.
- Define the **unified config schema** (pydantic models for `Machine`, `Session`, `Scan`, `Recon`). Lives in lftomo so library consumers can validate configs without depending on the CLI.
- Re-express lftomo's existing prep functions as Stages only after their imports/API are current — this is the validation step that says we got the ABC right. Add the regression tests these functions are currently missing.
- Implement first-class I/O for **both TIFF stack and HDF5 dxchange** behind one `Read` Stage with two backends. Symmetric write side.
- Copy the `localCT` test style: tiny generated TIFF/gain folders in `tmp_path`, no machine-specific absolute paths. Existing lftomo I/O tests that point at local drives should be converted or marked integration-only.

**Inside hatmehyt (repurposed as the CLI):**

- Open a `recon-cli-pivot` branch off current `master`. Land the strip/keep/refactor in one or a few reviewable commits before any algorithm wiring.
- Add package metadata first (`pyproject.toml` + `hatchling`) so `hatmehyt` can be installed/tested as the CLI sibling.
- **Strip:** `segment/`, `analyze/`, `recon/recon.py` and `recon/center_find.py` stubs, the empty `prep/phase.py` and `prep/stripe.py`. **Move out (to `lfcapture`):** the `capture/` subtree, except `capture/general.scan_prep()` which is metadata-only and stays on the CLI side for ingest.
- **Keep concepts, not code blindly:** `@hht_step`, `Window` + `Slider`, `GroupWithConfigsWrapper`, custom param types, and `handle/meta.py` remain the design source, but current bugs in config merging, window iteration, and `scan_prep()` placement need repair or rewrite.
- **Refactor:** `_build_command_chain` (`cli/cli.py:243`) may keep its windowed-stage dispatch for the inner `recon run` pipeline if it survives audit, but it stops being the top-level command builder. The top-level surface adds flat single-shot verbs alongside: `ingest`, `prep flats`, `run`, `convert`, `export`.
- Define the `recon` command surface in that order. Wire `run` to import lftomo Stages.
- Write the canonical-layout ingester (see "Input reshaping" below).
- Write config importers:
  - `from-recon-minimal-yaml`: machine + session + scan triplet → unified config.
  - `from-phase-contrast-ini`: INI sections → unified config.
  - `from-r254-hardcoded` / `from-b10-recon`: read the top of an R2.54 script (or a `b10_recon.py`) and emit a unified config. The latter is the migration the lab is actually in the middle of.

### Phase 2 — algorithm composition as Stages

The unified Stage system has two layers of registry:

- **Stage-class registry.** `read`, `normalize`, `stripe`, `phase`, `skew_correct`, `movement_correct`, `center`, `helical_correct`, `recon`, `super_resolve`, `write`. Each Stage class declares its expected input/output shape types plus its mutability/window/artifact/resource contract. Plugins (e.g. ChengLab super-res) can register out-of-tree.
- **Algorithm-option registry per Stage class.** Each Stage class enumerates the algorithm options users can pick. `stripe` has `vo`, `fw`, `all_of_the_stripe`. `phase` has `paganin`, `bac`. `recon` has `gridrec`, `sirt_cuda`, `sart_cuda`, etc. Different algorithms within a Stage class do mathematically different things; they are **not interchangeable** and consolidation across them would break regression tests against any reference that used a non-canonical choice.

**Where consolidation actually happens: within a single algorithm option, where multiple Python implementations of the same algorithm exist.** Typical cases:
- Multiple Paganin implementations across sources (lftomo, `recon_minimal/prep/phase`, possibly others) — pick one canonical impl after benchmark.
- **`recon_minimal.py` wraps several TomoPy algorithms with fixes for bugs in TomoPy's own implementations.** In those cases, the canonical impl is `recon_minimal`'s wrapper, *not* upstream TomoPy. Action item: catalog which algorithms have recon_minimal-side fixes, check whether the fixes have been upstreamed, and either pin TomoPy to a version that has them or carry the wrapper forward as canonical.

The benchmark + ≤10% perf-regression rule applies to within-algorithm impl consolidation only. Across-algorithm consolidation doesn't happen — different algorithms stay as separate options.

#### Stage-class × algorithm-option × impl matrix

| Stage class | Algorithm options (kept as separate, user-selectable) | Per-algorithm impl selection (within-algorithm consolidation) |
|---|---|---|
| **read** | `tiff_stack`, `hdf5_dxchange` | `tiff_stack`: lftomo `byteread_helper` vs `recon_minimal.distribute_read` — benchmark, pick one. `hdf5_dxchange`: lftomo's wrapper is the only tested impl; canonical. |
| **normalize** | `weighted_normalize` (with config flags for pre/post-gain blending and 50%-θ gain switching — these are *flags within one algorithm*, not separate algorithms) | Candidate impls: lftomo `weighted_normalize`, `recon_minimal.clean_data`, `phase_contrast/funcs.py`'s 50%-θ version. Benchmark, pick canonical. |
| **stripe** | `vo`, `fw`, `all_of_the_stripe` — **all three kept, all different mathematical behavior** | `vo`: lftomo `remove_sort_vo` + `remove_dead_vo`. `fw`: lftomo `remove_stripe_fw`. `all_of_the_stripe`: TomoPy `remove_all_stripe` (used in `phase_contrast` and B10 R2.54). Each has one canonical impl; no within-algorithm consolidation question for stripe today. |
| **phase** | `paganin`, `bac` — different algorithms, both kept | `paganin`: **multiple impls exist** — lftomo `phase_retrieval`, `recon_minimal/prep/phase`, possibly `phase_contrast/funcs.py` if it has a Paganin path; benchmark across them, *watch for recon_minimal correctness fixes vs upstream*, pick canonical. `bac`: `phase_contrast/funcs.py:716, 947` (uses `hotopy.holo.BronnikovAidedCorrection`) is canonical; **B10 R2.54 ll.997, 1241 uses the older `hotopy.phase.BronnikovAidedCorrection` submodule path — API drift, not a candidate for canonical**; `b10_recon.py:25, 91–113` has the cleanest multiprocessing-per-Theta concurrency wrapper around the same `hotopy.holo` call. **NOT in `localCT/`** (its classical pipeline is gain → attn → stripe → center → gridrec, no phase retrieval per `app_super_recon.md:56`). Parameter exploration via the generalized sweep section; R2.54's `hoto_tomo_algo_list` ingests into `sweep: kind: list`. |
| **skew_correct** | `cv2_affine` | Single impl, lifted from `phase_contrast` / B10 R2.54 (same code). |
| **movement_correct** | `gain_shift_registration` | Single impl, same source. |
| **center** | `vo`, `pc`, `entropy_local` — **all three kept** | `vo`: `tomopy.find_center_vo` is canonical (check for recon_minimal fixes). `pc`: `tomopy.find_center_pc` is canonical (check for recon_minimal fixes). `entropy_local`: `recon_minimal.find_center` entropy path is the only impl. |
| **helical_correct** | `index_remap` | Lift `dan_scripts/berkeley_2021/allinwonderful_v1.5_A1_Helical.py::parallel_fill_array_helically`. Verify the berkeley_2020 + argonne_2022 variants are equivalent before picking; if they diverge meaningfully, treat as multiple impls of the same algorithm and benchmark. Resolves `recon_minimal` TODO at l.148. |
| **recon** | `gridrec`, `sirt_cuda`, `sart_cuda`, `cgls_cuda`, `em_cuda`, `fbp_cuda`, `fp_cuda`, `bp_cuda` — **all kept**, different algorithms (gridrec is FBP-family; SIRT/SART/CGLS/EM are iterative; the *_CUDA variants are ASTRA-backed GPU impls) | `gridrec`: **TomoPy is canonical IF unverified by recon_minimal review; if `recon_minimal.py` has a fix-wrapper around TomoPy's gridrec, that becomes canonical**. ASTRA `*_CUDA`: TomoPy's ASTRA bridge (`recon_minimal` ll.1202–1209 and B10 R2.54 ll.3119–3131 use the same bridge) is canonical. |
| **super_resolve** | `drht`, `ldm` — different algorithms, both kept (off by default) | `ReconChengLab/localCT/recon/super_recon/`. Both research-grade. Needs ChengLab consent. LDM needs missing checkpoints recovered. |
| **write** | `tiff_slices`, `neuroglancer`, `hdf5_volume` — different output formats | `tiff_slices`: lftomo `write_slice` is canonical. `neuroglancer`: `ReconChengLab/Neuroglancer/ct_to_ng.py` ("do whatever with"); B10's `ng_transformed/remove_gz.py` is a post-processing utility, not a writer-impl candidate. `hdf5_volume`: new, trivial via h5py. |

Sequencing within Phase 2: implement Read (both backends) → Normalize → Stripe (just `vo` first) → Center (just `vo`) → Recon (just `gridrec`) → Write (just `tiff_slices`) first. That's a working pipeline that covers the dominant lab path with one algorithm option each. Adding the other algorithm options (`fw`/`all_of_the_stripe` stripe variants; `pc`/`entropy_local` center; `sirt_cuda`/etc. recon; `paganin`/`bac` phase; helical; skew + movement; super-res; neuroglancer + hdf5_volume output) is incremental — each lands as a registration into an existing Stage class.

**Catalog of recon_minimal's TomoPy-fix wrappers.** Before Phase 2 starts, walk `recon_minimal.py` against the TomoPy functions it imports and flag every place where the wrapper does more than parameter-marshalling — those are the places where the canonical impl is recon_minimal's, not TomoPy's. Each flagged wrapper gets a triage decision: (a) upstream the fix to TomoPy and adopt upstream, (b) carry as canonical impl with a deprecation pointer, or (c) confirm the fix has already been upstreamed and adopt a pinned-version upstream. This catalog is a Phase-1.5 deliverable — needs to land before any `recon`-class Stage swaps a recon_minimal wrapper for upstream.

### Test data strategy

Confirmed with Sorotassu on 2026-05-29:

**Three tiers.**

- **Tier 1 — synthetic phantoms.** Generated in code (Shepp-Logan, deliberately-striped sinograms, deliberately-misaligned COR). Live in `tests/fixtures/` of each repo; run on every commit; seconds per test. Catch arithmetic regressions and ABC-shape bugs. **No real data needed; the lab does not supply these.**
- **Tier 2 — small real datasets for integration tests.** 3 datasets minimum, each ≤200 projections single sectional FOV (downsampled if the originals are larger), targeting ≤3 min wall-clock per integration test:
  1. clean parallel-beam scan with heavy striping — TIFF stack
  2. strong phase contrast — HDF5 dxchange
  3. helical scan — either format, smallest available
- Optional broader coverage: one more per {acquisition system × energy} combination from B10 (PCRBox 14keV, PCRBox 20keV, RedBox 14keV).
- **Tier 3 — one full-scale reference dataset.** TB-scale, single representative volume. Powers Phase-2 perf benchmarks (the ≤10% regression gate), Dask out-of-core validation, end-to-end sanity. In-place on Yarrkha; test harness takes `--ref-dataset <path>`.

**Data hosting.** All Tier-2 and Tier-3 datasets are non-public for now. They live on Yarrkha (or other lab-internal storage). The CI architecture follows: **a self-hosted GitHub Actions runner registered on Yarrkha** picks up Tier-2 and Tier-3 workflows via `runs-on: [self-hosted, yarrkha]`; Tier-1 synthetic tests stay on GitHub-hosted runners. If/when datasets become publishable, lifting them to Zenodo/S3/LFS is a test-harness config change with no test-code impact.

**Reference recons.** Each Tier-2 (and the Tier-3) dataset needs a frozen reference reconstruction it asserts against. This is the contract regression tests check.

- **When generated:** any time before Phase 2 starts. Recommended now, with the script that's considered canonical production for that dataset (R2.54 for B10 lineage, `b10_recon.py` for the in-flight migration, `recon_minimal.py` for Sorotassu's own work).
- **When asserted:** regression-against-reference assertions only start mattering at the start of Phase 2 (when Stages begin replacing real algorithm impls). Phase 1 ABC work uses Tier-1 synthetic data; no reference recons needed there.
- **Immutability:** the reference is the contract with the past. Do not re-run the producing script later to "refresh" it. If the unified pipeline diverges from a reference, the burden is on the unified pipeline to prove the divergence is intentional and an improvement.
- **Exact ordering is part of the reference.** BAC/stripe/log/skew ordering differs across lineages (`b10_recon.py` vs. R2.54 vs. `phase_contrast`), so `pipeline.yaml` must pin the stage order literally. Do not normalize those differences away in templates until reference tests prove equivalence.

**Reference-archive shape**, dropped alongside each dataset:

```
session-root/
  scan.yaml
  projections/         # TIFF or HDF5
  flats/
  reference/
    recon/             # reference slice stack — feature-rich ~10-slice chunk OK
      slice_00000.tif
      ...
    metadata.yaml      # producing script path + git SHA, host (Yarrkha?), date,
                       # operator, runtime, env notes (SLURM, GPU)
    pipeline.yaml      # FULL pipeline specification (per "Pipeline specification"
                       # section above) — which Stages were applied, in what order,
                       # which algorithm-impl each used, exact params each got.
                       # Different references will have different pipeline shapes
                       # because they come from different sources (R2.54 vs
                       # recon_minimal vs b10_recon vs phase_contrast etc.) —
                       # this file pins the shape so the regression assertion
                       # knows what pipeline to run.
                       # Stored in expanded form (not just `pipeline_template: …`)
                       # so reference reproduction survives template drift.
```

A feature-rich chunk (~10 slices through interesting structure) is plenty for most regression assertions and saves disk. Full-volume reference reserved for Tier-3.

The `pipeline.yaml` is what makes references portable: regression test reads the dataset, loads the reference `pipeline.yaml`, runs the unified pipeline with that exact spec, compares output to `reference/recon/` slices. Same machinery whether the reference came from R2.54, `recon_minimal`, or anywhere else — the differences are entirely in what the `pipeline.yaml` declares.

**What you don't need to send.**
- Synthetic / phantom data — generated in code.
- Per-VOI variants — per-VOI differences are config (COR, alpha range, chunk size) and live in Stage params, not in test data. One representative per acquisition system + energy is enough.
- Capture-side data — `lfcapture`'s testing is its own concern.
- Anything specifically for the Dask backend — Tier-3 size is what makes Dask out-of-core mode actually do work.

**Open before Phase 2 starts.**

- Publication / license status of the Tier-2 datasets (so the eventual public-mirror lift is unblocked when wanted).
- Whether any candidate already has a "golden" reference recon that can be lifted directly (saves a re-run of the producing script).

### Phase 3 — capture/acquisition lives in `lfcapture`

New sibling repo. Receives:

- `dan_scripts/capture.py` (Euresys Coaxlink + ESP301).
- `localCT/hardware/` (camera + multi-stage abstraction — PI / ACS / Newport XPS / harvester / egrabber / opencv; if pulled, this is the one piece of localCT needing ChengLab consent for `lfcapture`).
- `synchrotron/ALS_B8/R3_Berkeley8_v0.20_gui.py` (tkinter + napari B8 acquisition GUI).
- `capture/general.py` (mostly — `scan_prep()` stays with hatmehyt for ingest-side metadata) and `capture/camera.py` from hatmehyt as it's stripped.

`lfcapture` emits data into the canonical on-disk layout (`scan.yaml` + `projections/` + `flats/`); the recon tool consumes from there. The two sides share the `Scan` pydantic model from lftomo, not code.

Why split it out: every monolith we've surveyed (R2.54, `recon_dan_v1.py`) is half-acquisition. Maintaining the boundary is what makes both sides finishable. The `lfcapture` name mirrors `lftomo`'s convention; future tomography-adjacent libraries (`lfsegment`?) can pick up the prefix.

### Phase 4 — retire and redirect

Each lineage gets:

- A deprecation README pointing at the unified tool.
- A `migrate.py` that converts its config / hardcoded block into the unified YAML.
- The legacy script remains available as a tagged release for reproducibility, but no new VOI-forks.

Order of retirement (least disruptive first):
1. `recon_minimal.py` — config importer is mechanical; this one is your own; lowest political cost.
2. B10 R2.54 family — once `b10_recon.py`'s author signs off and the BAC + helical Stages exist, the R2.54 forks become redundant.
3. `phase_contrast/recon_v1.py` — needs BAC Stage to be a perfect drop-in.
4. `original/recon_dan_v1.py` — keep available; document as historical.
5. `hatmehyt` — archive the repo with a README pointing at where its CLI patterns landed.

---

## Input reshaping

### The current chaos

| Source | Projections | Flats/darks | Metadata |
|---|---|---|---|
| `recon_minimal` | TIFF stack, dir-scanned | YAML-pointed dirs, then pre-medianed | YAML triplet (machine/session/scan) |
| `hatmehyt` | TIFF glob `*.tif*` | flatfield logic exists; never wired | CSV/YAML/Google Sheets via `meta.FMT` |
| `lftomo` | TIFF dir or HDF5 `exchange/data` | `{predark,pregain,postdark,postgain}_median.tif` siblings | inline `CTFov` namedtuple |
| `phase_contrast` | `projections/` of TIFFs | `gains/{gain,gain_post,dark,dark_post}.tif` | INI |
| B10 R2.54 | HDF5 dxchange primary, TIFF fallback | same `gains/*` layout | hardcoded ll.1–300 |

### Proposed canonical layout

Both projection formats are first-class. A session is a directory with a `scan.yaml`, a `projections/` payload (either form), and optional flats.

```
session-root/
  scan.yaml              # energy_keV, pixel_size_um, ssd_cm,
                          # n_projections, degrees_per_fov, n_fov,
                          # helical: {pitch_um_per_deg, ...},
                          # detector: {width, height, dtype},
                          # acquisition: {beamline, system, date, operator},
                          # projections.format: "tiff_stack" | "hdf5_dxchange",
                          # projections.path: relative path below

  # projections payload — ONE of:
  projections/
    *.tif                # TIFF stack, natural-sorted; OR
    *.tiff               #   (single multi-page TIFF also OK; auto-detected)
  projections/
    data.h5              # HDF5 dxchange: exchange/data + data_white + data_dark

  # flats — optional if HDF5 carries them; otherwise either:
  flats/
    pregain/  postgain/  predark/  postdark/      # raw, OR
    pregain_median.tif  postgain_median.tif       # pre-computed (lftomo-compatible)
    predark_median.tif  postdark_median.tif

  scanlog.txt            # optional; frame-timing for movement correction
  recon.yaml             # per-recon overrides (algorithm, COR, phase params, slice range)
```

Rationale:

- **Neither TIFF nor HDF5 is "the" internal format.** The lab uses TIFF stack extensively and it has real performance advantages on large datasets in practice (independent slice reads, no contention through a single HDF5 file handle, simpler partial-failure semantics). HDF5 dxchange is also first-class because synchrotron data lands in that form, TomoPy ingests it natively, and it's a clean single-artefact deliverable for archives. The canonical pivot is `scan.yaml`'s metadata, not the binary container.
- **Conversion between them is a utility, not an ingest step.** `recon convert --to {tiff_stack, hdf5_dxchange}` exists for users who explicitly want to convert (e.g. delivering an archive, or switching to TIFF for speed on a particular pipeline). The pipeline itself reads either form natively.
- **Flats are accepted raw or pre-medianed** (the four `*_median.tif` files lftomo uses). The medianed form is a cache; if it's absent the pipeline computes it.
- **`scan.yaml` and `recon.yaml` are split** because scan metadata is per-dataset and immutable ("this is the scan we took") while recon parameters are per-run and tweaked ("here's what we tried this time"). Merging them is what makes the R2.54 per-VOI fork pattern feel necessary in the first place.
- **`recon.yaml` is a full pipeline specification, not an overrides file.** It declares which Stages run, in what order, which algorithm-impl each one uses, and what params each gets. The pipeline shape itself is the configuration — see "Pipeline specification" below.

### Tooling around it (commands provided by the CLI sibling project)

- **`recon ingest <src> --layout {berkeley8, als_b9, local_ct, dxchange, recon_minimal, autodetect}`** — emits a `scan.yaml` plus a `projections/` symlink farm or pointer file. No copy of the raw data; you just get a canonical view. Autodetect when filenames are unambiguous.
- **`recon prep flats <session>`** — reads raw flats, writes the four `*_median.tif` siblings. Replaces the B10 flat-helper scripts, including the `print_times.py` `writer.writerow(v)`-before-assignment bug. Idempotent.
- **`recon run <session> [--config recon.yaml]`** — runs the pipeline. Stages from the config; sensible defaults; emits `recon_run.yaml` (à la hatmehyt's `hht_run.yaml`) with the resolved parameter set for reproducibility.
- **`recon convert <session> --to {tiff_stack, hdf5_dxchange}`** — explicit format conversion. Not part of the pipeline; an offered utility.
- **`recon export <session> --format {neuroglancer, hdf5_volume}`** — chunked/precomputed output. Lifts `Neuroglancer/ct_to_ng.py` (no ChengLab consent needed — `Neuroglancer/` is "do whatever with") and adds a trivial HDF5 volume writer.

---

## Pipeline specification

Different sources do different things to the data — R2.54 runs BAC + skew correction + movement correction + gridrec; `recon_minimal.py` runs Paganin + entropy-local centering + gridrec/ASTRA; `b10_recon.py` runs BAC + gridrec on H5; `phase_contrast/recon_v1.py` runs gain blending + skew + movement + BAC + gridrec. Each one is a **specific sequence of Stages with specific algorithm choices**, and the unified tool needs to specify that sequence cleanly in `recon.yaml`. This matters in three places: writing recon configs by hand, ingesting legacy configs (R2.54, INI, hardcoded blocks) into the unified form, and pinning the pipeline shape inside a reference recon's `params.yaml` for regression assertion.

### Shape

`recon.yaml` declares a `pipeline:` list of Stage configs in order. Each entry has a `kind` (the Stage class), an `algorithm` (which impl, when multiple exist), and a `params:` block (algorithm-specific, validated by that Stage's pydantic schema). Omitting a `kind` from the list omits the Stage entirely.

```yaml
# recon.yaml — full pipeline specification
pipeline_template: r254-canonical      # optional; loads a shipped preset.
                                       # Anything below this line overrides it.

pipeline:
  - kind: read
    backend: tiff_stack                # tiff_stack | hdf5_dxchange
    params: {glob: "projections/*.tif"}

  - kind: normalize
    algorithm: weighted_normalize
    params: {pre_post_blend: theta_50}

  - kind: stripe
    algorithm: vo                      # vo | fw | all_of_the_stripe (TomoPy)
    params: {size: 71}

  - kind: phase                        # OMIT this block entirely to skip phase retrieval
    algorithm: bac                     # bac | paganin
    params: {alpha: 2.5, gamma: 1.75}

  - kind: skew_correct                 # optional
    params: {angle_deg: 0.26}

  - kind: movement_correct             # optional
    params: {shifts: [0.0, -1.0376]}

  - kind: center
    algorithm: vo                      # vo | pc | entropy_local
    params: {smooth: 1.0}

  - kind: recon
    algorithm: gridrec                 # gridrec | sirt_cuda | fbp_cuda | ...
    params: {filter_name: shepp}

  - kind: write
    backend: tiff_slices               # tiff_slices | neuroglancer | hdf5_volume
    params:
      naming_template: "{sample_id}_{slice:05d}"
      dtype: float32
```

Key semantics:

- **Stage omission is the off-switch.** R2.54's `if hoto_tomo_algo_list[0][0]: …` runtime gate becomes "include or omit the `phase:` block in the config." No `enabled: false` flag at the Stage level (the sweep-section per-combo `enabled:` flag is unrelated and only applies inside a sweep list).
- **Order is significant.** The Pipeline builder validates shape compatibility at build time (using the `expected_input` / `produced_output` shape types on each Stage), so e.g. inserting `phase` after `center` would fail before any work runs.
- **Stage contracts are significant too.** Shape compatibility is necessary but insufficient: the builder also needs to know whether a Stage mutates its input, which axes it windows over, what metadata it requires, what artifacts it emits, and whether it requires GPU/large scratch memory. This is especially important for sweep fork semantics and for Dask/SharedNP coexistence.
- **Stage-class registry.** New Stage classes register themselves (out-of-tree plugins can register too); the validator checks `kind:` against the registry.
- **Algorithm-impl registry per Stage class.** `stripe` declares `vo` / `fw` / `all_of_the_stripe` as registered impls; `recon` declares `gridrec` / `sirt_cuda` / etc. New impls register against an existing Stage class.

### Pipeline templates / presets

Reproducing R2.54's behaviour shouldn't require hand-writing 50 lines of YAML every time. The library ships a `pipelines/` directory of canonical presets, versioned alongside the code:

| Preset name | Reproduces | Notes |
|---|---|---|
| `r254-canonical` | R2.54_Berkeley8_hpc_v1.81 default config | BAC + skew + movement + gridrec |
| `b10-recon-canonical` | `b10_recon.py` default | BAC + gridrec on H5 |
| `recon-minimal-default` | `recon_minimal.py` default | Paganin + entropy-local + gridrec |
| `phase-contrast-bac` | `phase_contrast/recon_v1.py` default | gain blending + skew + movement + BAC + gridrec |
| `recon-minimal-astra-sirt` | `recon_minimal.py` with ASTRA SIRT_CUDA | GPU path |

A user invokes one via `recon run <session> --pipeline-template r254-canonical` or via `pipeline_template: r254-canonical` in their `recon.yaml`. Anything declared in the user's `pipeline:` list overrides the preset entry-by-entry (matched on `kind:`); Stages declared in the preset but absent from the user's `pipeline:` stay; Stages declared in neither are omitted. Standard merge semantics.

### Versioning

Templates are versioned with the library (`r254-canonical@v1.0`). A reference recon's `params.yaml` records the resolved template version explicitly so reproduction is stable even if the template later evolves:

```yaml
# inside reference/params.yaml
resolved_from: r254-canonical@v1.0
pipeline:
  # ... the full expanded pipeline as-resolved, not just the template ref.
  # If the template ever changes, the reference still reproduces its own past
  # because the expanded form is locked in.
```

The expanded-form-locked rule applies to all reference archives, not just template-based ones. Storing the resolved pipeline (not just the template ref) is what makes references reproducible after template drift.

### Migration importers

Each lineage's importer (`from-r254-hardcoded`, `from-b10-recon`, `from-phase-contrast-ini`, `from-recon-minimal-yaml`) emits a full `recon.yaml` with the pipeline shape encoded, not just a few params:

- `from-r254-hardcoded`: parses R2.54's top-300-line hardcoded block; emits `pipeline: …` listing whichever Stages were actually wired (BAC iff `hoto_tomo_algo_list[0][0]`; skew iff non-zero; etc.).
- `from-b10-recon`: emits a `b10-recon-canonical` shape, parameters lifted directly.
- `from-phase-contrast-ini`: maps INI sections to Stage entries.
- `from-recon-minimal-yaml`: takes the YAML triplet, emits a `recon-minimal-default` shape with the triplet's overrides applied.

Each importer is also the operational way to *generate* a template — run the importer on the canonical example config from each lineage, hand-clean, commit to `pipelines/`.

---

## Generalized sweep shape

Several Stages have parameters where lab users routinely want to explore values rather than commit to one — BAC's `(α, γ)` is the canonical case (R2.54's `hoto_tomo_algo_list = [[run?, α, γ], ...]` exists for exactly this), but the same shape applies to COR-around-an-initial-guess, Paganin α, stripe-removal α, reconstruction-algorithm choice, slice-range sub-runs for parameter validation, and so on. Rather than re-invent the sweep idiom per Stage (the way R2.54 hardcodes it for BAC alone), the unified plan introduces one cross-cutting `Sweep` abstraction.

### Where the sweep lives

**At the `Pipeline` builder, not inside `Stage`.** Individual Stages stay sweep-unaware — they take their resolved params, do their work, return state. The sweep is a Pipeline-time construct: when the builder sees a `sweep:` block on a Stage's config, it forks the pipeline at that Stage, keeping everything upstream shared (one read, one normalize, one stripe-remove) and replaying everything downstream once per sweep combo.

This matches R2.54's implicit semantics (one sino-build, N BAC variants, N reconstructed volumes) and generalizes it to any Stage.

### Sweep spec shapes

Four shapes cover everything the lab has been doing in the surveyed code:

```yaml
# 1. Explicit list of param-dicts (R2.54 compat).
sweep:
  kind: list
  runs:
    - {alpha: 1.6, gamma: 1.2}
    - {alpha: 2.5, gamma: 1.75, enabled: false}   # per-combo skip (R2.54's run? flag)
    - {alpha: 5.0, gamma: 2.0}

# 2. Cross-product grid.
sweep:
  kind: grid
  axes:
    alpha: [1.0, 2.5, 5.0]
    gamma: [1.2, 1.75]
  # → 6 combos

# 3. Linspace / arange along one axis, fixed on the rest.
sweep:
  kind: linspace
  axis: {name: alpha, start: 0.5, stop: 10.0, n: 20}
  fixed: {gamma: 1.5}

# 4. Zip (pairwise across axes, not cross-product).
sweep:
  kind: zip
  axes:
    alpha: [1.0, 2.0, 3.0]
    gamma: [0.5, 0.6, 0.7]
  # → 3 combos, not 9
```

The `enabled: false` per-entry flag (form 1 only) is the direct ergonomic equivalent of R2.54's `[run?, α, γ]` — lets users keep historic combos in the config for reproducibility while turning them off for current runs.

**Algorithm-level sweeps** (sweeping *which algorithm option*, not just its params) use the `list` shape with each entry overriding the Stage's `algorithm:` key alongside its params:

```yaml
- kind: stripe
  sweep:
    kind: list
    runs:
      - {algorithm: vo, size: 71}
      - {algorithm: fw, level: 4, sigma: 2}
      - {algorithm: all_of_the_stripe, snr: 3}
```

Each combo runs the stripe Stage with a different algorithm option entirely, including different param shapes. Useful for "which stripe algorithm works on this dataset" exploration — exactly the across-algorithm comparison the Phase-2 framing explicitly does *not* consolidate.

### Pipeline fork semantics

```
   read → normalize → stripe → phase {sweep over (α, γ)} → center → recon → write
                              │
                              ├── (α=1.6, γ=1.2) → center → recon → write_α1.6_γ1.2/
                              ├── (α=2.5, γ=1.75) → center → recon → write_α2.5_γ1.75/
                              └── (α=5.0, γ=2.0)  → center → recon → write_α5.0_γ2.0/
```

- Upstream of the swept Stage: shared, computed once.
- Swept Stage + downstream Stages: replayed per combo.
- State is forked at the swept-Stage boundary (`SharedNPArray` copy-on-fork or per-combo allocation, depending on whether the swept Stage mutates the shared array — that's a Stage contract concern, not a sweep concern).
- Multiple sweeps in one Pipeline: cross-product. `Pipeline.expand_sweeps()` returns the flattened combo list at build time so the runner knows the total work upfront.

### Output naming

Per-combo outputs must be distinguishable. The naming template lives on the writer Stage (or any Stage producing artifacts) as a pydantic field:

```yaml
write:
  naming_template: "{sample_id}_α{phase.alpha:.2f}_γ{phase.gamma:.2f}"
  # resolves at runtime per combo
```

Template variables resolve from the combo's swept values plus session metadata. Default template if not specified: `"{sample_id}_{sweep_hash}"` where `sweep_hash` is a short hex digest of the combo's param-dict, so users get distinct dirs without having to think about naming.

### Reproducibility

The emitted `recon_run.yaml` records, per combo:
- the resolved param-dict (literal values, not the sweep spec that produced them);
- the output path the combo wrote to;
- timing + git SHAs (lftomo + hatmehyt + lfcapture if relevant).

Re-running a single combo from a sweep is then a `recon run --combo <hash>` operation against the same `recon_run.yaml`, no sweep machinery needed.

### Sweep edge cases worth flagging now

- **Sweeping early Stages (e.g. read with multiple input dirs).** Degenerates to running the whole pipeline N times. Fine but wasteful if Stages downstream of the swept one could've shared work — flag at builder time when shared-prefix is empty.
- **Empirical / data-derived parameters.** b10_recon.py computes γ from data (l.84–94). If the sweep spec includes `gamma: derived` (sentinel value), the Stage computes it at runtime per combo and the run record captures the resolved value. Keep this in the schema as an explicit allowed value rather than a string-or-number polymorphism.
- **Sweeps that cross a windowed Stage.** lftomo's `TomoWindow` machinery already handles strided iteration per slice/window; sweep × window is a 2D iteration. Probably allowed but worth a tested example before declaring stable.
- **Disk-budget guard.** A 6-axis grid sweep silently writes 6× the data. The builder should log expected total combos + estimated output volume at startup and abort if a `--max-sweep-combos N` guard is exceeded (default high, override low).

### Phase ordering for the sweep abstraction

- **Phase 1:** the `Stage`/`PipelineState` ABC accepts a `sweep: Optional[SweepSpec] = None` on any Stage's pydantic params. The `Pipeline.expand_sweeps()` builder method exists but only the `list` and `grid` shapes are implemented. Test with a synthetic Stage that takes one numeric param.
- **Phase 2:** when BAC lands as a Stage, it's immediately sweep-capable because the abstraction is already in place. R2.54's `hoto_tomo_algo_list` ingester becomes a one-shot translator: `[[True, 2.5, 1.75], [False, 5.0, 1.2]]` → `kind: list, runs: [{α: 2.5, γ: 1.75}, {α: 5.0, γ: 1.2, enabled: false}]`.
- **Phase 2 / 3:** `linspace` and `zip` sweep shapes added as Stages start using them.

---

## Dask consideration

lftomo currently parallelizes via `SharedNPArray` + `multiprocessing.Pool` — single-node, in-memory. Works well at lab-CT scale. Synchrotron datasets routinely hit "doesn't fit in RAM" and "wants to span nodes" regimes where Dask is the natural fit (lazy compute graph, out-of-core arrays, distributed scheduler).

**Decision: Dask is a Phase-1 design constraint, not a Phase-1 implementation commitment.**

- The `Stage` ABC's `PipelineState.array` field must be typed `Union[SharedNPArray, dask.array.Array, np.ndarray]` so wiring a Dask backend later is additive — a new Stage subclass or a new backend flag, not an ABC change.
- The `Pipeline` builder needs to know the chosen backend at construction time so it can pick the matching Stage impls (don't mix backends mid-pipeline).
- Dask and SharedNP coexist sensibly: SharedNP for **within-node IPC** (fast shared memory between processes on one box); Dask for **cross-node compute** and **out-of-core arrays larger than node RAM**. A typical recon at SLURM scale would use Dask at the top level to distribute Stages across workers, with each worker using SharedNP internally for its multiprocessing pool.
- Actual Dask wiring happens in Phase 2 or 3 — once Stages exist for the in-memory path, adding a Dask-backed variant of each (or a Dask-aware `Stage.run`) is incremental.

The cheapest way to validate this is in Phase 1: write the `Stage` ABC, implement one Stage with the NumPy path, then write a no-op Dask backend that just wraps NumPy in a `dask.array.from_array` to prove the ABC doesn't pin the backend. If that prototype is awkward, the ABC needs rework before anything else gets built on top.

---

## Risks / things I'm not sure about

These deserve to be flagged rather than papered over:

- **lftomo's prep functions are untested and stale against current utility names.** The infrastructure is tested; the algorithms aren't, and `prep/flat.py`, `prep/phase.py`, and `prep/stripe.py` still import old `SharedNP` / `DataWindow` style APIs. Before declaring lftomo the base, Phase 1 needs import cleanup plus regression tests against generated fixtures and known datasets.
- **lftomo packaging currently underdeclares runtime dependencies.** The source imports `h5py`, `tifffile`, `tomopy`, `scipy`, `pywt`, `numexpr`, `natsort`, and `skimage`, but `pyproject.toml` only declares a small subset. CI and import smoke tests will be noisy until the env/dependency story is corrected.
- **uv-only would be too brittle for the recon runtime.** uv is excellent pip-compatible tooling, and worth using for editable local installs inside an existing env. But TomoPy, ASTRA, hotopy, CUDA-adjacent packages, and beamline/hardware dependencies are the wrong dependency set to force through a PyPI-only virtualenv workflow. Conda/mamba remains the operational substrate.
- **B10's lftomo-based scripts are migration prototypes, not proof the checked-out lftomo API is ready.** They import old lftomo names (`DataWindowSet`, `SharedNP`, `distribute_read`) that are not present in the current package. That is still a strong signal that B10 wants to move toward lftomo, but it adds a concrete compatibility/porting task.
- **Algorithm impls labelled "same" may not be performance-equivalent.** Several rows in the Phase 2 table have multiple candidate impls (TIFF reader, Vo stripe, Paganin, helical). On TB-scale datasets a 30%-slower impl can mean hours per recon — the lab's current users may be implicitly depending on the speed envelope of whatever impl they're using today, and a silent swap could regress production. Benchmark on a representative dataset before retiring any non-canonical impl; gate swaps on a ≤10% regression rule of thumb.
- **LDM checkpoints are missing** (per the `localCT/app_super_recon.md` description vs. what's on disk). Don't promise super-res inpainting until those are located.
- **`print_times.py` has an actual bug** (`writer.writerow(v)` before `v` is assigned). Worth fixing in place even before the unification work starts, so the lab isn't producing bad CSVs in the meantime.
- **hatmehyt's implementation is rougher than its concepts.** The current code has config-merge, window-iterator, and packaging gaps. The CLI pivot should preserve the ideas after audit/fix, not assume the existing `cli.py` and `window.py` can be wired directly to lftomo.
- **Git history is available now, but not deeply analyzed here.** The current workspace clones are not shallow; the old depth-1 warning no longer applies locally. A short history review may still be useful, but it is no longer a blocker just to see past commits.
- **ChengLab consent for `localCT` lifts.** Per Sorotassu's note, this is the only outside-the-house source — `localCT` is the lift that needs an explicit conversation (BAC holotomography in `phase_contrast/` is also Cheng-lab code; both should be covered in the same ask). All other lineages — dan_scripts, hatmehyt, lftomo, b10_recon, the rest of B10 — are Sorotassu's or "do whatever with".
- **The "best base library" recommendation is opinionated.** Reasonable people could argue for `localCT` instead of lftomo, on the grounds that it's the most production-shaped package in the set. The counter is that `localCT` is half hardware-control by mass; the recon part of it is smaller than lftomo's. The 2-part structure proposed here partially defuses this: lftomo as library can drop into localCT, so a localCT-centric user doesn't have to abandon their orchestration to adopt the unified algorithms.
- **localCT has patterns worth copying even if it is not the base.** Its `ReconstructionRequest`/result dataclasses, scan-folder inspection, run summaries, TomoPy runtime bootstrap, and tiny generated TIFF tests are concrete assets. Ignoring them would mean reinventing solved scaffolding.
- **Dask coexistence is unproven for this codebase.** The "SharedNP within nodes + Dask across nodes" picture is the standard recipe, but lftomo's `SharedNPArray` was designed against `multiprocessing`, not Dask. The Phase 1 no-op Dask backend prototype is the cheapest way to catch a fundamental incompatibility early.

---

## Suggested immediate next moves

In rough order of cost-to-benefit:

1. **Open an architecture issue** in `den-sq/lftomo` capturing this plan so the conversation is durable. Cross-link from a parallel issue in `ChengLabResearch/ReconChengLab` to bring localCT maintainers in for the super-res consent conversation only.
2. **Stabilize lftomo imports and dependency metadata.** Move package metadata to `pyproject.toml` + `hatchling`, add/update conda-first environment docs, and make `import lftomo.util.window`, `import lftomo.prep.flat`, `import lftomo.prep.phase`, and `import lftomo.prep.stripe` smoke-testable in the documented env. `uv pip` can be documented as the fast editable-install path inside that env.
3. **Make the old-lftomo-API decision.** Either add a temporary compatibility shim for B10/notebooks (`DataWindowSet`, `SharedNP`, `distribute_read`) or port those scripts and prep modules to current `TomoDataSet` / `TomoWindow` / `SharedNPArray` immediately.
4. **Fix the `print_times.py` bug** in B10. One-liner; unrelated to the rest of the plan.
5. **Draft the canonical `scan.yaml` schema** as a pydantic model and validate it can absorb each of the five lineages' metadata. Cheapest way to find the unification's first showstopper if there is one.
6. **Prototype the `Stage` ABC inside lftomo** with explicit mutability/window/artifact/resource contracts, then re-express *one* lftomo function (recommend `weighted_normalize`) through it, with both a NumPy backend and a stub Dask backend. Concrete code to react to is more useful than more prose, and the Dask stub forces the ABC to be backend-agnostic from day one.
7. **Add Tier-1 synthetic fixtures using the localCT style.** Generated tiny TIFF/gain folders in `tmp_path`, no absolute lab paths. Use this to make the import/API cleanup visible in CI before bigger reference data lands.
8. **Open hatmehyt's `recon-cli-pivot` branch.** Strip (`segment/`, `analyze/`, `recon/recon.py`, `recon/center_find.py`, empty `prep/phase.py`+`prep/stripe.py`), move `capture/` placeholder for the `lfcapture` migration, preserve and repair the CLI concepts, retitle the README, and add package metadata. Set up the `[capture]` extras hook (empty for now). Land as a coherent PR to make the lineage visible in `git log`.
9. **Add the `lfcapture` package skeleton** in the existing repo with a README pointing at the migration plan and its own `lfcapture` console-script stub. Receives `capture/` from hatmehyt + `dan_scripts/capture.py` as the first imports; hardware abstractions later.
10. **Catalog `recon_minimal.py`'s TomoPy-fix wrappers.** Phase-1.5 deliverable; gates Phase-2 work on the `recon`, `center`, and `phase` Stage classes. For each TomoPy function `recon_minimal.py` wraps non-trivially, document: what does the wrapper do beyond marshalling args? is the fix upstream already? if not, is it worth upstreaming? Per wrapper, triage to (a) upstream and adopt upstream, (b) carry as canonical with a deprecation pointer, or (c) confirm upstream already has it and adopt pinned upstream. Without this, Phase 2's "canonical impl per algorithm" rule can't be applied accurately to TomoPy-backed Stages.
