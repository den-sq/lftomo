# Reconstruction Unification Buildout Plan

Updated after local review of the checked-out `lftomo`, `hatmehyt`, `dan_scripts`, `B10Scripts`, `lfcapture`, and `ReconChengLab/localCT` trees.

## Current filing assumptions

- File detailed issues for Phase 0, Phase 1, Phase 1.5, and foundational test-data work.
- Keep Phase 2 / 3 / 4 as epics until the Phase 1 API and Stage contracts settle.
- `lfcapture` now exists, but is effectively an empty initial commit. File `lfcapture` work directly in `den-sq/lfcapture` rather than using `hatmehyt` as a temporary home.
- Package management decision is settled: use `pyproject.toml` + `hatchling` for package metadata/builds, keep conda/mamba environment files as the recon/capture runtime contract, and treat `uv pip` as optional fast glue inside those envs.
- If project-board linkage is required, the GitHub token still needs `read:project` / `project` scope. Repo issues can be filed without that linkage.

## Updated issue inventory

| # | Title | Repo | Phase | Notes |
|---|---|---|---|---|
| 1 | Standardize lftomo packaging and conda-first runtime envs | lftomo | 0/1 | Move to `pyproject.toml` + `hatchling`, update conda/mamba env docs, document optional `uv pip install -e ".[dev]"`, and add import smoke tests. |
| 2a | Port `lftomo/prep/{flat,phase,stripe}.py` to current utility API | lftomo | 0/1 | Self-contained inside lftomo: replace `DataWindowSet` / `SharedNP` / `distribute_read` with `TomoDataSet` / `TomoWindow` / `SharedNPArray` across the three prep modules. No old-name compatibility shim. Update or add prep-module tests against the new API. |
| 2b | Port `lftomo/notebook/*.ipynb` to current utility API | lftomo | 1 | Same renames as #2a, applied to example notebooks. Smoke-runnable end-to-end in the documented conda env. Split from #2a because notebook diffs are noisy. |
| 2c | Port `b10_recon.py` to current lftomo API (active-use copy) | lftomo | 1 | Produces a current-API copy of `b10_recon.py` for ongoing development against checked-out lftomo. **Independent of the snapshot-frozen reference at `lftomo@ddfcdf6`** (which #28e uses to generate the b10_recon producer reference and which must NOT be modified). Coordinate filename / branch convention so the two copies of `b10_recon.py` do not collide. |
| 3 | Define Stage / PipelineState / Pipeline ABCs with explicit contracts | lftomo | 1 | Include shape, mutability, window axes, required metadata, artifacts, resource hints, and Dask stub backend. |
| 4 | Pipeline-spec validator: shape and contract compatibility | lftomo | 1 | Depends on #3; validates ordering, mutation/fork semantics, artifacts, and backend choice. |
| 5 | Define unified config schema (`Machine` / `Session` / `Scan` / `Recon`) | lftomo | 1 | Pydantic models importable by CLI and downstream consumers. |
| 6 | Validate `scan.yaml` schema against all five surveyed lineages | lftomo | 1 | Depends on #5. |
| 7 | Lift localCT request/result/run-summary patterns into lftomo design | lftomo | 1 | Use `ReconstructionRequest`, folder inspection, JSON summaries, TomoPy bootstrap, and tiny-TIFF test style as reference patterns. |
| 8 | Tier-1 synthetic fixtures and hermetic tiny-TIFF tests | lftomo | 1 | Shepp-Logan, striped, misaligned, and flat/gain fixtures generated in `tmp_path`. |
| 9 | Convert machine-specific lftomo tests to hermetic or integration-only | lftomo | 1 | Existing absolute-path data references should not run in normal CI. |
| 10 | Re-express `weighted_normalize` as first Stage proof | lftomo | 1 | Depends on #2a (prep ports), #3 (ABCs), #4 (validator). |
| 11 | Re-express Paganin phase-retrieval algorithms as two Stages + tests | lftomo | 1/2 | **Two distinct algorithms, both kept as separate algorithm options under the `phase` Stage class** (not multiple impls of the same algorithm): (a) **standard phase retrieval** — Paganin filter for phase-only retrieval; (b) **one-shot phase/absorption** — coupled phase+absorption retrieval, Sorotassu's canonical re-implementation. Each lands as its own `phase` algorithm option with its own canonical impl. No benchmark exercise — there is one canonical impl per algorithm. Depends on #3 (ABCs), #4 (validator). |
| 12 | Re-express Vo stripe algorithms as Stages + tests | lftomo | 1/2 | `remove_sort_vo`, `remove_dead_vo`, `remove_large_vo`. |
| 13 | Re-express Fourier-wavelet stripe removal as Stage + tests | lftomo | 1/2 | `remove_stripe_fw`. |
| 14 | First-class TIFF stack read Stage | lftomo | 1/2 | Benchmark current `byteread_helper` against `recon_minimal.distribute_read`. |
| 15 | First-class HDF5 dxchange read Stage | lftomo | 1/2 | `exchange/data`, `data_white`, `data_dark`; clarify direct-offset assumptions. |
| 16 | TIFF slices write Stage | lftomo | 1/2 | Start with current `write_slice`, then add dtype/range semantics. |
| 17 | Pin exact per-lineage pipeline order in reference specs | lftomo | 1/2 | BAC/stripe/log/skew order differs across B10, R2.54, and phase_contrast. |
| 18 | Catalog `recon_minimal.py` TomoPy-fix wrappers | lftomo | 1.5 | Gates `recon`, `center`, and `phase` Stage canonicalization. |
| 19 | Fix B10 `print_times.py` pre-assignment bug | lftomo | 0 | `writer.writerow(v)` before `v` is assigned; not part of algorithm unification. Filed under lftomo per the "loose items file under the target repo" rule, since lftomo is the unification target B10 is moving onto. |
| 20 | hatmehyt packaging baseline | hatmehyt | 1 | Add `pyproject.toml` + `hatchling` metadata before CLI pivot work. |
| 21 | hatmehyt recon-cli-pivot: strip stubs and audit CLI concepts | hatmehyt | 1 | Preserve ideas, repair defects instead of copying blindly. Issue body to enumerate known defects to audit: (a) config-merge correctness in `GroupWithConfigsWrapper` / `--configs` handling; (b) `Window` / `Slider` iteration semantics in `cli.py:310` and `window.py`; (c) `scan_prep()` placement (currently `capture/general.py:48`, needs to move to the CLI ingest side); (d) packaging gaps (covered separately by #20). Each defect gets a yes/no "fixable in pivot" call as audit output. |
| 22 | Top-level recon command-surface stubs | hatmehyt | 1 | `ingest`, `prep flats`, `run`, `convert`, `export`. |
| 23 | Canonical-layout ingester | hatmehyt | 1 | Depends on schema validation. |
| 24a | Config importer: `from-recon-minimal-yaml` | hatmehyt | 1 | Translates `recon_minimal.py`'s machine + session + scan YAML triplet into the unified `recon.yaml` shape. Depends on #5. |
| 24b | Config importer: `from-phase-contrast-ini` | hatmehyt | 1 | Translates `phase_contrast/init_*.ini` INI sections into the unified `recon.yaml`. Depends on #5. |
| 24c | Config importer: `from-r254-hardcoded` | hatmehyt | 1 | Parses R2.54's top-300-line hardcoded block and emits a unified `recon.yaml`. Includes `hoto_tomo_algo_list` → sweep-`kind: list` translation. Depends on #5. |
| 24d | Config importer: `from-b10-recon` | hatmehyt | 1 | Reads a `b10_recon.py` (current-API copy) and emits a unified `recon.yaml`. Depends on #5 and on #2c for the current-API b10_recon source. |
| 25 | Add lfcapture package skeleton and console script | lfcapture | 1/3 | **Status: completed** in `den-sq/lfcapture@449fe1a` (2026-06-03). Repo skeleton, `pyproject.toml` on hatchling, `lfcapture` Click app with `plan` / `run` / `status` stubs, smoke tests. File and immediately close at filing time, with a link to the commit, so the issue exists on the board for audit-trail purposes. |
| 26a | Migrate `dan_scripts/capture.py` into lfcapture | lfcapture | 3 | Euresys Coaxlink 201 + ESP301 motion-stage adapter. Sorotassu-owned source; no consent concerns. Depends on #25. |
| 26b | Migrate hatmehyt `capture/` subtree into lfcapture | lfcapture | 3 | Coordinates with #21's strip step. `scan_prep()` stays on the hatmehyt ingest side per #21 audit defect (c). Depends on #25. |
| 26c | Lift `localCT/hardware/` abstractions into lfcapture | lfcapture | 3 | **Consent-gated**: ChengLab conversation must close before this issue moves out of Backlog. Camera + multi-stage backends (PI / ACS / Newport XPS / harvester / egrabber / opencv). Depends on #25 + ChengLab consent thread (tracked separately, not as an issue). |
| 27 | Self-hosted GitHub Actions runner on Yarrkha | lftomo | 1 | Enables non-public Tier-2/Tier-3 reference workflows. |
| 28a | Tier-2 dataset infrastructure: placement, `scan.yaml`, reference-archive layout | lftomo | 1/2 | Land three Tier-2 datasets on Yarrkha with canonical layout (`scan.yaml` + `projections/` + `flats/`). Validate `scan.yaml` against #6's pydantic schema. Define the `reference/<producer>/{recon/, metadata.yaml, pipeline.yaml}` directory convention. Wire pytest fixtures + runner-label routing. Depends on #6, #27. |
| 28b | Generate R2.54 producer references | lftomo | 1/2 | Run R2.54 (`R2.54_Berkeley8_hpc_v1.81.py` or VOI-matched variant) against applicable Tier-2 datasets. Freeze under `reference/r254/`. Depends on #28a, #17. |
| 28c | Generate recon_minimal producer references | lftomo | 1/2 | Run `dan_scripts/recon_minimal.py` against applicable Tier-2 datasets. Freeze under `reference/recon_minimal/`. Depends on #28a, #17. |
| 28d | Generate phase_contrast producer references | lftomo | 1/2 | Run `ChengLabResearch/phase_contrast/recon_v1.py` against applicable Tier-2 datasets. **Pull-only** — no upstream push. Freeze under `reference/phase_contrast/`. Depends on #28a, #17. |
| 28e | Generate b10_recon producer references against `lftomo@ddfcdf6` | lftomo | 1/2 | Install `lftomo @ ddfcdf628254f0dadd219b076fa01da527ca6123`; run the pre-port `b10_recon.py` against applicable Tier-2 datasets. Freeze under `reference/b10_recon/` with `metadata.yaml` recording both b10_recon SHA and `lftomo@ddfcdf6`. **Independent of #2c**'s current-API port (must not touch the same `b10_recon.py` file used here). Depends on #28a, #17. |
| E1 | Phase 2: algorithm composition as Stages | lftomo | 2 | Epic checklist of Stage classes and algorithm options. |
| E2 | Phase 3: lfcapture build-out and hatmehyt capture wrapper | lfcapture / hatmehyt | 3 | Epic after capture package skeleton exists. |
| E3 | Phase 4: retire and redirect legacy recon scripts | varies | 4 | Deprecation READMEs, import shims, tagged legacy releases; excludes hatmehyt itself. |

## Labels to create

Recommended common labels:

- `phase:0`, `phase:1`, `phase:1.5`, `phase:2`, `phase:3`, `phase:4`
- `architecture`, `api-drift`, `dependencies`, `package-management`, `testing`, `reference-data`
- `cli`, `capture`, `algorithm`, `migration`, `consent-needed`

## Notes for filing

- **No issues are to be filed against `ChengLabResearch/*` repos** (including `ReconChengLab` and any `localCT/` work). Pulling code in is fine; pushing back is not, and tracking lives on the lftomo/hatmehyt/lfcapture project board with cross-repo references where useful. Backend-switch work touching `localCT` itself stays on a local branch or a fork.
- Add a tracker comment on the Phase 2 epic for ChengLab consent around `localCT/recon/super_recon/` (`drht`, `ldm`) and any `localCT/hardware/` lift into `lfcapture`. The consent thread is a tracker note, not an issue.
- Do not turn every Phase 2 algorithm option into a detailed issue until the Stage ABC and validator have landed.
- B10's `b10_recon.py` remains a strong migration signal, but issue text should call it a prototype pinned to an older lftomo API (`SharedNP` / `DataWindowSet` / `distribute_read`), not a current working consumer of this checkout. The remediation path is immediate porting, not compatibility shims.
- Reference recons are organized as a producer suite (R2.54, `recon_minimal.py`, `phase_contrast/recon_v1.py`, `b10_recon.py` against pinned old lftomo `ddfcdf6`) — one reference per (producer × dataset) combination, not one per dataset. Issues that reference Tier-2/3 references should make the multi-producer shape explicit and separate this snapshot-reference producer (#28e, against `lftomo@ddfcdf6`) from #2c's immediate port of `b10_recon.py` to current lftomo for ongoing development.
- Issue text should not promise uv-only installs. The supported path is conda/mamba first for compiled scientific and hardware dependencies, then `uv pip` or pip for editable local packages and PyPI-only extras inside that env.
- `hatmehyt` is not a transitional CLI. Per the 2026-06-03 decision, it stays as the canonical CLI long-term; no Phase-4 retirement issue against the hatmehyt repo itself.
