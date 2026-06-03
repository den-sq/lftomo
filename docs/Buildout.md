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
| 2 | Port old lftomo consumers and prep modules to current API | lftomo | 0/1 | Immediate port is the decision: update B10/notebooks and `prep/flat.py`, `prep/phase.py`, `prep/stripe.py` from `DataWindowSet` / `SharedNP` / `distribute_read` to `TomoDataSet` / `TomoWindow` / `SharedNPArray`; no old-name compatibility shim. |
| 3 | Define Stage / PipelineState / Pipeline ABCs with explicit contracts | lftomo | 1 | Include shape, mutability, window axes, required metadata, artifacts, resource hints, and Dask stub backend. |
| 4 | Pipeline-spec validator: shape and contract compatibility | lftomo | 1 | Depends on #3; validates ordering, mutation/fork semantics, artifacts, and backend choice. |
| 5 | Define unified config schema (`Machine` / `Session` / `Scan` / `Recon`) | lftomo | 1 | Pydantic models importable by CLI and downstream consumers. |
| 6 | Validate `scan.yaml` schema against all five surveyed lineages | lftomo | 1 | Depends on #5. |
| 7 | Lift localCT request/result/run-summary patterns into lftomo design | lftomo | 1 | Use `ReconstructionRequest`, folder inspection, JSON summaries, TomoPy bootstrap, and tiny-TIFF test style as reference patterns. |
| 8 | Tier-1 synthetic fixtures and hermetic tiny-TIFF tests | lftomo | 1 | Shepp-Logan, striped, misaligned, and flat/gain fixtures generated in `tmp_path`. |
| 9 | Convert machine-specific lftomo tests to hermetic or integration-only | lftomo | 1 | Existing absolute-path data references should not run in normal CI. |
| 10 | Re-express `weighted_normalize` as first Stage proof | lftomo | 1 | Depends on #2-#4. |
| 11 | Re-express Paganin phase retrieval as Stage + regression tests | lftomo | 1/2 | Depends on API cleanup and Stage contract. |
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
| 24 | Config importers | hatmehyt | 1 | `from-recon-minimal-yaml`, `from-phase-contrast-ini`, `from-r254-hardcoded`, `from-b10-recon`. |
| 25 | Add lfcapture package skeleton and console script | lfcapture | 1/3 | Repo exists; add README, package layout, `pyproject.toml` + `hatchling`, and `lfcapture` entry point. |
| 26 | Migrate capture sources into lfcapture | lfcapture | 3 | `dan_scripts/capture.py`, hatmehyt `capture/`, selected localCT hardware abstractions after consent. |
| 27 | Self-hosted GitHub Actions runner on Yarrkha | lftomo | 1 | Enables non-public Tier-2/Tier-3 reference workflows. |
| 28 | Tier-2 datasets: placement, `scan.yaml`, producer-suite reference recons | lftomo | 1/2 | Three small real datasets. For each dataset, identify which legacy producers apply (R2.54, `recon_minimal.py`, `phase_contrast/recon_v1.py`, `b10_recon.py` against pinned old lftomo `ddfcdf6`) and generate one reference per applicable producer. Reference layout per dataset includes `reference/<producer>/recon/` + `pipeline.yaml`. **Depends on #17** for pinning producer-specific pipeline order before references are frozen. |
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
- Reference recons are organized as a producer suite (R2.54, `recon_minimal.py`, `phase_contrast/recon_v1.py`, `b10_recon.py` against pinned old lftomo `ddfcdf6`) — one reference per (producer × dataset) combination, not one per dataset. Issues that reference Tier-2/3 references should make the multi-producer shape explicit and separate this snapshot-reference producer from Buildout #2's immediate port to current lftomo.
- Issue text should not promise uv-only installs. The supported path is conda/mamba first for compiled scientific and hardware dependencies, then `uv pip` or pip for editable local packages and PyPI-only extras inside that env.
- `hatmehyt` is not a transitional CLI. Per the 2026-06-03 decision, it stays as the canonical CLI long-term; no Phase-4 retirement issue against the hatmehyt repo itself.
