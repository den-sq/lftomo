# Tomographic Reconstruction Pipeline Survey

**Date:** 2026-05-29
**Author:** Tava (assisted analysis for Sorotassu)
**Scope:** five distinct reconstruction-script lineages used in/by the lab
**Method:** parallel structural surveys (Explore agents) of each source, followed by hand synthesis

## Sources surveyed

| # | Source | Role per Sorotassu |
|---|---|---|
| 1 | [`den-sq/dan_scripts`](https://github.com/den-sq/dan_scripts/) | Sorotassu's current production — focus on `recon_minimal.py` + includes; archived `recon.py` ignored; helical lives in older archived scripts |
| 2 | [`den-sq/hatmehyt`](https://github.com/den-sq/hatmehyt/) | Never-completed CLI redesign |
| 3 | [`den-sq/lftomo`](https://github.com/den-sq/lftomo/) | More recent approach, intended as a better baseline |
| 4 | [`ChengLabResearch/ReconChengLab`](https://github.com/ChengLabResearch/ReconChengLab/) | Parallel lineage from another lab member |
| 5 | `/media/sorotassu/great_sarak/remote_copy/B10Scripts.tar.gz` | Modified scripts from yet another lab member |

The user's stated path `/media/sorotassu/great_sarak/B10/` was actually `…/remote_copy/B10/`; the working scripts are in `B10Scripts.tar.gz` adjacent to it.

---

## 1. `den-sq/dan_scripts` — `recon_minimal.py` (current production)

**Headline:** Click-based CLI, three-YAML-config driven, sliding-window memory model, TomoPy+ASTRA backend, custom Paganin phase. Helical is **stubbed** here — working helical lives in archived `berkeley_*` scripts.

### Pipeline (lines, function names from `recon_minimal.py`)

- **Read** — `distribute_read()` (l.682) over TIFF stack via `byteread_helper` (l.663). Multi-threaded sinogram buffering.
- **Normalize** — `clean_data()` (l.466). Either `tomopy.normalize()` (l.491) or custom weighted pre/post (l.265).
- **Stripe removal** — `prep.stripe.remove_dead_vo` / `remove_large_vo` / `remove_sort_vo` (ll.517–525, imported l.29).
- **Phase retrieval** — `prep.phase.phase_retrieval()` (l.1188). Custom Paganin FFT.
- **Centering** — `tomopy.find_center_vo` (l.553), `tomopy.find_center` (l.557), or entropy-local (l.562). Polynomial tilt fit ll.577–613.
- **Reconstruction** — `tomopy.recon()` with `gridrec` (l.1198) or `tomopy.astra` (l.1209) for CUDA/iterative.
- **Write** — `recon_write()` (l.616), per-slice TIFF.

### Algorithms exposed (enum `cli.RA`, ll.148–156)

`GRIDREC` (default), `FP_CUDA`, `BP_CUDA`, `FBP_CUDA`, `SIRT_CUDA`, `SART_CUDA`, `CGLS_CUDA`, `EM_CUDA`.

### Inputs

- TIFF stack, read sinogram-ordered (Y × Θ × X) via `tifffile.memmap`.
- Four flats: `PREGAIN`, `POSTGAIN`, `PREDARK`, `POSTDARK` (enum from `util.data_store.FLAT`), pre-medianed to `{flat}_median.tif`.
- **YAML triplet** required: machine (hardware/storage), session (location/name), scan (`Degrees Per Fov`, `Projections Per Fov`, `Height`, `Width`, `Energy`, `Resolution`-microns, `SSD`-cm, `FOV Count`).
- Path discovery via `DataStore` (l.889) from machine+session+project (ll.78–84).

### Outputs

Per-slice 2D TIFF at `{ds.recon}/{sample_id}_{y_index}.tif` (ll.1237–1238). Configurable dtype via `--out_dtype` (l.852, default `np.float32`). No HDF5 output.

### CLI

Click command `reconstruct()` (l.794). `CommandWithConfigsWrapper` (l.776) overlays a `--run_conf` YAML over defaults. Key options: `--slice_range`, `--window_size`, `--recon_algorithm`, `--center_find` (multi), `--phase_alpha`, `--rot_center`, `--tilt`, `--job_id`/`--job_segments`/`--job_index` for parallel jobs. Flags `LOCAL_GAINS`, `LOCAL_THETA`, `SKIP_CENTER_NORMALIZATION`, `THREAD_READ` (l.836).

### Capture

`capture.py` — Euresys Coaxlink 201 GenICam camera (l.16) + ESP301 motion stage (l.78). Outputs TIFFs or memory-mapped NPYs with timestamped scan log.

### Helical

**Not in `recon_minimal.py`.** TODO comment at l.148 ("Check for helical"), l.511 ("Different for helical?"), l.1147 ("non-zero fall speed means helical") — detection only, no coordinate transform. Working helical in:
- `berkeley_2021/allinwonderful_v1.5_A1_Helical.py` — `parallel_fill_array_helically()` ll.91–116
- `berkeley_2020/allinwonderful_v1.56_A1_Helical.py`
- `argonne_2022/scripts/allinwonderful_v1.5_A1_Helical.py`

### Coupling / reusability

Main reconstruction loop (ll.1125–1242) tightly integrates I/O, normalization, phase, rotation, recon. Hard to extract individual stages. Cleanly liftable: `SharedNP` memory abstraction (`util/mem.py` ll.27–149); `prep.phase.*` (89 lines); `prep.stripe.*`. No checkpoint/resume.

---

## 2. `den-sq/hatmehyt` — stalled CLI framework

**Headline:** Excellent CLI scaffolding (Click, chained sub-commands, `@hht_step`, `Window`/`Slider`, `GroupWithConfigsWrapper`). **Zero algorithms implemented.** Phase and stripe modules are empty files. No TomoPy/ASTRA imports anywhere. Last commit Sep 15 2022 (`eb22673`).

### What's implemented vs stubbed

| Module | Status | Notes |
|---|---|---|
| `recon/recon.py` | Stub | Log-only; no algorithm |
| `recon/center_find.py` | Stub | Decorator + log |
| `prep/prep.py` | Stub | Empty body |
| `prep/flatfield.py` | Partial | 436 lines; frame-timing, vertical-travel, scintillator mask; not wired to CLI |
| `prep/phase.py` | **Empty file** | 0 lines |
| `prep/stripe.py` | **Empty file** | 0 lines |
| `capture/general.py` | Partial | 77 lines; `scan_prep()` is functional metadata handler |
| `capture/camera.py` | Stub | Log-only |
| `segment/segment.py` | Stub | Log-only; `WINUSE.YES` flag set |
| `analyze/cell.py` | Stub | Log-only |
| `handle/meta.py` | Complete | 272 lines; CSV/YAML/Google Sheets/credential mgmt |
| `handle/transfer.py` | Partial | `get_record`/`write_record` CLI steps |

### CLI design (the reason it matters)

- Click `chain=True` group (`cli/cli.py` ll.223–237), invoked via `hatmehyt prep --stub X reconstruct --stub Y …`.
- `GroupWithConfigsWrapper` (ll.52–76): `--configs` accepts YAML files, merges into Click context.
- Custom param types: `Range` (l.79), `Crop` (l.96), `Frange` (l.120), `KVList` (l.140), `EnumParameter` (l.151).
- `@hht_step()` decorator (ll.163–196): wraps a stage function with memory context, metadata extraction, and a dispatcher keyed on `WINUSE` (enum l.19–29: `YES`/`SET`/`NO` for sequential / window-setting / non-windowed).
- `Window` dataclass (`window.py` ll.13–80): 3D image slice with vertical/projection/horizontal ranges, padding, offsets; methods `shift()`, `offset()`, `padded()`, `fresh()`, `done()`, `unpadded()`, `yx()`, `px()`.
- `Slider` class (referenced `cli.py` l.310): iterates windows with configurable stride/crop.
- `shared_np` (`mem.py` ll.49–80): `multiprocessing.shared_memory` wrapper for inter-process large-array passing.
- Command chaining (`cli.py` ll.243–286 `_build_command_chain()`): chains windowed/window-setting/non-windowed commands into blocks managing memory lifecycle.

### Inputs (where wired)

CSV/YAML/Google Sheets metadata via `meta.fetch_meta()`. `capture/general.scan_prep()` (ll.48–77) expects `projections`, `sections`, `degrees_per_section`, `vertical.start`/`stop`, `resolution`, `crop` (ProjDim), `helical` boolean, optional `log` for frame timing. Image globs via `tifffile`.

### Outputs

`hht_run.yaml` (cli.py l.361–362) with executed parameters; `meta.write_meta()` to CSV/YAML/Sheets. **No reconstruction output exists** — the recon path is stubbed.

### Why it stalled (best guess from tree)

Final merge of "implement-window-handling" branch (Sep 2022). All recon/seg/analyze modules created as empty stubs on that merge. TODOs in `flatfield.py` ll.419/425 ("deal with multiple exposures", "Simplify"). Reads as: chassis was finished and proven; engine never installed.

### Lift candidates

`@hht_step`, `Window`+`Slider`, `shared_np`, `meta.FMT`+`fetch_meta`/`write_meta`, `GroupWithConfigsWrapper`. No plugin registry / Stage ABC / Pipeline class — those still need invention.

---

## 3. `den-sq/lftomo` — more recent library

**Headline:** Library, no CLI. Strong shared-memory + windowing + I/O infrastructure. TomoPy backend. **Migration prototype exists** in B10's `b10_recon.py`, but it imports older lftomo names (`DataWindowSet`, `SharedNP`, `distribute_read`) that are not present in the current package — strong intent signal, not a current working consumer.

### Modules

- `prep/flat.py` (160 lines) — `weighted_normalize` (ll.44–71), `calculate_shift` (ll.78–146). Stubs: `shift_normalize`, `shifting_flat` (ll.74–75, 148–149).
- `prep/phase.py` (160 lines) — Paganin retrieval (TomoPy rewrite): `phase_retrieval` (ll.113–124) spawns multiprocessing pool over `_phase_retrieval` (ll.127–159) per projection slice; handles height padding for FFT stability.
- `prep/stripe.py` (221 lines) — Fourier-Wavelet (`remove_stripe_fw` ll.160–182 → `_remove_stripe_fw` ll.190–220), Vo median (`remove_sort_vo` ll.31–51), dead-line interp (`remove_dead_vo` ll.120–157).
- `util/mem.py` — `SharedNPArray`, `SharedNPManager`.
- `util/window.py` — `TomoWindow`, `TomoDataSet`, strided iteration; projection or sinogram order swappable via shape type (ll.249–258).
- `util/io.py` — `byteread_helper` (ll.39–67), `get_file_set` for HDF5/TIFF (ll.70–98), `read_window` distributed reader (ll.115–177), TomoPy center-find integration, `write_slice` per-slice TIFF with optional int conversion (ll.261–273).
- `util/shapes.py` — typed dataclasses `DataShape` / `ProjOrder` / `SinoOrder` / `ReconOrder` / `ImgSlice`.
- `util/log.py` — Click-based colored logging + memory monitoring.

### Backend

TomoPy (`tomopy.recon`, `find_center_vo`, `find_center_pc`); NumExpr (`numexpr.evaluate` in phase.py l.155); SciPy (`medfilt2d`, `median_filter`, `binary_dilation`, `uniform_filter1d`, `interpolate.interp2d`); PyWavelets for FW stripe; tifffile/h5py for I/O. **No ASTRA, no GPU path yet.**

### Inputs

- Dir of natsorted single-page TIFFs, multi-page TIFF, or HDF5 with `exchange/data` (ll.76, 255–257 in io.py).
- Sinogram or projection order via shape type.
- Pre-medianed flats: `predark_median.tif`, `pregain_median.tif`, `postdark_median.tif`, `postgain_median.tif` (FLAT enum, flat.py ll.17–41). Notebook reads them into `SharedNP` separately (ll.110–117).
- FOV/angles via `CTFov` namedtuple (window.py l.17); no sidecar metadata files. Angles derived from FOV (ll.199–204).

### Outputs

Per-slice TIFF via `write_slice` to e.g. `output/recon/{5-digit-index}.tif`. float32 internal; `np_convert` can map to int16/uint16 (io.py ll.25–36). Center-find diagnostics → `cf_data/s{start}_{center:.2f}.tif` (ll.252–253).

### CLI

**None.** `click` imported only for colored logging (log.py l.9). Driven from `notebook/recon.ipynb` and from `b10_recon.py`.

### Architecture

- Light dataclasses (`SharedNPArray`, `TomoDataSet`, `TomoWindow`).
- **Functions, not stages.** No `Pipeline`/`Stage` ABCs, no registry. Extension means writing new functions and manually sequencing them.
- Shared memory is the central design pattern; well-suited to absorbing other scripts if they adopt `SharedNPArray`.
- Tests (~1100 lines across `tests/util/`): `test_shape`, `test_window`, `test_mem`, `test_io`, `test_log`. Solid on infrastructure. **Zero tests on `prep/*` functions.**

### Capture

None.

### Recent activity

depth=1 clone — single visible commit ("I/O and tests", `ae28851`). Package version 0.0.1. Recent direction is I/O + test consolidation; not yet production-shaped.

---

## 4. `ChengLabResearch/ReconChengLab` — parallel lineage

**Headline:** Multi-branch tree, no top-level README. Hosts the **ancestor of B10's R2.54** (`original/recon_dan_v1.py`), a config-driven refactor (`phase_contrast/`), a clean modular package (`localCT/`), a beamline acquisition GUI (`synchrotron/`), and Neuroglancer prep tools. Distinctive feature: **Bronnikov-aided correction (BAC) holotomography** via external `hotopy` lib.

### Top-level

| Dir | Role |
|---|---|
| `original/` | Archived baseline monoliths (`recon_dan_v1.py` ~4500 lines, `hotopagv2.py` ~5000 lines, `recon_dan_v1_sweep*.py` parameter sweeps, `inits.py`, `funcs.py`) |
| `phase_contrast/` | INI-driven refactor — `recon_v1.py` (~350 lines), `funcs.py` (~1500 lines, phase-heavy), `utils.py` (INI + HPC env detection) |
| `phase_fringes/` | Fringe-domain phase **analysis** (not retrieval); `fringe_utils.py` + task runners (`run_task1_distance_phase_contrast.py` for SSD studies 30–190 mm) |
| `synchrotron/ALS_B8/` | `R3_Berkeley8_v0.20_gui.py` — tkinter + napari acquisition GUI for B8 beamline; `sadiff.py` SAD diffraction setup |
| `localCT/` | Modular Python package with GUI/CLI, hardware abstraction (camera, PI/ACS/Newport stages, EPICS hooks), `main_super_recon.py`, `recon/pipeline.py`+`inspection.py` |
| `Neuroglancer/` | `ct_to_ng.py` (3D TIFF → cloudvolume precomputed, histogram inspection), `STEP_1_cloudvolume_precom_from_tiff_*.py`, mip-gen shell pipelines |
| `utils/` | `parallel_processing.py`, `run_recon_main.py` (legacy), gain/noise/memory helpers |
| `test/` | Single test script |
| `SinoSuperRez/` | **Empty stubs**; real super-res lives in `localCT/recon/super_recon/` |

### Canonical pipeline (`phase_contrast/recon_v1.py`)

```
Load INI → discover/sort projection TIFFs →
sinogram generation (parallel, memmap):
    gain_averaging (taper-blended pre/post gains)
    helical pitch correction (vertical_velocity per section)
    skew correction (cv2.warpAffine)
    movement correction (gain-shift registration)
sinogram post (per chunk, multiprocess):
    all_of_the_stripe_removal (multi-algorithm)
    [if holo enabled] hotopy.holo.BronnikovAidedCorrection(α, γ)
    [optional] bilateral filter
center finding (TomoPy find_center_pc / find_center_vo / manual) →
TomoPy gridrec recon (parallel by slice) →
edge trim → rotation → crop →
write TIFF stacks (write_full, write_bin2/4)
```

### Backend

TomoPy main; pyfftw FFT; hotopy (external) for BAC holotomography; cv2 for skew/bilateral; scipy interp/FFT/ndimage; numpy/tifffile/h5py/psutil; cloudvolume/igneous/zarr for Neuroglancer.

Flags `astratoolbox=False`, `ASTRA_ALGO=False` — no ASTRA or custom CUDA in active code.

### Distinctive

- **Bronnikov-aided correction** (`phase_contrast/funcs.py`): `BronnikovAidedCorrection(image.shape, alpha, gamma)` per-projection, between gain normalization and sinogram assembly. Projection-domain phase retrieval (not image-domain Paganin).
- **No classical Paganin** found.
- **`phase_fringes/`** is diagnostic — gradient + high-pass for fringe width/contrast. Orthogonal to recon.
- **`localCT/`**: clean modular ROI/high-res lab CT; GUI-driven; full hardware stack; designed for sectional + helical; super-res in `recon/super_recon/`:
  - **DRHT** (Dense Residual Hierarchical Transformer, 2023): sparse-view angular upsampling, patch-based (128×128), attention-weighted loss.
  - **LDM** (physics-guided latent diffusion, ICIP 2025): mask-inpainting for arbitrary missing-view / limited-angle; **code present, LDM checkpoints missing**.
- **`synchrotron/ALS_B8/R3_Berkeley8_v0.20_gui.py`**: tkinter GUI for live preview, projection capture, metadata logging. Acquisition only.
- **`Neuroglancer/ct_to_ng.py`**: TIFF → cloudvolume precomputed, full histogram inspection (uint16 65536-bin, percentiles, saturation), optional igneous mip generation, workarounds for intervaltree null-interval crashes.

### Inputs

```
proj_dir/
  projections/         # individual TIFFs per angle (img_NNNN.tif), memmap-read uint16
  gains/
    gain.tif, gain_post.tif
    dark.tif, dark_post.tif
```

Metadata via INI (`init_*.ini`) or hardcoded (`original/recon_dan_v1.py` ll.46–87). Optional `scanlog_.txt` for frame timing. `read_h5=False` by default but parsing fallback exists.

### Outputs

Per-slice TIFF stacks (full + bin2 + bin4 variants). Neuroglancer precomputed via `ct_to_ng.py`. `localCT/app_super_recon` writes per-stage TIFF + JSON stats + preview images + histograms.

### CLI

- `original/` + `phase_contrast/`: hardcoded paths + INI; no CLI; edit-and-rerun.
- `localCT/`: proper package CLI — `python -m localct.main --config config/sample_system.yaml`, `python -m localct.main_super_recon`, `localct-super-recon` console script.
- `Neuroglancer/`: standalone scripts, no CLI.

### `original/`

Archived monoliths — `recon_dan_v1.py` is the ancestor of B10 R2.54. Shows pre-refactor history.

### Lift candidates (per the survey)

**Modular / plugin-ready:** stripe removal, COR estimation, Neuroglancer precompute export, gain/dark with time blending, DRHT super-res.

**Entangled:** holotomography (BAC tightly bound to projection-domain gain+stripe order), helical pitch correction, movement/skew registration, `localCT` hardware control.

**Missing:** LDM inpainting code (described in `app_super_recon.md` ll.41, 96 but absent), LDM checkpoints.

---

## 5. B10 scripts (`B10Scripts.tar.gz`)

**Headline:** Two acquisition systems × energies × VOIs, each holding a per-VOI hand-edited fork of `R2.54_Berkeley8_hpc_v1.81.py` (3,240 lines, descendant of `ReconChengLab/original/recon_dan_v1.py`). **Each VOI dir also contains a `b10_recon.py` (~150 lines)** that imports `lftomo` against an older utility API (`DataWindowSet`, `SharedNP`, `distribute_read`) — a migration prototype, not a runnable consumer of current lftomo. The intent is to migrate off R2.54 onto lftomo; the path to that endpoint requires porting (or shimming) b10_recon to the current lftomo names first.

### Canonical script

`PCRBox_Octo0/14kev/voi1/R2.54_Berkeley8_hpc_v1.81.py` — 3,240 lines, Dec 31 2024. Per-VOI copies differ **only in hardcoded constants**:

| Line | Field | voi1 | voi2 |
|---|---|---|---|
| 98 | `sinogram_chunk_size` | 400 | 500 |
| 119–120 | alpha range | 0.5–10 | 2.5 only |
| 197 | `center_of_rotation_px_matx` | 1264.5 | 1271.5 |
| 274 | `chunk_overlap` | 20 | 0 |

Fork-by-edit-in-place pattern.

### Variants in same dirs

- `_alt`: aggressive memory tuning (chunk_size 24), output to `32bit_reconstructed_FS_baseline`, narrowed α/β search, `find_center_pc=False`.
- `_h5_support` (voi1/14kev only): adds `read_h5=True` flag (l.76) + H5 branch in `parallel_sectional_write_…` (ll.1076–1100) using `h5py.File['exchange']['data'][idx]`. Outputs bin4 16-bit.
- `_used_to_moveCor` (RedBox_Octo3 only): differs in one line — `movement_correction=[False, 0.0, -1.0376]` vs base `[False, 0.26…, -1.707…]`. Historical experiment.
- `b10_recon.py`: **completely different lineage**. ~150 lines. Sorotassu's own lftomo-based rewrite of the R2.54 monolith (author path `/home/dnorthover/Cheng Code/lftomo`; `dnorthover` is Sorotassu). Imports `lftomo` against an **older utility API**: uses `SharedNP` (now `SharedNPArray` in current lftomo), `DataWindowSet` (now `TomoDataSet`), and `distribute_read` (no longer present). Structured logging, native H5 (`/exchange/data`), `hotopy.holo.BronnikovAidedCorrection` with empirical γ computation (ll.84–94). **As of 2026-05-29 this file will not run against checked-out lftomo without porting or a compatibility shim** — it documents migration intent, not a working consumer.

### Pipeline (canonical R2.54)

```
read (H5 dxchange primary, TIFF fallback) →
normalize (flat-field + 50%-theta gain switching) →
remove_all_stripe (TomoPy multi-algorithm, l.630) →
[optional] hotopy.holo.BronnikovAidedCorrection (l.2011) →
center (find_center_pc=True or manual) →
[optional] cv2.warpAffine for skew/movement (ll.1192–1279) →
TomoPy gridrec (ncore=32, l.3135) OR ASTRA SIRT_CUDA (ll.3119–3131) →
per-slice TIFF write (ll.3172–3191), optional bin4 16-bit
```

### Backend

TomoPy main; pyfftw; hotopy (`hotopy.holo.BronnikovAidedCorrection`); cv2; ASTRA optional; multiprocessing.Pool (no MPI, no cupy).

### Inputs

- Primary: HDF5 dxchange (`exchange/data`, `exchange/data_white`, `exchange/data_dark`).
- Fallback: TIFF stack in `projections/` + flats in sibling `gains/{gain,gain_post,dark,dark_post}.tif`.
- Optional scanlog (`scanlog_.txt`, l.52); file-name natural-key sort.

### Outputs

`rec_NNNNN.tif` per slice in `32bit_reconstructed_{sample_name}` (local) or `/gpfs/.../32bit_reconstructed_{uid}_360_pass2` (HPC, l.3150). 32-bit float; optional 16-bit bin4. If phase applied, output dir suffixed `_BAC_alpha_gamma` (l.3156).

### CLI

**None.** All config hardcoded in lines 1–300. Per-VOI dirs are independent forks. `b10_recon.py` is the alternative — Path objects, relative dirs, structured.

### HPC

`_hpc_` is literal:
- ll.1995–2008: SLURM env detection (`$SLURM_NODELIST`, `$SLURM_NTASKS`); if present, switches output to `/gpfs`, sets `apply_hpc_settings=True`.
- Bumps chunk to 200, disables find_center, forces full_recon.
- ll.3200–3237: tmp cleanup from `/dev/shm` and `/tmp`.
- TomoPy gridrec with ncore=32.

### Top-level helpers

| File | Purpose |
|---|---|
| `httpserver.py` (28 lines) | CORS-enabled static HTTP server, port 8080 — Neuroglancer hosting |
| `print_and_flats.py` (62 lines) | Extracts pre/post flats from H5, writes **medians** (`np.median`) to `yolo_flats/`. No bug — `k, v = sorted_scan_list[x]` is assigned (l.36) before `writer.writerow(v)` is called (l.37). |
| `print_times.py` | **Sibling-but-not-identical** to `print_and_flats.py`: writes **means** (`np.mean`), and **has bug** — `writer.writerow(v)` at l.38 runs *before* `k, v = sorted_scan_list[x]` at l.39, so `v` is unbound on the first iteration and stale on subsequent ones. (Survey originally misattributed the bug to `print_and_flats.py`; the actually-buggy file is `print_times.py`.) |
| `ng_transformed/remove_gz.py` (10 lines) | Decompresses Neuroglancer precomputed `.gz` chunks in-place |

### Lineage relative to dan_scripts

`Berkeley8` label → ALS B8 beamline descendant. Direct ancestor is `ReconChengLab/original/recon_dan_v1.py`, not `dan_scripts/berkeley_*`. So this is **the Cheng lineage adapted by a third party**, not a fork of Sorotassu's work.

---

## Notes on method

- Each per-source survey was produced by an Explore agent against the cloned tree (`/tmp/recon-survey/<repo>/`).
- All four GitHub repos: shallow (depth=1) clones. Full git history not consulted.
- The B10 archive was extracted from `/media/sorotassu/great_sarak/remote_copy/B10Scripts.tar.gz`.
- Line numbers above are from the depth=1 clones as of 2026-05-29.
- Where a survey said "ambiguous" or "checkpoints missing" or "stubbed", that's preserved here — please don't read certainty into it.
