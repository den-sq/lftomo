# Tier-2 reference data

Tier-2 data remains on lab-internal storage and is exercised only by a runner
with `[self-hosted, recon-data]`. The repository contains path fixtures and
layout validation, not the projection data itself. The exact installed
metadata is versioned in `tests/tier2_metadata/`; each file is copied to its
dataset root as `scan.yaml`.

## Dataset registry

| Fixture | Stable path on the Linux runner | Acquisition notes | Applicable producers |
| --- | --- | --- | --- |
| `dataset_striping_tiff` | `/mnt/wsl/PHYSICALDRIVE2p1/data/AAA592` | Berkeley B7 AAA592 center section 6, 24 keV, 3,937 projections over 180°, 11,648 × 3,000 pixels, 0.5 µm pixel size. This is the `striping_tiff` / `recon_minimal` reference dataset, replacing the earlier AAA588 candidate whose raw scan and cited historical reconstruction did not match. The recovered thresholded copy contains 3,935 TIFFs; the historical job used all 3,937 and recorded no dropped frames. Exact run configuration and producer commit are archived on `Z:` under `AAA592/data/history/24keV_center_section_6/7`. | `r254`, `recon_minimal` |
| `dataset_phase_contrast_hdf5` | `/mnt/wsl/PHYSICALDRIVE2p1/B10/PCRBox_Octo0/20kev/voi2.1` | B10 dxchange HDF5, 20.0003 keV, 1,969 projections over 180°, 2,560 × 2,160 pixels, 0.637 µm effective pixel size. Existing recon: `alt_recon/recon`. | `r254`, `b10_recon` (old lftomo pinned as documented in the unification plan) |
| `dataset_phase_contrast_tiff` | `/mnt/z/gpfs/Daphnia_Phase_ALS_B8_Oct_2023/AAA523_110mm_20keV/1696573090_1969projs_175000ussection2_projections_tiff` | AAA523 Daphnia ALS B8 sectional TIFF scan, 20 keV, 1,969 projections over 180°, 5,008 × 4,440 pixels, 0.365 µm pixel size, 110 mm propagation distance. Completed phase_contrast BAC 2/1.25 output is present beside the input. | `phase_contrast` |
| `dataset_helical_tiff` | `/mnt/z/gpfs/phenome/Argonne/2022_April/initial_setup/AAAPhantom_25.45kV/scan/1650859932_32000projs_60000us_Helical_Whole_Phantom_Scan` | 25.45 keV, 32,000 TIFF projections, 60 ms exposure, 7,968 × 3,056 pixels, 0.7046 µm pixel size; helical motion is approximately 2,792 detector rows per 360°. Put generated reconstructions below this dataset root. | `r254` |
| `dataset_dropped_frame_tiff` | `/mnt/wsl/PHYSICALDRIVE2p1/ru_samples/AAA388_20kV/1653171676_3937projs_200000us_redo_scan` | AAA388 two-drop TIFF acquisition: 3,937 planned slots and 3,935 recorded projections. Fourth-column hardware buffer timestamps are authoritative and resolve missing planned indices 2,837 and 2,888; wall-clock timestamps are not used. This is a metadata/import fixture and does not require a canonical reference-output layout. | angle-import validation only |

Each path can be overridden without changing the tests:

- `LFTOMO_DATASET_STRIPING_TIFF`
- `LFTOMO_DATASET_PHASE_CONTRAST_HDF5`
- `LFTOMO_DATASET_PHASE_CONTRAST_TIFF`
- `LFTOMO_DATASET_HELICAL_TIFF`
- `LFTOMO_DATASET_DROPPED_FRAME_TIFF`

The Windows `Z:` drive maps to
`\\main.vast.pennstatehealth.net\Files\Labs\Keith_Cheng_Lab` and is exposed to
WSL as `/mnt/z` with DrvFS. From Windows PowerShell, mount it with:

```powershell
wsl -d Ubuntu -u root -- mkdir -p /mnt/z
wsl -d Ubuntu -u root -- mount -t drvfs Z: /mnt/z
```

The existing Windows logon supplies the share credentials. A Yarrkha runner
with the data mounted directly at `/gpfs`, or any workstation with a different
mount point, should set the corresponding environment variables.

## Canonical layout

Each fixture resolves to a directory with this shape:

```text
dataset-root/
  scan.yaml
  projections/
  flats/
  scanlog.txt                  # optional
  reference/
    <producer>/
      recon/
      recon.sha256
      metadata.yaml
      pipeline.yaml
```

`scan.yaml` must parse as `lftomo.config.Scan`; paths inside it should be
relative to `dataset-root`. HDF5 data may keep whites and darks in the dxchange
file, but the `flats/` directory is still retained as the canonical slot (and
may be an empty directory or a symlink). Existing folders named `gains/` can be
exposed as `flats/` with a directory symlink; projection data should likewise
be linked rather than copied when an overlay is needed.

Reference slots are created now, but outputs are populated by the per-producer
reference issues. Once populated, every slot must contain all four entries
shown above. `recon.sha256` is produced from the `recon/` directory with:

```bash
find recon -type f -print0 | sort -z | xargs -0 sha256sum > recon.sha256
```

`metadata.yaml` records the producer script path and commit, host, date,
operator, runtime, and environment notes. `pipeline.yaml` is the complete,
expanded stage order and parameter set used by that producer.

## Frozen reference inventory

| Dataset | Producer | Slices | Frozen | Versioned manifest |
| --- | --- | --- | --- | --- |
| `phase_contrast_hdf5` | `r254` | 1076–1085 | 2026-07-13 | `tests/tier2_metadata/references/phase_contrast_hdf5/r254/` |
| `phase_contrast_hdf5` | `b10_recon` | 1076–1085 | 2026-07-13 | `tests/tier2_metadata/references/phase_contrast_hdf5/b10_recon/` |
| `striping_tiff` | `recon_minimal` | 1800–1809 | 2026-07-13 | `tests/tier2_metadata/references/striping_tiff/recon_minimal/` |
| `phase_contrast_tiff` | `phase_contrast` | 2000–2009 | 2026-07-14 | `tests/tier2_metadata/references/phase_contrast_tiff/phase_contrast/` |

These archives were lifted from completed historical volumes rather than
rerunning their producers. The recon_minimal archive has exact producer and
configuration provenance; the unversioned R2.54 and b10_recon archives record
that their surviving scripts were modified after the historical runs and are
therefore not executable numerical contracts. Frozen references are immutable:
do not rerun a producer to refresh them.

## Running the checks

The default test run collects but skips Tier-2 tests. On a host with the
registered datasets mounted and canonical metadata installed, run:

```bash
pytest tests/test_tier2_datasets.py --run-tier2
```

The manually dispatched `Recon-data Tier-2 validation` workflow runs the same
command on `[self-hosted, recon-data]`.
