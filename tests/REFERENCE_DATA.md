# Tier-2 reference data

Tier-2 data remains on lab-internal storage and is exercised only by the
`[self-hosted, yarrkha]` workflow. The repository contains path fixtures and
layout validation, not the projection data itself. The exact installed
metadata is versioned in `tests/tier2_metadata/`; each file is copied to its
dataset root as `scan.yaml`.

## Dataset registry

| Fixture | Stable path on the Linux runner | Acquisition notes | Applicable producers |
| --- | --- | --- | --- |
| `dataset_striping_tiff` | `/mnt/z/gpfs/Influenza_APS_2022/Mouse_Lungs/AAA588_25.45kV/Scan/1651106162_77000projs_90000us_Mouse_Lung_PTA_Normal_Real_Scan` | 25.45 keV, 77,000 TIFF projections, 90 ms exposure, 14,192 × 2,960 pixels, 0.7046 µm pixel size (10 mm detector width / 14,192 horizontal pixels); deliberately heavy striping. Existing recon: `Z:\gpfs\Influenza_APS_2022\COVID_Influenza_Progression\AAA588\data\20kV_mouse2_lung_untreated_PTA_afterbeamup_recon_p26`. | `r254`, `recon_minimal` |
| `dataset_phase_contrast_hdf5` | `/mnt/wsl/PHYSICALDRIVE2p1/B10/PCRBox_Octo0/20kev/voi2.1` | B10 dxchange HDF5, 20.0003 keV, 1,969 projections over 180°, 2,560 × 2,160 pixels, 0.637 µm effective pixel size. Existing recon: `alt_recon/recon`. | `phase_contrast`, `r254`, `b10_recon` (old lftomo pinned as documented in the unification plan) |
| `dataset_helical_tiff` | `/mnt/z/gpfs/phenome/Argonne/2022_April/initial_setup/AAAPhantom_25.45kV/scan/1650859932_32000projs_60000us_Helical_Whole_Phantom_Scan` | 25.45 keV, 32,000 TIFF projections, 60 ms exposure, 7,968 × 3,056 pixels, 0.7046 µm pixel size; helical motion is approximately 2,792 detector rows per 360°. Put generated reconstructions below this dataset root. | `r254` |

Each path can be overridden without changing the tests:

- `LFTOMO_DATASET_STRIPING_TIFF`
- `LFTOMO_DATASET_PHASE_CONTRAST_HDF5`
- `LFTOMO_DATASET_HELICAL_TIFF`

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

## Running the checks

The default test run collects but skips Tier-2 tests. On a host with the three
datasets mounted and canonical metadata installed, run:

```bash
pytest tests/test_tier2_datasets.py --run-tier2
```

The manually dispatched `Tier-2 dataset validation` workflow runs the same
command on `[self-hosted, yarrkha]`.
