# Configuration

lftomo's configuration is a set of Pydantic models in `lftomo.config.models`,
each of which reads and writes YAML. This page explains what each one is for and
how to prepare the two files a reconstruction needs. Field-by-field details
(types, defaults, constraints) are in the
[configuration model reference](api/config.md), which is generated from the
models and so always matches what the code validates.

## Which files a run uses

| Model | Typical file | Holds | Used by `lftomo-run` / `run_reconstruction()` |
|---|---|---|---|
| `Scan` | `scan.yaml` | Facts about one acquisition: energy, geometry, angles, where the projections are | Yes |
| `Recon` | a pipeline file, e.g. `recon.yaml` | The ordered reconstruction Stages and their parameters, output location | Yes |
| `Machine` | `machine.yaml` | Per-deployment storage roots, scratch, thread and memory budgets | No |
| `Session` | `session.yaml` | Per-session sample, operator, location, project | No |

The runner reads exactly two files, a `Recon` and a `Scan`. `Machine` and
`Session` are defined and validated but not consumed by the reconstruction
runner; you do not need them to reconstruct. Per-run resource settings that do
exist live on `Recon` (`process_workers`, `worker_timeout_seconds`) and on the
command line (`--window-rows`).

All models reject unknown keys (`extra="forbid"`), so a misspelled field name
fails validation with the field name in the error.

## Scan

A `Scan` describes one acquisition and nothing about how to reconstruct it. Keep
it to measured facts.

### Required fields

| Field | Meaning | Where to find it |
|---|---|---|
| `energy_keV` | Beam energy | Beamline/instrument metadata (dxchange: `measurement/instrument/monochromator/energy`, in eV) |
| `pixel_size_um` | Effective pixel size at the sample | Detector metadata (dxchange: `measurement/instrument/detector/pixel_size`, in mm) |
| `n_projections` | Number of recorded projection images actually present | Length of the projection stack |
| `degrees_per_fov` | Angular coverage per field of view, usually 180 or 360 | Scan setup |
| `detector.pixel_size_um` | Detector pixel size | Detector metadata |
| `detector.rows`, `detector.columns` | Projection image height and width. Optional in the model, but the runner plans its read windows from them and fails without them. | Projection stack shape |
| `acquisition.n_projections` | Number of planned acquisition slots, including any dropped frames | Scan setup; equal to `n_projections` when nothing was dropped |
| `acquisition.rotation_end_deg` | End of the rotation, endpoint-exclusive (see below) | Scan setup / angle dataset |
| `projections` | Where the projections are (see below) | — |

### Fields to set when they apply

- `ssd_cm`: sample-to-scintillator distance. Required by propagation-based phase
  Stages that use physical units (the Paganin family); BAC does not use it. For
  dxchange: `measurement/instrument/camera_motor_stack/setup/camera_distance`,
  in mm.
- `acquisition.rotation_start_deg` (default 0) and `acquisition.exposure_ms`.
- `n_fov`, `helical`: multi-field-of-view and helical acquisitions.
- `metadata`: free-form, lineage-specific values. It also carries legacy
  four-map flat fields (`metadata.flat_fields`), see
  [Calibration](#calibration).

### Angles

`rotation_start_deg` to `rotation_end_deg` is divided into
`acquisition.n_projections` endpoint-exclusive slots. A scan that records both
its first and last angle (for example 1969 images from 0° to 180° inclusive)
therefore has `rotation_end_deg = 180 + 180/1968`, one step past the last
angle. With `180.0` the angles do not fit the slots, and reading fails with
`acquisition index … is outside … planned slots`.

`acquisition.angle_source` selects where per-projection angles come from:

- omitted: uniform slots, no dropped frames;
- `kind: hdf5`: an angle dataset in the projection file (default
  `exchange/theta`, in degrees);
- `kind: hardware_timestamps`: a TIFF scan log; gaps become dropped slots;
- `kind: explicit`: inline acquisition indices and/or angles.

When `n_projections` and `acquisition.n_projections` differ (dropped frames), an
explicit `angle_source` is required. The full contract is in
[Irregular projection angles](irregular-projection-angles.md).

### Projections

Two storage formats are supported, selected by `projections.format`:

```yaml
# dxchange HDF5
projections:
  format: hdf5_dxchange
  path: scan.h5
  data_path: exchange/data          # defaults shown
  white_path: exchange/data_white
  dark_path: exchange/data_dark
```

```yaml
# TIFF stack
projections:
  format: tiff_stack
  path: projections                 # directory
  pattern: "*.tif*"
```

A relative `projections.path` (and a relative `hardware_timestamps` path) is
resolved from the directory containing `scan.yaml`, not from the working
directory.

Example Scan files for several lab lineages are kept as test fixtures in
[`tests/fixtures/scan_yaml_lineages/`](https://github.com/den-sq/lftomo/tree/master/tests/fixtures/scan_yaml_lineages).
Their numbers are illustrative; copy the structure, not the values.

## Recon (the pipeline file)

A `Recon` is an ordered list of Stages plus run options:

```yaml
output_path: recon
pipeline:
  - kind: read
    algorithm: hdf5_dxchange
    params: {order: projection, calibration_policy: required}
  - kind: normalize
    algorithm: weighted
    params: {pre_subtracted: true}
  - kind: phase
    algorithm: bronnikov_aided
    params: {alpha: 1.6, gamma: 1.2}
  - kind: stripe
    algorithm: fw
    params: {wavelet: sym20, sigma: 2.0}
  - kind: center
    algorithm: manual
    params: {value_px: 1282.75}
  - kind: recon
    algorithm: gridrec
  - kind: write
    algorithm: tiff_slices
    params: {dtype: float32}
```

### Stages

Each entry is a `StageConfig`: `kind` plus an optional `algorithm` select the
Stage (`kind.algorithm`, e.g. `stripe.fw`), and `params` are passed to that
Stage's constructor. The available Stage names are the keys of
`lftomo.pipeline.stages.STAGE_REGISTRY`:

```shell
python -c "from lftomo.pipeline.stages import STAGE_REGISTRY; print('\n'.join(sorted(STAGE_REGISTRY)))"
```

`params` is not schema-checked at YAML load. Each Stage validates its own
parameters when it is constructed, which the runner does for every Stage before
it reads any projection data. A misspelled parameter therefore still fails
before any data is read, but the error comes from the Stage constructor. A
Stage's accepted parameters are its constructor arguments, in
`lftomo/pipeline/stages/`.

The runner also checks the order of Stages before running: each Stage declares
the data layout and domain (transmission or attenuation) it accepts and
produces, so for example a Stage that needs transmission data cannot follow the
negative-log step.

An optional `id` names a Stage occurrence. IDs are needed for
[candidate spaces](candidate-spaces.md): a `sweep` on a Stage, or named
`variants`, expand one `Recon` into several concrete runs.

### Output

- `output_path`: the directory slices are written to. A relative path is
  resolved from the working directory. `--output` on the command line (or
  `output_path=` in Python) overrides it.
- When a `Recon` expands to several candidate runs, each is written to its own
  `candidate-<key>` subdirectory beneath `output_path`.
- When an output path is set, every run writes `recon_run.yaml` next to its
  slices, containing the resolved `Recon` (`resolved_recon`) and the
  streaming window plan that was used (`streaming_dataset`). Rerunning
  `resolved_recon` does not restore that plan; the window settings
  (`--window-rows`, `--extra-pad-y`, or `window_config=`) have to be passed
  again, as they come from the run, not the `Recon`.

### Paths inside the pipeline file

The runner resolves two kinds of relative Stage parameter path from the
directory of the pipeline file: a normalize Stage's `calibration_manifest`, and
the legacy `pre_gain`/`post_gain`/`pre_dark`/`post_dark` TIFFs. This is why a
bundled template that names such files should be copied next to your data
before being edited; see the [Quickstart](quickstart.md#2-pick-a-pipeline-template).

The runner passes other path parameters to the selected Stage unchanged, and
that Stage's contract determines how they are interpreted. For example,
`write.hdf5_volume`'s `path: results/recon.h5` writes to `results/recon.h5` under
the working directory, wherever the pipeline file lives, while
`center.entropy_local`'s `diagnostic_subdirectory: qc/centers` is placed relative
to `Recon.output_path` and must be a safe relative path; an absolute value is
rejected when the Stage is built.

### Bounding a run

- `read` Stage `y_start` / `y_stop`: global, half-open detector rows to
  reconstruct. Output slice names keep global row numbers.
- `read` Stage `theta_stride`: use every Nth projection, for quick tests.
- `--window-rows N` (or `window_config=` in Python): stream the selected rows
  through windows of N rows, bounding memory. See
  [Windowed streaming](windowed-streaming.md).

## Calibration

Normalization needs flat (white) and dark fields. The runner resolves them
before constructing Stages, in this order:

1. a calibration bundle selected in the pipeline file (`calibration_manifest` on
   the normalize Stage) or passed by a Python caller;
2. legacy four-map fields in `scan.yaml` (`metadata.flat_fields`, relative to
   `scan.yaml`);
3. legacy `pre_gain`/`post_gain`/`pre_dark`/`post_dark` paths on the normalize
   Stage (relative to the pipeline file);
4. whites and darks embedded in a dxchange HDF5 file (`read` Stage
   `calibration_policy: optional` or `required`).

For a dxchange file that already contains `exchange/data_white` and
`exchange/data_dark`, option 4 needs no extra configuration. Bundles, their
validation and their compatibility checks are described in
[Calibration bundles](calibration-bundles.md).

## Writing YAML from Python

Every model has `to_yaml()` and `from_yaml()`, and validates on construction, so
a Scan can be built from values read out of an acquisition file and saved:

```python
from lftomo.config import Acquisition, Detector, HDF5DxchangeProjections, Scan

scan = Scan(
	energy_keV=20.0003, pixel_size_um=0.637, ssd_cm=8.18,
	n_projections=1969, degrees_per_fov=180.0,
	detector=Detector(pixel_size_um=0.637, rows=2160, columns=2560),
	acquisition=Acquisition(
		n_projections=1969, rotation_end_deg=180 + 180 / 1968,
		angle_source={"kind": "hdf5", "path": "exchange/theta"},
	),
	projections=HDF5DxchangeProjections(format="hdf5_dxchange", path="scan.h5"),
)
open("scan.yaml", "w").write(scan.to_yaml())
```

To validate existing files without running anything:

```python
from lftomo.runner import load_recon, load_scan

recon, recon_path = load_recon("recon.yaml")   # or a bundled template name
scan, scan_path = load_scan("scan.yaml")
```

`load_recon` validates the document; Stage parameters are only checked when the
Stages are built (see [Stages](#stages)).
