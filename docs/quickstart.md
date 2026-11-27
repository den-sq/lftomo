# Quickstart

This walks one acquisition from raw projections to reconstructed slices, first
with the `lftomo-run` command and then with the same call from Python. It uses a
dxchange HDF5 scan and the bundled `b10-recon-canonical` pipeline; the same
steps apply to a TIFF-stack scan with a different Scan `projections` block (see
[Configuration](configuration.md)).

## 1. Install

Follow the [installation instructions in the README](https://github.com/den-sq/lftomo#installation).
They create the supported conda environment and install lftomo into it. Add the
`bac` extra if the pipeline uses Bronnikov-aided correction, as
`b10-recon-canonical` does.

Check that the command is available:

```shell
lftomo-run --help
```

## 2. Pick a pipeline template

lftomo ships four pipeline templates in `lftomo/pipelines/`:

| Template | Input | Origin |
|---|---|---|
| `recon-minimal-canonical` | TIFF stack, sinogram order | `dan_scripts/recon_minimal.py` lineage |
| `phase-contrast-canonical` | TIFF stack, projection order, external gain/dark TIFFs | phase-contrast lineage |
| `b10-recon-canonical` | dxchange HDF5 with embedded whites/darks | B10 `b10_recon.py` (voi2.1) |
| `r254-canonical` | dxchange HDF5 with embedded whites/darks | B10 R2.54 lineage |

Each template is a complete, ordered `Recon` document that reproduces its
originating script. The acquisition-specific values in it, such as the rotation
center, the projection count and the output directory, are the originating
dataset's. Treat them as a starting point, not as defaults for your scan.

A template can be passed by name (`b10-recon` or `b10-recon-canonical`), but
for your own data, copy it next to your files and edit the copy. The
calibration paths a pipeline file names (for example the gain/dark TIFFs in
`phase-contrast-canonical`) resolve from the directory of that pipeline file, so
a template used by name would look for them inside the installed package.

```shell
python -c "from lftomo.runner import resolve_pipeline_path; print(resolve_pipeline_path('b10-recon'))"
cp "$(python -c "from lftomo.runner import resolve_pipeline_path; print(resolve_pipeline_path('b10-recon'))")" recon.yaml
```

There is no `init` or scaffolding command; copying a template (or writing a
model with `to_yaml()`, see [Configuration](configuration.md#writing-yaml-from-python))
is the supported way to start.

## 3. Write `scan.yaml`

The Scan file holds the facts of the acquisition. For a dxchange HDF5 scan whose
`exchange/data` is 1969 × 2160 × 2560 and whose `exchange/theta` runs from 0° to
180° inclusive:

```yaml
energy_keV: 20.0003
pixel_size_um: 0.637
ssd_cm: 8.18
n_projections: 1969
degrees_per_fov: 180.0
detector:
  pixel_size_um: 0.637
  rows: 2160
  columns: 2560
acquisition:
  n_projections: 1969
  rotation_start_deg: 0.0
  rotation_end_deg: 180.0914634   # 180 + 180/1968: one step past the last angle
  angle_source:
    kind: hdf5
    path: exchange/theta
    units: degrees
projections:
  format: hdf5_dxchange
  path: scan.h5                  # relative to this scan.yaml
```

Read every value from your acquisition: energy, pixel size and sample-to-
scintillator distance from the instrument metadata (in this file,
`measurement/instrument/...`), the projection count and detector shape from the
`exchange/data` shape. The [Configuration guide](configuration.md#scan) explains
each field.

The `rotation_end_deg` line is the one most often wrong. lftomo divides
`rotation_start_deg` to `rotation_end_deg` into `n_projections`
endpoint-exclusive slots. A scan that records both 0° and 180° has 1968 steps
of 180/1968°, so its exclusive end is 180 + 180/1968. Setting `180.0` there
fails with `acquisition index 1969 is outside 1969 planned slots`. See
[Irregular projection angles](irregular-projection-angles.md).

## 4. Adjust the pipeline

Open the copied `recon.yaml` and set at least:

- the rotation center: `center` → `manual` → `value_px` (and `tilt` if the axis is
  tilted), or replace the stage with `center.vo` / `center.pc` to search for it;
- `theta_count` on the `read` stage, to your projection count;
- `output_path`, or pass `--output` on the command line.

For a first run, restrict the detector rows on the `read` stage so the run is
quick:

```yaml
  - kind: read
    algorithm: hdf5_dxchange
    params: {order: projection, calibration_policy: required, theta_count: 1969,
             y_start: 1080, y_stop: 1090}
```

`y_start`/`y_stop` are global, half-open detector rows; output slices keep their
global row numbers.

## 5. Run from the command line

```shell
lftomo-run recon.yaml scan.yaml --output recon/
```

The runner validates both files, builds and validates every Stage, resolves the
calibration (here the whites and darks embedded in the HDF5 file), and only then
reads projections. It prints one line per run:

```text
run 0 [f6623d9703a3767a]: /path/to/recon
```

`recon/` then contains one float32 TIFF per reconstructed row (`01080.tif`, …)
and `recon_run.yaml`, a run record (not a pipeline file). Its `resolved_recon`
section is the `Recon` that produced the slices, with every `Recon` field filled
in, and `streaming_segments` records the directional window plans used. Saved on its
own as a YAML file, `resolved_recon` reruns the same pipeline, but the window
plan is not restored from it: pass the same window options again. BAC and slope
Paganin use Theta batches. Slope and affine Paganin read the complete physical
detector height even when an output row selection is requested. Legacy
alpha Paganin still follows its Y window and padding extent, which can change
its numerical result. See [windowed streaming](windowed-streaming.md).

Useful options: `--json-output` prints machine-readable run summaries;
`--window-rows N` streams the volume through windows of N detector rows to bound
memory; `--window-theta N` controls projection batches for phase retrieval
(defaults to `--window-rows`);
`--intermediate-root PATH` selects temporary storage for direction changes;
`--gpu` declares CUDA support for GPU Stages.

## 6. Run the same thing from Python

```python
from pathlib import Path

from lftomo.runner import run_reconstruction


def main() -> None:
	results = run_reconstruction("recon.yaml", "scan.yaml", output_path="recon")
	for result in results:
		print(result.output_path, sorted(p.name for p in Path(result.output_path).glob("*.tif")))


if __name__ == "__main__":
	main()
```

The `if __name__ == "__main__":` guard is required: the runner starts worker
processes, which import your script again. See the [Python library guide](library.md)
for the return value and the optional controls.

## Troubleshooting

- **`insufficient shared memory: planned peak ...`** The runner keeps its working
  arrays in POSIX shared memory (`/dev/shm`) and checks their planned peak against
  the free space before reading projections. Containers often limit it. Docker's
  default of 64 MB may hold only one row of a large scan. The required capacity
  also depends on the phase batch and selected detector extent. Raise the limit
  (for Docker, `--shm-size`) or reduce the working set with `--window-rows`,
  `--window-theta`, or the `y_start`/`y_stop` source selection.
- **`Bus error`, `BrokenProcessPool`, or a worker dies with no traceback.** Usually
  the same shortage, in memory the check doesn't count. Vo stripe Stages
  (`stripe.vo_*`, used by `r254-canonical` and `recon-minimal-canonical`) allocate
  scratch of about twice the normalized working array on top of the planned peak,
  and other processes can use `/dev/shm` after the check. Raise the limit; fewer
  rows per window also helps until each window is down to one row.
- **`unknown pipeline ...`**: the first argument is neither an existing file nor a
  bundled template name; the error lists the available templates.
- **`normalization requires an explicit calibration bundle, legacy four-map fields,
  or embedded HDF5 whites/darks`**: no flat/dark source was found. See
  [Calibration bundles](calibration-bundles.md#runner-resolution) for the
  resolution order.
