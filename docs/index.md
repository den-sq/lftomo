# lftomo

lftomo is the lab's library for large-format tomography reconstruction. A
reconstruction is described by two YAML files: a **Scan** (facts about one
acquisition) and a **Recon** (an ordered pipeline of Stages). It runs with the
`lftomo-run` command or the equivalent Python call.

## Getting started

- [Quickstart](quickstart.md): from a raw scan to reconstructed slices, on the
  command line and from Python.
- [Configuration](configuration.md): what the Scan and Recon files contain, which
  values to take from your acquisition, how paths and calibration are resolved.
- [Using lftomo from Python](library.md): `run_reconstruction()`, its return
  value, and the optional windowing and calibration controls.

Installation is described in the
[README](https://github.com/den-sq/lftomo#installation).

## Reference

- [Configuration models](api/config.md): every field, generated from the
  validating models.
- [Pipeline API](api-pipeline.md): Stage contracts, managed storage and
  dispatch.
- [Reconstruction conventions](reconstruction-conventions.md),
  [filtering conventions](filtering-conventions.md),
  [irregular projection angles](irregular-projection-angles.md).

## Going further

- [Candidate spaces](candidate-spaces.md): sweeping parameters and pipeline
  variants in one Recon.
- [Calibration bundles](calibration-bundles.md): validated flat/dark and beam
  models shared across scans.
- [Experiment operations](experiment-operations.md): searching, scoring and
  promoting pipeline parameters.
- [Windowed streaming](windowed-streaming.md): bounding memory on large volumes.

## Examples

- [Reconstruction notebook](https://github.com/den-sq/lftomo/blob/master/notebook/recon.ipynb)
  and [B10 notebook](https://github.com/den-sq/lftomo/blob/master/notebook/recon_b10.ipynb).
- [`examples/`](https://github.com/den-sq/lftomo/blob/master/examples/README.md):
  the current B10 driver and the experiment template.

## Project and operations

- [Pipeline-core epic acceptance](epic-acceptance.md)
- [Recon-data runner](recon-data-runner.md)
