# lftomo
Large Format Tomography Package

## Installation

Use the conda/mamba environment as the runtime contract for reconstruction
deployments. TomoPy and other compiled scientific packages should come from
conda-forge / bioconda rather than a standalone PyPI-only environment.

```shell
mamba env create -f environment.yml
conda activate lftomo
```

Then install the local checkout into the activated environment. `uv pip` is the
fast editable-install path when it is available; plain `pip` is also fine.

```shell
uv pip install -e ".[dev,tomopy]"
# or: pip install -e ".[dev,tomopy]"
```

Bronnikov-aided correction is optional because it executes through hotopy.
Install it only on deployments that use `phase.bac`:

```shell
pip install -e ".[bac]"
```

Do not treat a uv-only virtualenv as the supported deployment path for
TomoPy-heavy reconstruction work. Use it only as pip-compatible glue inside the
conda environment.

## Run a pipeline

Pass a pipeline YAML (or a bundled canonical template name) and a Scan YAML to
the runner. Relative projection paths in the Scan are resolved from the Scan
file, and `--output` overrides the pipeline's output directory.

```shell
lftomo-run recon-minimal scan.yaml --output recon/
lftomo-run phase-contrast-canonical scan.yaml --output recon/ --json-output
```

Stage sweeps expand automatically. Multiple concrete runs are written beneath
`sweep-0000`, `sweep-0001`, and so on. Use `--gpu` only when the runtime has the
CUDA dependencies required by a selected GPU Stage.

Angle sampling, gridrec defaults, and the optional TomoPy/NumPy compatibility
contract are documented in
[`docs/reconstruction-conventions.md`](docs/reconstruction-conventions.md).

## Tests

Unit tests should be hermetic: generate any required files under `tmp_path` or
load static fixtures from `tests/`. Tests that require local beamline data,
machine-mounted paths, or hardware are integration tests and must be marked with
`@pytest.mark.integration`.

```shell
pytest -q
pytest -q --run-integration
```

`--run-integration` runs lab-local tests on a host with the `recon-data`
capability. On other hosts those tests remain skipped unless
`LFTOMO_RUN_INTEGRATION=1` is set explicitly. The default test run keeps those
checks skipped so a clean checkout can run without lab-local files.

Self-hosted runner architecture, service operations, storage prerequisites, and
recovery procedures are documented in
[`docs/recon-data-runner.md`](docs/recon-data-runner.md).
