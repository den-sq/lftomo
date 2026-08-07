# lftomo
Large Format Tomography Package

## Installation

Use the conda environment as the compiled runtime and default-test contract for
reconstruction deployments. TomoPy and other compiled scientific packages
should come from conda-forge / bioconda rather than a standalone PyPI-only
environment. Supported Python versions are 3.11 through 3.14.

```shell
CONDA_CHANNEL_PRIORITY=strict conda env create --solver libmamba -f environment.yml
conda activate lftomo
```

Then install the local checkout into the activated environment. `uv pip` is the
fast editable-install path when it is available; plain `pip` is also fine.

```shell
uv pip install -e ".[dev,tomopy]"
# or: pip install -e ".[dev,tomopy]"
python -m pip check
python -m pytest -q
```

To qualify the environment from scratch, remove the named environment first
and repeat those commands. The environment file intentionally excludes docs
and optional algorithms so its dependency solution is not affected by tools a
reconstruction or default unit test does not use.

Bronnikov-aided correction is optional because it executes through hotopy.
Install it only on deployments that use `phase.bac`:

```shell
pip install -e ".[bac]"
```

Do not treat a uv-only virtualenv as the supported deployment path for
TomoPy-heavy reconstruction work. Use it only as pip-compatible glue inside the
conda environment.

Build documentation in a separate virtual environment. The `docs` extra is the
source of truth for documentation tools and is not part of `environment.yml`:

```shell
python -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install -e ".[docs]"
python -m pip check
python -m mkdocs build --strict
```

## Run a pipeline

Pass a pipeline YAML (or a bundled canonical template name) and a Scan YAML to
the runner. Relative projection paths in the Scan are resolved from the Scan
file, and `--output` overrides the pipeline's output directory.

```shell
lftomo-run recon-minimal scan.yaml --output recon/
lftomo-run phase-contrast-canonical scan.yaml --output recon/ --json-output
```

Finite parameter, algorithm, and named structural candidates expand
automatically. Multiple concrete runs are written beneath stable
`candidate-<key-prefix>` directories, and JSON summaries contain a directly
rerunnable resolved `Recon`. See
[`docs/candidate-spaces.md`](docs/candidate-spaces.md) for Stage IDs, typed
domains, conditional algorithm cases, structural variants, and preflight
guards. Use `--gpu` only when the runtime has the CUDA dependencies required
by a selected GPU Stage.

Angle sampling, gridrec defaults, and the optional TomoPy/NumPy compatibility
contract are documented in
[`docs/reconstruction-conventions.md`](docs/reconstruction-conventions.md).
Filtering dtype behavior is recorded in
[`docs/filtering-conventions.md`](docs/filtering-conventions.md), and operators
using DRHT/LDM must follow
[`docs/super-resolution-security.md`](docs/super-resolution-security.md).

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
