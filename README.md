# lftomo
Large Format Tomography Package

## Installation

Use `environment.yml` as the compiled runtime and default-test contract for
current reconstruction deployments. It pins the corrected native `libtomo`
build from the [TomoPy fork's published channel](https://github.com/den-sq/tomopy/tree/716c0d2b2826e54b99e8041dc7bdf01fefdd9acd/conda),
with the unchanged conda-forge TomoPy 1.15.3 Python package. Other compiled
scientific packages come from conda-forge / bioconda. Supported Python versions
are 3.11 through 3.14.

```shell
CONDA_CHANNEL_PRIORITY=strict conda env create --solver libmamba -f environment.yml
conda activate lftomo
```

The current native package supports Linux x86-64, requires glibc 2.28 or newer,
and retains CUDA 12.9 support. Gridrec itself runs on the CPU. For a CPU-only
host, prefix the environment creation command with `CONDA_OVERRIDE_CUDA=12.9`
to install the same package; GPU algorithms still require a compatible NVIDIA
driver and device. Keep the channel URL and exact `libtomo` build pin in the
environment file so later solves retain the gridrec corrections.

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

Legacy producer references, including the `b10_recon_current` oracle, use a
separate stock environment. Create it and repeat the editable installation above:

```shell
CONDA_CHANNEL_PRIORITY=strict conda env create --solver libmamba -f environment-legacy.yml
conda activate lftomo-legacy
```

Use the current environment for other work. The gridrec correction changes
reconstruction values, so compare frozen producer references in the legacy
environment. The self-hosted `recon-data` runner keeps that stock runtime.

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

New to lftomo? The [Quickstart](docs/quickstart.md) walks a scan from raw
projections to slices; [Configuration](docs/configuration.md) explains the Scan
and pipeline files; [Using lftomo from Python](docs/library.md) covers
`run_reconstruction()`.

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
`@pytest.mark.integration`. Hermetic tests that run the reconstruction runner or
an experiment end to end, compare against a stored reference reconstruction, or
need a real optional engine (hotopy, ASTRA, Torch) are system tests, marked
`@pytest.mark.system`. A test whose cases take 0.5 s or more together belongs
there too.

```shell
pytest -q
pytest -q --run-system
pytest -q --run-integration
```

The default run is the unit suite. `--run-system` adds the system tests; the
full scientific suite below runs them.

`--run-integration` runs lab-local tests on a host with the `recon-data`
capability. On other hosts those tests remain skipped unless
`LFTOMO_RUN_INTEGRATION=1` is set explicitly. The default test run keeps those
checks skipped so a clean checkout can run without lab-local files.

The separate [full scientific suite](.github/workflows/full-suite.yml) is an
informational workflow, not a required pull-request check. It runs the unit and
system suites (`--run-system`) after relevant pushes to `master`, and can also be
dispatched manually. It consumes the current `environment.yml` and selects
Python 3.12 from its supported range. Its job summary records environment setup,
source install, test duration, and installed size; the micromamba setup log records cache and
cold-download behavior, while an artifact retains the resolved package list.
Pull requests continue to use the lightweight checks in
[CI](.github/workflows/ci.yml). Lab-data integration and Tier-2 reference checks
remain separate and opt-in.

Self-hosted runner architecture, service operations, storage prerequisites, and
recovery procedures are documented in
[`docs/recon-data-runner.md`](docs/recon-data-runner.md).
