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
uv pip install -e ".[dev]"
# or: pip install -e ".[dev]"
```

Do not treat a uv-only virtualenv as the supported deployment path for
TomoPy-heavy reconstruction work. Use it only as pip-compatible glue inside the
conda environment.

## Tests

Unit tests should be hermetic: generate any required files under `tmp_path` or
load static fixtures from `tests/`. Tests that require local beamline data,
machine-mounted paths, or hardware are integration tests and must be marked with
`@pytest.mark.integration`.

```shell
pytest -q
pytest -q --run-integration
```

`--run-integration` is intended for Yarrkha or another machine with the expected
local data mounts. The default test run keeps those checks skipped so a clean
checkout can run without lab-local files.
