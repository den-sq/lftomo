# Using lftomo from Python

`lftomo-run` is a thin wrapper around one function,
`lftomo.runner.run_reconstruction()`. Calling it from a script runs exactly the
same validation, calibration resolution, streaming and output as the command
line, and returns a summary of each run.

## A complete script

```python
from pathlib import Path

from lftomo.runner import run_reconstruction


def main() -> None:
	results = run_reconstruction(
		"recon.yaml",        # pipeline file, or a bundled template name such as "b10-recon"
		"scan.yaml",
		output_path="recon",  # optional; overrides Recon.output_path
	)
	for result in results:
		print(result.index, result.candidate_key[:16], result.output_path)
		print("  parameters:", result.parameters)
		print("  run record:", result.artifacts.get("recon_run"))
		print("  slices:", sorted(p.name for p in Path(result.output_path).glob("*.tif")))


if __name__ == "__main__":
	main()
```

Keep the `if __name__ == "__main__":` guard. The runner executes Stages in a
pool of worker processes started with the `spawn` method, and each worker
imports the main script. Without the guard, every worker starts its own
reconstruction and the run fails with broken worker connections.

Paths behave as on the command line: `scan.yaml`'s projection path is resolved
from the Scan file's directory; a normalize Stage's `calibration_manifest` and
legacy gain/dark paths from the pipeline file's directory; `output_path` and
any other relative Stage path from the working directory (see
[Configuration](configuration.md#paths-inside-the-pipeline-file)).

## Return value

`run_reconstruction()` returns `list[RunResult]`, one entry per concrete run.
A plain pipeline gives one run; a `Recon` with [candidate spaces](candidate-spaces.md)
gives one per candidate. Each `RunResult` has:

| Field | Meaning |
|---|---|
| `index` | Run number, in expansion order |
| `candidate_key` | Stable key identifying the concrete candidate; multi-candidate output goes to `candidate-<key>` beneath `output_path` |
| `parameters` | The swept parameter values chosen for this run (empty for a plain pipeline) |
| `output_path` | Directory the slices were written to |
| `resolved_recon` | The resolved `Recon` as a dict, with every `Recon` field filled in. `Recon.model_validate()` accepts it and its `to_yaml()` output reruns the same pipeline. Window settings are not part of it: pass the same `window_config` again to reproduce the run (its `metadata.streaming_dataset` records the plan that was used) |
| `artifacts` | Run artifacts, including `output_files`, `center` (the centers used), `calibration_resolution` (which flat/dark source was used), `recon_run` (path of the run record), `window_results`, and shared-memory accounting (`measured_shared_memory_bytes`, `allocation_events`) |

`lftomo-run --json-output` prints the same fields as JSON.

## Optional controls

```python
from lftomo.pipeline import WindowConfig
from lftomo.runner import run_reconstruction
from lftomo.util.shapes import Y

results = run_reconstruction(
	"recon.yaml", "scan.yaml", "recon",
	window_config=WindowConfig(rows=32, extra_pad=Y(0)),
)
```

(Called from a script, this belongs inside the guarded `main()` above.)

- `window_config`: stream the selected detector rows through windows of
  `rows` rows (the `--window-rows` option), with optional extra row padding.
  This bounds memory for large volumes. See [Windowed streaming](windowed-streaming.md).
- `calibration_bundle` / `verify_calibration_hashes`: pass a
  `lftomo.calibration.CalibrationBundleRef` (manifest path and calibration ID)
  to use a validated flat/dark bundle instead of fields in the scan or pipeline.
  See [Calibration bundles](calibration-bundles.md).
- `backend`: the compute backend; the default `NumpyBackend()` is what the CLI
  uses (`--gpu` passes a backend that declares CUDA support).
- `stage_observer`, `execution_observer`: callbacks for Stage start/finish and
  execution spans, for progress reporting.
- `state_executor`: used by the experiment subsystem; not needed for ordinary
  runs.

## Lower-level use

The runner composes public pieces that can also be used directly:

- `lftomo.runner.load_recon()` / `load_scan()` load and validate the two files
  (see [Configuration](configuration.md#writing-yaml-from-python)).
- `lftomo.pipeline.stages.create_stage()` builds one Stage from a `StageConfig`,
  and `lftomo.pipeline.Pipeline` runs a list of Stages on a `PipelineState`.
  This is useful for applying a single Stage to an in-memory array; the Stage
  unit tests in `tests/pipeline/` are compact examples. Contracts, managed
  storage and dispatch are described in the [Pipeline API](api-pipeline.md).

Other examples in the repository:

- [`examples/b10_recon_current.py`](https://github.com/den-sq/lftomo/blob/master/examples/b10_recon_current.py)
  ([usage](https://github.com/den-sq/lftomo/blob/master/examples/README.md)): a
  B10 driver built on the lower-level utility API rather than the runner.
- [`notebook/recon.ipynb`](https://github.com/den-sq/lftomo/blob/master/notebook/recon.ipynb)
  and [`notebook/recon_b10.ipynb`](https://github.com/den-sq/lftomo/blob/master/notebook/recon_b10.ipynb).
- [Experiment operations](experiment-operations.md): searching and promoting
  pipeline parameters with `lftomo-experiment`.
