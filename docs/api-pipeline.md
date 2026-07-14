# Pipeline API

The pipeline layer composes tomography reconstruction work as an ordered list
of `Stage` objects. Execution is intentionally sequential at the pipeline
level; parallelism stays inside individual Stages.

Each Stage declares a `StageContract` before it runs:

- shape consumed and produced, usually one of `ProjOrder`, `SinoOrder`, or
  `ReconOrder`
- whether it mutates the working array in place
- window axes, if the Stage iterates over a subset of the array
- required scan/recon metadata fields
- artifacts consumed and produced
- cooperative resource hints such as memory multiplier, CPU preference, and GPU
  requirement

`PipelineState` carries the working array, validated `Scan` and `Recon` config
models, an artifact registry, and the selected backend. The initial backends are
`NumpyBackend` and `DaskStubBackend`. The Dask stub proves the ABC can carry a
Dask-shaped array without introducing a scheduler dependency.

## Validation Rules

`Pipeline` validates the ordered Stage list before execution. The validator
checks:

- adjacent Stage shape compatibility through `expected_input` and
  `produced_output`
- required `Scan` metadata fields
- artifact consumers only reference artifacts produced upstream
- GPU-requiring Stages do not run under CPU-only backends
- sweep/fork points do not reuse a shared prefix after an upstream Stage has
  mutated input state

Each failure mode raises a specific validation exception that includes the Stage
index/name and contract field that failed.
