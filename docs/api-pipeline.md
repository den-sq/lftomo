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
