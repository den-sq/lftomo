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
models, an artifact registry, the selected compute backend, and the run's
storage allocator. The initial compute backends are
`NumpyBackend` and `DaskStubBackend`. The Dask stub proves the ABC can carry a
Dask-shaped array without introducing a scheduler dependency.

## Managed working storage

Production runs enter `PipelineRunContext`, which owns one `SharedNPManager`
for the complete run. `SharedNPManager` implements `StorageBackend` directly,
and the handles it returns are `SharedNPArray` objects implementing
`ManagedBuffer` directly; there is no wrapper around the existing shared-memory
objects. Pipeline Stages open the process-local NumPy view with
`working_view(state)`. Plain NumPy arrays remain valid for hermetic Stage tests.

All production Stage allocations go through the manager. A handle sent to a
worker is a non-owning attachment and cannot unlink the segment. Normal and
exceptional context exit unlink all tracked segments. The manager also runs
outside an isolated trial's process group and watches its owner, allowing it to
unlink tracked segments when a timed-out trial is terminated with `SIGKILL`.

The prefix cache's immutable path uses a separate `FileBackedNPArray`, which is
a first-class read-only `ManagedBuffer`. Workers reopen its `.npy` mapping by
path; mutating suffixes continue to receive an independent writable copy.

Read Stages also establish three distinct coordinate records on `PipelineState`:

- `dataset`: the full source `TomoDataSet`, including the complete angle vector;
- `source_window`: the immutable global source selection, position, stride, and
  padding; and
- `buffer_spec`: the current allocation's shape, dtype, and layout.

Layout conversion and reconstruction update `buffer_spec`; they do not rewrite
source coordinates. Output slice numbering therefore uses the source window's
global Y position plus the buffer-local row.

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
