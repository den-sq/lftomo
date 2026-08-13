# Pipeline API

The pipeline layer composes tomography reconstruction work as an ordered list
of `Stage` objects. Stage ordering remains sequential; a run-scoped executor
dispatches each Stage's independent locality units without changing that order.

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
objects. At a Stage or state boundary, the single access idiom for the
process-local NumPy view is `np.asarray(state.working_array)`. This is a
no-copy conversion for managed buffers and plain NumPy arrays; tests pin that
contract. Code that deliberately needs a private copy uses
`np.asarray(state.working_array).copy()` rather than
`np.array(state.working_array)`.

Direct `state.working_array.array()` is not the Stage-boundary idiom because
plain NumPy arrays, which remain valid for hermetic Stage tests, do not expose
that method. Low-level worker kernels and storage code that receive a
`ManagedBuffer` handle directly still use its `array()` method to open the
process-local mapping.

All production Stage allocations go through the manager. A handle sent to a
worker is a non-owning attachment and cannot unlink the segment. Normal and
exceptional context exit unlink all tracked segments. The manager also runs
outside an isolated trial's process group and watches its owner, allowing it to
unlink tracked segments when a timed-out trial is terminated with `SIGKILL`.
Because a process-tree reaper may target the detached manager directly, its
`SIGTERM` handler synchronously unlinks every tracked segment before exiting;
the owner-death watch and explicit termination therefore share the same cleanup
operation.

Out-of-place Stages use a two-phase handoff: allocate a prospective managed
output, fill it while the input remains live, publish it as the next
`PipelineState`, and only then retire the old logical role. A failed Stage
therefore leaves the input current. Retirement never unlinks storage; it makes
the physical segment eligible for an explicitly named alias later in the same
run.

Two aliases are recognized rather than inferred generically. Raw staging may
back reconstruction output after normalization has retired it, proved by the
planner's liveness schedule. Projection-order normalized input may back TomoPy
output because the supported TomoPy boundary materializes that input before it
writes `init_recon`; the installed runtime is exercised by the test suite and
the hidden copy remains in the external-engine memory estimate. External
prefix/cache/trial pins prevent either reuse while they are live. Set
`Recon.buffer_aliasing: false` to disable all aliases, or
`Recon.tomopy_input_aliasing: false` to disable only the engine-dependent pair.

The validated `PipelinePlan` exposes each Stage's input/output `BufferSpec`, an
allocation/alias/retirement schedule, and separate peak estimates for shared
memory, per-worker private scratch, and external-engine temporary storage.
Flat/dark calibration arrays and detector-sized phase filters are also
manager-issued artifacts declared by their producer contracts.

## Run-scoped dispatch

`PipelineRunContext` owns one persistent process pool as well as the storage
manager. `STAGE_DISPATCH` classifies every `STAGE_REGISTRY` entry, and the
registry-completeness test makes a new Stage fail until it is assigned process,
thread, serial, or engine dispatch (or a documented exemption). Module-level
worker kernels receive reopenable `ManagedBuffer` handles rather than volume or
slice arrays. Projection kernels mutate disjoint views; bounded sinogram chains
gather projection-order row scratch inside the worker and scatter once; and
layout, fused-affine, helical, and super-resolution fills write manager-owned
prospective outputs before the parent commits the handoff.

I/O remains on I/O-oriented paths. TIFF output submits `(handle, index, path,
dtype)` tasks, while single-file HDF5 and Neuroglancer writers retain their
bounded chunk behavior. TomoPy, ASTRA, torch, and GPU Stages remain serial at
the orchestration level and receive an internal-core budget capped by the run's
worker count. Plain ndarray-only execution is deliberately serial so hermetic
tests never depend on process scheduling.

`Recon.process_workers` selects the pool size; the production default is at
most four workers. `Recon.worker_timeout_seconds` applies a per-task deadline.
A shared/canonical reconstruction YAML should normally leave both unset so it
stays portable across machines; the fields are per-run overrides until a
runner-level machine-profile interface is added.
A timeout kills the worker process tree, waits for mutation to stop, discards
the broken pool, and lazily creates a fresh pool for later work. Context exit
closes the executor before the manager unlinks its buffers.

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

Runner-level streaming is specified in [windowed-streaming.md](windowed-streaming.md).
It resolves Stage pad contracts before allocation, rejects unsatisfied full-axis
requirements, records retained/fresh transitions, and treats a full-volume run
as the one-window case.

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
