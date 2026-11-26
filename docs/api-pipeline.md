# Pipeline API

The pipeline layer composes tomography reconstruction work as an ordered list
of `Stage` objects. Stage ordering remains sequential; a run-scoped executor
dispatches each Stage's independent locality units without changing that order.

For a worked implementation, registration edits, and a runnable test, see
[Adding a Stage](adding-a-stage.md).

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

## Stage authoring and parameters

`Stage` owns construction: its nested `Params(StageParams)` model validates
configuration into immutable `stage.params`, then `contract_for(cls, params)`
resolves the instance contract. `StageParams` rejects extra keys. Scalar bounds
and choices declared with Pydantic fields appear in
`StageClass.Params.model_json_schema()`. `StageConfig` checks supplied values
against that model while allowing required inputs to remain unresolved in
pipeline templates; Stage construction requires all inputs. Serialized config
mappings remain at the input boundary. Runtime code reads the typed model.

Use a class-level `StageContract` for invariant facts. The default `contract_for`
returns it; parameter-dependent implementations return
`dataclasses.replace(cls.contract, ...)`. Runtime dependency preflight reads
class-level `runtime_requirements`; planning and execution read the resolved
instance contract. Parameters are not mirrored as individual Stage attributes.

`SinogramStage` supplies the assigned Y loop for `run_sinogram(sinogram, state)`;
`ProjectionStage` supplies the assigned Theta loop for
`run_projection(image, state)`. Both adapters pass a numerical row or image view
and accept in-place mutation or returned replacement values. The accompanying
state retains the full working buffer and held source window; its `work_range`
identifies the current invocation's assigned sample. Plain `Stage` remains
available for orchestration, managed output
handoffs, and external engines. There is no separate locality enum: bounded
planning uses the row-kernel capability and contract, while `STAGE_DISPATCH`
selects the executor.

`PipelineState.work_range` is a `DataRange` in packed buffer sample ordinals,
independent of source coordinates and storage layout. `None` assigns the full
buffer. Use `state.work_indices("Theta")` or `state.work_indices("Y")` for work
loops, and `state.work_index()` for vectorized operations over the assignment.
`state.work_index(other_shape)` expresses the same assignment in another
semantic layout. The full array's shape describes storage, not the work extent
of a dispatched invocation.

For source coordinates, map each assigned ordinal through
`state.source_window.source_index(axis, index)`. This accounts for the held
window's source start, realized halo, and sampling stride. Source Theta
positions `[2, 5, 8]`, for example, occupy packed buffer ordinals `[0, 1, 2]`.
`TomoWindow.internal` retains source stride and must not be used as a packed
buffer index. Row/image views at numerical boundaries do not create child
windows or change the held window's identity.

## Calibrated Paganin material lines

`phase.paganin` and `phase.paganin_filter` accept an optional `slope` in metres
for the line `delta = material_alpha + slope * mu`. Their existing `alpha`
parameter remains the historical delta/beta filter coefficient; it is not the
material-line intercept. Omitting `slope` preserves the existing arithmetic,
FFT extent, and defaults exactly.

```yaml
- kind: phase
  algorithm: paganin
  params:
    slope: 1.0e-9
    material_alpha: 2.0e-8
    envelope_m: 0.0003
    fft_pad: true
```

Nonzero `material_alpha` requires both `slope` and `envelope_m`. The envelope is
total spatial thickness in metres, supplied as a scalar, detector Y/X array,
or Theta/Y/X array broadcastable over the full source. The canonical source
window selects its rows and projection indices, including strided selections.
Array envelopes are normalized once into a manager-owned window buffer before
dispatch. Numerical kernels receive views selected by their assignment; the
serialized envelope is removed from worker configuration carriers. Original configuration
remains unchanged for replay.
The affine operation performs one approximate defect-correction step; it does
not make the affine inversion exact. These plain values can be copied from the
mctutil calibration export without importing that package.

For the slope path, `fft_pad: true` (the default) edge-fills each numerical FFT
canvas to `next_fast_len(int(1.5 * dimension))`. `TomoWindow` owns this canvas,
including asymmetric padding and interior crops. This generated numerical
padding is separate from the source halo described in
[windowed-streaming.md](windowed-streaming.md). The FFT remains extent-sensitive;
arbitrary finite halos do not establish exact whole-volume equivalence.
`phase.paganin_filter` returns filtered transmission. `phase.paganin` applies
the terminal negative log with a slope-path default `epsilon: 1.0e-6`; an
explicit `epsilon` retains its existing clipping policy. The legacy path keeps
`epsilon: 1.0e-8`.

`lftomo.prep.phase.retrieve_single` exposes the same slope/affine calculation
for composite algorithms. Its default numerical dtype is float32, matching the
calibration reference. The shared phase filter/denominator is manager-owned
and prepared once before projection dispatch. FFT, edge-fill and gradient results are
temporary arrays created by the numerical libraries. SciPy FFT uses one worker
inside each projection task; the existing run executor owns parallelism.

## Managed working storage

Production runs enter `PipelineRunContext`, which owns one `SharedNPManager`
for the complete run. `SharedNPManager` implements `StorageBackend` directly,
and the handles it returns are `SharedNPArray` objects implementing
`ManagedBuffer` directly; there is no wrapper around the existing shared-memory
objects. All nonempty working buffers implement `TypedBuffer`, which owns their
semantic `DataShape`, dtype, and layout. `ManagedBuffer` adds the reopenable
storage contract. Borrowed NumPy or Dask inputs use `LocalArray(array, shape)`;
they remain local and execute serially. At a Stage or state boundary, the single access idiom for the
process-local NumPy view is `np.asarray(state.working_array)`. This is a
no-copy conversion for managed buffers and borrowed NumPy arrays; tests pin that
contract. Code that deliberately needs a private copy uses
`np.asarray(state.working_array).copy()` rather than
`np.array(state.working_array)`.

The Stage boundary uses the NumPy conversion protocol uniformly. Low-level
worker kernels and storage code that receive a
`ManagedBuffer` handle directly still use its `array()` method to open the
process-local mapping.

All production Stage allocations go through the manager. A handle sent to a
worker is a non-owning attachment and cannot unlink the segment. Normal and
exceptional context exit unlink all tracked segments. The manager also runs
outside an isolated trial's process group and watches its creating process
through `multiprocessing.parent_process().join()`. This follows the owner even
when the manager's OS parent is a forkserver. If the owner dies, including from
`SIGBUS` or `SIGKILL`, the manager unlinks its segments and exits. Pool workers
use the same owner watch to exit without waiting for tasks or shutdown messages
from the dead owner. Closing these helpers also releases inherited output pipes.
Because a process-tree reaper may target the detached manager directly, its
`SIGTERM` handler synchronously unlinks every tracked segment before exiting;
the owner-death watch and explicit termination therefore share the same cleanup
operation. After a fatal owner exit, Python's resource tracker may report and
remove abandoned pool semaphores whose owner-side finalizers could not run.
Normal context shutdown still closes the executor before the manager.

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
the hidden copy is outside the shared-memory allocation schedule. External
prefix/cache/trial pins prevent either reuse while they are live. Set
`Recon.buffer_aliasing: false` to disable all aliases, or
`Recon.tomopy_input_aliasing: false` to disable only the engine-dependent pair.

The validated `PipelinePlan` exposes each Stage's input/output `BufferSpec` and
an allocation/alias/retirement schedule. These are unbound allocation
descriptors; runtime Stages read shape, dtype, and layout from the current
typed buffer instead of retaining a matching spec on state. Its `shared_memory_peak_bytes` sums
physical allocations in that schedule; aliases add no storage, and retired
segments remain allocated until run-context shutdown. This estimate excludes
allocations absent from the schedule, worker-private scratch and external-engine
temporaries.
Flat/dark calibration arrays and detector-sized phase filters are also
manager-issued artifacts declared by their producer contracts.

Bounded segment and composite scratch is declared by `Stage.bounded_work_requests` and
resolved by `collect_bounded_work` before allocation. `BufferSpec` owns each
request's shape and dtype; same-key requests share one compatible allocation
for the complete segment. Runtime collection derives its input descriptor with
`BufferSpec.from_buffer` at this planning boundary and discards it after
allocation. Generic execution owns allocation, initialization,
assignment-based views and retirement through the existing storage backend.
The `bounded_work_allocations` artifact records each resolved role, shape,
dtype and byte count, with `indexing="buffer_sample_ordinals"`. Vo's former
`vo_scratch_allocations` report is replaced
by this algorithm-independent report; benchmark result fields keep their
existing meaning. Multimaterial planes, including the reconstruction-layout
previous-label buffer, use this same hook and lifecycle; their initialization
and memory plan derive from these requests. Direct NumPy composite execution
allocates only requests marked `required_for_fallback` through `allocate_output`
and removes temporary keys
on completion or failure. Mixed-axis composites report no single unit axis.

Requests are inspectable without execution. Non-sinogram Stage requests are
included in the predictive allocation schedule. Bounded sinogram scratch is
still omitted from the shared-memory peak; [#360](https://github.com/den-sq/lftomo/issues/360)
tracks accounting for it once per segment, including standalone sinogram dispatch.

Before starting a run's manager or reading projection pixels, the runner checks
the largest padded windows' planned shared-memory peak against available
`/dev/shm` space on Linux. For directional sweeps it replays the allocation
schedules across windows and segments, including intermediate reloads,
per-role reservation maxima, and retirement between windows. Planning and the
manager share the same alias-selection policy. Those same plans supply the
run's buffer reservations. See [windowed streaming](windowed-streaming.md) for
direction selection and intermediate storage lifecycle.
Insufficient capacity raises an error naming both byte counts and suggesting
smaller `--window-rows` or a larger `/dev/shm` (`--shm-size` in Docker). This is
a capacity snapshot, not a reservation against concurrent allocations. Platforms
without Linux's `/dev/shm` convention keep their existing allocation behavior.

## Run-scoped dispatch

`PipelineRunContext` owns one persistent process pool as well as the storage
manager. `STAGE_DISPATCH` classifies every `STAGE_REGISTRY` entry, and the
registry-completeness test makes a new Stage fail until it is assigned process,
thread, serial, or engine dispatch (or a documented exemption). Projection,
bounded-sinogram, affine-fill, and multimaterial workers receive full reopenable
`ManagedBuffer` handles, the held source window, and an assigned `DataRange`.
Projection kernels mutate only their
assigned views; bounded sinogram chains
gather projection-order row scratch inside the worker and scatter once; and
layout, fused-affine, helical, and super-resolution fills write manager-owned
prospective outputs before the parent commits the handoff.

I/O remains on I/O-oriented paths. TIFF output submits `(handle, index, path,
dtype)` tasks, while single-file HDF5 and Neuroglancer writers retain their
bounded chunk behavior. Engine-dispatched Stages, including TomoPy, ASTRA and
Torch super-resolution, remain serial at the orchestration level and receive
an internal-core budget capped by the run's worker count. Plain ndarray-only
execution is deliberately serial so hermetic tests never depend on process
scheduling.

BAC runs in projection workers and declares
`ResourceHints(cpu_thread_pool="torch")`. When creating the pool, the executor
sets a per-worker native-thread ceiling of
`max(1, available_cpus // process_workers)`. Before an enabled BAC kernel runs,
the worker lowers Torch's intra-op thread count to that ceiling if necessary.
It preserves lower settings, including explicit `OMP_NUM_THREADS` or
`MKL_NUM_THREADS` limits, and keeps the cap for that worker's lifetime. Torch
loads only when a Stage declares that pool; disabled BAC declares none.
Parent/serial Torch settings and other native libraries are unaffected. This
limits each pipeline independently; concurrent experiment candidates retain
their separate worker budgets.

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
Serialized shape and dtype are consumed when restoring that typed buffer,
rather than retained as another runtime geometry record.

Read Stages also establish distinct geometry records on `PipelineState`:

- `dataset`: the full source `TomoDataSet`, including the complete angle vector;
- `source_window`: the immutable global source selection, position, stride, and
  padding; and
- `working_array.shape` and `working_array.dtype`: the current buffer's
  allocation geometry and dtype; the shape type supplies its layout.

Layout conversion and reconstruction publish a new typed buffer; they do not rewrite
source coordinates. Output slice numbering therefore uses the source window's
global Y position plus the buffer-local row. The invocation's `work_range`
records scheduling only and does not duplicate these geometry records.

Runner-level streaming is specified in [windowed-streaming.md](windowed-streaming.md).
It resolves Stage pad contracts before allocation, rejects unsatisfied full-axis
requirements, derives manifest window records from `TomoDataSet`, and treats a
full-volume run as the one-window case.

## Validation Rules

`Pipeline` validates the ordered Stage list before execution. The validator
checks:

- adjacent Stage shape compatibility through `accepts` and `produces`
- required `Scan` metadata fields
- artifact consumers only reference artifacts produced upstream
- GPU-requiring Stages do not run under CPU-only backends
- sweep/fork points do not reuse a shared prefix after an upstream Stage has
  mutated input state

Each failure mode raises a specific validation exception that includes the Stage
index/name and contract field that failed.
