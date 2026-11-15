# Agent instructions

These instructions apply to the entire `lftomo` repository. Keep this file
concise: exact behavior belongs in the referenced source, documentation, and
regression tests rather than in a second specification here.

## Before implementation

- Reuse before reinventing. Inspect the repository for the owning abstraction,
  helper, or contract before adding one. Ask: "Am I adding a second way to do
  what a named canonical facility already does?"
- Surface a load-bearing choice before implementing it when the issue does not
  determine correctness, ownership or lifecycle, concurrency, resource
  behavior, persistence, compatibility, or an operator-visible contract.
  State the alternatives, existing seam inspected, proposed choice, and
  consequences. Stop for explicit approval if the choice changes scope or
  introduces an overlapping mechanism.
- Make the smallest cohesive change that composes with, and where necessary
  extends, the owning seam. A smaller local diff is not preferable when it
  creates a parallel architecture.
- Treat a new warning caused by the change as a finding: fix it or cite and
  explain it in the implementation record.

## Canonical ownership

### Window and padding geometry

- `lftomo.util.window.TomoDataSet` and `TomoWindow` own source shape, window and
  interior extents, padding extents, allocation shape, and coordinate
  translation. Use `DataRef.with_pad`, `rescope`, and `TomoWindow.internal`
  where their contracts apply.
- An algorithm may use a numerical padding primitive to fill a buffer, but it
  must consume geometry derived from and expressed by the canonical window
  model. It must not independently decide or represent that geometry.
- Source-context requirements use semantic `DataShape` values and resolve into
  `TomoWindow.pad_before` / `pad_after`; generated affine canvas padding uses
  `TomoWindow.with_output_padding`. The removed `AxisPad` and `WindowRegion`
  types must not be restored. Do not add another padding model. Consult
  `docs/windowed-streaming.md` and `tests/util/test_window.py`.

### Managed storage and work planes

- `lftomo.storage.TypedBuffer` owns bound buffer shape, dtype, and layout.
  Execution reads those facts from the buffer; `LocalArray` adapts borrowed
  NumPy or Dask arrays at input. `BufferSpec` is only an unbound allocation or
  serialization descriptor, never a second geometry record on runtime state
  or worker payloads. Deserialize it into a typed buffer once, and derive it
  from a typed buffer only at a planning or serialization boundary.
- `lftomo.storage.ManagedBuffer`, `StorageBackend`, and `PipelineRunContext`,
  implemented for shared memory by `lftomo.util.mem.SharedNPArray` and
  `SharedNPManager`, own production storage and lifecycle.
- Route owned or reusable work planes and material scratch through that seam.
  Bounded Stage work is declared through `Stage.bounded_work_requests` using
  `BufferSpec`; the generic execution lifecycle shares requests across the
  segment, allocates them, derives views from assigned buffer ranges and
  retires them. Workers retain the held window and full buffer; their assigned
  `DataRange` owns only scheduling, and `TomoWindow` translates sample ordinals
  to source coordinates. Keep
  algorithm-specific dtype, capacity, padding and initialization policy in the
  requesting Stage, not the executor. Temporary work is separate from published
  `managed_artifacts` and prefix-cache state.
- Declare unavoidable output temporaries created by external numerical
  libraries. Do not promise zero allocation when the external API cannot
  provide it.
- Consult `docs/api-pipeline.md`, `tests/pipeline/test_managed_storage.py`, and
  `tests/pipeline/test_buffer_planning.py`.

### Parallel dispatch

- In the order-aware Stage and Vo execution path, preserve the run-scoped
  process executor and gather-once / run-chain / scatter-once design in
  `lftomo.pipeline.execution` and `lftomo.pipeline.dispatch`.
- Do not create nested pools. Keep reusable scratch indexed by source/window
  row and independent of worker count.
- This does not prohibit separately justified bounded concurrency for I/O or
  event persistence. For CPU-heavy work, explicitly account for the supported
  Python versions, the GIL, external-engine threading, and process ownership;
  threads are not a silent default.
- Keep `lftomo.pipeline.stages.STAGE_DISPATCH` complete. Consult
  `docs/order-aware-layout.md`, `docs/stripe-chain-regression.md`,
  `tests/pipeline/test_process_dispatch.py`, and
  `tests/pipeline/test_bounded_segments.py`.

### Logging and progress events

- `lftomo.util.log.Logger`/`LOG` are the human and operator logging seam.
  `lftomo.experiment.events.ProgressEvent` and `ProgressEventWriter` are the
  structured durable progress seam. They are complementary, not
  interchangeable.
- Do not add direct timestamp formatting, implicit CWD log files,
  `print`-based operational logging, or a private progress stream. Presentation
  belongs in `hatmehyt`; lftomo owns the structured event and logging contract.
- Consult `tests/util/test_log.py` and `tests/experiment/test_lifecycle.py`.

### Acquisition and pipeline contracts

- Preserve projection provenance through `lftomo.angles`, the acquisition
  models in `lftomo.config.models`, and `TomoDataSet.angles` and
  `acquisition_indices`. The authoritative semantics and regression coverage
  are in `docs/irregular-projection-angles.md`, `tests/test_angles.py`, and
  `tests/test_tier2_irregular_angles.py`.
- Extend the public `Stage`, `StageContract`, and `PipelineState` contracts in
  `lftomo.pipeline.core` and the public configuration models in
  `lftomo.config.models`; do not fork them into an algorithm-local lifecycle or
  configuration path. Consult `docs/api-pipeline.md` and
  `tests/pipeline/test_core.py`.

## Verification and review record

- Run the smallest relevant test selection while iterating, then the supported
  non-Tier-2 suite when the change is broad enough to warrant it. Do not run
  data-dependent Tier-2 or acceptance work unless the task calls for it and
  the required external data is available.
- For architecture-sensitive behavior, prove the canonical seam was exercised;
  output equivalence alone is insufficient. The proof is PR evidence, not a
  new test: state the seams the change uses, the existing seams searched, and
  paste the evidence that each was exercised (an allocation report, lifecycle
  events, a dispatch trace, or a public-adapter identity check) in the PR
  template's `Shared-basis audit` and `Verification` sections.
- Permanent seam enforcement is one repository-wide gate per seam, such as
  `tests/pipeline/test_window_geometry_ownership.py`. A change that introduces
  a seam adds its gate; a change that extends a seam extends the existing gate
  if it does not already cover the extension. Do not add tests that only
  assert how one change is wired to a seam.

## Code Review Rules

- Every implementation PR must complete the repository PR template's
  `Shared-basis audit` and retain any load-bearing decisions that were surfaced
  before implementation.
- Treat an undeclared overlapping mechanism as a defect even when output tests
  pass. Require the PR's seam evidence when the change concerns storage,
  windowing, logging, lifecycle, dispatch, or another canonical ownership seam.
- Treat a new seam without a repository-wide gate, or a test that only asserts
  one change's wiring to a seam, as a finding.
