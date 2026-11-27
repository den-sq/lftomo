# Windowed streaming contract

Windowing is runner/machine configuration, not candidate configuration. Use
`--window-rows`, `--window-theta`, `--window-columns`, and `--extra-pad-y` on
the runner CLI, or pass `WindowConfig` to `run_reconstruction()`. Omitting all
three size limits produces one full-selection window through the same driver.

`resolve_streaming_segments` groups adjacent Stages by iteration direction.
`StageContract.window_axes` declares allowed directions;
`required_window_axis` fixes the direction and `preferred_window_axis` supplies
a preference. Otherwise the planner uses `WindowConfig.axis` (CLI
`--window-axis`, default `Y`) when allowed, or another allowed direction that
does not split a required full axis. Required/preferred directions must be
among the Stage's allowed axes and cannot contradict `requires_full_axes`.
These directions describe source windows, independently of buffer layout.

Enabled slope `phase.paganin` / `phase.paganin_filter` and BAC / BAC correction
prefer `Theta`. Thus read → normalize → slope Paganin (or BAC) → stripe → recon
runs as Y → Theta → Y sweeps. Theta windows contain the complete selected
detector height and width plus the phase filter's real source context. A row
selection is an output region of interest: slope/affine Paganin and BAC read the
complete physical detector height instead of edge-filling at the selection
boundary. Their sampled FFT kernel has global support; a finite halo based on
the continuum exponential decay does not reproduce the full-detector result.
The canonical window model clips source context at physical detector edges;
internal numerical FFT padding remains separate.

For slope/affine Paganin and BAC, narrowing the output row selection therefore
does not reduce the phase input height. Theta batch size still bounds the
number of complete detector images held at once, and the preceding normalized
intermediate includes all physical rows. Whole-selection runs with an output
ROI likewise retain the complete detector height through phase retrieval.

BAC checks the completed held window before downstream processing. A
non-finite result raises `BACNonFiniteError` with the affected source projection
and acquisition indices; no output pixels are replaced. The `bac_qc` artifact
contains counts for every projection in that window. The runner writes
`bac_qc.json` under the explicit output directory (beside volume-file outputs).
On failure it marks the report incomplete and includes the failing window's
counts; later windows have not run. Without an output destination the same QC
is available as `error.qc`. Borrowed-array callers that omit source geometry
receive explicitly labelled buffer ordinals, not inferred source coordinates.
The existing `clip_input` rule remains unchanged; it is not a new B10 zinger
rule.

In a bounded run, the default Theta batch equals `rows` / `--window-rows`;
`theta` / `--window-theta` overrides it. When only columns are bounded, the
default Theta batch equals the selected detector height. Every batch is capped
by the available projection samples. Matching row and projection batch sizes
gives similar input-buffer sizes when the selected Y and Theta extents are
similar, and lets multiple projections share dispatch overhead. Window sizes
count samples, including when the read selection strides through projection indices.

For example:

```sh
lftomo-run pipeline.yaml scan.yaml --window-rows 64 --window-theta 4 \
  --intermediate-root /scratch/lftomo
```

The pointwise negative-log and beta-scale Stages also support X windows.
Whole-selection execution needs no direction transitions or intermediate files.
It uses the same real phase context as the Theta plan. The lower-level
`resolve_streaming_plan` remains an explicit single-direction API for existing
callers; it does not segment a run.

Three quantities remain separate in the resolved plan:

1. Each Stage declares an algorithm requirement with
   `StageContract.required_pad`. Slope and affine Paganin require all physical
   Y source rows; legacy alpha Paganin uses the canonical `_phase_pad`
   calculation. Bilateral filtering uses its kernel radius, and movement uses
   the ceiling of its maximum displacement. Requirements use semantic
   `DataShape` values; `fixed` is symmetric unless `fixed_after` is supplied.
2. `WindowConfig.rows`, `theta`, and `columns` select core samples per window
   along their respective axes.
3. `WindowConfig.extra_pad` is the optional operator override. It is symmetric
   unless `extra_pad_after` is supplied.

Before and after requirements compose independently by componentwise addition
through the streaming segment.
For example, filters requiring two and three Y rows require five rows when
chained; `max(2, 3)` is insufficient because the second filter consumes the
first filter's neighborhood-dependent output. The requested padding stored on
`TomoDataSet` is the componentwise maximum of each composed side and the
corresponding user padding. The dataset realizes that request against physical
source bounds when it emits each window; the resolver does not calculate or
carry a second "effective" value. The dataset's serialized document in
`recon_run.yaml` records the selection, sampling stride, window schedule, and
padding request as the single replayable streaming geometry for each segment.
Downstream halo coverage propagates backward: an earlier segment materializes
the complete source interval needed by the next segment, including real
neighbors outside the requested output selection. It never replaces missing
intermediate neighbors with edge filling. Operator `extra_pad` applies to each
segment; parity comparisons must use the same resulting phase extent.

Source-context padding and generated output padding are distinct. Source
padding is carried as `TomoWindow.pad_before` / `pad_after`, and is realized
against the physical source bounds by `TomoDataSet`. The fused affine stage's
`pad_x` creates a wider published canvas instead; `with_output_padding()`
derives that canvas without pretending it can read pixels outside the source.

`TomoDataSet` is itself the reiterable window plan. Its global `pos`/`shape`
own the selected source bounds, `step` owns the directional window schedule, and
`sample_step` owns sampling within each emitted `TomoWindow`. There is no
parallel selection window or adapter iterator. Every iteration reads
`window.with_pad()` from the physical source or preceding intermediate and runs the normal managed
Pipeline. TIFF
names use global detector Y; HDF5 and Neuroglancer writers place each interior
block at its output-row offset. Shared storage reserves the largest planned
window capacity and reuses the same named physical roles, so edge-window sizes
do not make memory grow with the window count when buffer reuse is enabled.
Fragment allocation plans include the bounded intermediate input. The runner
replays the segment allocation schedules, accounting for the largest
reservation of each shared role and reuse between windows and sweeps, before
starting the shared-memory manager. Disabling reuse counts each window's
allocations separately, including automatic-center prepasses.

At a direction change, the preceding sweep writes its complete output to an
NPY intermediate through `FileBackedNPArray`. `PipelineRunContext` owns these
files and their mappings; the next sweep copies only its current padded window
into managed shared storage. `TomoWindow.packed_index` translates global source
coordinates into stored sample positions. Acquisition angles and indices stay
on the canonical datasets, including strided or missing projections. All
segments share the same run-owned process executor.

`WindowConfig.intermediate_root` / `--intermediate-root` selects the filesystem
location (system temporary storage by default). Each run, candidate, and rerun
creates a fresh unique subdirectory. Existing files are never overwritten or
used as implicit restart inputs. By default the context removes its directory
on success or failure. The shared-memory manager also removes non-retained
directories when the owner exits abruptly, including SIGKILL, through the same
owner-exit watch that reclaims shared memory. `retain_intermediates=True` /
`--retain-intermediates` keeps files after normal or abrupt exit; files from a
failed run may be incomplete and are not resumable.

Before each candidate starts, the run log reports the resolved root, each
boundary's estimated file bytes, and their coexisting peak. All boundary files
remain on disk until that candidate exits, so this peak is their sum, including
NPY headers and excluding filesystem overhead. Free space is sampled at that
point, including the effect of any retained earlier candidates. Insufficient
space produces an advisory warning; it does not refuse the run. Select a root
with enough capacity for the reported files. The `intermediate_storage` record
in the resolved plan, result artifacts, and `recon_run.yaml` preserves these
estimates and the sampled available bytes.

Successful `recon_run.yaml` manifests also record segment geometry,
intermediate paths, completion, layout, dtype, domain, and retention policy.

Direction boundaries currently require unchanged source geometry and no
window-specific artifacts consumed across the boundary. Plans that would
cross such a boundary fail in preflight. Geometry-changing Stages and center
finding still work within one segment; automatic centers after phase retrieval
replay from that segment's processed intermediate. `phase.multimaterial` keeps
its existing internal mixed-axis execution and is not decomposed into sweeps.
Its internal reconstruction requires the complete selected Theta axis, so the
planner keeps it in a Y sweep even when the default direction is Theta.

Automatic center finding uses the **replay** policy. For each window, the
driver first executes the actual read-through-center processed prefix in its
own managed buffers, detaches the center artifacts, retires those buffers, and
then replays the prefix for the main reconstruction while replacing the center
Stage with the saved artifact. It never runs center finding on raw input.
Manual centers remain ordinary metadata Stages and need no pre-pass.

A Stage may declare `requires_full_axes=("Y", ...)`. The planner chooses a
compatible direction or fails before allocation with the Stage and axis named.
Whole-volume mode remains valid. Prefetch remains separate work in #181.

FFT-extent-sensitive Stages warn only when the plan partitions a declared
sensitive axis. Legacy alpha Paganin in Y windows still uses fixed window and
padding parity: first
`examples/b10_recon_current.py`, and then a frozen Tier-2 producer reference
when that producer used the same extent. Windowed-vs-whole FFT equivalence is
not claimed for that plan. Reference `metadata.yaml` should record window sizes, composed
and requested padding, and the global source/core interval that produced each
comparison.
