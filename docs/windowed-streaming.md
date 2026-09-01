# Windowed streaming contract

Windowing is runner/machine configuration, not candidate configuration. Use
`--window-rows` and `--extra-pad-y` on the runner CLI, or pass `WindowConfig`
to `run_reconstruction()`. Omitting the row limit produces one full-selection
window through the same driver.

Three quantities remain separate in the resolved plan:

1. Each Stage declares an algorithm requirement with
   `StageContract.required_pad`. Paganin uses the canonical `_phase_pad`
   calculation, bilateral filtering uses its kernel radius, and movement uses
   the ceiling of its maximum displacement. Requirements use semantic
   `DataShape` values; `fixed` is symmetric unless `fixed_after` is supplied.
2. `WindowConfig.rows` selects the core rows held by one machine-sized window.
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
padding request as the single replayable streaming geometry.

Source-context padding and generated output padding are distinct. Source
padding is carried as `TomoWindow.pad_before` / `pad_after`, and is realized
against the physical source bounds by `TomoDataSet`. The fused affine stage's
`pad_x` creates a wider published canvas instead; `with_output_padding()`
derives that canvas without pretending it can read pixels outside the source.

`TomoDataSet` is itself the reiterable window plan. Its global `pos`/`shape`
own the selected source bounds, `step` owns the Y window schedule, and
`sample_step` owns sampling within each emitted `TomoWindow`. There is no
parallel selection window or adapter iterator. Every iteration reads
`window.with_pad()` from the physical source and runs the normal managed
Pipeline. TIFF
names use global detector Y; HDF5 and Neuroglancer writers place each interior
block at its output-row offset. Shared storage reserves the largest planned
window capacity and reuses the same named physical roles, so edge-window sizes
do not make memory grow with the window count.

Automatic center finding uses the **replay** policy. For each window, the
driver first executes the actual read-through-center processed prefix in its
own managed buffers, detaches the center artifacts, retires those buffers, and
then replays the prefix for the main reconstruction while replacing the center
Stage with the saved artifact. It never runs center finding on raw input.
Manual centers remain ordinary metadata Stages and need no pre-pass.

A Stage may declare `requires_full_axes=("Y", ...)`. A multi-window plan then
fails before allocation with the Stage and axis named; there is no silent local
substitute. Whole-volume mode remains valid. Retention and prefetch remain
separate work in #180 and #181; this implementation carries no unused
transition model for those future features.

Paganin and BAC are FFT-extent-sensitive. A windowed plan names them in a
warning. Their parity oracle is the same fixed window and padding: first
`examples/b10_recon_current.py`, and then a frozen Tier-2 producer reference
when that producer used the same extent. Windowed-vs-whole FFT equivalence is
not claimed. Reference `metadata.yaml` should record `window_rows`, composed
and requested padding, and the global source/core interval that produced each
comparison.
