# Windowed streaming contract

Windowing is runner/machine configuration, not candidate configuration. Use
`--window-rows` and `--extra-pad-y` on the runner CLI, or pass `WindowConfig`
to `run_reconstruction()`. Omitting the row limit produces one full-selection
window through the same driver.

Three quantities remain separate in the resolved plan:

1. Each Stage declares an algorithm requirement with
   `StageContract.required_pad`. Paganin uses the canonical `_phase_pad`
   calculation, bilateral filtering uses its kernel radius, and movement uses
   the ceiling of its maximum displacement.
2. `WindowConfig.rows` selects the core rows held by one machine-sized window.
3. `WindowConfig.extra_pad` is the optional operator override, expressed with
   the canonical semantic-axis shapes (`Y`, `Proj`, or `ProjOrder`).

Requirements compose by componentwise addition through the streaming segment.
For example, filters requiring two and three Y rows require five rows when
chained; `max(2, 3)` is insufficient because the second filter consumes the
first filter's neighborhood-dependent output. Effective padding is the
componentwise maximum of that composed requirement and the user padding.
The resolved selection, window size, all three derived pad records, warnings,
and every core/source interval are written to `recon_run.yaml`.

`TomoDataSet` is the source-shape authority and constructs every planned
`TomoWindow`. A window retains the requested logical core and symmetric pad
requirement, then derives the source extent clipped independently at each edge
of the configured read selection. `TomoWindow.internal` is the core expressed
in contiguous buffer-local coordinates, and `allocation_shape` is derived from
the clipped source extent and stride. The plan contains those canonical
windows directly; manifest intervals and retained/fresh transitions are
records derived from them, not another geometry model.

Every iteration reads its canonical source extent, runs the normal managed
Pipeline, and exposes only `TomoWindow.internal` reconstruction rows to writers. TIFF
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
substitute. Whole-volume mode remains valid. Each logical window transition
also records adjacency, retained and fresh global intervals, and a
`prefix_valid` marker. The marker is always false in this issue: retention and
prefetch consume this seam in #180 and #181 but are not implemented here.

Paganin and BAC are FFT-extent-sensitive. A windowed plan names them in a
warning. Their parity oracle is the same fixed window and padding: first
`examples/b10_recon_current.py`, and then a frozen Tier-2 producer reference
when that producer used the same extent. Windowed-vs-whole FFT equivalence is
not claimed. Reference `metadata.yaml` should record `window_rows`, composed
and effective padding, and the global source/core interval that produced each
comparison.
