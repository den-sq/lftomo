# Multimaterial phase correction

`phase.multimaterial` consumes flat/dark-corrected transmission and returns
corrected attenuation for the usual reconstruction stages. It uses a material
table at the scan's effective energy. It does not import mctutil.

```yaml
kind: phase
algorithm: multimaterial
params:
  mu_table: [60.8, 100.0, 140.0]       # m^-1, material order matters
  delta_table: [9.12e-8, 1.50e-7, 1.20e-7]
  weights: null
  n_iter: 1
  omega: 0.7
  air_ratio: 0.05
  sino_sigma: 1.0
  classify_sigma: 1.5
  label_smooth: 3
  pad: true
  padding: kernel
```

The material and correction parameters follow the supplied single-distance
reference. The stage fits the table **through the origin**, retaining optional
material weights. That
fit is the sole owner of the operative slope and off-line residuals. It does
not accept a second, independently specified fit. `n_iter=0` returns the
slope-form retrieval with the selected padding, while still checking the
reference's final classification. `omega=1` applies a full frozen-label correction in one step;
the default `omega=0.7` applies a deliberately partial correction.

For each iteration, the stage reconstructs each row using the reference's
working Ram-Lak FBP, segments high-confidence material cores with watershed,
and forward-projects the material labels. It filters the synthesized phase
residual and updates `C = (1 - omega) * C + omega * dB`. The corrected data are
always `B_original - C`. The internal FBP keeps the reference's centered,
parallel detector geometry and equal angle weights; it does not replace or
reconfigure the downstream reconstruction algorithm.

`correction_norms` and `label_change` are always published in
`PipelineState.artifacts`, including empty histories when appropriate. The
runner retains them in each `window_results` entry. The first history contains
one RMS correction per update; the second compares successive labels used for
updates and has `max(n_iter - 1, 0)` entries. Both histories cover the entire
padded window buffer, including halo rows; with finite halos, overlapping rows
contribute to each neighboring window's history. Re-segmenting with `n_iter > 1`
has no contraction guarantee and can drift when residuals have opposite signs.
These histories are required evidence for assessing that behavior.
Read them through `RunResult.artifacts["window_results"][i]["artifacts"]` or
the runner's `--json-output`. `recon_run.yaml` records resolved configuration
and window geometry; it does not store these diagnostic histories.

## Consuming a material-table export

`lftomo.config.phase.multimaterial_config` is a one-way adapter for
`mctutil.phase.material_table` version 1. Pass the decoded JSON and an existing
acquisition `Scan`:

```python
from lftomo.config.phase import multimaterial_config

scan, stage_config = multimaterial_config(export_document, acquisition_scan)
assert scan is acquisition_scan
# Insert stage_config after normalization and before reconstruction.
# scan.to_yaml() and stage_config.to_yaml() produce ordinary lftomo config.
```

The supplied Scan remains authoritative: the adapter returns the same object
and checks that its energy matches the export's target `effective_energy_kev`.
The source `calibration.energy_kev` may differ after an energy transfer.
Geometry kinds must match. Parallel geometry checks effective pixel size and
propagation distance; cone geometry checks both native distances and detector
pixel pitch. Scan alone continues to derive the effective cone geometry.
Comparisons convert scan lengths to the export's metres and allow a relative
tolerance of `1e-12` with zero absolute tolerance for unit-conversion roundoff.
Missing scan propagation distance or a mismatch raises a `ValueError` naming
the field and both values. Update acquisition metadata or export a compatible
table explicitly before retrying; the adapter never fills or replaces scan
energy or geometry.

The adapter consumes ordered coefficients and
`calibration.parameters.weights` into the stage configuration. An affine fit
in the export is intentionally replaced by the reference loop's through-origin
fit. The export's spatial affine envelope is not a multimaterial-loop input.
Neither the source object nor the supplied Scan is mutated.

Unsupported schema versions and harvested transfers marked
`extreme_weights=True` are rejected. The latter flag identifies an unreliable
basis representation; exporting the table does not validate it. Harvested
tables retain their experimental scientific status and source-energy
uncertainty limitations. This consumer adds no new calibration guarantee.

## Window, storage, and execution ownership

Every halo row participates in retrieval, classification, and correction on
each pass; downstream reconstruction/writing trims the canonical interior.
Projection angles come from `TomoDataSet.angles` selected by the source window,
including strided acquisition selections. No packed row list is treated as a
new physical grid.

The projection-wide air median and transverse FFT depend on context. Windows
follow the existing fixed-window-plus-padding parity contract and emit the
extent-sensitivity warning; no exact finite-halo equivalence is claimed.

With the default `padding: kernel`, the fitted slope sets the decay length
`L = sqrt(distance_m * slope)`. Retrieval and each correction pass request
`ceil(8 * L / pixel_m)` real source rows per side. The numerical FFT canvas
supplements the realized halo independently on each Y side, edge-filling only
the deficit; X receives the full radius on both sides. Only axes that need
edge fill round up to a fast FFT length. If real halo covers both Y sides,
there is no generated Y padding or Y rounding. The continuous kernel's mass
outside a square with half-width `8L` is bounded by `2 * exp(-8)`, approximately
**0.0671%**. This tail bound does not guarantee output error, exact window
equivalence, or convergence of the multimaterial correction.

Use `padding: reference` for the previous numerical extent,
`next_fast_len(int(1.5 * dimension))`, including its behavior in the presence
of real source halo. This mode preserves the frozen reference results; with
full source context it also preserves exact interior parity with whole-source
processing for the same angle selection. Kernel mode can use different
numerical canvases for those windows because it accounts for their individual
halos. `pad: false` retains periodic FFT behavior in either mode. Source and
numerical padding use the same canonical `TomoWindow` geometry as affine
Paganin retrieval; the legacy Paganin path is unchanged.

The parent coordinates iteration barriers through the run's existing process
executor. Row workers run FBP, segmentation, and material projection; projection
workers run retrieval and correction with one FFT thread. There are no nested
pools. The existing managed `phase_filter` facility prepares the slope
denominator once for the window; both retrieval and correction workers reuse
that same buffer through every iteration. The producer's canonical filter
shape also supplies memory planning. Baseline, current retrieval, correction,
synthesis, phase residual, and (when needed) previous-label planes are managed,
indexed by source/window rows, and retired on success or failure. Their
shape/dtype specifications also feed
memory planning. NumPy/SciPy FFT, gradient, FBP, watershed, and projection return
arrays are per-kernel numerical temporaries, as is the `np.pad` edge-fill
canvas. No reusable managed FFT canvas is allocated, and the stage does not
retain a history of reconstructed volumes.
