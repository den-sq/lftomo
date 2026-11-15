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
```

The parameters mirror the supplied single-distance reference. The stage fits
the table **through the origin**, retaining optional material weights. That
fit is the sole owner of the operative slope and off-line residuals. It does
not accept a second, independently specified fit. `n_iter=0` returns the
original slope-form retrieval, while still checking the reference's final
classification. `omega=1` applies a full frozen-label correction in one step;
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
updates and has `max(n_iter - 1, 0)` entries. Re-segmenting with `n_iter > 1` has
no contraction guarantee and can drift when residuals have opposite signs.
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
# Insert stage_config after normalization and before reconstruction.
# scan.to_yaml() and stage_config.to_yaml() produce ordinary lftomo config.
```

The adapter applies the export's energy and native geometry to the returned
Scan and preserves the remaining acquisition metadata. Parallel inputs convert
metres to the existing configuration units. Cone inputs go into Scan's native
cone geometry and Detector's native pixel pitch; Scan alone derives effective
parallel geometry. The adapter consumes ordered coefficients and
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

The projection-wide air median and transverse FFT depend on context. Full
source context gives the same result for each window's interior as processing
the whole source with the same angle selection. Smaller windows follow the
existing fixed-window-plus-padding parity contract and emit the existing
extent-sensitivity warning; no exact finite-halo equivalence is claimed.
The planner composes a phase-padding requirement for retrieval and each
correction pass. Numerical FFT edge padding separately uses the shared
`TomoWindow` canvas from affine Paganin retrieval.

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
arrays are per-kernel numerical temporaries; the stage does not retain a
history of reconstructed volumes.
