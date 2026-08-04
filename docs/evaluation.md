# Reconstruction evaluation

`evaluate.metrics` is a preserving `ReconOrder` / `RECON` Stage. It reads
configured reconstruction slices, emits a versioned JSON-compatible `metrics`
artifact, and leaves array values, layout, and domain unchanged.

```yaml
- id: quality
  kind: evaluate
  algorithm: metrics
  params:
    protocol_version: "1"
    windows:
      - {id: top, slice_start: 0, slice_stop: 4, weight: 1}
      - {id: bottom, slice_start: 28, slice_stop: 32, weight: 1}
    metrics:
      - {kind: reference, reference_artifact: evaluation_reference}
      - {kind: validity, clip_min: -0.1, clip_max: 3.0}
      - {kind: ring, smoothing_sigma: 2, inner_radius: 4}
      - kind: roi_noise_cnr
        background: {y_start: 0, y_stop: 32, x_start: 0, x_stop: 32}
        signal: {y_start: 64, y_stop: 96, x_start: 64, x_stop: 96}
    aggregation:
      method: weighted_mean
      invalid: fail
      window_weights: {top: 1, bottom: 1}
    preview:
      scale_min: -0.1
      scale_max: 3.0
      crop: {y_start: 0, y_stop: 128, x_start: 0, x_stop: 128}
      blinded: true
```

Reference arrays are supplied explicitly in the named Pipeline artifact and
must have the complete reconstruction shape. Background and signal ROIs are
also explicit; the Stage never guesses a region when one is missing or
malformed.

## Protocol-1 metric set

The registry is intentionally small:

- `reference.nrmse`, `reference.correlation`, and
  `reference.normalized_bias` reproduce the existing Tier-2 comparison
  formulas, including reference-dynamic-range normalization.
- `validity.nonfinite_count` is a hard-validity value and
  `validity.clipped_fraction` measures saturation at explicit bounds.
- `artifact.ring_score` computes the RMS high-frequency residual of the
  centered radial mean profile, normalized by image dynamic range. Synthetic
  concentric-ring fixtures establish its monotonic response. Its default center
  is the geometric image center; an explicit center may be supplied.
- `roi.background_noise` is the configured background standard deviation and
  `roi.cnr` is absolute signal/background mean contrast divided by that
  standard deviation.

Each registry definition declares direction, units, normalization, required
inputs, default aggregation, undefined behavior, protocol version, slice or
volume resource scope, and a memory multiplier. Kernels receive one 2-D slice
at a time; the Stage records the maximum kernel shape and never copies a full
reconstruction volume. Reference metrics need additional slice-sized float64
work arrays, reflected by the Stage's 4x slice-memory hint.

Every raw per-slice/window observation is retained. Aggregates record method,
invalid policy, observation/sample/invalid counts, units, direction,
normalization, and protocol version. Invalid values either fail the aggregate,
are explicitly excluded, or use a configured finite worst value. They are
never silently converted to zero.

## Ranking

`pareto_rank` is a pure deterministic non-dominated-sort utility. It honors
per-objective minimize/maximize direction, applies hard constraints first,
keeps ties and all non-dominated candidates, and gives missing/non-finite
objective values a non-winning invalid rank. Output ordering is by candidate
key, independent of trial completion order.

An optional affine-normalized weighted score is available only when its terms,
directions, offsets, scales, and missing-value behavior are explicitly
configured. It does not replace or mutate raw metrics or Pareto results.

Entropy and generic sharpness are not protocol-1 universal objectives: both
can reward noise, ringing, or over-enhancement and can disagree with expert or
downstream-task judgment. They may be added later only with a versioned,
validated scientific protocol.
