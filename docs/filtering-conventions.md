# Filtering conventions

## Projection zinger replacement

`filter.zinger` implements the [B10 rule selected in
#374](https://github.com/den-sq/lftomo/issues/374#issuecomment-7222682430).
For each normalized transmission projection, it computes the median `M` of
each pixel's 3×3 neighbourhood, including the pixel itself, and replaces `T`
with `M` only where `T > 1.5 * M`. Equality leaves the pixel unchanged. The
neighbourhood and factor are fixed; `enabled` is the only Stage parameter.

All medians use the original input projection, so replacements are
simultaneous and do not influence neighbouring decisions. The Stage preserves
transmission and handles each projection independently, with no neighbouring
projections or flat-dependent threshold. At physical detector edges it uses
nearest-value extension, equivalent to reflect extension for this 3×3 kernel.
The Stage declares one pixel of source context on each side of Y and X;
`TomoDataSet` and `TomoWindow` resolve that context from real detector pixels
and clip it at physical edges. A selected-row boundary therefore uses its real
neighbouring row rather than treating it as a detector edge.

Opt in after normalization and before phase retrieval. For example, use these
two Stage entries for BAC; the zinger Stage can also precede Paganin:

```yaml
  - kind: filter
    algorithm: zinger
    params: {enabled: true}
  - kind: phase
    algorithm: bac
    params: {alpha: 1.6, gamma: 1.2}
```

The bundled canonical pipelines retain their frozen producer order. The
existing BAC `clip_input` option also retains its AAA523 behavior; enabling
`filter.zinger` does not alter that option. See the [windowed streaming
guide](windowed-streaming.md) for the Theta-first B10 invocation.

## Bilateral-filter dtype behavior

`filter.bilateral` uses OpenCV's `bilateralFilter`, which accepts 8-bit and
32-bit floating-point images but rejects `float64`. The Stage intentionally
converts each projection image to `float32` before filtering.

The filtered values are assigned back into the existing pipeline working array:

- a `float32` input remains `float32`;
- a `float64` input array remains `float64`, but its filtered values were
  calculated at `float32` precision;
- integer input arrays retain their integer storage dtype when results are
  assigned back, so normalization to floating point should precede filtering
  when fractional values matter.

This conversion is deterministic and covered by the bilateral Stage tests. It
is not a promise of full `float64` numerical precision.
