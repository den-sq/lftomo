# Filtering conventions

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
