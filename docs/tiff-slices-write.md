# TIFF slices write contract

The registered `write.tiff_slices` Stage writes each first-axis `(Z, X)` plane
of a reconstruction array to an independent TIFF file below
`Recon.output_path`.

- Output defaults to `float32`; `int16` and `uint16` use the existing
  `lftomo.util.io.np_convert` range conversion.
- Names default to `{sample_id}_{slice:05d}.tif`. `sample_id` resolves from the
  Stage parameter, then `Recon.metadata`, then `Scan.metadata`, with `recon` as
  the final fallback. Templates may include their own `.tif` or `.tiff` suffix.
- `slice_start` offsets indices for windowed reconstruction batches.
- Each worker receives a reopenable reconstruction handle, slice index, path,
  and dtype; no NumPy slice is pickled. There is no shared output file or
  inter-slice lock. Setting `processes` above one submits to the pipeline run's
  persistent process pool; ndarray-only tests remain serial.

Integer conversion follows `np_convert` exactly, preserving compatibility with
the legacy `write_slice` output. Consumers requiring physical-value retention
should keep the default `float32` output.
