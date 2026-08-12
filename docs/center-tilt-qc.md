# Center-vs-row tilt fitting and QC

`center.vo` supports `tilt_fit: polynomial` for estimating the rotation center
at several detector rows and producing a center value for every row in the
current window.  Fits use source detector-row coordinates, including a
window's Y offset and step.  The reported `row_span` is therefore a detector
row span; center coordinates are never mixed into that calculation.

The fit iteratively rejects residual outliers using the median absolute
deviation.  `tilt_outlier_sigma` controls the clipping threshold (default 3.5),
`tilt_max_iterations` bounds clipping (default 5), and `tilt_min_inliers` can
raise the minimum retained sample count.  The `center_fit` artifact includes
the sampled local and source rows, measured centers, coefficients, residuals,
inlier mask, rejected rows, fit iterations, and residual RMS and maximum.

Residual QC is optional.  Set `tilt_residual_cutoff_px` to reject fits whose
inlier residual RMS exceeds the cutoff.  `tilt_qc_policy: reject` raises a
stage error (and permits the configured entropy fallback to run); `force`
retains the fit but marks it `forced`.  Passing fits are marked `accepted`.
The artifact records the policy, status, reasons, and cutoff.

No quantization is applied by default.  If `tilt_quantization_px` is set, each
predicted row center is rounded to the nearest multiple of that pixel value;
the unquantized profile remains in the provenance artifact.
