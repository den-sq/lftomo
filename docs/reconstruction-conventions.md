# Reconstruction conventions

## TomoPy dependency scope

Core configuration, orchestration, and non-TomoPy consumers support the
package's ordinary NumPy dependency. TomoPy reconstruction requires the
`tomopy` extra and NumPy 1.x because TomoPy 1.x uses copy semantics that changed
in NumPy 2:

```shell
pip install -e ".[tomopy]"
```

For real reconstruction deployments, prefer `environment.yml`; it obtains the
compiled stack from conda and pins `numpy<2`. Development environments that run
the reconstruction tests should install `.[dev,tomopy]` after creating that
conda environment.

## Projection angles

When no explicit `angles` artifact is supplied, gridrec and center Stages use
`Acquisition.rotation_start_deg` (inclusive) through
`Acquisition.rotation_end_deg` (exclusive), with one uniformly spaced angle per
projection. In other words, they use `numpy.linspace(..., endpoint=False)`.
This matches the canonical producers: a 0–180° scan with 1,800 projections uses
0.1° steps and does not repeat the 180° endpoint. Continuous or irregular scans
can provide the exact one-dimensional radians array as the `angles` artifact.

## Gridrec defaults

`recon.gridrec` deliberately defaults to TomoPy's `shepp` filter and applies a
circle mask with `circle_mask_ratio=1.0`. The filter is explicit because a bare
`tomopy.recon()` default can differ by TomoPy version; comparisons must use the
same filter. Historical ad-hoc producers commonly used a mask ratio of `0.95`,
so parity configurations must record that value rather than relying on the
Stage default.
