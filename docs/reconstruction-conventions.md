# Reconstruction conventions

## TomoPy dependency scope

Core configuration, orchestration, and non-TomoPy consumers support the
package's ordinary NumPy dependency. TomoPy reconstruction requires TomoPy
1.15.3 or newer because that release supports NumPy 2 copy semantics:

```shell
pip install -e ".[tomopy]"
```

For real reconstruction deployments, prefer `environment.yml`; it obtains the
compatible NumPy/TomoPy compiled stack from conda. Development environments
that run the reconstruction tests should install `.[dev,tomopy]` after creating
that conda environment.

## Physical voxel geometry

The reconstruction lattice uses detector-index coordinates. Its physical pixel
pitch is `Scan.pixel_size_um`: the supplied sample pitch for parallel scans, or
the detector pitch divided by the Fresnel magnification for native cone scans.
The same geometry supplies the effective distance and pixel pitch used for
phase filtering and its padding. Reconstruction algorithms and their amplitude
conventions are unchanged; see [cone-beam configuration](configuration.md#cone-beam-fresnel-geometry).

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
