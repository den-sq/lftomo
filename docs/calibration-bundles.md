# Calibration bundle core

Calibration bundles persist flat/dark and beam-model artifacts independently of
one reconstruction. A bundle is a directory containing a versioned
`calibration.yaml` and external array payloads. The manifest records component
hashes, coordinate frames, detector coverage, provenance, compatibility facts,
and validation results; it never embeds full detector arrays.

This core deliberately separates three responsibilities:

- `lftomo.calibration` defines manifests, compatibility, validation, and bounded
  readers.
- The runner resolves which source is selected and records overrides.
- Builder APIs and scientific calibration algorithms create payloads.

## Component validity

Compatibility is evaluated per component rather than through one global
"similar scan" decision.

| Component | Required validity scope |
| --- | --- |
| `detector_static` | detector, optics, detector ROI/orientation, static-artifact epoch |
| `dark` | detector, ROI, exposure, temperature/time validity when recorded |
| `dynamic_beam` / `beam_keyframes` | detector, beamline/optics, energy, camera segment, physical time axis |
| `beam_basis` | compatible detector/beamline/optics plus a training library covering the target beam state |
| `beam_schedule` | beamline, energy, camera segment and beam state |
| `beam_residual` | detector, beamline/optics, energy, detector region, camera segment and beam state |
| `camera_registration` | detector/optics, motor axes, units, signs, and detector coordinates |

Missing target facts are mismatches when a component declares that fact. The
default policy raises `CalibrationCompatibilityError`. An override requires a
non-empty reason and returns a serializable `CompatibilityReport` containing all
diagnostics; later runner integration stores that report in run provenance.
There is no nearest-calibration selection in this API.

## Shift and stretch of beam bands

`normalize.beam_band_shift` is a RAW-to-TRANSMISSION stage for vertically
drifting beam texture. Its only parameter is `schedule_component` (default
`beam_schedule`). Select the bundle through the runner's existing
`calibration_manifest` option. It supports projection and sinogram layout and
both Y and Theta windows.

The schedule is a `beam_schedule` component in beam coordinates with axes
`[Theta, Parameter]` and two columns, `(dy, m)`: detector-row shift and positive
magnification. Its time axis is `projection_index`, and sampling is `nearest`.
Time-axis values name acquisition slots: the stage maps the source ordinal
through `TomoDataSet.acquisition_indices`, preserving dropped-frame provenance.
Each processed acquisition slot must have an exact schedule entry; the stage
does not interpolate away measured frame-to-frame jitter. Its typed
`band_shift` field carries the model:

```yaml
band_shift:
  yc: 1080
  dy_end: 30.75
  sigma_x: 60
  s_max: 1.2
  white_component: pre_gain
  dark_component: pre_dark
  residual_component: beam_residual
```

The white (`detector_static`) and dark (`dark`) are measured, unsubtracted
full-detector `(Y, X)` fields. The optional residual is a `beam_residual`
component in beam coordinates, also `(Y, X)`, containing the measured
X-smoothed log-texture change left after the shift/stretch fit. `dy_end` must
be nonzero when using that residual. Omit `residual_component` for no residual;
setting all `m` entries to one then gives shift-only normalization.

For `g = max(W-D, 1)` and `b = gaussian_filter1d(log(g), sigma_x, axis=X)`,
the stage samples `b(yc + (y-yc-dy)/m, x)` with cubic interpolation and nearest
edge treatment, then divides `P-D` by
`g * exp(shifted_b-b) * exp(clip(dy/dy_end, 0, s_max)*R)`. Gaussian filtering uses
nearest edge treatment along X, matching the B10 harness. Without a residual
the last exponent factor is absent. All numeric model settings and component
references belong to the calibration component; they are not duplicated in
Stage parameters.

Calibration preparation reads complete detector planes so the cubic spline
prefilter sees the same boundaries for a full run and a row selection.
Projection reads need no extra halo. The derived gain, dark, texture, spline
coefficients and optional residual use run-managed temporary storage and are
retired after dispatch, including failures. The bundle reader fills these
planes in at most 64-row selections, bounding HDF5 materialization as well as
the finite-value check. Per-projection warp and line work
use the generic declared-work lifecycle. SciPy retains its internal line
temporaries. Static buffer planning includes declared projection work, while
bundle-dependent preparation is visible in the actual storage allocation
events; its size is not known from the input `BufferSpec` alone.

## Coordinate and time bounds

Static arrays are addressed in detector coordinates and read only for the
requested `[y_start, y_stop)` window. A dynamic payload must declare exactly one
`Theta` or `T` axis and an explicit `TimeAxis`.

`CalibrationBundle.read_component()` selects one projection index or evaluates
one requested physical time. Keyframe interpolation loads at most the two
bracketing Y windows. It does not construct a `(Theta, Y, X)` gain stack.
Algorithms needing their own interpolation can use `read_time_bracket()` and
operate on those two bounded arrays.

Supported payloads are `.npy`, HDF5 datasets, and memory-mappable TIFF. HDF5
descriptors name a dataset explicitly. Relative payload paths may not escape the
bundle directory.

## Minimal manifest

```yaml
schema_version: '1.0'
calibration_id: als832-epoch-1
created_at: '2026-06-11T15:00:00Z'
producer:
  package: lftomo
  version: 0.1.0
  git_commit: abc123
  parameters: {}
sources:
  - path: /data/scan-a.h5
compatibility:
  detector_id: pco-edge
  beamline: ALS 8.3.2
  optics_id: 20keV-110mm
  energy_keV: 20
  region: {y_start: 0, x_start: 0, height: 2160, width: 2560}
  static_artifact_epoch: from-20260611-143029
  camera_segment: camera-a
components:
  detector:
    kind: detector_static
    coordinate_frame: detector
    payload:
      path: components/detector.npy
      format: npy
      sha256: 0000000000000000000000000000000000000000000000000000000000000000
      shape: [2160, 2560]
      dtype: float32
      axes: [Y, X]
    compatibility: {}
    interpolation: nearest
    validation_results: []
    metadata: {}
validation_results: []
metadata: {}
```

Use `validate_bundle(..., verify_hashes=True)` when accepting a newly created or
transferred bundle. Hashing is streamed in 8 MiB blocks. Routine window reads do
not repeatedly hash complete payloads.

## Runner resolution

`lftomo.runner.resolve_calibration()` resolves calibration before Stage
construction and complete-pipeline validation. The resolution order is:

1. `calibration_manifest` on a normalization Stage in `recon.yaml`;
2. a caller-supplied `CalibrationBundleRef`;
3. explicit legacy `Scan.metadata["flat_fields"]` four-map input;
4. explicit legacy `pre_gain`, `post_gain`, `pre_dark`, and `post_dark` Stage
   paths;
5. whites and darks embedded in a dxchange HDF5 source.

If YAML and a direct caller both select a bundle, their resolved manifest path
and calibration ID must agree. The runner never searches for a nearby bundle.
It validates the selected bundle and its component compatibility before
constructing Stages. A mismatch raises by default; `compatibility_policy:
override` also requires `compatibility_override_reason` and persists the full
diagnostic decision.

For legacy normalization algorithms, a bundle can provide components named
`pre_gain`, `post_gain`, `pre_dark`, and `post_dark`. Only the active detector Y
window is read. Embedded HDF5 fallback similarly reads only the first and last
white/dark Y windows and suppresses a second, full calibration-stack read.

Every run records a serializable `calibration_resolution` artifact. External
bundle runs additionally seed the typed `calibration_bundle` reference and
`calibration_compatibility` report. Direct callers use:

```python
from lftomo.runner import run_reconstruction

results = run_reconstruction(
    "recon.yaml",
    "scan.yaml",
    calibration_bundle=bundle_ref,
)
```

## Bounded bundle construction

`CalibrationBuildSession` writes into a hidden sibling staging directory and
publishes the target directory with one atomic rename only after every payload,
descriptor, hash, and validation result passes. A crash leaves a `BUILDING.json`
marker; an explicit or context-manager failure leaves `BUILD_FAILED.json` and
the partial payloads for diagnosis. `inspect_build()` distinguishes those states
from complete and corrupt bundles.

Array components are NPY memory maps filled sequentially along a caller-selected
axis. `append()` retains only the current iterator chunk; `close()` flushes and
hashes the payload in bounded blocks. `write_table_component()` applies the same
contract to numeric row chunks and records explicit column names.

```python
from lftomo.calibration import CalibrationBuildSession

with CalibrationBuildSession(
    "/calibrations/trip-1",
    calibration_id="trip-1",
    producer={"package": "lftomo", "version": "0.1.0", "parameters": params},
    compatibility={"detector_id": "pco-edge", "region": detector_region},
) as build:
    build.add_source(scan_path, uuid=scan_uuid, hash_source=True)
    writer = build.create_array_component(
        "beam_keyframes",
        shape=(n_keyframes, rows, columns),
        dtype="float32",
        axes=("T", "Y", "X"),
        kind="beam_keyframes",
        coordinate_frame="beam",
        time_axis=time_axis,
    )
    for keyframe_batch in bounded_keyframe_reader:
        writer.append(keyframe_batch)
    writer.add_validation_result(qc_result)
    writer.close()
    bundle_ref = build.finalize()
```

Source records accept paths, UUIDs, acquisition times, caller-supplied hashes,
and algorithm metadata such as detector crop/binning/orientation, camera
segment, and static-artifact epoch. `hash_source=True` computes a streamed hash;
it does not materialize the source file. Algorithm builders add component or
bundle `ValidationResult` objects, including confidence metrics, before
finalization.

## Conventional per-session medians

`lftomo.prep.flat_median.generate_four_map_medians()` is the deliberately
smaller legacy-compatible workflow. It accepts explicit `HDF5FrameSeries` or
`TIFFFrameSeries` readers for pre-gain, post-gain, pre-dark, and post-dark,
median-combines one detector-row chunk at a time, and atomically publishes:

```text
pre_gain.tif
post_gain.tif
pre_dark.tif
post_dark.tif
flat_medians.yaml
```

The TIFFs are float32. `FourMapMedianResult.flat_fields` and
`scan_metadata()` preserve the exact order expected by `normalize.weighted` and
the runner. The YAML sidecar records source paths/datasets, frame ranges/counts,
source shape/dtype, detector crop/binning/orientation, output shape/dtype, chunk
size, caller build parameters, and per-role non-finite input counts. Recoverable
NaN/inf samples are excluded from the per-pixel median; a pixel with no finite
sample fails the atomic build. Source arrays are treated as immutable even when
a TIFF reader returns a writable disk-backed view.

```python
from lftomo.prep.flat_median import (
    generate_four_map_medians,
    hdf5_four_map_sources,
)

result = generate_four_map_medians(
    hdf5_four_map_sources("session.h5"),
    "session-flat-medians",
    y_chunk=32,
)
scan.metadata.update(result.scan_metadata())
```

The HDF5 convenience function uses `exchange/data_white` and
`exchange/data_dark` for both pre and post by default. For sessions with
distinct datasets or frame ranges, construct four explicit `HDF5FrameSeries`
instances. TIFF input may be one multi-page stack or a sequence of consistently
shaped single-frame files. The output directory must not already exist, so a
failed run can never expose a mixture of old and partial maps.

## Static detector / dynamic beam decomposition

`decompose_static_dynamic()` implements the source model in log space. A
Gaussian separates fine detector-scale structure from the broad beam envelope;
an optional long directional median assigns elongated horizontal or vertical
bands to the beam; and the temporal median retains stable compact residuals in
the detector field `D`. Dynamic snapshots are defined as `B[k] = F[k] / D`, so
`D * B[k]` reconstructs each cleaned input flat exactly.

This is an identifiability prior, not a proof of physical origin. A compact
beam feature that never moves can be classified as detector-static. With the
directional prior enabled, an elongated detector scratch parallel to the beam
bands can be classified as beam texture. More flat motion and frames reduce
the first ambiguity; disabling the directional prior avoids the second but
requires adequate temporal beam motion.

`build_static_dynamic_components()` applies the model in haloed Y windows and
writes directly through `CalibrationBuildSession`. The detector component has
kind `detector_static` and is valid by detector, optics, detector coordinates,
and static-artifact epoch. The beam component has kind `dynamic_beam` and is
valid by beam/optics/energy and one camera segment. Each camera move starts a
new dynamic component; `segment_time_bracket()` refuses to bracket outside the
named segment, so interpolation cannot cross a move. `D` may carry across a
camera move only while detector pixels, optics/focus, and artifact epoch remain
unchanged.

`register_beam_offset()` performs bounded old-to-new ZNCC registration on row
and column beam profiles after optionally dividing out `D`. Each axis reports
the winning ZNCC, competing-peak ambiguity, peak margin, measured shift, and
resolved shift. An encoder prior limits the refinement search and becomes the
resolved value when confidence or ambiguity fails policy; this is especially
important along stripes, where the aperture problem makes one axis weak even
when its winning ZNCC looks high. `calibrate_motor_axis()` derives its typed
pixel scale and sign from that resolved value and returns the full axis
diagnostic for persistence as a `camera_registration` component or
compatibility metadata. The `env_sigma` argument remains accepted for parity
with the source model and provenance, but does not currently tune the bounded
ZNCC calculation.

### RA-00808 validation finding

The supplied B11 medians and `als832_refs/manifest.csv` confirm a hard camera
move between `20260611_143029_cyrothin1` (`i0_y=-25`) and
`20260611_145551_cyrothin2` (`i0_y=-40`). Bounded registration on a 2× view
measures `dy=-15.11` view pixels, or `-30.22` detector pixels: approximately
`2.01 px` per recorded i0-y unit. The stripe-parallel X profile has negligible
ZNCC in this pair and must not be trusted without an encoder prior.

The new artifact first visible in `20260611_143029_cyrothin1` moves in detector
coordinates with the later camera move. It is therefore beam/optic-attached,
not a detector-pixel defect, and must remain in the dynamic beam component. It
does not justify changing the detector-static epoch for `D`; it does require a
new beam state at its first appearance and a new camera segment at each move.
No interpolation is permitted between those segment-specific dynamic
components.

## Two-flat beam tracking normalization

`normalize.beam_tracking` is the order-aware replacement for weighted
normalization when pre/post beam structure drifts during a scan. It accepts
projection or sinogram layout, prefers projection order, preserves layout, and
is the pipeline's sole RAW-to-TRANSMISSION transition. For each global
projection index it evaluates only one Y-windowed gain; it never constructs a
`(Theta, Y, X)` gain stack.

The safe-default interpolator retains the log-space endpoint mean—including
detector-static `D`—and transports only the smoothed log ratio, where `D`
cancels. A serializable linear or piecewise `BeamSchedule` maps global scan
fraction to beam time. `fit_schedule: true` estimates that mapping from sampled
projection air pixels and records air count/fraction, fitted beam times, and
residual standard deviation in `beam_tracking_qc`. Missing air or a degenerate
endpoint direction fails explicitly. Legacy four-map input remains supported;
bundle input names can be configured through `component_names`.

A schedule may extrapolate beam structure beyond its endpoint beam times, but
dark interpolation is clamped to the measured pre/post interval. A Theta-window
executor must seed `projection_start` and total `projection_count` artifacts;
the stage rejects a partial projection batch without those global coordinates
instead of silently assigning incorrect schedule fractions.

On the five timestamped AAA392 rescan5 projections documented by the source
package, a bounded 192×1024 air-region validation found a nonzero initial beam
time and reduced mean held-out air residual from 0.01086 (naive weighted
endpoints) to 0.01067. The improvement is modest for this subpixel-motion scan;
the schedule is the material correction. More complex non-monotonic events
require keyframes rather than forcing a two-endpoint schedule.

## Flat-series keyframes

`build_flat_series_digest()` samples the midpoint of each equal-length source
interval and takes a short centered median (normally 3–5 frames). It reads and
writes by detector-row window through `CalibrationBuildSession`; it never block
averages an interval or accumulates full-resolution snapshots. The
`beam_keyframes` component records physical/frame time, source frame indices,
source path/pattern and first/last frame, crop/binning, camera segment,
intensity and centroid trajectory, hashes, and maximum keyframe gap.

`normalize.beam_keyframes` requires projection times on that same explicit time
axis. It reads only the two bracketing Y windows and evaluates one gain per
projection, while preserving the detector-static content common to both
keyframes. Queries outside the component time range fail. `maximum_gap` can
enforce an observability policy; even without a hard limit, QC states that an
event occurring wholly between keyframes cannot be recovered. Camera segments
remain hard boundaries through component compatibility.

Choose spacing from the shortest beam event that must be recovered, not merely
from total scan length: use at least two samples across that event and prefer
three or more when its shape is nonlinear. Treat interpolation uncertainty as
increasing with the bracketing gap. A gap at or below the validated median
training gap is the baseline regime; a 1–2× gap is elevated uncertainty, and a
gap above 2× should be rejected with `maximum_gap` unless held-out validation
supports it. This is an observability policy rather than a statistical error
bar—events wholly inside any gap remain unknowable from the digest.

On the copied 1,200-frame 10-minute white-field series, the real-data test uses
20 five-frame keyframes and holds out every inter-keyframe midpoint. The compact
piecewise model lowers median held-out error relative to interpolation between
only the series endpoints while touching only the sampled frames and a 32-row
working window.

## External beam-basis normalization

`build_beam_basis()` is the opt-in path for scans with no trustworthy in-scan
flat. It accepts one bounded random-access source or a
`ConcatenatedFrameSource` made from compatible external scans. The builder makes
two detector-row-windowed passes: the first accumulates only an N-by-N Gram
matrix in log space; the second writes the mean and retained PCA fields directly
through `CalibrationBuildSession`. `HDF5RandomFrameSource` supports explicit
dataset/frame ranges, and `crop` limits both source reads and the persisted
detector region. The component records its source scan identifiers, explained
fractions, training residual, coefficient ranges, detector crop, and camera
segment.

`normalize.beam_basis_fit` consumes that selected `beam_basis` component and an
optional detector-static component. It accepts both projection and sinogram
layout, prefers projection order, preserves layout, and is the pipeline's sole
RAW-to-TRANSMISSION transition. It fits a handful of coefficients independently
for each projection and therefore never constructs a full gain stack. A 2-D
shared air mask or a layout-matching 3-D mask may be supplied as the `air_mask`
artifact. Without one, the fallback selects bright pixels whose transmission
against the basis mean is near one.

The automatic mask assumes that every projection contains a sufficiently large,
well-exposed air region. It is not appropriate when the sample or holder covers
the whole detector. `exposure_quantile` and `edge_quantile` configure the
bright-field and edge exclusions. Likely sample edges are removed, the mask is
eroded, and a robust residual trim rejects false-positive sample pixels before
the final fit. That trim assumes true air is the majority of the candidate
mask: when gross contamination violates this assumption, the MAD centre may
follow the sample rather than reject it. Fit-residual, coefficient-range, and
out-of-basis QC are therefore required backstops, not optional decoration.
QC reports air count/fraction, edge overlap, rejected-mask fraction, residual,
basis rank/conditioning, coefficient stability, training-range extrapolation,
and relative out-of-basis error. Insufficient air and rank-deficient coverage
always fail; configurable limits either fail the Stage or, with
`fail_on_qc: false`, leave explicit per-projection violations for diagnostic
runs.

The basis can represent only beam states present in its compatible training
library. A low PCA training residual does not authorize use across a camera
move, energy/optics change, or unrepresented beam event. Populate
`training_coverage` with the source scans and keep normal compatibility checks
enabled. The RA-00808 integration test trains across three independent scans at
the same recorded camera/SSD configuration, improves reconstruction of a clean
flat held out from a fourth scan over the static mean, and reduces the fitted
residual on real projection air pixels. It reads a 128-row crop in 32-row
windows rather than materializing any full scan.
