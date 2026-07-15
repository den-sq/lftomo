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
| `camera_registration` | detector/optics, motor axes, units, signs, and detector coordinates |

Missing target facts are mismatches when a component declares that fact. The
default policy raises `CalibrationCompatibilityError`. An override requires a
non-empty reason and returns a serializable `CompatibilityReport` containing all
diagnostics; later runner integration stores that report in run provenance.
There is no nearest-calibration selection in this API.

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
when its winning ZNCC looks high. `calibrate_motor_axis()` returns typed units,
pixel scale magnitude, sign, and the full axis diagnostic for persistence as a
`camera_registration` component or compatibility metadata.

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
