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
