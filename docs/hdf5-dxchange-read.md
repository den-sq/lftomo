# HDF5 dxchange read contract

The registered `read.hdf5_dxchange` Stage reads the paths declared by
`HDF5DxchangeProjections`.

- `exchange/data` (or configured `data_path`) is mandatory and must be a
  `(Theta, Y, X)` dataset. A missing or incorrectly shaped dataset is a clear
  configuration error.
- `exchange/data_white` and `exchange/data_dark` are optional by default. When
  both exist, bounded copies are exposed as `white_fields` and `dark_fields`
  artifacts. If either is absent, neither artifact is emitted; downstream
  configuration may provide separate flat/dark files.
- `calibration_policy="required"` makes either missing calibration dataset an
  error. `calibration_policy="ignore"` skips calibration reads even when the
  datasets exist.

Projection-order reads fill manager-issued working storage with
`Dataset.read_direct` when the source is contiguous. Sinogram-order reads use a
bounded Theta scratch followed by transpose/scatter because h5py requires a
writable C-contiguous destination. Chunked or compressed sources take the same
bounded fallback path. No path materializes or transposes a second full volume.

The legacy `util/io.py` HDF5 branch derived a raw byte offset from
`Dataset.id.get_offset()`. That only works for contiguous, uncompressed HDF5
storage and returns no usable offset for chunked or compressed datasets. The
Stage makes no direct-offset assumption and supports contiguous, chunked, and
compressed dxchange files uniformly.

## Added noise realisations for mosaic spectra

To produce a `NOISY` reconstruction, add the optional `noise` mapping to the
ordinary HDF5 read Stage. Keep the same source, row range, normalization, phase,
stripe, center and reconstruction parameters for both runs:

```yaml
pipeline:
  - kind: read
    algorithm: hdf5_dxchange
    params:
      y_start: 800
      y_stop: 864
      noise:
        gain: 2.05
        read: 66
        seed: 0
  # The ordinary processing Stages follow unchanged.
```

For `FULL`, omit `noise` (or set it to `null`). An empty mapping enables the
defaults shown above. Each raw pixel receives a zero-mean Gaussian draw with
variance `gain * max(P - dark, 0) + read`, where `dark` is the pixelwise median
of all frames in the configured `dark_path`. `read` is a variance in squared
raw detector units, not a standard deviation. Calibration pixels are unchanged.
The added noise is not clipped or converted back to unsigned integers: the read
buffer uses the canonical window's working dtype, normally float32. Source
metadata retains the on-disk dtype.

The noise path always requires a non-empty dark dataset with the source detector
shape. `calibration_policy` continues to control which calibration artifacts are
published: if that policy would omit the fields, the noise path reads and retires
its own managed dark input without publishing it downstream. No white dataset
is required solely for noise addition.

Each full detector row uses NumPy PCG64 initialized from a SeedSequence containing
`[seed, acquisition_slot, source_row]`. Acquisition slots come from the canonical
dataset's provenance, including dropped projections. Source rows come from the
canonical window, including padding. Thus overlapping samples are identical
across read layouts, projection subsampling and window sizes. This deliberately
does not reproduce the old harness's block-dependent random sequence; spectra
can differ through finite-sample variation while using the same variance model.

The existing generic bounded-work lifecycle allocates and retires three float64
detector-row vectors. The median calculation also uses NumPy's private
`dark_frame_count × detector_width` temporary for each row, preserving the loaded
dark input for ordinary normalization. Noise generation is serial within the
read Stage and does not create a worker pool.

Finish both pipelines with the ordinary registered volume writer:

```yaml
  - kind: write
    algorithm: npy_volume
```

`write.npy_volume` writes float32 `reconstruction.npy` under `output_path`, or
writes directly to a `.npy` `output_path`/Stage `path`. Windowed runs fill one
mapped volume in selected source-row order, without assembling a second volume
in memory. Existing files are protected unless `overwrite: true` is explicit.
The mapping is closed after each write; the durable file remains after the run.
The array axes are lftomo's `(Y, Z, X)`: slice number followed by image row and
column, matching mctutil's `(Z, N, N)` terminology for square reconstructions.

For example, after making `full.yaml` and `noisy.yaml` differ only in the read
noise option:

```shell
lftomo-run full.yaml scan.yaml --output FULL
lftomo-run noisy.yaml scan.yaml --output NOISY
mctutil mosaic spectra FULL/reconstruction.npy NOISY/reconstruction.npy spectra.json --box 768
```

The spectra command owns one shared `--box`/`--centre Y X` crop for both inputs;
use the same reconstruction geometry and the desired rotation-axis center.
lftomo adds no second crop convention. TIFF writers remain available for ordinary
reconstructions, but `mosaic spectra` currently requires NumPy block inputs.
