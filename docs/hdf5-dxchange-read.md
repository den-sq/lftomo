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
