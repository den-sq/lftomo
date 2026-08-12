# Irregular and dropped-frame projection angles

`Scan.n_projections` is the number of recorded projection arrays or files.
`Acquisition.n_projections` is the number of planned trigger/acquisition slots.
They are equal for an ordinary uniform acquisition.  If they differ, the scan
must declare a non-uniform `angle_source`; read validation still requires the
physical source count to equal the recorded count.

Every read produces three artifacts:

- `acquisition_indices`: strictly increasing zero-based planned-slot indices,
  one for each recorded projection;
- `angles`: the corresponding per-projection angles in radians;
- `angle_provenance`: source, count, convention, and selection metadata.

The acquisition convention divides the interval from `rotation_start_deg` to
`rotation_end_deg` into `Acquisition.n_projections` endpoint-exclusive slots.
Explicit indices and angles must agree with that convention.  TIFF scan logs
can declare `angle_source.kind: hardware_timestamps`; timestamp gaps are
converted to per-interval slot increments and cumulatively summed.  HDF5 scans
can use `angle_source.kind: hdf5` to import an angle dataset.  `explicit`
accepts embedded acquisition-index and/or degree arrays for small or generated
metadata.

Theta start/stop/stride selection applies one identical slice to projection
data, indices, and angles.  The full arrays remain on `TomoDataSet`, while the
selected arrays are carried in `PipelineState.artifacts` to center and
reconstruction stages.  No dummy projection is inserted for a missing slot.

The historical `get_thetas` helper has no call sites and is not migrated.  The
active historical `skip_adjustment` path is also not copied: it applied local
drop offsets independently and could produce duplicate indices.  Hardware-gap
increments here are cumulative, so a drop shifts every later projection.
