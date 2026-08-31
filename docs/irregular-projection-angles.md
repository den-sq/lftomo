# Irregular and dropped-frame projection angles

`Scan.n_projections` is the number of recorded projection arrays or files.
`Acquisition.n_projections` is the number of planned trigger/acquisition slots.
They are equal for an ordinary uniform acquisition.  If they differ, the scan
must declare a non-uniform `angle_source`; read validation still requires the
physical source count to equal the recorded count.

Source discovery validates two aligned arrays:

- `acquisition_indices`: strictly increasing zero-based planned-slot indices,
	one for each recorded projection;
- `angles`: the corresponding per-projection angles in radians.

The acquisition convention divides the interval from `rotation_start_deg` to
`rotation_end_deg` into `Acquisition.n_projections` endpoint-exclusive slots.
Explicit indices and angles must agree with that convention.  TIFF scan logs
can declare `angle_source.kind: hardware_timestamps`; timestamp gaps are
converted to per-interval slot increments and cumulatively summed.  HDF5 scans
can use `angle_source.kind: hdf5` to import an angle dataset.  `explicit`
accepts embedded acquisition-index and/or degree arrays for small or generated
metadata.

Hardware-timestamp inference anchors the first recorded projection to planned
slot zero.  A leading dropped frame is therefore not observable from the
recorded timestamp intervals and is deliberately treated as a shifted
acquisition origin.  Under that convention, a count deficit with no interior
gap is assigned to the tail: for an ordinary long scan with its final frame
missing, the last acquisition index is less than
`Acquisition.n_projections - 1`.  The endpoint-calibrated nominal interval can
be biased by boundary loss, however, and this inference is only valid while the
observed interval ratios still round to their correct slot counts.  Use an
`explicit` index or angle source when boundary-slot identity must be recovered
independently of the first-record anchor.

The full arrays remain on `TomoDataSet`. Theta start/stop/stride selection is
recorded once by `TomoWindow`; center, reconstruction, and super-resolution
derive their selected angles directly from that dataset/window pair. They are
not republished as `PipelineState.artifacts`. No dummy projection is inserted
for a missing slot.

The historical `get_thetas` helper has no call sites and is not migrated.  The
active historical `skip_adjustment` path is also not copied: it applied local
drop offsets independently and could produce duplicate indices.  Hardware-gap
increments here are cumulative, so a drop shifts every later projection.
