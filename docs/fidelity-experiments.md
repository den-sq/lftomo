# Fidelity experiments

Multi-fidelity experiments screen every candidate on small, representative windows and
promote only recorded winners to more expensive levels. Fidelity is part of `TrialId`:
window coordinates, halo, projection bounds and stride, projection limit, resource
fraction, memory/output limits, and timeout are comparison-bearing facts rather than
informal runner flags.

## Tuning and held-out windows

Every window declares its role. Tuning windows drive ranking; held-out windows are
retained in decision records and reports but are never passed into promotion scoring.
Use several separated row windows when artifacts vary over detector height.

```yaml
windows:
  - {id: top, role: tuning, y_start: 128, y_stop: 144, halo: 8}
  - {id: middle, role: tuning, y_start: 1016, y_stop: 1032, halo: 8}
  - {id: blind-bottom, role: held_out, y_start: 1900, y_stop: 1916, halo: 8}
fidelities:
  - id: screen
    windows: [top, middle, blind-bottom]
    projection_stride: 8
    projection_limit: 256
    resource_fraction: 0.25
    memory_gb: 8
    max_output_bytes: 268435456
  - id: finalist
    windows: [top, middle, blind-bottom]
    projection_stride: 1
    memory_gb: 32
promotion:
  name: successive_halving
  protocol_version: "1"
  reduction_factor: 4
  min_survivors: 2
  ties: include_all
  invalid: exclude
  failures: exclude
output:
  screening: metrics_and_previews
  final_fidelity_id: finalist
  write_full_finalists: true
```

The executor performs each listed row window independently. It injects the global row
range plus halo into the read stage, passes only the inner local range to evaluation, and
applies projection bounds, stride, and limit before reconstruction. Strided reads attach
the corresponding original acquisition angles, so projection values and theta metadata
cannot drift apart. `dry-run` validates every candidate at every fidelity level.

Slice observations are first aggregated within their window. Window values are then
aggregated separately by role using the configured mean, median, or explicit window
weights. Raw per-window values, the tuning aggregate, and the held-out aggregate remain
available for audit.

## Promotion decisions

Successive halving uses the configured constraints and Pareto objectives. It has explicit
rules for ties, invalid metrics, and failed attempts; candidate keys provide the final
stable ordering for `stable_cut`. `include_all` can promote more than the nominal cutoff
when candidates have an identical boundary signature. Manual decisions are supported by
the library API and are marked `manual_override` with the complete ranking retained.

Each boundary writes
`promotions/<from>--<to>/active.json`. The record contains its input fingerprint, ranks,
cutoff, tie signature, promoted keys, metric values, and parent trial IDs. Next-level
results carry the decision ID and parent trial ID as promotion lineage. A clean resume
reuses the same decision. Changed attempts or protocol inputs make the stored boundary
stale; callers must either stop or explicitly invalidate it, which moves the old record
to `quarantine/promotions/` before publishing the replacement.

## Blinded previews and finalist output

`export_blinded_previews` applies one fixed scale and crop contract to every candidate,
writes deterministic opaque labels, and produces a separate private mapping back to
candidate keys. PNG and TIFF output are supported, and the byte guard is checked before
each file is published. Do not vary scale, crop, or selected slice by candidate: that
would make a visual comparison biased even if filenames remain blinded.

Writer stages are stripped throughout screening unless screening writers were explicitly
requested. With `write_full_finalists: true`, they are restored only when a promoted trial
is executing the configured final fidelity. The final fidelity should therefore describe
the intended full output window/projection scope; promotion does not silently turn a
small screening window into a full-volume reconstruction.
