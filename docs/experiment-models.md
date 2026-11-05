# Experiment configuration and identity

The experiment schema is an envelope around the canonical
[`Recon` candidate space](candidate-spaces.md). It does not have its own
parameter, algorithm, or structural-variant fields.

```yaml
schema_version: 1
name: stripe-study
candidate_space:
  pipeline:
    - {id: read, kind: read, algorithm: hdf5_dxchange}
    - id: stripe
      kind: stripe
      sweep:
        kind: cases
        cases:
          - {name: fw, algorithm: fw, params: {sigma: 2}}
          - {name: vo, algorithm: vo_sort, params: {size: 21}}
sampler: {name: exhaustive, seed: 17}
resources: {max_concurrency: 2, memory_gb: 32}
metrics:
  - {id: nrmse, metric: reference.nrmse, protocol_version: "1", direction: minimize}
objectives:
  - {metric_id: nrmse, role: objective}
windows:
  - {id: top, y_start: 0, y_stop: 16}
  - {id: bottom, y_start: 1008, y_stop: 1024}
fidelities:
  - {id: screen, windows: [top, bottom], projection_stride: 4}
aggregation: {name: mean, metric_ids: [nrmse], invalid: fail}
output: {screening: metrics_only}
```

Schema validation resolves metric, objective, window, fidelity, aggregation,
and output-policy references without running a reconstruction. Unknown fields
and unsupported schema versions fail. Exhaustive sampling accepts only finite
candidate spaces; Sobol and random references require an explicit budget and
may consume the same continuous domains defined by `Recon`.

## Candidate key versus trial ID

A candidate key hashes only canonical resolved pipeline content. It is stable
when candidate enumeration changes and does not include an output location.
A trial ID adds the identity-bearing inputs needed to decide whether a result
is scientifically and numerically reusable:

- dataset content and acquisition identity;
- resolved scan configuration and calibration hashes;
- package version, commit, dirty state, runtime/backend, dependencies, and
  relevant hardware facts;
- exact realized fidelity windows, before/after pads, and projection sampling;
- metric protocol names, versions, and configuration hashes.

The complete sweep-free `Recon` remains embedded for direct rerun. Labels,
notes, dataset/output paths, and sampler provenance do not change
compatibility. The sampler seed and coordinate remain recorded in every trial
to explain candidate discovery, but identical resolved candidates are reusable
across samplers. Compatibility compares normalized
identity-bearing fields and reports paths such as
`dataset.content_hash` or `metric_protocols[...].protocol_version`, not only
a bare hash mismatch.

`WindowReference` remains the YAML boundary. Its `to_window` method resolves
bounds, fidelity sampling, and requested halo through `TomoWindow.from_bounds`.
`FidelityIdentity.windows` maps the existing window IDs directly to those
`TomoWindow` objects. There is no additional window class. Its
`held_out_window_ids` records the held-out subset from configuration; the other
IDs are tuning windows. The shared geometry converter supplies the documents
used for persistence and hashing. Execution and cache sizing consume the same
windows, including clipped edge pads. Requests that resolve to the same geometry
have the same window identity; different realized pads cannot share one.

Programmatic `ExperimentInputs` require `source`, a full-source `TomoWindow`
carrying physical shape, raw dtype, and work dtype. The CLI derives it from the
scan. Both trial execution and Tier-2 planning resolve fidelities through
`FidelityIdentity.from_reference`. Stride and projection limits remain in the
configuration; the resulting window's step and extent own realized sampling.

This changes internal trial and plan documents and their hashes. Old records
containing only requested bounds/halo cannot establish realized geometry and
are not accepted as compatible records. Experiment YAML and its version remain
unchanged. `WindowReference.from_window` requires the physical shape and emits
the smallest scalar halo that reproduces the window; geometry that cannot fit
the YAML schema is rejected. The original requested halo cannot be recovered
after clipping.

## Manifest lifecycle

`ExperimentManifest` records the versioned config and append-unique attempt
IDs. Each `TrialAttemptRecord` has a monotonic sequence slot and one strict
state:

- `pending`: no execution or final fields;
- `running`: a start time and no final state;
- `succeeded`: start/finish times and one matching `FinalResult`;
- `failed` or `interrupted`: start/finish times, structured failure, and no
  successful result.

A final result contains the rerunnable `Recon`, raw metric slots, Pareto
slots, finalized outputs, and optional promotion lineage. Output records and
final results require `complete: true`; a partial attempt cannot validate as
successful. The future persistent runner must publish these complete records
atomically and append new attempts rather than overwriting earlier evidence.
