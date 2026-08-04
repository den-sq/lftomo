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
- sampler seed plus coordinate or explicit case;
- exact fidelity windows, halos, and projection stride;
- metric protocol names, versions, and configuration hashes.

The complete sweep-free `Recon` remains embedded for direct rerun. Labels,
notes, and dataset/output paths belong to `TrialDescription`; changing them
does not change compatibility. Compatibility compares normalized
identity-bearing fields and reports paths such as
`dataset.content_hash` or `metric_protocols[...].protocol_version`, not only
a bare hash mismatch.

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
