# Candidate spaces

The `Recon` model is the only vocabulary for describing pipeline candidates.
Ordinary runs exhaust finite candidates directly. Experiment orchestration may
sample continuous domains, but it consumes the same models and does not define
a second parameter, algorithm, or pipeline-structure language.

## Stable Stage occurrence IDs

Use `id` to identify an occurrence when a pipeline repeats a Stage kind or
when a sweep or structural variant must remain stable under insertion and
reordering:

```yaml
pipeline:
  - {id: read, kind: read, algorithm: hdf5_dxchange}
  - id: stripe-fw
    kind: stripe
    algorithm: fw
  - id: stripe-vo
    kind: stripe
    algorithm: vo_sort
    sweep:
      kind: list
      params: {size: [11, 21, 31]}
```

IDs start with a letter and contain only letters, digits, `_`, `-`, and
`.`. They are unique within a pipeline. Existing canonical YAML without IDs
remains valid: ordinary parameter sweeps use the legacy positional
`stage[0]` qualification. Algorithm cases and experiment records should use
IDs, and structural variants require an ID on every Stage. Diagnostics use the
ID whenever one is present.

## Parameter domains and algorithm cases

The legacy `list`, `grid`, `linspace`, and `zip` forms retain their
original values and ordering. A `domain` sweep adds one typed representation:

```yaml
sweep:
  kind: domain
  params:
    filter: {kind: categorical, values: [hann, parzen]}
    center: {kind: discrete, values: [1023.5, 1024.0, 1024.5]}
    alpha: {kind: log, start: 0.0001, stop: 0.01, count: 5}
```

`discrete` and `categorical` domains are finite. `linear` and `log`
domains are finite when `count` is present and continuous when it is omitted.
The plain runner rejects continuous domains with a diagnostic directing the
caller to an experiment sampler.

Keep parameters conditional on their algorithm in named `cases`:

```yaml
- id: stripe
  kind: stripe
  sweep:
    kind: cases
    cases:
      - name: fourier
        algorithm: fw
        params: {wname: db5}
        domains:
          sigma: {kind: linear, start: 1, stop: 3, count: 3}
      - name: sorting
        algorithm: vo_sort
        domains:
          size: {kind: discrete, values: [11, 21, 31]}
```

This produces only valid FW and Vo parameter combinations; it never crosses
`sigma` with `vo_sort` or `size` with `fw`.

## Explicit structural variants

Structural variation is a named allow-list of complete Stage-ID sequences:

```yaml
variants:
  - name: fw-then-vo
    stages: [read, normalize, stripe-fw, stripe-vo, recon, write]
  - name: vo-only
    stages: [read, normalize, stripe-vo, recon, write]
```

Only listed structures are materialized; the implementation never generates
arbitrary permutations. Every selected structure becomes a complete,
sweep-free `Recon` and is validated by the normal Stage constructors and
pipeline contracts before reconstruction. Phase structures remain subject to
the phase semantic contracts.

Repeated stripe occurrences also remain subject to the
[ordered stripe-chain regression invariant](stripe-chain-regression.md):
candidate materialization may select a listed order but execution may never
reorder it or replace bounded row scratch with an implicit full-volume
transition.

## Identity, output, and guards

Each resolved candidate has a SHA-256 candidate key derived from its canonical
pipeline content. The key is independent of enumeration order and distinct
from experiment trial identity, which additionally includes dataset,
environment, fidelity, and metric protocol. Multi-candidate plain runs write to
`candidate-<key-prefix>` directories and JSON summaries include the full key,
qualified values, output path, artifacts, and directly rerunnable resolved
`Recon`.

`candidate_limits.max_candidates` guards Cartesian growth before
materialization. When detector dimensions are available,
`candidate_limits.max_estimated_output_bytes` also guards the estimated
float32 reconstruction output before calibration or projection data is read.
Raising either limit must be explicit in the YAML.
