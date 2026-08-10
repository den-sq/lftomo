# Experiment runner

The experiment runner evaluates resolved pipeline candidates as independent trials. It
uses the candidate-space and identity contracts described in [Experiment
models](experiment-models.md), validates every complete resolved pipeline before reading
projection data, and writes stable, machine-readable trial records and leaderboards.

## Choose a sampler

Use `exhaustive` for a finite list, grid, zip, case, or pipeline-variant space. An
optional budget is a safety ceiling: the runner fails instead of silently truncating a
larger enumeration. Use a seeded `sobol` sampler for typed continuous domains. Every
Sobol coordinate, resolved parameter, seed, and candidate key is recorded, and duplicate
resolved candidates execute only once. Seeded `random` sampling is also available for
diagnostic comparisons.

```yaml
schema_version: 1
name: phase-alpha-screen
candidate_space:
  pipeline:
    - id: phase
      kind: phase
      algorithm: paganin
      sweep:
        kind: domain
        params:
          alpha: {kind: log, start: 0.0001, stop: 0.1}
sampler: {name: sobol, seed: 20260804, budget: 32}
resources:
  max_concurrency: 4
  memory_gb: 32
  max_output_bytes: 1073741824
metrics:
  - id: rings
    metric: artifact.ring_score
    protocol_version: "1"
    direction: minimize
objectives:
  - {metric_id: rings}
windows:
  - {id: middle, y_start: 500, y_stop: 516, halo: 8}
fidelities:
  - {id: screen, windows: [middle], projection_stride: 4}
aggregation: {name: mean, metric_ids: [rings]}
output: {screening: metrics_only, max_bytes: 1073741824}
```

## Validate and run

The scan YAML must provide an immutable SHA-256 value in
`metadata.dataset_content_hash`. Validation checks the experiment schema and registered
metric protocol without opening projection payloads. Dry-run also resolves, deduplicates,
identifies, and validates every complete candidate while reporting output roots and
resource estimates.

```console
lftomo-experiment validate experiment.yaml
lftomo-experiment dry-run experiment.yaml scan.yaml --root results
lftomo-experiment run experiment.yaml scan.yaml --root results
```

Each trial has a deterministic `results/trials/<trial-id>/` root. Attempt records are
atomically replaced as they move through pending, running, and terminal state, so an
interruption cannot publish a successful final result. The root contains
`experiment-results.json`, `experiment-results.yaml`, `leaderboard.md`, and a manifest.
The leaderboard preserves raw metric values and deterministic Pareto rank; it does not
collapse unlike objectives into an arbitrary scalar score.

Screening defaults to `metrics_only`, which removes writer stages and prevents a
parameter sweep from unexpectedly producing full-volume output. Raising an output guard
requires an explicit `--override-limits`. GPU experiments also require declared GPU slots
through the library policy; this first runner deliberately does not share mutable stage
state or partial pipeline prefixes between candidates.

`peak_memory_bytes` is a true per-trial measurement only when the executor supplies it
in `TrialExecution`. Otherwise the runner records the process-wide `RUSAGE_SELF`
high-water mark. Concurrent thread-based trials therefore share the fallback value; use
an executor-supplied measurement when isolated per-trial memory accounting matters.

## Runtime output accounting

Output estimates are preflight warnings and early guards; they are not the runtime
accounting source. Complete pipeline writers place TIFFs in an attempt-scoped staging
root. After all windows finish, the runner measures and hashes the complete set, takes
the experiment output-ledger lock, checks every applicable cumulative budget, and only
then atomically replaces the published files. A rejected attempt removes its staging
files and cannot contribute a finalist result.

The counted roles are `screening_output`, `finalist_output`, and `blinded_preview`.
Pipeline YAML, attempt records, progress records, prefix-cache snapshots, and temporary
publication files are not final output and do not consume these fields. Their storage is
covered by the result, cache, and scratch path-resource policies instead. Unknown-size
formats need enough scratch space for one complete in-flight artifact per writer; when
no scratch root is supplied, the runner uses a hidden result-local staging directory so
the final replacement still has a same-filesystem atomic boundary.

The three byte fields have distinct cumulative scopes over counted roles:

- `resources.max_output_bytes` limits all committed output in the experiment root;
- `FidelityReference.max_output_bytes` limits committed output at that fidelity; and
- `output.max_bytes` limits the configured output roles across the invocation.

The ledger serializes check-and-publication, so concurrent finalists cannot both spend
the same remaining bytes. A replacement at the same logical path subtracts the prior
file before adding the new one, which prevents forced reruns from double counting.
Exactly-at-limit publication is allowed; one byte over raises `OutputLimitExceeded`
before a durable replacement. `output-ledger.json`, per-attempt records, summaries, and
`output.committed`/`output.limit_exceeded` events retain the actual bytes and effective
limits. Resume reconstructs the ledger from the latest successful attempt records,
validates recorded sizes and hashes, and quarantines unrecorded files under the managed
`published/` tree.

Resource limits remain deliberately identity-bearing: they define the bounded protocol
under which the candidate set is allowed to finish. `--override-limits` does not rewrite
that identity; it disables both estimate and runtime rejection for the invocation and is
recorded in the replay bundle, manifest metadata, summaries, and accounting snapshots.
Blinded preview export uses the same temporary-write, measured-byte,
`OutputLimitExceeded`, and serialized-publication semantics for its policy limit.

## Progress events

`progress.jsonl` is an append-only, derived live tail. Atomic attempt projections in the
result store remain authoritative for status and recovery. Schema-one readers must
accept events that predate optional display fields; new schema-one writers populate the
fields whenever the corresponding context is known.

Each scheduled fidelity emits `fidelity.started`, one `fidelity.progress` event after
every terminal execution/reuse/skip outcome, and `fidelity.completed`. The coordinator
that owns worker submission also owns these counts, so concurrent producers cannot race
them. `completed_candidates` is the sum of succeeded, failed, interrupted, skipped, and
reused outcomes. `remaining_candidates` excludes active work:

```text
completed_candidates + active_candidates + remaining_candidates = total_candidates
```

The fidelity total is the set actually scheduled at that level. After promotion it is
the promoted set, not the original sampler population.

Candidate events retain stable hashes while also carrying a one-based sampler position,
the sampler total, and selected structural variant. Window and Stage events carry
one-based positions and totals from the resolved fidelity and resolved candidate
pipeline, respectively. Boundary events report measured trial/window/Stage elapsed
time. Executors may emit `stage.heartbeat` or `window.heartbeat` through their active
execution context when a long-running implementation has no finer internal units; a
percentage is never inferred. Failure events include a redacted one-line message,
failure type, and durable traceback context.

## Resume and rerun

`run` initializes a versioned filesystem store. The store retains an immutable identity
document and append-safe attempt sequence for every trial. A per-trial atomic claim file
prevents two processes from executing the same trial concurrently. Inspect the store or
resume it with:

```console
lftomo-experiment inspect results
lftomo-experiment resume experiment.yaml scan.yaml --root results
```

A clean resume reuses a compatible successful result. Failed and interrupted attempts
are retained but require `--retry-failed` or `--retry-interrupted`; the store never turns
one into a success implicitly. `rerun` always creates a new attempt and can be restricted
to resolved trial identities:

```console
lftomo-experiment rerun experiment.yaml scan.yaml --root results \
  --trial-id 6e2f...full-trial-id
```

Compatibility is checked against the complete trial identity, including dataset and
calibration hashes, resolved candidate, code and environment, fidelity windows, and
metric protocols. Sampler coordinates remain recorded provenance but are deliberately
non-identity, so the same resolved candidate can reuse a result across samplers. The
default mismatch policy is `fail`; select
`--mismatch skip` to persist an auditable incompatible record or `--mismatch rerun` to
run under the new identity. Diagnostics name concrete fields such as
`dataset.content_hash`.

An abandoned running claim is never guessed from ordinary elapsed time. After checking
that its owner is gone, recovery must be explicitly requested with `--recover-stale` and
an appropriate `--stale-after` threshold; use `--retry-interrupted` to then start a new
attempt. Invalid records and partial output directories are moved under `quarantine/`
rather than being accepted as cache hits.
