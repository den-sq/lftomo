# Experiment operations

This is the end-to-end runbook for candidate selection, independent trials, persisted
resume, multi-fidelity promotion, blinded review, and finalist output. Start from
[`examples/experiment-tier2-template.yaml`](../examples/experiment-tier2-template.yaml)
for a complete pipeline that deliberately contains two ordered stripe occurrences and a
separated Paganin-filter/negative-log phase chain.

## What belongs in one experiment

An algorithm family is only a set of interchangeable implementations for one authored
role—for example the choices for the `stripe-primary` occurrence. It is not a reason to
split execution into separate pipelines. A candidate is always a complete ordered
pipeline, and one candidate may contain several occurrences with the same role:

```text
normalize -> stripe-primary -> stripe-sort -> phase-filter -> negative-log -> recon
```

The sweep can vary the algorithm or typed parameters of any occurrence, and structural
variants can include or omit explicitly named occurrences. Resolution never deduplicates
by role: `stripe-primary` and `stripe-sort` remain distinct, ordered stages. Every trial
receives a deep-copied resolved `Recon` and newly constructed Stage objects, so mutable
state cannot leak between candidates.

Phase composition has an additional scientific boundary. `paganin_filter` and
`bac_correction` preserve transmission and may be chained before one
`negative_log`. The terminal log is its own stage; do not add one to each filter. Domain
validation rejects a correction after the log, a second log, or reconstruction without a
terminal transmission-to-attenuation conversion. Fused compatibility algorithms remain
valid alternatives but must not be followed by another negative log.

## Prepare, validate, and estimate

The experiment YAML owns the candidate space, sampler seed/budget, objectives, windows,
fidelities, guards, and promotion rules. The scan YAML owns immutable acquisition facts
and must include `metadata.dataset_content_hash`. Paths and labels are descriptive;
dataset/configuration/calibration hashes, code/environment, resolved candidate, fidelity,
and metric protocol form `TrialId`. Sampler coordinates remain recorded provenance but
do not prevent reuse of an identical resolved candidate discovered by another sampler.

```console
lftomo-experiment validate experiment.yaml
lftomo-experiment dry-run experiment.yaml scan.yaml --root /scratch/run-2026-08-04
```

For a multi-fidelity config, dry-run emits a preflight block for every level. Review:

- requested versus unique/deduplicated candidates and all resolved stage IDs/algorithms;
- exact output roots and identity values;
- each row window, halo, theta bounds, stride, and projection limit;
- validation failures for the complete pipeline at that fidelity;
- concurrency after GPU/memory limits and the output estimate versus all guards.

Exhaustive budgets are ceilings, never truncation. Sobol and random samplers require an
explicit budget and record every unit-cube coordinate and resolved value. Do not use
`--override-limits` until the estimate and destination capacity have been independently
checked; the override is persisted as an operator action, not evidence that output is
safe.

## Run and review

Set a private, run-specific blinding key when screening previews are configured. Store
the key outside the public result bundle; record only its non-secret key ID in a Tier-2
plan.

```console
export LFTOMO_PREVIEW_BLINDING_KEY='load-from-your-secret-store'
lftomo-experiment run experiment.yaml scan.yaml --root /scratch/run-2026-08-04
```

Screening executes each global row window with its declared halo and projection sampling.
Evaluation sees the inner local rows only. Per-slice values aggregate within each window,
then tuning and held-out windows aggregate separately. Promotion uses tuning metrics,
constraints, and Pareto objectives only. Held-out values remain in the decision record
for later audit. The configured tie/invalid/failure policy is applied before candidate-key
stable ordering.

The root layout is versioned and inspectable:

```text
store.json
manifest.json
trials/<trial-id>/identity.json
trials/<trial-id>/attempts/<attempt-id>.json
promotions/<from>--<to>/active.json
previews/<blind-id>/*.png
private/preview-mapping/entries/<blind-id>.json
private/preview-mapping/mapping.json
experiment-results.json
fidelity-results.json
leaderboard.md
```

Attempt files move atomically through pending/running/terminal state. A successful result
contains raw metrics and Pareto rank; failures cannot contain a final result. Blinded
filenames use only opaque IDs and every candidate uses the same scale, crop, and selected
slice rule. Keep the entire `private/` tree access-controlled and share only `previews/`
during blinded review. Screening strips writer stages. Only promoted trials at
`final_fidelity_id` regain the configured full writer.

## Inspect, resume, and rerun

```console
lftomo-experiment inspect /scratch/run-2026-08-04
lftomo-experiment resume experiment.yaml scan.yaml --root /scratch/run-2026-08-04
lftomo-experiment resume experiment.yaml scan.yaml --root /scratch/run-2026-08-04 \
  --retry-failed --retry-interrupted
lftomo-experiment rerun experiment.yaml scan.yaml --root /scratch/run-2026-08-04
```

A clean resume reuses successful attempts and the matching promotion decision without
execution. Identity mismatches default to failure with field-level diagnostics. Choose
`--mismatch skip` to retain an incompatible terminal record, or `--mismatch rerun` to run
under the new identity. Failed/interrupted attempts require their explicit retry flags;
`rerun` always allocates a new attempt and preserves prior history.

Live claim files prevent duplicate execution. A stale claim is not automatically stale
work: verify its host/PID first, then use `--recover-stale --stale-after SECONDS` together
with `--retry-interrupted`. Recovery records the abandoned run as interrupted and moves
partial data to quarantine. Changed lower-fidelity results invalidate the active promotion
boundary. Review the change before using `--invalidate-stale-promotions`; the old decision
is quarantined, never overwritten as though it had not existed.

## Record and audit a Tier-2 protocol

Real-data comparisons require pinned, tamper-evident provenance before projection payloads
are opened: the dataset and scan/calibration hashes, clean Git commit,
dependency/hardware environment, environment-lock hash, projection-manifest hash, metric
protocol hashes, exact windows, resource bounds, preview contract, blinding key ID, and
finalist policy. `protocol_id` content-addresses these facts; the write and audit paths
reject a plan whose current payload no longer matches that ID. The model itself is not
strictly frozen.

```console
lftomo-experiment tier2-plan experiment.yaml scan.yaml \
  --environment-lock environment.yml \
  --projection-manifest projections.sha256 \
  --blinding-key-id review-key-2026-08 \
  --max-candidates 32 \
  --max-screen-window-rows 32 \
  --max-screen-projections 256 \
  --max-output-bytes 536870912000 \
  --max-memory-gb 64 \
  --output tier2-plan.json
```

The planner refuses dirty/unknown code, unblinded or missing preview contracts, fewer than
two tuning windows, no held-out window, unbounded screening projections, excessive trial
counts/windows, or absent memory/output guards. Hash the actual projection manifest—not a
mount path—and archive the plan before `run`.

After the bounded real-data run and blinded review:

```console
lftomo-experiment tier2-audit tier2-plan.json /scratch/run-2026-08-04
```

The audit requires the recorded experiment ID/preview contract, every raw metric, Pareto
ranks, held-out values in promotion records, and finalized finalist output. The optional
`tests/test_tier2_experiment.py` performs the same audit when
`LFTOMO_TIER2_EXPERIMENT_PLAN` and `LFTOMO_TIER2_EXPERIMENT_ROOT` are set and pytest is
invoked with `--run-tier2`.

## Test tiers and limits

The default hermetic suite proves a known winner through real repeated-stripe and split
phase stages, deterministic promotion, persistence/resume, previews, content-addressed
protocol planning, and finalist output. It also covers failures, interruption, mismatch
policies, concurrent claims, guards, ties, and stale decisions with small synthetic data.

Real Tier-2 execution is intentionally not a per-push CI job. It depends on mounted lab
data, pinned environment and projection manifests, sufficient scratch capacity, and a
human-controlled blinding secret. Run it deliberately on the recon-data capability and
publish only the audit/report artifacts required for review. Screening results demonstrate
performance on their declared windows and projection sample; they do not prove a candidate
is globally best, physically correct, or safe on all acquisitions. Final selection remains
a scientific decision supported by—rather than delegated to—the leaderboard.

Common-prefix reuse is a separate opt-in optimization. See [Common-prefix
cache](prefix-cache.md) for its fail-closed Stage contracts, explicit scratch/resource
bounds, immutable snapshot rules, independent verification, and benchmark gate. It does
not change trial identity or make caching the default.
