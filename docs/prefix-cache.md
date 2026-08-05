# Common-prefix cache

Common-prefix caching is an optional experiment optimization. It does not change
candidate keys, trial IDs, metrics, promotion, persistence, or resume behavior.
Independent execution remains the default and the diagnostic reference path.

## Enable it deliberately

The cache requires an explicit scratch directory plus disk and fork-memory bounds:

```console
lftomo-experiment dry-run experiment.yaml scan.yaml --root results \
  --prefix-cache \
  --prefix-cache-root /scratch/lftomo-prefix-cache \
  --prefix-cache-max-disk-bytes 53687091200 \
  --prefix-cache-max-memory-bytes 8589934592

lftomo-experiment run experiment.yaml scan.yaml --root results \
  --prefix-cache \
  --prefix-cache-root /scratch/lftomo-prefix-cache \
  --prefix-cache-max-disk-bytes 53687091200 \
  --prefix-cache-max-memory-bytes 8589934592 \
  --prefix-cache-max-entries 64 \
  --prefix-cache-on-error rebuild \
  --prefix-cache-verify-fraction 0.05
```

`dry-run` reports one decision per trial and fidelity window: the canonical prefix ID,
Stage IDs, stop reason, estimated snapshot bytes, and whether the suffix requires an
explicit mutation-isolation copy. An estimate that exceeds either bound is a visible
`bypass`, not a partially cached run. Dry-run creates no cache files.

## Fail-closed prefix planning

Candidates are compared by canonical ordered `StageConfig` and `StageContract` content,
not Stage display names or enumeration positions. A Stage is eligible only when its
contract explicitly sets `cacheable_prefix=True`. Planning stops before a writer,
evaluation, GPU, forking, externally stateful, nondeterministic, or otherwise
insufficiently contracted Stage. A parameter, algorithm, order, layout/domain contract,
resource contract, or stable Stage occurrence change ends the shared prefix.

Runtime cache identity additionally includes:

- dataset content and acquisition identity;
- scan/calibration configuration, code, environment, and backend;
- fidelity and row/projection window;
- the initial working-array dtype, shape, and content hash;
- initial layout and data domain;
- canonical prefix configs and contracts;
- deep content fingerprints for scan metadata and input artifacts.

Unsupported artifact types do not fall back to `repr`: they bypass or fail according to
`--prefix-cache-on-error`. A miss compares prior manifests for the same prefix and records
field paths such as `runtime.dataset.content_hash` or `artifacts.calibration_bundle`.

## Snapshot and fork safety

Each entry is constructed in `builds/`, fsynced, integrity-hashed, and atomically renamed
under `entries/<prefix-id>/<cache-key>/`. `manifest.json` is written last in the temporary
directory. Published snapshot files are never updated; last-access records live
separately so LRU accounting does not mutate scientific evidence.

The working array is stored as NumPy `.npy` and restored with a read-only memory map.
When any suffix Stage declares `mutates_input=True`, the executor first makes an explicit
bounded array copy. Otherwise the mapping remains read-only, so an incorrect mutation
contract fails instead of changing the snapshot. Artifacts are restored as a deep fork,
including nested dictionaries, lists, arrays, and metadata. Downstream failure,
interruption, or cancellation therefore cannot alter another trial or the cache.

Artifact state uses integrity-checked Python serialization because Stage artifacts may
retain typed calibration references. Treat the scratch directory as trusted local
storage; do not load a cache supplied by an untrusted user.

## Recovery, bounds, and fallback

A process and filesystem lock serialize publication and eviction. Kernel lock ownership,
not elapsed-time guessing, establishes whether recorded lock metadata is stale. Abandoned
build directories, stale claims, corrupt payloads, and sampled equivalence mismatches move
under `quarantine/`; they are never accepted as hits.
Quarantine shares the disk bound with active entries; the oldest quarantined evidence is
pruned when admitting a new snapshot would otherwise exceed that bound.

The store enforces actual serialized bytes in addition to preflight estimates. LRU
eviction is bounded by both `max_disk_bytes` and `max_entries`; access markers are outside
immutable entries. A required array fork is also checked against `max_memory_bytes`.
Error policy is explicit:

- `rebuild`: quarantine corrupt/partial state and publish a fresh snapshot;
- `bypass`: preserve independent execution for the affected trial;
- `fail`: stop instead of concealing any cache problem.

No mode guesses that a weak identity, unknown Stage, unsupported backend, or oversized
snapshot is safe to share.

## Equivalence verification and benchmarking

`--prefix-cache-verify-fraction` selects a deterministic subset of hits by trial/window
ID. For each selected hit, the prefix is executed independently and arrays plus nested
artifacts are compared with the declared numerical tolerances. A mismatch quarantines
the entry and bypasses it (or fails under `on_error=fail`). The default is `0.05` whenever
the cache is enabled. Keep verification nonzero during rollout, and consider `1.0` for an
initial representative qualification run before reducing the fraction based on measured
cost and confidence.

Every cached window records `build_seconds`, `load_seconds`, `snapshot_bytes`,
`read_bytes`, `written_bytes`, `cache_bytes_after`, fork strategy, invalidation fields,
recovery count, and evictions. Trial attempt records already provide end-to-end elapsed
time and process peak memory. Benchmark representative screening and full-volume shapes
with the same pinned dataset/configuration in this order:

1. independent cold run;
2. cache cold/build run;
3. cache warm run with a nonzero verification fraction;
4. independent repeat to expose ordinary run-to-run variance.

Compare end-to-end time, attempt peak memory, cache read/write bytes, final cache size,
metrics, outputs, and array/artifact equivalence. Record detector/projection/window shape,
storage type, concurrency, and cache bounds with the results. Caching stays opt-in unless
those measurements show a material benefit without exceeding the declared budgets. It
should be tabled for workloads whose common prefix is short, whose fork copy dominates,
or whose scratch traffic offsets the saved computation.

## Current limits

The implementation supports trusted, local, NumPy-backed scratch storage. It deliberately
does not provide cross-cluster coordination, GPU snapshots, live mutable state sharing,
algorithm reordering, or automatic enablement. Prefix build, load, publication, and
eviction currently share a global cache lock; benchmark concurrent first-build workloads
before relying on the cache for throughput. Per-prefix locking can be added later if
contention proves material without changing cache identity or snapshot semantics.
