# Pipeline-core epic acceptance

Issue #185 is the opt-in closure run for the pipeline-core redesign in #173. It is
not part of ordinary pull-request CI. Run it once on a self-hosted machine with the
canonical `phase_contrast_hdf5` Tier-2 dataset mounted, and repeat it only when an
assembled acceptance record is required.

The command retains every generated TIFF, experiment attempt, cache snapshot,
measurement, JSONL progress event, and proof record under an operator-selected
result directory. Choose a new or empty result directory outside the repository
and dataset tree.

```bash
lftomo-acceptance run \
  /mnt/wsl/PHYSICALDRIVE2p1/B10/PCRBox_Octo0/20kev/voi2.1 \
  /path/outside/the/repository/lftomo-epic-173-acceptance \
  --workers 2
```

The equivalent source-tree invocation is:

```bash
python -m lftomo.acceptance run DATASET_ROOT RESULT_ROOT --workers 2
```

`DATASET_ROOT` must contain the canonical `scan.yaml`, the projection source,
the four pregain/postgain/predark/postdark median TIFFs under `gains/` or
`flats/`, and `reference/b10_recon`. Use `--flat-root` if the median TIFFs live
elsewhere. If `scan.metadata.dataset_content_hash` is absent, the command hashes
the projection HDF5 file once before execution; `--dataset-content-hash` accepts
an already established immutable SHA-256 value.

The invocation directory also retains float32 four-map TIFFs with the B10
oracle's exact pre-subtraction applied (`pre_gain -= pre_dark`,
`post_gain -= post_dark`). Pipeline runs use those resolved maps while the named
oracle reads the original medians, so the normalization inputs are identical.

Human-readable progress is rendered by lftomo's shared logger and retained under
`logs/`. The same events are recorded structurally in `acceptance-progress.jsonl`.
The final machine record is `acceptance-record.json`, and `acceptance-summary.md`
is formatted for posting to issue #185.

## Mode 1: windowed pipeline

The mode selects global rows `[1068, 1092)` as two 12-row windows with zero user
pad. The current-oracle script, current-oracle reproduction pipeline, and
frozen-B10 reproduction pipeline are three independent reads of those rows; no
run consumes another run's output. Progress messages identify both the run and
the window interval. The mode records:

- exact TIFF parity between `examples/b10_recon_current.py` and the equivalent
  split `bac_correction` plus `negative_log` pipeline;
- comparison of the historical B10-family pipeline against the frozen
  `b10_recon` slices 1076–1085, whose producer used the same 12-row extent;
- an explicit non-applicability record for `recon_minimal`, whose frozen output
  belongs to a different dataset and 64-row producer extent;
- non-gating memory observations: run-level manager allocations plus sampled
  private-memory deltas for every Stage invocation;
- `/dev/shm` inspection after normal completion and after `SIGKILL` of an
  isolated owner process group during its first window.

The shared-memory observation comes from unique manager allocations and remains
run-level. Private memory is sampled for each Stage invocation and recorded with
its window, Stage index, dispatch kind, process-tree peak, and per-process peaks;
`/dev/shm` mappings are excluded so shared storage is not counted twice. These
measurements are diagnostic telemetry only. The acceptance run makes no memory
estimate, upper-bound promise, or pass/fail claim from them.

## Mode 2: whole-volume experiment

This mode passes no runner `WindowConfig`: every trial processes its selected
four-row volume as one complete allocation. Two BAC-gamma candidates execute
concurrently. Their common read/normalize prefix is cached with explicit disk
and memory bounds. The record requires:

- one cache miss and one hit with one stable prefix ID and cache key;
- overlapping successful trial intervals;
- balanced manager pin/unpin events and no alias event against a pinned name;
- per-candidate TIFF fingerprints identical to a second independent no-cache
  experiment, while the two candidates remain distinct;
- no live shared-memory segment after both experiments complete.

This split preserves D7: windowed prefix caching remains deferred to #184.

## Posting the closure record

Review `acceptance-record.json` and the retained outputs before posting. When
both modes pass, the generated summary can be added to #185 with:

```bash
gh issue comment 185 --repo den-sq/lftomo \
  --body-file /path/to/RESULT_ROOT/acceptance-summary.md
```

The issue should close only after the real Tier-2 record is attached. The
hermetic tests validate the harness and lifecycle proofs but do not substitute
for that recorded run.
