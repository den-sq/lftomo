# Public experiment lifecycle

`lftomo.experiment.ExperimentLifecycle` is the supported orchestration boundary for
library callers and the `lftomo-experiment` CLI. It exposes structured `validate`,
`preflight`, `run`, `inspect`, `status`, `resume`, and `rerun` operations. Presentation,
signals, and operator prompts remain caller responsibilities.

```python
from lftomo.experiment import CancellationToken, ExperimentLifecycle, ResumePolicy

token = CancellationToken()
api = ExperimentLifecycle(runner, cancellation=token, callback=render_event)
plan = api.preflight()
summary = api.run()
status = api.status()

# From another thread or a signal handler owned by the caller:
api.request_cancellation("operator requested stop")

# A later process can reconstruct the invocation from the result root.
replayed = ExperimentLifecycle.from_replay("/data/experiment-results")
replayed.resume(ResumePolicy(retry_interrupted=True))
```

## Durable state and live events

Each attempt has an atomically replaced `progress/<attempt-id>.json` projection containing
its fidelity, window, Stage occurrence/algorithm, state, timing, counters, and failure
context. `ExperimentLifecycle.status()` reconstructs current state from these projections
and terminal attempt records only.

`progress.jsonl` contains public, versioned `ProgressEvent` records for live rendering.
Concurrent producers pass through one locked sequence writer. For every attempt update,
the authoritative projection is replaced before its event is appended. Consequently a
missing or truncated JSONL tail can lose display updates but cannot change experiment
status. Full tracebacks include the active Stage and window and pass through secret
redaction before either projection or event persistence.

Cancellation is cooperative. Once requested, the scheduler stops claiming new trials and
waits for in-flight executors to reach checkpoints between windows and Stages. It records
each affected attempt as explicitly resumable, flushes events, and returns structured
`CancellationDetails`. Compatible successes remain reusable.

## Replay bundle

After preflight succeeds and before the first claim, the result root receives:

```text
invocation/replay-bundle.json
invocation/normalized-experiment.yaml
invocation/normalized-scan.yaml
```

The bundle contains normalized scientific configuration, stable dataset/config/code/
environment identities, and execution, output, path, and cache policies. It does not
contain secret values. When preview blinding is active, only a short key fingerprint is
recorded so a caller can verify an externally supplied key. Original YAML files and the
original interactive session are not required for compatible resume or rerun.
