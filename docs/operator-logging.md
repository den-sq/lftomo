# Operator logging

`lftomo.util.log.Logger` and `LOG` are the canonical human/operator logging
interface. The module-level `log` instance writes its configured severities to
stdout or stderr and does **not** create a file by default.

File output is explicit:

```python
from lftomo.util.log import LOG, Logger

with Logger(log_files={"run": ("/results/run.log", LOG.STATUS | LOG.TIME)}) as log:
    log.write("Reconstruction", "started", LOG.STATUS)
```

`log.set_logdir(path)` is the convenience opt-in used by the acceptance runner.
Path sinks are opened lazily, flushed after each line, and closed by
`Logger.close()` or context-manager exit. Streams supplied by the caller are
borrowed and are never closed by the logger.

The elapsed-time and memory columns are observations made when a line is
formatted. They are not Stage or execution-segment attribution. Versioned
experiment `ProgressEvent` JSONL is a separate redacted machine-readable stream;
formatted log lines never replace it.

Prior releases implicitly selected an `imaging_log_*.txt` file in the caller's
current directory. That behavior was removed: callers that need a file must
select one explicitly.
