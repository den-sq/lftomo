# localCT Pattern Decisions

This note records the `localCT` infrastructure patterns lifted into lftomo for
den-sq/lftomo#8. The implementation is pull-only: no changes are pushed back to
`ChengLabResearch/ReconChengLab`.

## Decisions

- `ReconstructionRequest` / result dataclasses: **adapt**. lftomo now has small
  request/result dataclasses in `lftomo.localct_patterns` for typed run
  boundaries.
- Scan-folder inspection: **adapt**. `inspect_scan_folder()` returns a typed
  result for projections, flats, darks, and metadata files before reconstruction
  logic decides how to proceed.
- JSON run summaries: **adopt**. `RunSummary`, `write_run_summary()`, and
  `read_run_summary()` provide a stable JSON run-record shape.
- TomoPy runtime bootstrap: **adapt**. `check_tomopy_runtime()` performs the
  initial importability check without importing TomoPy at module import time.
- Tiny generated TIFF tests: **adopt**. `tests/test_localct_patterns.py` uses
  generated 2x2 TIFFs in `tmp_path`; broader Tier-1 fixtures remain in the later
  fixture issue.

## Attribution

These are infrastructure patterns adapted from
`ChengLabResearch/ReconChengLab/localCT`, not direct code copies. Super-resolution
research code and hardware abstractions remain out of scope here.
