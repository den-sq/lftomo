# B13 — Read: window as driver, one TomoDataSet per run

**Wave B · size M/L · issue #262 · depends on B6 (#260), B10 (#261) · in-wave consumers B14 (#263), B16 (#265) · freezes B6/B10 APIs**

Line references exact against `670eca8` (post-Wave-A master).

## Context

Read stages already build TomoDataSet/TomoWindow (`_bind_source_state`,
read.py:156, called at :352/:476) — but the window is an *output* of the
read, not its driver, and the dataset story is duplicated:

**The single-dataset finding (2026-08-31).** The plan-time dataset
(streaming.py:168-170) is built *incomplete* — no angles/indices,
defaulted dtype — which is why `_bind_source_state` rebuilds a fresh
TomoDataSet per pipeline execution (N per windowed run, 2N with
auto-center prepass), overwriting the `window_plan.dataset` the runner
handed in. Two TomoDataSet constructions exist so that neither is
authoritative.

The angle fallback compounds it: `TomoDataSet` synthesizes angles from
the FOV via linspace (window.py:288-293) when none are passed — the
plan-time dataset always takes that fallback, while the read-time
rebuild passes real acquisition angles. Same object, two different angle
truths depending on construction site.

util/io's tiff readers (`distribute_tiff_read` io.py:69,
`distribute_tiff_sinogram_read` io.py:117) take loose scalars and
hand-roll byte-offset math that `ContigMemIter`/`MemAddressIter` already
express — the exemplar `read_window` in the same module routes through
them.

## Work items

1. **`from_bounds`-driven read** — read stages construct their window
   via `TomoWindow.from_bounds` (B10) from the planned bounds; the
   six-loose-ints assembly inside `_bind_source_state` is replaced by
   the factory call. The in-source shape-vs-step comment moves into the
   factory's contract (that subtlety is what B10 encapsulates).
2. **Complete the plan-time dataset; delete the read-time rebuild** —
   one metadata probe at plan time (shape + dtype + angles +
   recorded-vs-declared validation, one file open) completes
   streaming's dataset; `_bind_source_state` stops constructing a
   TomoDataSet and becomes bulk I/O + validation against the handed
   dataset + planned window. End-state: **one TomoDataSet creation per
   run**; StreamingPlan iterates TomoWindows off it (the driver-side
   iterator already works this way).
3. **Angles: acquisition-model arrays as the only path** — wire
   `TomoDataSet.angles` to the acquisition-model angle arrays and
   deprecate the fov-linspace fallback (window.py:288-293). Read stages
   already pass angles explicitly; after item 2 the plan-time dataset
   does too, so the fallback's only remaining callers are tests —
   deprecate loudly, delete in a later wave.
4. **util/io tiff readers take a TomoWindow** and route through
   `ContigMemIter`/`MemAddressIter` like `read_window`, deleting the
   hand-rolled byte-offset math. External function signatures may change
   — both are internal (no consumer constraint touches util/io).

## Deletion ledger

- `_bind_source_state`'s TomoDataSet rebuild (the N-per-run
  construction) and its six-loose-ints window assembly.
- The fov-linspace angle fallback path as a production path
  (deprecation now, deletion later — it is the *only* Wave B deletion
  allowed to be staged).
- io.py:69/:117 byte-offset arithmetic.
- The read-stage half of the selection algebra duplicated with
  `planning._read_spec` (B10 landed the shared derivation; this deletes
  the twin).

## Acceptance

- Tests: one-TomoDataSet-per-run provenance pin (object identity from
  plan to state across a windowed run, including the auto-center
  prepass); plan-time dataset carries real angles (recorded-vs-declared
  validation exercised against a fixture with mismatched header);
  tiff read window equivalence against the current byte-offset outputs.
- Provenance: `_bind_source_state` constructs no TomoDataSet; grep pin.
- Numerical suites unchanged — any diff is a migration bug, not an
  accepted change.
- This merge **freezes the B6/B10 APIs** (no-consumer-no-freeze).
