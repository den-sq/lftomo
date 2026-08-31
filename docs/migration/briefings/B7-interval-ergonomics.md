# B7 — Interval ergonomics (G8)

**Track C-early · size S/M · issue #277 · depends on B4 (#243, merged) · in-wave consumer B9 (#278)**

Line references exact against `f0e4e89` (post-Wave-B master).

## Context

The workaround is `tuple[int, int]`. Construction is already fine
(`Y.drange(a, b, step)` exists, used at streaming.py:208) — the canonical
object *loses to the tuple everywhere else in its life*:

- **Forced step:** `DataRange(start, stop, step)` requires all three
  (shapes.py:306+). `StreamingPlan.selection` has no stride, so making it a
  DataRange asserts an iteration nobody means.
- **Clunky readback:** endpoints return as `dr.start.Y`/`dr.stop.Y` — two
  attribute hops + a repeated axis name vs `sel[0]`.
- **No comparison story:** canonical code converts *itself* to tuples to
  compare — streaming.py:80 (`(int(pos.Y), int(pos.Y+shape.Y)) == self.selection`)
  and the read-stage validation twin.

So the tuple wins on build+read+store+compare; the extension has to win all
four or the tuples regrow — that is the whole lesson of this gap.

## Work items

1. **`Y.interval(a, b)`** (and the general per-axis form) — interval
   semantics with step absent/defaulted, no iterator baggage. This is the
   piece `DataRange` structurally cannot be (it forces a stride).
2. **`start_stop` accessors on `DataRange` *and* `DataRef`** — project
   `pos`/`shape` per named axis to `(start, stop)`. Kills the two-hop
   `dr.start.Y`/`dr.stop.Y` readback and the start+length flavor.
3. **Equality against the interval form** — so streaming.py:80 and the read
   twin stop round-tripping through tuples to compare. Port those two sites
   in-PR.

## Out of scope (named, not folded)

- **Document/serialized form** of the interval arrives with **B9**
  (`to_document`), not here. B7 provides the in-memory ergonomics only.
- The runner's `plan.selection[0]` rebasing (runner.py:447-449, 508-510 — the
  ≥5 subtraction sites) is **B21**'s driver-authority work. B7 gives the
  primitive; B21 consumes it. Do not migrate the runner here.

## Deletion ledger

- streaming.py:80 self-comparison + the read-stage validation twin → interval
  equality.
- `dr.start.Y`/`dr.stop.Y` two-hop readbacks at the ported sites → `start_stop`.

## Acceptance

- `Y.interval(a, b)` constructs without a step and compares equal to the
  tuple form it replaces; `start_stop` on DataRange and DataRef returns the
  projected endpoints; the two self-comparison sites use equality, not tuple
  conversion.
- No new public surface beyond interval + `start_stop` + equality — the API is
  sized to the ports above (two-sided acceptance against the ledger).
- API not frozen until B9 (#278) merges against it.
