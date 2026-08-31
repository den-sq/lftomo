# B25 — localct_patterns document helpers

**Track C-early · size S · issue #279 · depends on B9 (#278)**

Line references exact against `f0e4e89` (post-Wave-B master).

## Context

`localct_patterns.RunSummary.to_json`/`from_json` (localct_patterns.py:66/71,
used at :119/:124) is the **third serialization idiom** in the repo — after
pydantic (`model_dump`) and pickle (`state.pkl`) — with a hand-rolled `Path`
coercer. B9 lands the one document contract; this folds the last idiom onto it
so nothing outside the contract serializes canonical objects.

## Work items

1. Reroute `RunSummary.to_json`/`from_json` onto the B9
   `to_document`/`from_document` contract (or a thin JSON wrapper over it),
   deleting the hand-rolled `Path` coercer.
2. **Wire-format check first:** if the on-disk JSON is an external contract any
   artifact/consumer depends on, preserve the serialized shape (adopt the
   document form internally, keep the JSON layout). If it's purely internal,
   adopt the document form directly.

## S-issue discipline

Done when the idiom is folded and the coercer is gone. This is the failure mode
of small tickets — work expanding to justify the issue. Nothing else rides
along; no new document fields, no `RunSummary` restructuring.

## Deletion ledger

- localct_patterns.py:66/71 hand-rolled `to_json`/`from_json` + the `Path`
  coercer → B9 document helpers.

## Acceptance

- `RunSummary` round-trips through the document contract; the hand-rolled
  coercer is gone; a `grep` for a third serialization idiom in `lftomo/` is
  empty.
- If the JSON wire format was preserved, a fixture pins byte-compatibility;
  if adopted wholesale, the internal-only status is stated.
