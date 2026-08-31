# B1 — prep/phase.py:138 dist/energy argument swap

**Wave 0 · size S · depends on nothing**

## Defect

`prep/phase.py:138` calls `_phase_pad(..., pixel_size, energy, dist)` but the
signature is `(dim, pixel_size, dist, energy)` — `dist` and `energy` are
transposed at this call site. The other `_phase_pad` call site is correct.
Result: wrong Paganin padding whenever this path runs. The bug is invisible
because the values travel as anonymous positional scalars — exactly the
failure class the wider migration removes; this fix is independent and
immediate.

## Implementation

1. Swap the arguments at `prep/phase.py:138` to match the signature.
2. Regression test pinning the Paganin *padded output* for a case where
   `dist != energy` numerically — the test must fail under the transposed
   call. A test with symmetric/equal values would pass either way and pin
   nothing.
3. While in file: verify the correct call site with the same test shape (no
   code change expected there).

## Acceptance

- Test fails on the pre-fix code, passes post-fix.
- No API or behavior change beyond the corrected padding.
- Nothing else rides along (S-issue guardrail: done when the ledger clears).
