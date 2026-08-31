# B3 — prep/flat.py:167 hardcoded reshape(40, 40)

**Wave 0 · size S · depends on nothing**

## Defect

`prep/flat.py:167` calls `reshape(40, 40)` — the literal is
`2 × scope` for the *default* `scope=ImgSlice(20, 20)` only. Any caller
passing a different scope gets either a crash (size mismatch) or a silently
wrong reshape.

## Implementation

1. Derive the reshape dimensions from the `scope` actually in effect
   (2·scope per axis, or whatever the surrounding arithmetic defines — read
   the enclosing function; the point is that the literal must become an
   expression of `scope`).
2. Regression test exercising a non-default scope (e.g. `ImgSlice(10, 15)`,
   asymmetric on purpose) through this path.

## Acceptance

- Test fails pre-fix (with a non-default scope), passes post-fix.
- Default-scope behavior byte-identical.
- Nothing else rides along.
