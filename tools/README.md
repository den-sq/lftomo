# tools/

Developer/CI helpers. Not part of the installed package.

## `test_redundancy.py` — coverage-redundancy screen

Flags tests whose coverage is a subset of another single test's
(`covered(T) ⊆ covered(S)`) — the tests most likely to be duplicative. Subset
containment (not global "adds zero coverage") is used deliberately: it
exonerates broad integration tests and flags only the narrow tests fully
contained in another.

**Advisory, not a gate.** Coverage is blind to assertions — two tests can walk
identical lines while asserting different properties. A flagged test is a
*candidate*; confirm it asserts nothing the containing test does before folding
(a scoped `mutmut` run on the shared lines, or a read). Provenance/contract
tests (which overlap behavioral tests by design) are excluded via
`--exempt-regex`.

Generate per-test coverage contexts, then run:

```sh
coverage run --branch --source=lftomo \
  --data-file=.coverage -m pytest        # needs dynamic_context=test_function
# or set it inline:
coverage run --branch --source=lftomo -m pytest \
  && :   # with [tool.coverage.run] dynamic_context = "test_function"

python tools/test_redundancy.py                       # full-suite audit
python tools/test_redundancy.py --changed origin/master   # PR screen: only tests added/changed vs a ref
```

`--changed` accepts a git ref (diffs added `def test_` in `tests/`) or a file of
pytest node ids. In `--changed` mode a non-empty result exits non-zero, so it
can annotate a PR — keep it informational, do not fail the build on it.

The container shown for a candidate is *a* superset, not necessarily the only
one; `[MUTUAL — identical coverage]` marks pairs that cover exactly the same
lines (either true duplicates or same-path/different-assertion — the highest-value
cases to confirm).
