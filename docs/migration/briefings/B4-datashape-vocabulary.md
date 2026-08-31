# B4 — DataShape vocabulary + accessor layer (G1/G5/G6)

**Wave A · size L · depends on B2 · in-wave consumers B11/B12**

One-file change (`util/shapes.py` + its tests), chunky but cleanly
reviewable. Two halves that are deliberately one issue: the named-axis
vocabulary the migration needs, and the accessor-layer performance fix —
they are the same machinery (the vocabulary needs cached per-class field
names; the perf fix is cached per-class field names).

## Context: why these exact extensions

The gap analysis classified ~100 named-axis consumer sites: most hold a
layout **class** (`state.layout`, `BufferSpec.layout`), not an instance. A
class can answer *order* questions (`axis_of("Y")` on `ProjOrder` the class)
but not *extent* questions (`.Theta` needs an instance). API must therefore
work on both class and instance where semantically possible — getting this
wrong reproduces `pipeline/layout.py` one level up.

Three of the gaps have their solution already in-repo, stranded — the
cheapest proof the gaps are real:

- `prep/flat._axis_slice` (flat.py:90-91) — a privately-invented
  `axis_slice`: scans `fields(window.shape)` for the name, indexes
  `window.index` positionally.
- `DataShape.transpose()` (shapes.py:58-70) — computes the permutation,
  **zero production callers** (tests only); 11 hand-rolled `np.transpose`
  sites + 9 hand-permuted shape-tuple constructions exist beside it. The
  missing half is only the apply side.
- `SharedNPArray.size(swap=)` (mem.py:286-298) — *is* the wanted `nbytes()`
  (correct dtype accumulator, field-merge override), stranded on the storage
  class instead of the shape.

## New API

On `DataShape` (usable on **class and instance**):

- `axis_of(name) -> int`
- `names` — per-class field-name tuple
- `point_index(**{axis: i})` — index tuple with `slice(None)` elsewhere;
  `projection_view(data, layout, i)` becomes
  `data[layout.point_index(Theta=i)]` for any layout

On `DataShape` (instance):

- `prod()`, `max_field()`, `nbytes(dtype)` — **one** accumulator policy
  (float64 → int, matching mem.py:135). Today 12+ hand-rolled nbytes sites
  use four different accumulator dtypes; mem.py:122 is overflow-exposed.
- `transposed_view(array, target) -> ndarray` — applies `transpose()` as a
  zero-copy `np.transpose` view. Also fix `transpose()`'s annotation (it
  returns a list, not a tuple). Reference implementation for the by-name
  spirit: geometry.py:301-305.

On `DataRef`:

- `axis_slice(name)` = `self.index[axis_of(name)]` — promote
  `prep/flat._axis_slice` to where it belongs (the prep call sites migrate
  in B26; this issue provides the canonical home).
- `nbytes()` (DataRef knows its dtype).

## Accessor layer (perf)

Measured (µs/call, CPython 3.11, 3-field shape):

| operation | current | shallow equivalent | factor |
|---|---|---|---|
| `asdict(shape)` | 1.72 | 0.13 (cached-names dict) | ~13× |
| `astuple(shape)` | 1.56 | 0.02 (attr tuple) | ~80× |
| `a + b` (`replace`+`fields` op) | 0.96 | 0.23 (fast-path ctor) | ~4× |
| raw tuple add (iterator's approach) | — | 0.12 | — |

Causes, ranked: (1) `asdict`/`astuple` recursively deep-copy — pure waste
for flat int fields, and they are the universal field accessors across
DataShape (`to`, `extract`, `cut`, `merge`, `transpose`, `__bool__`,
`drange`…) and DataRef (`index`, `step_shape`, `stride`, `rescope`…);
(2) per-operation `fields()` + `getattr` + `replace()` in `__mop`/`__op`;
(3) exception-driven fallback in `__mop` (shapes.py:75-81) — runs to the
`AttributeError` and retries, exceptions as control flow on a hot path.

Work items:

1. **Tier 1:** cache per-class field-name tuples in `__init_subclass__`;
   add shallow `as_dict()`/`as_tuple()`; ban `asdict`/`astuple` internally —
   route every internal call site through the shallow accessors. This is
   exactly the machinery `names`/`axis_of` needs; one implementation serves
   both.
2. **Tier 2:** same-class fast path in the operators
   (`type(self)(*map(op, self.as_tuple(), other.as_tuple()))`), general
   `replace` path kept for mixed shapes; kills the exception-driven
   fallback.
3. `slots=True` **on the DataShape leaf dataclasses only** — NOT on
   `DataRange`/`_DataRangeIter`: `DataRange.shape`/`length` are
   `cached_property`s and the iterator assigns dynamic attrs in
   `__post_init__`; both require `__dict__`.
4. **Rejected: `frozen=True`** — measured ~3× construction cost
   (0.18 vs 0.06 µs; `object.__setattr__` per field), and constructions
   dominate iterator yields. Immutability stays a stated convention (see B5
   seam note).
5. **`get_iter` reentrancy:** construct `_DataRangeIter(start, stop, step,
   conv)` directly instead of the transient `self.iter_conv` set/del dance
   (`get_iter` currently mutates `self` — not reentrant).
6. After landing: re-judge `_DataRangeIter`'s hand-rolled tuple cache by
   profile (0.12 µs vs ~0.23 µs post-fix — it stops being *necessary*;
   whether it stays is a measurement call, not a style call).

## Deletion ledger (what this API is sized against)

The deletions themselves land in B11/B12 (and later waves), but acceptance
of *this* issue is two-sided against the ledger: the API suffices for every
entry, and no public surface exists that no entry demands.

- 12+ hand-rolled nbytes/prod sites: runner.py:408, cli.py:333,
  mem.py:122/135/298, planning.py:113/267/271, execution.py:211-216 (×3),
  io.py:101/154/192; `astuple`+`np.max` at prep/stripe.py:326 and
  stages/stripe.py:290.
- 11 `np.transpose` permutation sites + 9 hand-permuted shape-tuple
  constructions (execution.py:517/586-589, planning.py:163, helical.py:101,
  read.py ×2, …).
- `pipeline/layout.py`'s 8 helpers + ~16 inline re-derivations (B11).
- 4 `_shape_tuple` variants (planning.py:104, mem.py:29, core.py:270,
  core.py:349) whose only job is feeding `np.prod` (B12/B5).
- 6 positional `.index[<literal>]` sites (streaming.py:91-93,
  runner.py:481/501, flat.py:48) — all assume Y at position 1.
- `DataRange.__len__`'s ravel-as-prod (shapes.py:262-266) — re-express on
  `prod()`; the in-source comment beside it ("Probably a sign DataShape
  design is off!") gets retired with it.

## Acceptance

- Extensions + tests in `tests/util/`; zero consumers outside shapes.py
  changed (internal shapes.py call sites rerouted to shallow accessors).
- Perf pin: a benchmark-style test or committed measurement showing
  `as_tuple` ≥10× over `astuple` on a 3-field shape, and the same-class
  operator fast path engaged (no exception raised in the hot path —
  assertable by mocking/counting).
- Class-and-instance semantics tested for `axis_of`/`point_index`/`names`.
- API not frozen until B11/B12 merge against it (no-consumer-no-freeze).
