# B2 — shapes.py hygiene

**Wave 0 · size S · depends on nothing · gates B4 (Wave A opener)**

Three independent cleanups in `util/shapes.py`, **one commit per topic**.
Wave 0 has no internal ordering, but B4 depends on this landing first (clean
shapes.py before the vocabulary/accessor work builds on it).

## 1. Fix `ImageStack` duplicate-Y

`util/shapes.py:188`: `class ImageStack(DataShape): Z: int; Y: int; Y: int`
— duplicate `Y`, missing `X`. Any adopter gets a 2-field shape. Fix to
`Z, Y, X` and add a field-set regression test (the class currently has no
production consumers, which is why the bug survived).

## 2. `Reverse*` renames (decided 2026-08-30)

- Rename `BackwardsSinoOrder` → `ReverseSinoOrder` (`shapes.py:186`, zero
  consumers — the rename is free).
- Add `ReverseReconOrder`: the CloudVolume/neuroglancer write order, exactly
  `reversed(ReconOrder)` — (Y,Z,X) → (X,Z,Y). It replaces the hand-rolled
  `np.transpose(data, (2,1,0))` at `write.py:306` in B19; adding the class
  now is deliberate: this is the last cheap moment before B9/G3 puts class
  names into serialized documents.

Naming decision: shape-relational names, not consumer names
(`CloudVolumeOrder`/`NeuroglancerOrder` rejected — they say nothing about
the shape and rot if external software changes).

## 3. Deprecate `NPString`

Self-flagged `"????"` in-source; no production use. Deprecation marker +
removal note, per the project's deprecation practice.

## Acceptance

- Three commits, one per topic; each independently revertable.
- Tests: `ImageStack` field-set pin; `ReverseReconOrder`/`ReverseSinoOrder`
  field order pinned (against their forward twins via `transpose()` is
  fine).
- Nothing else rides along.
