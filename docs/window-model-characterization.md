# Pre-#212 window-model characterization

This is the step-one baseline for #212. It records the existing canonical
`DataRef` / `TomoWindow` / `TomoDataSet` relationships before extending them.
The defects listed below are observations, not compatibility contracts.

## Existing relationships to preserve

For a middle output window `100:200` with symmetric Y padding 10, the current
operations already express the three required coordinate views:

| Operation | Y range | Meaning |
| --- | --- | --- |
| `window.index` | `100:200` | requested output in source coordinates |
| `window.with_pad(["Y"]).index` | `90:210` | padded source read |
| `window.internal.index` | `10:110` | requested output in padded-buffer coordinates |

`rescope()` remains the operation used to move or resize the same reference,
and the integer, step, and slice iterators continue to derive from its
`DataRange`.

The unpadded iterator also already sizes remainder windows correctly. For a
physical Y extent `0:1000`, selected output `100:1000`, and window size 400,
it emits `100:500`, `500:900`, and the short final window `900:1000`.

These relationships are pinned in `tests/util/test_window.py` without adding a
second geometry representation.

## Observed gaps to correct, not preserve

The current implementation cannot combine that baseline with correct physical
edge padding:

- Constructing `TomoDataSet` with a `DataRange` replaces the selected axes in
  its inherited shape with `over.stop - over.start`; the full physical source
  extent is then unavailable to window iteration.
- `DataRef.__init__()` handles unavailable padding below zero by moving the
  output position inward and shrinking its shape. This mutates the requested
  output instead of realizing less padding before it.
- `TomoWindowIter` subtracts symmetric padding when it cuts a final window at
  its range stop. In the non-divisible example, applying Y padding 10 through
  the current dataset path changes the requested output windows from
  `100:500`, `500:900`, `900:1000` to `110:510`, `510:910`, `910:980`.
- `with_pad()` has no physical upper-bound record and the model cannot yet
  carry distinct realized padding before and after a window.

#212 deliberately corrects these behaviors. The selected range describes
output; it is not the source clipping boundary. A separate physical source
shape on `TomoDataSet` will allow padding to read outside the selection, and
only a physical edge may reduce the realized padding. The eventual padded
version of the remainder test must therefore continue to emit `900:1000`
while clipping only its after-padding at physical row 1000.

This characterization does not select the name or public form of the future
physical-shape record or asymmetric-padding value. Those belong to the next
implementation steps and must extend the existing window seam narrowly.
