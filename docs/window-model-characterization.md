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

#212 corrects these behaviors in the canonical seam. The selected range
describes output; it is not the source clipping boundary.

## Core extension

`TomoDataSet.physical_shape` retains the full shape discovered from the source.
The inherited dataset `shape`, `pos`, and `frange` continue to describe the
selected output scope, so a selection such as `100:900` does not erase the
fact that physical rows `0:1000` exist.

`DataRef` and `TomoWindow` retain the existing symmetric `pad` constructor
argument and property. Internally and for asymmetric consumers, the same
objects expose ordinary semantic `DataShape` values as `pad_before` and
`pad_after`; no padding wrapper or second geometry model is introduced. The
single `pad` property remains available when both sides are equal and rejects
an asymmetric window so a caller cannot silently ignore one side.

`TomoDataSet.iter_with()` accepts requested `pad` and optional `pad_after`
values. Its iterator resolves each side against `physical_shape`, then creates
the canonical `TomoWindow` with the realized values. `with_pad()` expands by
`pad_before + pad_after`, while `internal` starts at `pad_before` and retains
the unpadded output shape.

Consequently, padding 10 around selected output `100:900` over physical rows
`0:1000` reads through `90:910`. For a non-divisible selection `100:1000`
with rows 400, the final output remains `900:1000`; its source read is
`890:1000`, with before-padding 10 and after-padding 0. Only the physical edge
clips padding, and clipping never shifts or shortens the requested output.

A directly constructed `DataRef`, `TomoWindow`, or `TomoWindowIter` without a
physical source record trusts the padding supplied by its caller. It does not
infer a source bound from the configured output range.
