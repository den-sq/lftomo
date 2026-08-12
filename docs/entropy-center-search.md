# Adaptive entropy-center search

`center.entropy_local` reconstructs one sinogram at a small set of candidate
centers and selects the candidate with the lowest Shannon entropy.  If that
candidate is at either end of the range, the stage moves the range by its full
width and repeats the search.  `max_expansions` bounds that process (default:
3); it counts additional searches after the initial sweep.

The `center_entropy` artifact records every sweep and one of three termination
states:

- `converged`: the best candidate is inside the final range.
- `cap-reached`: the best candidate is still on a boundary after the permitted
  number of expansions.
- `boundary-persisted`: moving the range leaves the same best candidate on the
  opposite boundary, so another shift would move away from it.

The artifact also records the selected center, expansion count, source slice,
candidate values, entropy values, and per-sweep boundary information.  This
makes a bounded non-converged result distinguishable from a successful search.

Set `diagnostic_images: true` to retain each masked candidate reconstruction as
a float32 TIFF.  The files are placed under `center-diagnostics` inside the run
output (configurable with `diagnostic_subdirectory`) and appended to the shared
`output_files` ledger, so later writer stages and streaming center prepasses do
not discard them.
