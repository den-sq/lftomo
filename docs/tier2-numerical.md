# Tier-2 numerical regression policy

Tier-2 numerical tests execute the bundled canonical producer pipeline against
the canonical `scan.yaml`, reconstruct the selected detector-row window, and
compare it with the immutable TIFF archive. They do not refresh references.
The runner must use the pinned `environment.yml` contract, particularly
`numpy<2`, because TomoPy 1.14 is not compatible with NumPy 2 copy semantics.

The harness bounds memory in three ways:

- it reads only the selected rows plus the producer's recorded processing halo;
- it crops away that halo before center finding and reconstruction;
- it hashes files in 8 MiB blocks and computes comparison metrics in 256-row
  image tiles; and
- it caps producer-era process/core settings at two workers, which does not
  change the algorithms but avoids replicating large buffers across 12–32
  workers.

Each reference TIFF must first match its `recon.sha256` entry. Numerical output
then passes only when all of these scale-independent conditions hold:

- normalized RMSE (RMSE divided by reference dynamic range) is at most `0.15`;
- Pearson correlation is at least `0.95`;
- absolute mean bias divided by reference dynamic range is at most `0.05`.

These tolerances accommodate historical environment and FFT-library drift while
still rejecting changes in anatomy, orientation, center, scaling, or processing
order. The JSON report records every checksum and metric.

By default the recon-data workflow evaluates every selected reference slice.
For a diagnostic smoke run, `LFTOMO_TIER2_SLICES=N` limits each contiguous
selection to its first `N` rows. A reference is skipped, with its archival reason
reported, when metadata explicitly says that the exact producer input is no
longer available. This currently applies to AAA592's recovered thresholded
variant, which has 3,935 projections rather than the historical 3,937 inputs.
