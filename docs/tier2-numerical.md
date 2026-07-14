# Tier-2 numerical regression policy

Tier-2 numerical tests execute the bundled canonical producer pipeline against
the canonical `scan.yaml`, reconstruct the selected detector-row window, and
compare it with the immutable TIFF archive. They do not refresh references.
The runner must use the pinned `environment.yml` contract, particularly
`numpy<2`, because TomoPy 1.14 is not compatible with NumPy 2 copy semantics.

The harness bounds memory in three ways:

- it reads only the selected rows plus the producer's recorded processing halo;
- for projection-domain producers, it retains that row window in projection
  order, reads and processes bounded Theta batches, and assembles them in a
  temporary disk-backed array rather than materializing the full volume in RAM;
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

The AAA523 phase-contrast reference preserves the producer's 140-row window
with 30-row overlaps (80 retained rows). Rows 2000--2009 therefore execute in
the aligned `[1920, 2060)` window, whose retained core is `[1950, 2030)`. Gain
normalization, geometry, movement correction, BAC, and filtering stay in
projection order; Fourier-wavelet stripe removal is applied one sinogram at a
time without transposing the full array. Reconstruction receives only the ten
selected rows. The reference runner defaults to four projections per batch;
`LFTOMO_REFERENCE_THETA_BATCH` may tune that bounded batch size without changing
the result, and `LFTOMO_TIER2_SCRATCH` selects the temporary-array filesystem.

The 2026-07-14 mounted AAA523 validation used eight-projection batches and
passed all 10 selected slices. NRMSE ranged from `0.00456` to `0.00499`, Pearson
correlation from `0.96259` to `0.96363`, and normalized bias from `0.00117` to
`0.00127`. Peak observed host use remained approximately 4.4 GiB with no swap.

By default the recon-data workflow evaluates every selected reference slice.
For a diagnostic smoke run, `LFTOMO_TIER2_SLICES=N` limits each contiguous
selection to its first `N` rows. A reference is skipped, with its archival reason
reported, when metadata explicitly says that the exact producer or its input is
no longer available. This applies to AAA592's recovered thresholded variant,
which has 3,935 projections rather than the historical 3,937 inputs, and to the
B10 R2.54 and `b10_recon.py` outputs, whose surviving unversioned scripts were
modified after their historical runs. Their immutable outputs remain useful
archives, but later script bytes are not presented as exact regression inputs.
