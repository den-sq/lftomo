# Examples

`experiment-tier2-template.yaml` is the complete experiment/promotion template used by
the experiment operations runbook. It demonstrates ordered repeated stripe occurrences,
a split phase-filter/negative-log chain, typed Sobol search, tuning and held-out windows,
blinded previews, successive halving, and finalist-only full output. Replace its dataset,
calibration, center, window, scale, and resource values with frozen facts from the target
acquisition before use.

## Active B10 reconstruction driver

`b10_recon_current.py` is the active-use copy ported to the current lftomo
utility API. It reads TIFF stacks or dxchange HDF5, normalizes with B10 flat
files, optionally applies BAC, reconstructs window-by-window, and writes TIFF
slices.

The pre-port `B10Scripts/**/b10_recon.py` files remain untouched. Those files,
paired with `lftomo@ddfcdf6`, are the snapshot-frozen producer for the Tier-2
reference suite. Do not use this active driver to refresh that reference.

Example:

```bash
python examples/b10_recon_current.py scan.h5 gains/ output/ \
  --center 1279.25 --window-size 16
```

For producer-aligned bounded comparisons, pass global half-open detector rows:

```bash
python examples/b10_recon_current.py scan.h5 gains/ output/ \
  --center 1282.75 --window-size 12 --y-start 1068 --y-stop 1092
```

Output filenames retain global row coordinates.
