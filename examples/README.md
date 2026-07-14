# Active B10 reconstruction driver

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
