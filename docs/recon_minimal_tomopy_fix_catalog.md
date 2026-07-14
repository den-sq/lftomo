# recon_minimal TomoPy-derived implementation catalog

This catalog records maintained local implementations in
`den-sq/dan_scripts`. They are coupled to sliding windows, reusable work
buffers, and named shared memory. They are not temporary patches and this
catalog creates no upstreaming or deprecation work.

Line numbers below refer to the surveyed `recon_minimal.py` checkout at the
time of issue #12. The implementation history is useful evidence: commit
`adc06fe` introduced phase retrieval as “original TomoPy values or corrected
paper implementation,” `735f7ff` fixed clipping/minus-log behavior, and
`e317596` introduced the three local Vo implementations with shared work-memory
reuse.

## `phase_retrieval`: Paganin phase retrieval

| Item | Local contract |
| --- | --- |
| Upstream identity | TomoPy Paganin `retrieve_phase` algorithm and its reciprocal-grid/filter helpers |
| Local source | `dan_scripts/prep/phase.py:22-144` |
| Entry point | `phase_retrieval(i, work_mem, phase_mem, pixel_size, energy, dist, tp=False, height_pad=False, thread_fallback=cpu_count())` |
| Window/memory behavior | Worker `i` mutates one `(Y, X)` projection in `SharedNP` storage. A read-only filter is created once in separate shared memory and reused by all workers. Optional vertical edge padding is derived from the physical propagation parameters. Horizontal padding is supplied earlier by the rotation/window path. |
| Phase-2 destination | The corrected-paper beta-valued path is the source for `phase.paganin_phase_absorption`; the standard phase-only option is registered separately as `phase.paganin`. The historical `tp=True` coefficient remains catalog evidence, not a third Stage option. |

Specific correctness behavior that was not in the upstream behavior when this
implementation was last checked:

1. The default filter implements the paper form
   `1 / (1 + distance * alpha * wavelength * |k|² / 4π)` and uses angular
   reciprocal coordinates (`2π / ((N-1) pixel_size)`). The `tp=True` branch
   preserves the older TomoPy coefficient and reciprocal-coordinate convention
   as an explicit, separately named compatibility algorithm.
2. Minus-log is deliberately delayed until after Fourier filtering. Values are
   clipped to the open interval `(1e-8, 0.99999999)` first, preventing NaN/Inf
   output at zero, negative, or saturated pixels. Commit `735f7ff` records the
   earlier failed advanced-index assignment and the corrected indexed write.
3. The corrected-paper result is divided by `4π / wavelength`, returning the
   beta-valued result required by the local pipeline rather than merely a
   filtered transmission image.
4. Padding is based on propagation distance, wavelength, and pixel size. It can
   choose a smaller FFT-friendly composite size instead of forcing the next
   power of two, and uses edge padding rather than zero padding to reduce
   boundary discontinuities.

Efficiency/integration changes go beyond argument marshalling: the filter is
precomputed and normalized once, workers mutate disjoint shared-memory planes,
temporary complex arrays are projection-sized, and NumExpr performs the
minus-log with its thread count reduced inside each process to avoid nested
oversubscription.

The Stage lift must begin with characterization tests around padding and the
`tp` split. The local snapshot calls `_phase_pad` from the worker with the final
two positional arguments in the opposite order from its declaration
(`prep/phase.py:119` versus `:68`); that call-site behavior must be resolved
explicitly rather than accidentally frozen.

## Vo stripe-removal family

All three functions operate on one Y sinogram (`i`) at a time in shared
sinogram memory. `j` selects a disjoint pair of reusable work-buffer slots, so
multiple processes avoid allocating full temporary volumes. They preserve the
Vo algorithm identities but are efficiency-oriented reimplementations, not
thin shared-memory wrappers.

### `remove_sort_vo`

- Local source/signature: `dan_scripts/prep/stripe.py:17-34`,
  `remove_sort_vo(sino_mem, work_mem, i, j, size=21, dim=1)`.
- Upstream identity: TomoPy `remove_stripe_based_sorting` / Vo sorting method.
- Reimplementation: transposes one sinogram into a contiguous work view,
  retains uint32 sort and inverse-sort indices in the paired work slot, applies
  the median filter in sorted space, and writes back in place. Index reuse and
  contiguous layout avoid the upstream function's per-call volume allocation.
- Phase-2 destination: `stripe.vo_sort`.

### `remove_large_vo`

- Local source/signature: `dan_scripts/prep/stripe.py:37-101`,
  `remove_large_vo(sino_mem, work_mem, i, j, snr=3, size=51,
  drop_ratio=0.1, norm=True)`; `_detect_stripe` is at lines 37-59.
- Upstream identity: TomoPy `remove_large_stripe` and Vo stripe detector.
- Reimplementation: clips the drop ratio, excludes sorted outliers from the
  mean, uses guarded division for zero smooth means, dilates the detected mask,
  optionally normalizes unaffected columns, and replaces masked columns from
  the sort-filter-sort-back work plane. The source sinogram copy is deliberate
  because it benchmarks faster than strided repeated access.
- Phase-2 destination: `stripe.vo_large`.

### `remove_dead_vo`

- Local source/signature: `dan_scripts/prep/stripe.py:104-164`,
  `remove_dead_vo(sino_mem, work_mem, i, j, snr=3, size=51, norm=True)`.
- Upstream identity: TomoPy `remove_dead_stripe` followed optionally by the
  large-stripe cleanup.
- Reimplementation: overlays scratch vectors and a boolean mask into otherwise
  unused bytes of the shared work slot, detects deviations from a uniform
  filter, skips interpolation when no bad column exists (the empty case crashed
  in the prior path), interpolates only bad columns, then optionally invokes
  the local large-stripe pass for residual normalization.
- Phase-2 destination: `stripe.vo_dead`.

## Reconstruction and center adapters

These paths are important Stage decisions but are not local mathematical
reimplementations. The catalog distinguishes them so shared-memory integration
is retained without falsely claiming a forked algorithm.

| Local source | What it does | Phase-2 decision |
| --- | --- | --- |
| `recon_minimal.py:319-339` `recon_helper` | Opens shared sinogram memory, reconstructs one slice with `tomopy.recon`, optionally writes it, and returns Shannon entropy. `generate_center_set` dispatches independent centers through a process pool. | `recon.gridrec` remains the upstream TomoPy kernel behind a local window adapter; the entropy sweep belongs to `center.entropy_local`. |
| `recon_minimal.py:532-574` `find_center` | Reads and normalizes selected windows, transposes projection order, and calls `tomopy.find_center_vo` or `tomopy.find_center`; local entropy search is the third option. | `center.vo` and `center.pc` use upstream kernels behind the local window adapter. `center.entropy_local` uses the local sweep. |
| `recon_minimal.py:1195-1211` main reconstruction dispatch | Selects `tomopy.recon(..., algorithm="gridrec")` or `tomopy.astra` with FP/BP/FBP/SIRT/SART/CGLS/EM CUDA methods and local iteration configuration. | `recon.gridrec` and each `recon.*_cuda` remain upstream compute kernels with the local window/shared-memory contract around them. |
| `recon_minimal.py:466-528` `clean_data` | Calls upstream `tomopy.normalize` when local weighted flats are disabled, but locally caps/logs data and dispatches the three Vo implementations by window. | Normalization is not a TomoPy reimplementation; keep `normalize.tomopy` and `normalize.weighted` distinct. Stripe choices use the local implementations above. |

## Canonicalization rules

- Maintain standard phase-only Paganin and the corrected-paper beta-valued
  phase/absorption path as separate algorithm names; do not hide the physical
  output difference behind a backend flag.
- Lift the three Vo implementations with their paired-work-buffer strategy and
  Tier-1 equivalence fixtures before composing `stripe.vo_family`.
- Keep TomoPy gridrec/CUDA and Vo/PC center kernels where they are direct calls;
  the local value is the Stage window adapter, resource control, and artifact
  contract.
- Treat the implementations in this document as locally maintained behavior.
  Their future changes are driven by local regression references and Stage
  contracts.
