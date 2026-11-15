# Paganin phase algorithm options

The `phase` Stage class retains two fused compatibility options and exposes
their operations separately for composition.

## `phase.paganin`

Standard single-distance Paganin phase retrieval for approximately homogeneous
objects. It applies the low-pass phase transfer function to each projection and
returns a dimensionless negative-log filtered projection. Existing YAML remains
supported. New pipelines can express the same numerical path as two ordered
occurrences:

```yaml
- kind: phase
  id: filter
  algorithm: paganin_filter
  params: {alpha: 0.06}
- kind: phase
  id: attenuation
  algorithm: negative_log
  params: {epsilon: 1.0e-8}
```

`phase.paganin_filter` preserves dimensionless transmission. Only
`phase.negative_log` performs the terminal conversion to attenuation.

## `phase.paganin_phase_absorption`

One-shot coupled phase/absorption retrieval. The finalized spelling is
`paganin_phase_absorption`. Its source is Sorotassu's corrected-paper branch in
`dan_scripts/prep/phase.py:22-144`, introduced by `dan_scripts@adc06fe` and
corrected for clipping/minus-log behavior by `735f7ff`. It delays minus-log
until after propagation filtering, clips invalid transmission, and scales by
`4π / wavelength` to return a length-valued projection proportional to the
line-integrated absorptive refractive-index component.

Both are based on the homogeneous-object, single-distance method described by
Paganin et al., “Simultaneous phase and amplitude extraction from a single
defocused image of a homogeneous object,” *Journal of Microscopy* 206 (2002),
DOI `10.1046/j.1365-2818.2002.01010.x`. The standard option is appropriate when
the dimensionless filtered negative-log signal is the desired pipeline
quantity. The coupled option preserves Sorotassu's historical beta-scaled
phase/absorption formulation; no frozen canonical producer currently selects
it.

The decomposed equivalent is `phase.paganin_filter`, then
`phase.negative_log` with `maximum: 0.99999999`, then `phase.beta_scale`.

The compatibility algorithms still perform negative-log internally and remain
terminal. Use the decomposed algorithms when another transmission-preserving
phase correction must be inserted. See the [phase-operation audit](phase-chain-audit.md)
for the full semantics, BAC decomposition, supported orders, and tolerances.

## Padding for calibrated material lines

When `phase.paganin` or `phase.paganin_filter` specifies `slope`, its default
`padding: kernel` derives the FFT extent from the low-pass operator. In metres,
the decay length is `L = sqrt(distance_m * slope)`; each edge requires
`ceil(8 * L / pixel_m)` pixels. The continuous kernel's mass outside the square
with half-width `8L` is at most `2 * exp(-8)`, approximately **0.0671%**.
This is a continuous-kernel tail bound, not a guarantee on retrieved-output
error. Discretization, image boundaries, affine correction, and the negative
log still affect the result.

The source-window planner requests real Y context for each retrieval pass
(two passes for an affine defect correction). The numerical FFT canvas then
subtracts the **realized** source halo independently on each Y side, and
edge-fills only the remaining deficit. X receives the full radius on both
sides. Axes needing edge fill round up to a fast FFT length, with any extra
pixels split between the two sides. When real halo covers both Y sides, Y is
neither edge-filled nor rounded. `TomoWindow` owns these extents and the crop
back to the held input buffer; downstream reconstruction trims source halo.

Set `padding: reference` to reproduce the previous slope/affine padding:
`next_fast_len(int(1.5 * dimension))` on each axis, irrespective of real source
halo. This preserves the frozen reference results. `fft_pad: false` disables
the numerical edge fill in either mode and uses a periodic FFT on the held
buffer. The legacy `alpha` path, selected by omitting `slope`, retains its
existing arithmetic, FFT extent, and source-padding behavior.

The shared denominator remains a managed buffer prepared before dispatch.
There is no reusable managed FFT canvas: `np.pad` in `lftomo.prep.phase`, FFT
results, and affine gradient results remain per-call numerical temporaries.
The prep functions `single_material_filter` and `retrieve_single` expose the
same `padding` choice, using `pad=False` for the periodic path.
