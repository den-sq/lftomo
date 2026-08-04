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
