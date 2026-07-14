# Paganin phase algorithm options

The `phase` Stage class registers two physically distinct options. They do not
share a Stage body and are not backend variants of one algorithm.

## `phase.paganin`

Standard single-distance Paganin phase retrieval for approximately homogeneous
objects. It applies the low-pass phase transfer function to each projection and
returns the negative-log projected signal. The canonical implementation starts
from `lftomo.prep.phase.phase_retrieval` but uses NumPy-backed Pipeline state
directly.

## `phase.paganin_phase_absorption`

One-shot coupled phase/absorption retrieval. The finalized spelling is
`paganin_phase_absorption`. Its source is Sorotassu's corrected-paper branch in
`dan_scripts/prep/phase.py:22-144`, introduced by `dan_scripts@adc06fe` and
corrected for clipping/minus-log behavior by `735f7ff`. It delays minus-log
until after propagation filtering, clips invalid transmission, and scales by
`4π / wavelength` to return the local beta-valued result.

Both are based on the homogeneous-object, single-distance method described by
Paganin et al., “Simultaneous phase and amplitude extraction from a single
defocused image of a homogeneous object,” *Journal of Microscopy* 206 (2002),
DOI `10.1046/j.1365-2818.2002.01010.x`. The standard option is appropriate when
the projected phase signal is the desired pipeline quantity; the coupled option
is for the producer lineage that consumes Sorotassu's beta-scaled
phase/absorption result.
