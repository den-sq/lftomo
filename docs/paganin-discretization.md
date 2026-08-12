# Paganin reciprocal-grid disposition

Issue #199 compared the current Paganin transfer with the audited
`dan_scripts@f601fc995ffbfc8757447c992cbb3f5fd1e36b19` implementation. The
decision is to **retain `numpy.fft.fftfreq` as the intentional discretization**.
The source grid is characterized and pinned as historical behavior, not adopted.

## Compared formulations

The current transfer uses the angular frequencies corresponding exactly to the
bins returned by `numpy.fft.fft2`:

```text
k = 2π * fftfreq(N, pixel_size)
H = 1 / (1 + distance * alpha * wavelength * |k|² / 4π)
```

This has one true DC bin, `H[0, 0] == 1`, and covers the discrete Fourier
Nyquist interval.

The audited source's `tp=False` branch constructs centered coordinates with an
`N - 1` denominator, applies `fftshift`, and normalizes the resulting transfer
by its maximum. That construction is not the frequency grid of an N-sample
DFT:

- for even N it contains no zero coordinate and max-normalizes four half-bin
  locations to one;
- it spans approximately ±2π/pixel rather than the DFT's ±π/pixel Nyquist
  interval;
- for odd N, shifting the centered grid with `fftshift` introduces the
  one-sample placement difference from the corresponding `ifftshift` mapping.

The current DC response follows from the physical formula and preserves image
mean. The source's maximum normalization conceals the absent even-grid DC
sample, and the odd-grid placement can change image mean. These are not reasons
to reinterpret the source behavior as a more accurate discretization.

## Source-derived characterization

`tests/fixtures/paganin_source.py` pins the audited commit and reproduces only
its reciprocal-grid/transfer behavior. `test_paganin_source_parity.py` compares
that reference with the current transfer across three tuning cases and two
held-out cases. The deterministic fixture includes smooth, edge, and periodic
content.

| Case | Shape | α | Energy | Pixel | Distance | Relative L2 | Max absolute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tuning low α | 32×48 | 0.02 | 20 keV | 0.637 µm | 11 cm | 0.002838 | 0.011453 |
| tuning reference α | 32×48 | 0.06 | 20 keV | 0.637 µm | 11 cm | 0.005075 | 0.018347 |
| tuning high α | 32×48 | 0.20 | 20 keV | 0.637 µm | 11 cm | 0.009214 | 0.031177 |
| held-out odd shape | 31×47 | 0.06 | 20 keV | 0.637 µm | 11 cm | 0.020920 | 0.029102 |
| held-out geometry | 33×64 | 0.08 | 25.45 keV | 0.7046 µm | 15 cm | 0.015347 | 0.025632 |

The differences are therefore material and shape-dependent, not rounding
noise. Existing split/fused equivalence checks remain useful but answer a
different question: they prove composition parity within the current
discretization.

## Source argument and padding quirks

The source call sites pass filter construction as `(alpha, pixel, distance,
energy)` and projection execution as `(pixel, energy, distance)`; those calls
match their respective signatures and need no compatibility shim. Two internal
behaviors must not be migrated:

1. its height-pad path calls `_phase_pad(..., pixel_size, energy, distance)`,
   swapping the helper's required `distance, energy` order;
2. it slices the Fourier-domain array before `ifft2` instead of inverse
   transforming the padded array and cropping the spatial result.

Current lftomo padding is expressed through the Stage's resolved source-halo
contract. The legacy `pad_y` compatibility parameter does not select either
broken source behavior. The source's `tp=True` coefficient convention is also
catalog evidence rather than another implementation of the current Stage.

## Compatibility result

No production algorithm or #185 reference/oracle anchor changes under this
decision. The completed `dd4ce95` acceptance continues to mean exactly what it
recorded. A future proposal to change the reciprocal discretization would need
a new scientific rationale, an explicit compatibility decision, and deliberate
re-baselining; #199 does not create that follow-up.
