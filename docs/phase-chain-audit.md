# Phase-operation semantics and composition decision

This is the decision record for issue #124 and the scientific input to issues
#125 and #126. The audit treats an algorithm's numerical operations separately
from the compatibility Stage that currently fuses them.

## Decision

The canonical Paganin and BAC paths are already multi-operation phase chains,
but their filter/correction and terminal negative-log operations are fused into
one Stage. They must be available in decomposed form:

```text
normalized transmission
  -> zero or more transmission-preserving phase filters/corrections
  -> exactly one negative-log conversion
  -> optional attenuation-preserving output scaling
```

All audited intermediates remain dimensionless transmission-like data. Retain
the monotonic `RAW -> TRANSMISSION -> ATTENUATION -> RECON` domain model; no
off-line projected-phase domain or explicit transition graph is required.
Domain compatibility already enforces every scientifically invalid ordering
identified by this audit, so no generic prerequisite/effect mechanism is
required either. Issue #125 therefore records and tests the sufficiency of the
existing contract rather than adding speculative machinery.

Negative-log and `TRANSMISSION -> ATTENUATION` are one invariant. A phase
filter or correction must not apply negative-log internally on the composable
path. Compatibility algorithms retain their fused behavior for existing YAML.

## Operation semantics

Normalized transmission is the dimensionless flat/dark-corrected ratio
`T = I / I0`. Attenuation is the reconstruction-ready projected domain; the
domain does not assert identical physical scaling for every algorithm.

| Operation | Input, output, units, and numerical behavior | Contract and valid neighbors |
| --- | --- | --- |
| Paganin filter (`phase.paganin_filter`) | Positive normalized transmission in; dimensionless filtered transmission out. Applies the homogeneous single-distance Fourier transfer function. It does not clip, scale, change sign, or apply `-log`. | `TRANSMISSION -> TRANSMISSION`; projection-local; projection and sinogram layouts, preferring projection; mutates when enabled; requires `energy_keV`, `pixel_size_um`, and `ssd_cm`; about 2x memory. It may precede another transmission correction or the terminal log, but cannot follow that log. |
| BAC correction (`phase.bac_correction`) | Positive normalized hologram/transmission in; corrected transmission `T / (1 - gamma*C)` out, where `C` is the regularized Laplacian correction. Optional input clipping is applied before the correction. The corrected value must remain finite and positive. It does not apply `-log`. | `TRANSMISSION -> TRANSMISSION`; projection-local; both layouts, preferring projection; projection copies with bounded thread concurrency and in-place writeback; no scan metadata; about 2x memory. It may precede another transmission correction or the terminal log, but cannot follow that log. Invalid numerical parameters fail at execution with the value-range cause. |
| Negative-log (`phase.negative_log`) | Positive transmission in; dimensionless `-log(T)` out. Optional `epsilon`/upper clipping occurs immediately before the log; `invert` is an output sign convention. | `TRANSMISSION -> ATTENUATION`; projection-local; both layouts, preferring projection; mutates when enabled; no metadata. This is the only terminal conversion in a decomposed chain. No transmission correction or second log may follow it. |
| Beta scale (`phase.beta_scale`) | Attenuation in; divides by `4*pi/lambda` to produce the historical length-valued coupled phase/absorption projection (centimetres with the current constants). It does not apply `-log`. | `ATTENUATION -> ATTENUATION`; projection-local; both layouts, preferring projection; mutates when enabled; requires `energy_keV`. It may follow the terminal log and cannot precede it. |
| Legacy Paganin (`phase.paganin`) | Fuses Paganin filter, lower clipping at `epsilon`, and `-log`. Output is a dimensionless filtered negative-log projection. | Compatibility `TRANSMISSION -> ATTENUATION` terminal Stage. It remains numerically unchanged and cannot compose with another enabled transmission-only phase operation. |
| Legacy coupled Paganin (`phase.paganin_phase_absorption`) | Fuses Paganin filter, clipping to `[epsilon, 0.99999999]`, `-log`, and beta scaling. Output is the historical length-valued projection. | Compatibility `TRANSMISSION -> ATTENUATION` terminal Stage. It remains numerically unchanged. |
| Legacy BAC (`phase.bac` / `phase.bronnikov_aided`) | HoToPy fuses BAC corrected transmission and `-log`; optional `invert` negates the final output. | Compatibility `TRANSMISSION -> ATTENUATION` terminal Stage. It remains numerically unchanged and retains the optional HoToPy dependency. |

The Paganin interpretation follows
[Paganin et al. (2002)](https://doi.org/10.1046/j.1365-2818.2002.01010.x).
The reciprocal-grid comparison and the decision to retain the DFT-bin
`fftfreq` convention are recorded in
[Paganin reciprocal-grid disposition](paganin-discretization.md).
[TomoPy's phase retrieval](https://tomopy.readthedocs.io/en/latest/_modules/tomopy/prep/phase.html)
likewise exposes the filtered array without applying negative-log. The BAC factorization follows
[De Witte et al. (2009)](https://doi.org/10.1364/JOSAA.26.000890) and HoToPy's
TIE-regime implementation, whose fused result is
`-log(T / (1 - gamma*C))`.

None of these operations produces or consumes a named artifact. All preserve
the current projection/sinogram layout and operate independently on detector
projection images.

## Production-motivated reference chains

The existing fused outputs are the frozen references for their decomposed
counterparts. This is stronger than inventing a new combined pipeline: it
requires the new Stage boundary to reproduce an operation already used in
production.

| Producer | Existing compatibility Stage | Equivalent repeated phase occurrences |
| --- | --- | --- |
| `recon_minimal` / AAA592 | `phase.paganin(alpha=0.06, epsilon=1e-8)` | `phase.paganin_filter(alpha=0.06)` -> `phase.negative_log(epsilon=1e-8)` |
| B10 | `phase.bronnikov_aided(alpha=1.6, gamma=1.2)` | `phase.bac_correction(alpha=1.6, gamma=1.2)` -> `phase.negative_log(epsilon=null)` |
| R2.54 | `phase.bronnikov_aided(alpha=2.4, gamma=1.2)` | `phase.bac_correction(alpha=2.4, gamma=1.2)` -> `phase.negative_log(epsilon=null)` |
| AAA523 | `phase.bronnikov_aided(alpha=2.0, gamma=1.25, clip_input=true)` | `phase.bac_correction(alpha=2.0, gamma=1.25, clip_input=true)` -> `phase.negative_log(epsilon=null)` |
| Coupled phase/absorption fixture | `phase.paganin_phase_absorption(alpha=0.06, epsilon=1e-8)` | `phase.paganin_filter(alpha=0.06)` -> `phase.negative_log(epsilon=1e-8, maximum=0.99999999)` -> `phase.beta_scale` |

The canonical manifests are
[`recon-minimal-canonical.yaml`](https://github.com/den-sq/lftomo/blob/master/lftomo/pipelines/recon-minimal-canonical.yaml),
[`b10-recon-canonical.yaml`](https://github.com/den-sq/lftomo/blob/master/lftomo/pipelines/b10-recon-canonical.yaml),
[`r254-canonical.yaml`](https://github.com/den-sq/lftomo/blob/master/lftomo/pipelines/r254-canonical.yaml),
and
[`phase-contrast-canonical.yaml`](https://github.com/den-sq/lftomo/blob/master/lftomo/pipelines/phase-contrast-canonical.yaml).
The split/legacy fixture comparison covers both supported layouts. Because an
explicit Stage boundary stores a float32 intermediate before the next
operation, the agreed relative tolerance is `3e-7` for the Paganin chains and
`2e-6` for the independent BAC comparison.

The legacy `pad_y` parameter is intentionally not part of the split Paganin
chain: the compatibility Stage currently stores but does not execute it. That
fidelity issue is separate from operation decomposition.

## Supported composition and order

More than one correction algorithm can be authored with repeated stable phase
occurrence IDs:

```yaml
pipeline:
  - kind: phase
    id: paganin-prefilter
    algorithm: paganin_filter
    params: {alpha: 0.06}
  - kind: phase
    id: bac-correction
    algorithm: bac_correction
    params: {alpha: 2.0, gamma: 1.25}
  - kind: phase
    id: attenuation
    algorithm: negative_log
    params: {epsilon: 1.0e-8}
```

Paganin-filter then BAC-correction and BAC-correction then Paganin-filter are
non-commuting but both domain-valid experiment candidates: each operation
still receives positive transmission-like data. Their different results are a
reason to compare candidates, not a reason for the generic validator to reject
one order.

The following orders are scientifically invalid and already fail through the
domain contract:

- any transmission correction after negative-log;
- two terminal negative-log conversions;
- beta scaling before the terminal conversion; and
- a complete reconstruction pipeline with no terminal conversion.

Numerical positivity cannot be proven during static validation. BAC rejects a
non-positive or non-finite corrected intermediate at execution; negative-log
either clips at configured bounds or, with `epsilon: null`, rejects invalid
input.

## Disabled/no-op behavior

Disabled legacy and split phase operations declare `PRESERVE`, do not mutate
or require metadata, and may remain authored on either side of the terminal
boundary. They do not satisfy the required terminal transition. This is an
existing regression invariant, not a separate semantic effect.

## Consequences for #125 and #126

- #125 adds no transition graph or effect fields. Synthetic contract tests
  prove that any number of `TRANSMISSION -> TRANSMISSION` operations compose
  before exactly one `TRANSMISSION -> ATTENUATION` operation and that the
  domain boundary rejects the audited invalid orders.
- #126 adds the split algorithms, numerical equivalence tests, multi-correction
  execution, both-layout coverage, and user-facing authoring guidance while
  preserving all fused compatibility algorithms.
