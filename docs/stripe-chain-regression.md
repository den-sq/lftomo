# Ordered stripe-chain regression invariant

Stripe removal is an authored sequence, not a set. In particular,
`r254-canonical` runs Fourier-wavelet removal followed by Vo sorting, and
`stripe.vo_family` runs its configured Vo operations exactly in list order.
The planner must never reverse or canonicalize these chains.

For compatible adjacent sinogram-local Stages over projection-order storage,
the execution invariant is:

1. gather one contiguous `(Theta, X)` detector-row scratch array;
2. run the complete authored Stage chain on that scratch array;
3. scatter the row once;
4. retain `ProjOrder` and the input data domain.

Tests compare this bounded path with direct contiguous `SinoOrder` execution
at `rtol=1e-6, atol=1e-6`. A reversed FW/Vo chain is also required to produce
a measurably different result on the non-commuting synthetic fixture. A
full-array implementation such as `stripe.all_of_the_stripe` is a planner
barrier and therefore splits adjacent bounded segments.

For a float32 volume shaped `(Theta, Y, X)`, one worker owns at most
`Theta * X * 4` scratch bytes. With `W` bounded workers, simultaneously live
row scratch is bounded by `min(W, Y) * Theta * X * 4`; it must not approach a
second `Theta * Y * X * 4` volume. The regression suite records actual scratch
shapes, asserts gather/scatter counts, caps live workers, and rejects layout
materializations in a compatible chain's plan. The isolated three-arm
`benchmarks/order_layout.py` benchmark remains available for representative
RSS and runtime measurements.

Candidate-space and later experiment code may vary stripe algorithms,
parameters, optional occurrences, and explicitly named structures. It must
preserve this per-candidate execution invariant and must not share mutable
stripe state across candidates.
