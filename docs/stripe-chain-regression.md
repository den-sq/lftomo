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

For a float32 volume shaped `(Theta, Y, X)`, a segment containing any local Vo
kernel allocates two manager-owned, rows-proportional work volumes: one float32
plane and one `uint32` sorting-index plane per source/window row. A segment
containing a Bottleneck line median also allocates one zero-initialized float32
pad plane shaped `(Theta, X + 2H)` per row, where `H` is the largest requested
line half-width in the fused segment. All operations copy into the same fixed
interior, so different kernel sizes safely reuse borders that are zeroed once.
Exact managed scratch is therefore `2 * Theta * Y * X * 4`, plus
`Theta * Y * (X + 2H) * 4` when that pad plane is required. Allocation remains
a pure function of the resolved window shape and authored kernels, not the
process-worker count. Each task retains the full buffer handle and held window;
its assigned `DataRange` selects the detector row and matching scratch planes.
The window translates that packed-buffer ordinal to a source position when
needed. These handles remain valid when the run-scoped process
pool is terminated and recreated. Whole-volume mode deliberately uses the same
topology and therefore accepts the correspondingly high ceiling; there is no
fallback or leasing path.

The Bottleneck path is limited to native float32/float64 sinograms whose values
all pass a per-sinogram `isfinite` check. A sinogram containing NaN or positive
or negative infinity retains `medfilt2d`, as do unsupported dtypes and
`vo_sort(dim=2)`'s square kernel. This preserves the earlier non-finite and
two-axis semantics rather than relying on different library behavior.

Projection-order gather and scatter still happen once per row inside each
worker through a managed handle. Sinogram-order rows are direct contiguous
views. ndarray-only hermetic execution is serial and may use ephemeral local
work. The regression suite records scratch allocation and retirement, asserts
worker-count independence and gather/scatter counts, and rejects layout
materializations in a compatible chain's plan.

`benchmarks/vo_family.py` compares the pre-#209 SciPy line median with the
Bottleneck line median under the same reusable work planes and real managed
process dispatcher. It reports elapsed time, exact managed scratch, sampled
private-process-tree memory, and an output fingerprint. Run one arm per process;
`--plan-only` reports B10 whole-volume allocation without attempting it:

```console
python -m benchmarks.vo_family --variant scipy      --operation sort  --shape 1969,12,2560 --workers 2
python -m benchmarks.vo_family --variant bottleneck --operation sort  --shape 1969,12,2560 --workers 2
python -m benchmarks.vo_family --variant bottleneck --operation chain --shape 1969,2160,2560 --plan-only
```

The isolated layout comparison in `benchmarks/order_layout.py` remains
available for representative layout/materialization measurements.

## #198 representative measurement

The completed opt-in measurement used Python 3.14 / NumPy 2.5.1, two process
workers, and a B10-shaped 1,969 × 12 × 2,560 float32 window. Each arm ran in a
fresh process. Times and sampled process-tree private-memory peaks are
observations, not regression gates:

| Operation | `dd4ce95`-equivalent seconds | Reusable seconds | Baseline peak | Reusable peak |
| --- | ---: | ---: | ---: | ---: |
| `vo_dead` | 2.153 | 2.094 | 690.8 MiB | 691.4 MiB |
| `vo_large` | 20.333 | 20.377 | 763.2 MiB | 690.6 MiB |
| `vo_sort` | 8.155 | 7.869 | 739.3 MiB | 655.1 MiB |
| complete chain | 28.908 | 27.634 | 767.6 MiB | 704.4 MiB |

Every corresponding output fingerprint matched exactly. The reusable arm
reported 483,901,440 managed scratch bytes for this window: 241,950,720 float
bytes and the same number of `uint32` index bytes, allocated once each for the
fused segment. The whole-volume report is 87,102,259,200 bytes (approximately
81.12 GiB) for 1,969 × 2,160 × 2,560, preserving the explicitly accepted high
ceiling.

## #209 representative measurement

The #209 before/after measurement used the same Python 3.14 / NumPy 2.5.1,
two-worker managed process dispatcher and B10-shaped 1,969 × 12 × 2,560
float32 window on both sides. The SciPy arm retains the #198 reusable float and
index planes; the Bottleneck arm adds the reusable pad plane. Times and sampled
process-tree private-memory peaks remain observations, not regression gates:

| Operation | SciPy seconds | Bottleneck seconds | Speedup | SciPy peak | Bottleneck peak | Exact fingerprint |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `vo_large` | 30.287 | 15.968 | 1.90× | 692.2 MiB | 925.9 MiB | `(1.000049142880016, 0.02007005319314297, 64.01603303845916)` |
| `vo_sort` | 20.269 | 10.617 | 1.91× | 672.8 MiB | 884.5 MiB | `(1.0000550083786948, 0.020015681260539667, 64.01633862739554)` |
| complete chain | 43.187 | 23.094 | 1.87× | 728.6 MiB | 903.4 MiB | `(1.0000631473376416, 0.02000801065109414, 64.01684959486074)` |

Each SciPy/Bottleneck pair produced the displayed fingerprint exactly. The
SciPy arm reports 483,901,440 managed scratch bytes. The Bottleneck arm reports
730,577,760 bytes
for `vo_large` and the complete chain (size 51), or 727,742,400 bytes for
`vo_sort` (size 21). The B10 whole-volume Bottleneck chain reports
131,503,996,800 bytes (approximately 122.47 GiB); this is the accepted
whole-volume ceiling, not a recommendation to select that mode.

Candidate-space and later experiment code may vary stripe algorithms,
parameters, optional occurrences, and explicitly named structures. It must
preserve this per-candidate execution invariant and must not share mutable
stripe state across candidates.
