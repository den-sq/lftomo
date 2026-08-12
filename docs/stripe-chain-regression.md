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
plane and one `uint32` sorting-index plane per source/window row. Exact managed
scratch is therefore `2 * Theta * Y * X * 4` bytes. The allocation is a pure
function of the resolved window shape, not the process-worker count. Each task
uses its already-dispatched row index to select the matching planes, so the
handles remain valid when the run-scoped process pool is terminated and
recreated. Whole-volume mode deliberately uses the same topology and therefore
accepts the correspondingly high ceiling; there is no fallback or leasing path.

Projection-order gather and scatter still happen once per row inside each
worker through a managed handle. Sinogram-order rows are direct contiguous
views. ndarray-only hermetic execution is serial and may use ephemeral local
work. The regression suite records scratch allocation and retirement, asserts
worker-count independence and gather/scatter counts, and rejects layout
materializations in a compatible chain's plan.

`benchmarks/vo_family.py` compares allocation-per-kernel and reusable-work
variants for each Vo operation and for the complete chain under the real
managed process dispatcher. It reports elapsed time, exact managed scratch,
sampled private-process-tree memory, and an output fingerprint. Run one arm per
process; `--plan-only` reports B10 whole-volume allocation without attempting
the approximately 81 GiB allocation:

```console
python -m benchmarks.vo_family --variant legacy   --operation sort  --shape 1969,12,2560 --workers 2
python -m benchmarks.vo_family --variant reusable --operation sort  --shape 1969,12,2560 --workers 2
python -m benchmarks.vo_family --variant reusable --operation chain --shape 1969,2160,2560 --plan-only
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

Candidate-space and later experiment code may vary stripe algorithms,
parameters, optional occurrences, and explicitly named structures. It must
preserve this per-candidate execution invariant and must not share mutable
stripe state across candidates.
