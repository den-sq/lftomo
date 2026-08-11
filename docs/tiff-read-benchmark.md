# TIFF read Stage benchmark

Issue #23 compares lftomo's current `read_window` / `byteread_helper` path with
the per-file dispatch used by `recon_minimal.distribute_read`.

## Tier-2 result

The benchmark used the mounted `striping_tiff` Tier-2 dataset, four explicitly
named projections (`proj_00000.tif` through `proj_00003.tif`), four full-width
rows at detector Y=1400, four reader threads, and five alternating runs. The
first run was discarded. Both readers produced identical `uint16` projection-
ordered arrays.

| Implementation | Median time |
| --- | ---: |
| `read_window` / `byteread_helper` | 1.599 s |
| `recon_minimal.distribute_read` strategy | 0.365 s |

The per-file strategy was 77% faster in this bounded comparison, well beyond
the project's 10% canonical-selection threshold. `read.tiff_stack` therefore
uses a current-API port named `distribute_tiff_read`: it inspects TIFF metadata
once, dispatches one contiguous read per projection, and supports bounded Y
windows so large Tier-2 scans do not require full-frame buffers.

For sinogram-order working storage, `distribute_tiff_sinogram_read` places each
source row directly at its strided destination in the manager-issued buffer.
The former contiguous projection read followed by a full-volume transpose copy
is no longer part of the Stage path, structurally eliminating that second
full-volume buffer. The removed strategy remains reproducible as the
`transition` arm of `benchmarks/order_layout.py`, alongside the `strided` arm,
so throughput and peak-memory comparisons can still be rerun without retaining
the transition in production. The producer-era
`sinogram_chunk_size`/`chunk_overlap` keys remain accepted as inert provenance
when loading older canonical configurations; window sizing belongs to the
runner-level streaming driver.

The benchmark intentionally names a tiny fixed subset rather than enumerating
all 77,000 projection paths. This keeps the check safe for WSL-mounted storage
while exercising the real Tier-2 file layout and filesystem.
