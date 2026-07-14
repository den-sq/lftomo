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

The benchmark intentionally names a tiny fixed subset rather than enumerating
all 77,000 projection paths. This keeps the check safe for WSL-mounted storage
while exercising the real Tier-2 file layout and filesystem.
