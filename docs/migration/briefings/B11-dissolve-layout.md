# B11 — Dissolve pipeline/layout.py

**Wave A · size M · depends on B4, B5**

## Context

`pipeline/layout.py` is the hub of the shadow-geometry system: 8 functions
that hardcode axis positions per layout
(`data.shape[0] if layout is ProjOrder else data.shape[1]`), using DataShape
classes as bare enum tags over raw ndarrays. Worse, the helpers are also
re-inlined at ~16 sites in modules that could import them — the system
leaked even where it was available. This issue makes every helper a
one-line delegation to the B4/B5 primitives (semantics preserved), migrates
the inline copies to the same calls, then deprecates the module.

## Consumer map (what must keep working, verbatim semantics)

| construct | consumers |
|---|---|
| `projection_view` | 26 sites: execution.py ×6, normalize ×7, phase ×8, geometry ×4, center ×2, filter ×1 |
| `n_projections` | 21 call sites + 6 inline re-derivations (execution.py:489, 522, 534, 545, 575) |
| `n_rows` | 14 call sites + 9 inline re-derivations (execution ×3, runner:498, normalize ×6) |
| `sinogram_view` / `sinogram_scratch` | 14 sites (execution, stripe, center) |
| `resolve_layout` / `configured_layout` | 50 sites — obsolete compatibility inputs; remove internal consumers and the shims |
| third system | recon.py:216-219 & 315-318: `theta_axis = 1 if sinogram_order else 0` — layout collapsed to a bool before axis math; migrate to `axis_of` |

## Work items

1. Reimplement each layout.py helper as a one-line delegation:
   `projection_view(data, layout, i)` → `data[layout.point_index(Theta=i)]`,
   `n_projections(data, layout)` → `data.shape[layout.axis_of("Theta")]`,
   `n_rows` → `axis_of("Y")`, sinogram variants likewise. Semantics
   byte-identical; the helpers become shims.
2. Migrate the **inline** copies to the same primitive calls:
   execution.py ×10, planning.py ×4, runner.py ×2, recon.py's
   `theta_axis`/`slice_axis` blocks. (Planning is the strongest case — it
   holds real instances via `BufferSpec.shape` post-B5 and its positional
   arithmetic discards extents it already has.)
3. Migrate the 6 positional `.index[<literal>]` sites
   (streaming.py:91-93, runner.py:481/501, flat.py:48) to
   `axis_slice("Y")` — windows are in scope at all of them.
4. Remove processing-stage `order`, `array_order`, and `sinogram_order`
	compatibility inputs. Readers retain their boundary layout declaration and
	explicit conversion stages retain their target declaration. Deprecate
	`pipeline/layout.py`; its named-axis view shims remain importable until
	removal after Wave B.
5. Delete `pipeline/stages/_layout.py` — the re-export shim has **zero
   importers** (verified in the sweep); dead code, delete outright.

Out of scope: the cross-process kernel entry points
(execution.py:72-76, 125-142, 273-335, 663-709) and util/io.py:181-194 —
workers receive `ManagedBuffer` + layout class with no instance marshalled.
That is a marshalling gap, not a vocabulary gap; B6 (unit dispatch) closes
it. Don't force it here.

## Provenance / acceptance

- Provenance test: no production module imports `pipeline/layout` except
  via the deprecation shim; `pipeline/stages/_layout.py` gone.
- Grep-gate (ratchet, counts may only decrease): `.shape[<int>]` and
  layout-conditional axis math (`if layout is`/`if sinogram_order`) outside
  boundary modules.
- All existing numerical tests unchanged — this issue is semantics-
  preserving by construction; any numerical diff is a bug in the migration,
  not an accepted change.
- Net-LOC: should land clearly negative (~16 inline derivations + shim
  deleted vs one-line delegations).
