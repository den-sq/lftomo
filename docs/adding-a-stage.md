# Adding a Stage

This guide adds a small sinogram-scaling Stage and a runnable test. Work from
the repository root in the
[development environment](https://github.com/den-sq/lftomo#installation). The example
keeps the arithmetic simple so the contract, registration, and execution path
are visible.

## Choose the execution pattern

`StageContract.locality` describes independent numerical work. Dispatch is a
separate declaration in `lftomo/pipeline/dispatch.py`:

| Work | Stage pattern | Dispatch group | Existing example |
| --- | --- | --- | --- |
| Independent projections | `Locality.PROJECTION`; `run(state)` handles the supplied projections, including a one-projection view. | `_PROJECTION_PROCESS` | `BilateralFilterStage` |
| Independent detector rows | `Locality.SINOGRAM`; implement `run_sinogram` and delegate `run` to the public helper below. Declare `bounded_views_from=(ProjOrder,)` when projection-order row views are supported. | `_BOUNDED_PROCESS` | `VoSortStripeStage` |
| A new output allocation | Allocate through `allocate_output`, fill it, then publish through `commit_output`; derive geometry from `BufferSpec` and `TomoWindow`. | `_OUT_OF_PLACE_PROCESS` for an existing supported fill path | `LayoutConvertStage` |
| An external engine owns parallelism | `run(state)` calls the engine with its assigned core budget; declare runtime dependencies and resource needs. | `_ENGINE` | `GridrecStage` |

The out-of-place group classifies a Stage; it does not generate a worker fill
implementation. A new fill algorithm must extend the existing managed fill
path. Keep Stage classes and worker callables at module scope so process workers
can import them. Reuse the run's storage manager and executor for owned scratch
and parallel work. See [Pipeline API](api-pipeline.md) for lifecycle details.

## Implement the row kernel

Create `lftomo/pipeline/stages/example_sino_scale.py`:

```python
import numpy as np

from lftomo.config import StageConfig
from lftomo.pipeline import (
    DataDomain, Locality, PRESERVE, PipelineState, Stage, StageContract,
    run_sinogram_stage,
)
from lftomo.util.shapes import ProjOrder, SinoOrder


class SinogramScaleStage(Stage):
    name = "example.sino_scale"
    contract = StageContract(
        accepts=(SinoOrder,), prefers=SinoOrder, produces=PRESERVE,
        accepts_domain=(DataDomain.ATTENUATION,), produces_domain=PRESERVE,
        locality=Locality.SINOGRAM, bounded_views_from=(ProjOrder,),
        mutates_input=True, cacheable_prefix=True, window_axes=("Y",),
    )

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    @classmethod
    def from_config(cls, config: StageConfig):
        return cls(**config.params)

    def run(self, state: PipelineState) -> PipelineState:
        return run_sinogram_stage(self, state)

    def run_sinogram(self, sinogram: np.ndarray, state: PipelineState) -> None:
        sinogram *= self.gain
```

`run_sinogram_stage` calls the kernel once per detector row, with a `(Theta, X)`
NumPy view and a row-scoped state. Its `BufferSpec` and `source_window` come from
the canonical unit iterator; use them for geometry and source coordinates.
The view may be strided. Mutate it and return `None`, or return a replacement
array which the helper copies into that row. The helper returns the original
parent state and creates no executor. `Pipeline` invokes the same kernel for
bounded chains through its existing run-scoped executor; ndarray tests run
serially. Stage authors need no private execution imports.

Declare layouts and numerical domains separately: `ProjOrder` and `SinoOrder`
describe memory axes; `RAW`, `TRANSMISSION`, `ATTENUATION`, and `RECON` describe
the values. This example accepts attenuation and preserves both domain and
layout. Set `produces_domain` explicitly when an algorithm changes the domain.
`cacheable_prefix=True` promises deterministic results fully represented by the
config, contract, backend, state inputs, and declared artifacts; leave it false
unless that promise holds. Declare required metadata, artifacts, padding, and
full-axis requirements when the algorithm needs them.

For every new `phase.*` Stage, explicitly set `fft_extent_sensitive=True` if its
FFT result depends on the detector-window extent, or `False` otherwise. The
registry-wide phase declaration test rejects the default `None`. Streaming
reads the instance contract to emit the existing full-height parity warning;
there is no phase-name list to update. This flag describes FFT sensitivity,
while `required_pad` and `requires_full_axes` describe input requirements.

Keep one class-level contract when parameters do not change its facts, as
above. Planning, validation, and execution read `stage.contract` on the
instance. Optional-dependency preflight reads
`StageClass.contract.runtime_requirements` before constructing a Stage, so those
requirements must be invariant and declared on the class. For parameter-based
fields, derive the instance contract with
`dataclasses.replace(type(self).contract, ...)` to retain the class declarations.
The class contract supplies defaults; the instance contract owns the configured
Stage's behavior.

## Register it and configure it

Make both edits before importing the Stage registry; registry/dispatch
completeness is checked at import time.

1. In `lftomo/pipeline/stages/__init__.py`, import
   `SinogramScaleStage` from `.example_sino_scale`, add
   `"example.sino_scale": SinogramScaleStage` to `STAGE_REGISTRY`, and add
   `"SinogramScaleStage"` to `__all__`.
2. In `lftomo/pipeline/dispatch.py`, add `"example.sino_scale"` to
   `_BOUNDED_PROCESS`. Each registry key belongs to exactly one dispatch group.

The YAML entry below can then follow an attenuation-producing Stage in an
existing reconstruction pipeline. `params` is passed to `from_config`; it
remains a free dictionary.

```yaml
- kind: example
  algorithm: sino_scale
  params:
    gain: 2.0
```

## Add the minimum test

Create `tests/pipeline/test_example_sino_scale.py`. It checks configured
construction, the in-place result in both layouts, and the planner's bounded
row path. The input is an explicit arithmetic example; no projection files are
read.

```python
import numpy as np
import pytest

from lftomo.config import (
    Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections,
)
from lftomo.pipeline import (
    BufferSpec, DataDomain, ExecutionKind, Pipeline, PipelineState,
)
from lftomo.pipeline.stages import SinogramScaleStage, create_stage
from lftomo.util.shapes import ProjOrder, SinoOrder


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
def test_sinogram_scale(layout):
    config = StageConfig(kind="example", algorithm="sino_scale", params={"gain": 2})
    scan = Scan(
        energy_keV=20, pixel_size_um=1, n_projections=4, degrees_per_fov=180,
        detector=Detector(pixel_size_um=1, rows=3, columns=5),
        acquisition=Acquisition(n_projections=4, rotation_end_deg=180),
        projections=TIFFStackProjections(format="tiff_stack", path="unused"),
    )
    shape = scan.projection_shape().to(layout)
    working = np.arange(60, dtype=np.float32).reshape(shape.as_tuple())
    expected = working * 2
    state = PipelineState(
        working_array=working, scan=scan, recon=Recon(pipeline=[config]),
        buffer_spec=BufferSpec(shape, working.dtype), domain=DataDomain.ATTENUATION,
    )
    stage = create_stage(config)
    pipeline = Pipeline(
        [stage], initial_layout=layout, initial_domain=state.domain,
    )

    result = pipeline.run(state)

    assert isinstance(stage, SinogramScaleStage)
    assert result.working_array is working
    assert result.layout is layout
    assert result.domain is DataDomain.ATTENUATION
    np.testing.assert_array_equal(working, expected)
    expected_kind = (
        ExecutionKind.BOUNDED_SINOGRAM if layout is ProjOrder
        else ExecutionKind.SEQUENTIAL
    )
    assert pipeline.plan.segments[0].kind is expected_kind
```

Run the example and registration checks:

```bash
python -m pytest -q tests/pipeline/test_example_sino_scale.py \
  tests/pipeline/test_process_dispatch.py::test_registry_dispatch_is_mechanically_complete
```

For a scientific algorithm, replace the scaling arithmetic and its expectation
with a relevant numerical reference. Add managed-storage or dispatch provenance
checks when the change introduces allocations or worker behavior; the existing
`test_managed_storage.py` and `test_bounded_segments.py` cover those seams.
