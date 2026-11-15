# Adding a Stage

This guide adds a small sinogram-scaling Stage and a runnable test. Work from
the repository root in the
[development environment](https://github.com/den-sq/lftomo#installation). The example
keeps the arithmetic simple so the contract, registration, and execution path
are visible.

## Choose the execution pattern

`SinogramStage` and `ProjectionStage` provide the unit-iteration adapters.
Use plain `Stage` for orchestration or a whole-array operation. Dispatch remains
a separate declaration in `lftomo/pipeline/dispatch.py`:

| Work | Stage pattern | Dispatch group | Existing example |
| --- | --- | --- | --- |
| Independent projections | Extend `ProjectionStage` and implement `run_projection(image, state)` for a `(Y, X)` view; the base handles Theta units. | `_PROJECTION_PROCESS` | `BilateralFilterStage` |
| Independent detector rows | Extend `SinogramStage` and implement `run_sinogram(sinogram, state)` for a `(Theta, X)` view; the base handles Y units and declares bounded views from `ProjOrder`. | `_BOUNDED_PROCESS` | `VoSortStripeStage` |
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
from dataclasses import replace

import numpy as np
from pydantic import Field

from lftomo.pipeline import DataDomain, PipelineState, SinogramStage, StageParams


class SinogramScaleStage(SinogramStage):
	name = "example.sino_scale"
	contract = replace(
		SinogramStage.contract,
		accepts_domain=(DataDomain.ATTENUATION,), cacheable_prefix=True,
	)

	class Params(StageParams):
		gain: float = Field(default=1.0, gt=0)

	def run_sinogram(self, sinogram: np.ndarray, state: PipelineState) -> None:
		sinogram *= self.params.gain
```

`SinogramStage.run` delegates to the public `run_sinogram_stage` adapter, which
calls the kernel once per detector row with a `(Theta, X)` NumPy view and a
row-scoped state. Its `BufferSpec` and `source_window` come from
the canonical unit iterator; use them for geometry and source coordinates.
The view may be strided. Mutate it and return `None`, or return a replacement
array which the helper copies into that row. The helper returns the original
parent state and creates no executor. `Pipeline` invokes the same kernel for
bounded chains through its existing run-scoped executor; ndarray tests run
serially. `ProjectionStage` similarly delegates to `run_projection_stage` for
Theta units. Each base supplies `run`; authors implement the numerical kernel.

For reusable work in a bounded sinogram segment, override
`Stage.bounded_work_requests(input_spec)` and return `BoundedWorkRequest`
values. Each request names its temporary artifact key and allocation role,
and owns a `BufferSpec` plus an optional scalar `fill_value`. Express numerical
padding through `TomoWindow` before deriving the allocation shape. The generic
executor allocates once for the segment through the run's `StorageBackend`,
passes only the current unit's reopenable handle to kernels, and retires it
after all units complete, including on failure. Plain ndarray execution keeps
the existing serial numerical fallback.

Stages sharing a key share one allocation: dtype, layout, initialization and
non-growing extents must agree. A request can explicitly permit larger capacity
along `grow_axes`; Vo median padding uses `("X",)` so a chain shares its largest
requested line canvas. Kernels must accept that capacity. Work must preserve
the dispatched unit-axis extent. The hook takes only the current window's
`BufferSpec`, so `collect_bounded_work(stages, input_spec)` can inspect exact
requirements before execution without reading pixels or allocating storage.
This work is temporary segment scratch, not a `managed_artifacts` producer or
prefix-cache output. Allocation reports are derived from the resolved requests
in `state.artifacts["bounded_work_allocations"]`.

Declare layouts and numerical domains separately: `ProjOrder` and `SinoOrder`
describe memory axes; `RAW`, `TRANSMISSION`, `ATTENUATION`, and `RECON` describe
the values. This example accepts attenuation and preserves both domain and
layout. Set `produces_domain` explicitly when an algorithm changes the domain.
`cacheable_prefix=True` promises deterministic results fully represented by the
config, contract, backend, state inputs, and declared artifacts; leave it false
unless that promise holds. Declare required metadata, artifacts, padding, and
full-axis requirements when the algorithm needs them.

For any Stage whose numerical result depends on input extent, declare the
affected axes in `extent_sensitive_axes`. For example, the phase filters use
`extent_sensitive_axes=("Y", "X")` because their two-dimensional FFTs depend on
both detector extents. Use `()` to explicitly declare no sensitivity. The
default `None` means undeclared; the registry-wide phase declaration test
rejects it for every `phase.*` Stage. Streaming reads the instance contract
and warns when it partitions a declared axis, currently `Y`; there is no
phase-name list to update. This property describes numerical sensitivity:
`required_pad` declares needed source context, while `requires_full_axes`
prohibits splitting an axis. Extent sensitivity allows splitting with the
warning that fixed-window parity does not imply whole-volume equivalence.

The nested `Params` model owns configured values. `StageParams` rejects unknown
keys and is immutable; use `Field` bounds and `Literal` choices for scalar
constraints, and model validators for relationships between fields. The base
constructor validates the supplied parameters into `stage.params` and resolves
`stage.contract` through `contract_for(cls, params)`. Kernels read `self.params`;
there is no second copy on the Stage. Inspect the schema with
`SinogramScaleStage.Params.model_json_schema()`.

Configuration validation imports the Stage registry to read these models.
Keep numerical engines and I/O dependencies inside the methods that use them,
so loading parameters and inspecting schemas works in the lightweight environment.

Keep one class-level contract when parameters do not change its facts, as
above. For parameter-dependent facts, override the `contract_for` classmethod
and return `dataclasses.replace(cls.contract, ...)` using the validated model.
The default returns the class declaration. Planning, validation, and execution
read the resolved instance contract. Optional-dependency preflight reads
`StageClass.contract.runtime_requirements` before construction, so those
requirements stay invariant on the class. The base owns parameter validation
and contract resolution; a Stage needs no constructor or `from_config` wrapper
for the ordinary case.

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
existing reconstruction pipeline. YAML keeps `params` as its serialized mapping;
`StageConfig` validates supplied keys and values against the registered `Params`
model. Required values may remain unresolved in a template until materialization.
`create_stage` requires the complete parameter model and supplies the occurrence
and dispatch metadata.

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
