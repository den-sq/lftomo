"""Reference parity and canonical ownership of the composite phase stage."""

import json
from pathlib import Path

import numpy as np
import pytest

from lftomo.config import Acquisition, ConeGeometry, Detector, Recon, Scan, TIFFStackProjections
from lftomo.config.phase import multimaterial_config
from lftomo.pipeline import BufferSpec, DataDomain, Pipeline, PipelineRunContext, PipelineState, validate_pipeline
from lftomo.pipeline.dispatch import DispatchKind, STAGE_DISPATCH
from lftomo.pipeline.stages import MultimaterialStage, TIFFStackReadStage, WeightedNormalizeStage
from lftomo.pipeline.stages import phase as phase_stages
from lftomo.storage import ManagedBuffer
from lftomo.util.shapes import ImgSlice, ProjOrder, SinoOrder, Y
from lftomo.util.window import CTFov, TomoDataSet, TomoWindow
from tests.fixtures.phase_multimaterial import compact_multimaterial_case, original_multimaterial_case


def _state(case, *, layout=ProjOrder, window=None, storage=None, executor=None):
	data = case["projections"]
	physical = ProjOrder(*data.shape)
	window = window or TomoWindow.from_bounds(physical, dtype=data.dtype)
	values = data[window.with_pad().index].copy()
	if layout is SinoOrder:
		values = values.transpose(1, 0, 2).copy()
	shape = layout(*values.shape)
	if storage is None:
		working = values
	else:
		working = storage.allocate(shape, values.dtype, role="normalized")
		working.array()[...] = values
	scan = Scan(
		energy_keV=case["energy_kev"], pixel_size_um=case["pixel_m"] * 1e6,
		ssd_cm=case["distance_m"] * 100, n_projections=physical.Theta,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=case["pixel_m"] * 1e6, rows=physical.Y, columns=physical.X),
		acquisition=Acquisition(n_projections=physical.Theta, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="unused"),
	)
	return PipelineState(
		working, scan, Recon(pipeline=[]), storage=storage, executor=executor,
		buffer_spec=BufferSpec(shape, values.dtype), domain=DataDomain.TRANSMISSION,
		dataset=TomoDataSet.from_range(
			CTFov(physical.Theta, 180, 1), physical, Y(physical.Y),
			dtype=data.dtype, angles=np.deg2rad(case["angles_deg"]),
		), source_window=window,
	)


def _stage(case, **params):
	return MultimaterialStage(mu_table=case["mu_table"], delta_table=case["delta_table"], **params)


@pytest.fixture(scope="module")
def reference():
	path = Path(__file__).parents[1] / "fixtures" / "phase_multimaterial_reference.npz"
	with np.load(path) as source:
		return dict(source), json.loads(path.with_suffix(".json").read_text())


def test_stage_contract_has_required_diagnostics_and_composed_halo():
	case = compact_multimaterial_case()
	stage = _stage(case, n_iter=2)
	assert stage.contract.produces_artifacts == ("phase_filter", "correction_norms", "label_change")
	assert stage.contract.managed_artifacts == ("phase_filter",)
	assert stage.contract.produces_domain is DataDomain.ATTENUATION
	assert stage.contract.required_pad.phase_fft_passes == 3
	assert STAGE_DISPATCH[stage.name].kind is DispatchKind.PROCESS
	assert not STAGE_DISPATCH[stage.name].projection_units


@pytest.mark.parametrize("n_iter", [0, 1, 2])
@pytest.mark.parametrize("noncollinear", [False, True])
def test_compact_fresnel_case_matches_unchanged_reference(reference, n_iter, noncollinear):
	arrays, _ = reference
	case = compact_multimaterial_case(noncollinear=noncollinear)
	state = _state(case)
	Pipeline([_stage(case, n_iter=n_iter)]).run(state)
	prefix = f"compact_{'noncollinear' if noncollinear else 'collinear'}_n{n_iter}"
	# The reference stores single-precision retrieval; angle unit conversion
	# at the canonical acquisition boundary may differ at floating-point roundoff.
	np.testing.assert_allclose(state.working_array, arrays[prefix + "_B"], rtol=1e-6, atol=1e-7)
	np.testing.assert_allclose(
		state.artifacts["correction_norms"], arrays[prefix + "_correction_norms"], rtol=1e-6, atol=1e-12,
	)
	np.testing.assert_array_equal(state.artifacts["label_change"], arrays[prefix + "_label_change"])
	assert state.domain is DataDomain.ATTENUATION


def test_original_three_material_fixture_same_rows_and_context_match_reference(reference):
	arrays, _ = reference
	case = original_multimaterial_case()
	state = _state(case)
	Pipeline([_stage(case)]).run(state)
	expected = arrays["original_full_n1_B"]
	np.testing.assert_allclose(state.working_array, expected, rtol=1e-6, atol=1e-7)
	np.testing.assert_allclose(
		state.artifacts["correction_norms"], arrays["original_full_n1_correction_norms"], rtol=1e-6, atol=1e-12,
	)
	assert state.artifacts["label_change"] == []


def test_one_step_damping_is_partial_correction_from_original_baseline(reference):
	arrays, _ = reference
	case = compact_multimaterial_case(noncollinear=True)
	full = _state(case)
	Pipeline([_stage(case, omega=1.0)]).run(full)
	np.testing.assert_allclose(full.working_array, arrays["compact_noncollinear_n1_omega1_B"], rtol=1e-6, atol=1e-7)
	baseline = arrays["compact_noncollinear_n0_B"]
	damped = _state(case)
	Pipeline([_stage(case)]).run(damped)
	np.testing.assert_allclose(
		damped.working_array, baseline - 0.7 * (baseline - full.working_array), rtol=1e-6, atol=1e-7,
	)
	assert not np.array_equal(full.working_array, damped.working_array)


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
def test_full_context_windows_and_strided_angles_match_whole_array(layout):
	case = compact_multimaterial_case(n_rows=4, row_varying=True, noncollinear=True)
	physical = ProjOrder(*case["projections"].shape)
	whole_window = TomoWindow.from_bounds(physical, theta=(0, physical.Theta, 2))
	whole = _state(case, layout=layout, window=whole_window)
	Pipeline([_stage(case, n_iter=2)]).run(whole)
	for start in (0, 2):
		window = TomoWindow.from_bounds(
			physical, y=(start, start + 2), theta=(0, physical.Theta, 2), halo=Y(physical.Y),
		)
		state = _state(case, layout=layout, window=window)
		Pipeline([_stage(case, n_iter=2)]).run(state)
		index = layout.point_index(Y=window.internal.axis_slice("Y"))
		expected = layout.point_index(Y=slice(start, start + 2))
		np.testing.assert_array_equal(np.asarray(state.working_array)[index], np.asarray(whole.working_array)[expected])
		for key in ("correction_norms", "label_change"):
			assert state.artifacts[key] == whole.artifacts[key]
		np.testing.assert_array_equal(state.artifacts["phase_filter"], whole.artifacts["phase_filter"])
		assert state.source_window is window


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
def test_managed_process_dispatch_owns_scratch_and_retires_on_success(monkeypatch, layout):
	case = compact_multimaterial_case(n_rows=4, row_varying=True, noncollinear=True)
	stage = _stage(case, n_iter=2)
	serial = _state(case, layout=layout)
	Pipeline([stage]).run(serial)
	denominator_calls = []
	build_denominator = phase_stages.slope_denominator

	def record_denominator(*args):
		denominator_calls.append(args)
		return build_denominator(*args)
	monkeypatch.setattr(phase_stages, "slope_denominator", record_denominator)
	with PipelineRunContext(max_workers=2) as context:
		state = _state(case, layout=layout, storage=context.storage, executor=context.executor)
		flats = np.ones((4, state.buffer_spec.shape.Y, state.buffer_spec.shape.X), dtype=np.float32)
		state.scan.metadata["flat_fields"] = flats
		state.artifacts["flat_fields"] = context.storage.allocate(
			ImgSlice(*flats.shape[-2:]), np.float32, role="test_flat_fields",
		)
		calls = []
		original = context.executor.submit_all

		def record(kernel, arguments):
			arguments = list(arguments)
			assert all(isinstance(args[0], ManagedBuffer) for args in arguments)
			for args in arguments:
				worker_scan = next(value for value in args if isinstance(value, Scan))
				assert "flat_fields" not in worker_scan.metadata
				if kernel.__name__ in ("run_material_retrieval_kernel", "run_material_correction_kernel"):
					assert isinstance(args[-1], ManagedBuffer)
					assert args[-1] is state.artifacts["phase_filter"]
			calls.append((kernel.__name__, len(arguments)))
			return original(kernel, arguments)
		monkeypatch.setattr(context.executor, "submit_all", record)
		pipeline = Pipeline([stage])
		result = pipeline.run(state)
		np.testing.assert_array_equal(np.asarray(result.working_array), serial.working_array)
		for key in ("correction_norms", "label_change"):
			assert result.artifacts[key] == serial.artifacts[key]
		np.testing.assert_array_equal(np.asarray(result.artifacts["phase_filter"]), serial.artifacts["phase_filter"])
		assert len(denominator_calls) == 1
		assert state.scan.metadata["flat_fields"] is flats
		assert [name for name, _ in calls] == [
			"run_material_retrieval_kernel", "run_material_classification_kernel",
			"run_material_correction_kernel", "run_material_classification_kernel",
			"run_material_correction_kernel", "run_material_classification_kernel",
		]
		expected_roles = {"multimaterial_" + name for name in stage.work_specs(state.buffer_spec.shape)}
		events = context.storage.allocation_events
		assert expected_roles <= {event["role"] for event in events if event["action"] == "retire"}
		plan = validate_pipeline(
			[TIFFStackReadStage(order="projection" if layout is ProjOrder else "sinogram"),
				WeightedNormalizeStage(), stage], state.scan, mode="complete",
			dataset=state.dataset, source_window=state.source_window,
		)
		planned = {event.role: event.bytes for event in plan.allocation_schedule if event.role in expected_roles}
		assert planned == {
			"multimaterial_" + key: shape.nbytes(dtype)
			for key, (shape, dtype) in stage.work_specs(state.buffer_spec.shape).items()
		}
		planned_filter = next(event for event in plan.allocation_schedule if event.role == "phase_filter")
		assert planned_filter.bytes == result.artifacts["phase_filter"].shape.nbytes(np.float64)
		assert all("phase_filter" in event.live_roles for event in plan.allocation_schedule
			if event.role in expected_roles)
		assert len([event for event in events if event["action"] == "allocate"
			and event["role"] == "phase_filter"]) == 1


def test_missing_material_core_retires_scratch_without_publishing_diagnostics():
	case = compact_multimaterial_case()
	with PipelineRunContext(max_workers=1) as context:
		state = _state(case, storage=context.storage, executor=context.executor)
		before = np.asarray(state.working_array).copy()
		stage = MultimaterialStage(mu_table=[50, 120, 10000], delta_table=[1e-7, 2.4e-7, 2e-5])
		with pytest.raises(ValueError, match="no core pixels for class 3"):
			Pipeline([stage]).run(state)
		np.testing.assert_array_equal(np.asarray(state.working_array), before)
		assert "correction_norms" not in state.artifacts
		roles = {"multimaterial_" + key for key in stage.work_specs(state.buffer_spec.shape)}
		assert roles <= {event["role"] for event in context.storage.allocation_events if event["action"] == "retire"}


def test_borrowed_managed_input_without_allocator_uses_serial_scratch(monkeypatch):
	case = compact_multimaterial_case(noncollinear=True)
	expected = _state(case)
	Pipeline([_stage(case)]).run(expected)
	with PipelineRunContext(max_workers=1) as context:
		state = _state(case, storage=context.storage, executor=context.executor)
		state.storage = None

		def unexpected_dispatch(*args):
			raise AssertionError("unmanaged scratch must not be sent as process output")
		monkeypatch.setattr(context.executor, "submit_all", unexpected_dispatch)
		Pipeline([_stage(case)]).run(state)
		np.testing.assert_array_equal(np.asarray(state.working_array), expected.working_array)


def _calibration(case, kind="parallel"):
	geometry = {"kind": "parallel", "pixel_m": case["pixel_m"], "distance_m": case["distance_m"]}
	if kind == "cone":
		geometry = {
			"kind": "cone", "source_to_sample_m": 0.2, "sample_to_detector_m": 0.2,
			"detector_pixel_m": 2 * case["pixel_m"],
		}
	return {
		"schema": "mctutil.phase.material_table", "schema_version": 1,
		"effective_energy_kev": case["energy_kev"], "geometry": geometry,
		"material_table": {"mu": list(case["mu_table"]), "delta": list(case["delta_table"]),
			"line_fit": {"alpha": 1e-8, "slope": 9e-10, "eps": [0, 0]}},
		"calibration": {"parameters": {"weights": [1, 2]}}, "transfer": None,
	}


def _calibration_scan(case, kind):
	scan = _state(case).scan
	if kind == "cone":
		scan = scan.model_copy(update={
			"geometry": ConeGeometry(source_to_sample_m=0.2, sample_to_detector_m=0.2),
			"detector": scan.detector.model_copy(update={"pixel_size_um": 2 * case["pixel_m"] * 1e6}),
		})
	return scan


@pytest.mark.parametrize("kind", ["parallel", "cone"])
def test_export_boundary_preserves_scan_and_refits_with_original_weights(kind):
	case = compact_multimaterial_case(noncollinear=True)
	source_scan = _calibration_scan(case, kind)
	scan_before = source_scan.model_dump()
	artifact = _calibration(case, kind)
	# A transferred table targets this scan even when its source calibration
	# was acquired at a different energy.
	artifact["calibration"]["energy_kev"] = 0.8 * case["energy_kev"]
	before = json.dumps(artifact, sort_keys=True)
	scan, config = multimaterial_config(artifact, source_scan, n_iter=2)
	stage = MultimaterialStage(**config.params)
	assert scan is source_scan
	assert scan.model_dump() == scan_before
	assert stage.params.n_iter == 2
	assert stage.params.weights == (1, 2)
	assert stage.params.line_fit["alpha"] == 0
	assert stage.params.line_fit is stage.params.line_fit
	assert "line_fit" not in stage.params.model_dump()
	assert "geometry" not in stage.params.model_dump()
	assert "calibration" not in stage.params.model_dump()
	assert scan.energy_keV == artifact["effective_energy_kev"]
	assert scan.pixel_size_um == case["pixel_m"] * 1e6
	assert scan.magnification == (2 if kind == "cone" else 1)
	assert json.dumps(artifact, sort_keys=True) == before


@pytest.mark.parametrize("kind", ["parallel", "cone"])
def test_export_boundary_accepts_unit_conversion_roundoff(kind):
	case = compact_multimaterial_case()
	scan = _calibration_scan(case, kind)
	artifact = _calibration(case, kind)
	artifact["effective_energy_kev"] = np.nextafter(artifact["effective_energy_kev"], np.inf)
	for field, value in artifact["geometry"].items():
		if field != "kind":
			artifact["geometry"][field] = np.nextafter(value, np.inf)
	resolved, _ = multimaterial_config(artifact, scan)
	assert resolved is scan


@pytest.mark.parametrize("kind,field", [
	("parallel", "effective_energy_kev"), ("cone", "effective_energy_kev"),
	("parallel", "pixel_m"), ("parallel", "distance_m"),
	("cone", "source_to_sample_m"), ("cone", "sample_to_detector_m"), ("cone", "detector_pixel_m"),
])
def test_export_boundary_rejects_acquisition_mismatches_without_mutation(kind, field):
	case = compact_multimaterial_case()
	scan = _calibration_scan(case, kind)
	scan_before = scan.model_dump()
	artifact = _calibration(case, kind)
	metadata = artifact if field == "effective_energy_kev" else artifact["geometry"]
	metadata[field] *= 1 + 1e-8
	before = json.dumps(artifact, sort_keys=True)
	with pytest.raises(ValueError, match=field):
		multimaterial_config(artifact, scan)
	assert scan.model_dump() == scan_before
	assert json.dumps(artifact, sort_keys=True) == before


@pytest.mark.parametrize("kind", ["parallel", "cone"])
def test_export_boundary_rejects_different_geometry_kind(kind):
	case = compact_multimaterial_case()
	scan = _calibration_scan(case, kind)
	artifact = _calibration(case, "cone" if kind == "parallel" else "parallel")
	with pytest.raises(ValueError, match="kind"):
		multimaterial_config(artifact, scan)


def test_export_boundary_requires_native_cone_match_even_when_effective_geometry_matches():
	case = compact_multimaterial_case()
	reference_scan = _calibration_scan(case, "cone")
	scan = reference_scan.model_copy(update={
		"geometry": ConeGeometry(source_to_sample_m=0.3, sample_to_detector_m=0.15),
		"detector": reference_scan.detector.model_copy(update={"pixel_size_um": 1.5 * case["pixel_m"] * 1e6}),
	})
	assert scan.pixel_size_um == pytest.approx(reference_scan.pixel_size_um)
	assert scan.ssd_cm == pytest.approx(reference_scan.ssd_cm)
	assert scan.magnification != reference_scan.magnification
	with pytest.raises(ValueError, match="source_to_sample_m"):
		multimaterial_config(_calibration(case, "cone"), scan)


def test_export_boundary_rejects_unknown_parallel_propagation_distance():
	case = compact_multimaterial_case()
	scan = _calibration_scan(case, "parallel")
	scan = scan.model_copy(update={"geometry": scan.geometry.model_copy(update={"ssd_cm": None})})
	with pytest.raises(ValueError, match="distance_m"):
		multimaterial_config(_calibration(case), scan)


def test_export_boundary_rejects_unknown_schema_and_extreme_transfer():
	case = compact_multimaterial_case()
	scan = _state(case).scan
	artifact = _calibration(case)
	artifact["schema_version"] = 2
	with pytest.raises(ValueError, match="schema/version"):
		multimaterial_config(artifact, scan)
	artifact["schema_version"] = 1
	artifact["transfer"] = {"extreme_weights": True}
	with pytest.raises(ValueError, match="extreme_weights"):
		multimaterial_config(artifact, scan)
