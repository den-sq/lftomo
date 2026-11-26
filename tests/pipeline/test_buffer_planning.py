from importlib.metadata import version

import h5py
import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	BufferAction,
	DataDomain,
	PipelineRunContext,
	PipelineState,
	WindowConfig,
	validate_pipeline,
)
from lftomo.pipeline.stages import (
	AffineSkewMovementStage,
	DRHTStage,
	GridrecStage,
	HelicalIndexRemapStage,
	LayoutConvertStage,
	ManualCenterStage,
	NegativeLogStage,
	PaganinFilterStage,
	TIFFStackReadStage,
	VoLargeStripeStage,
	VoSortStripeStage,
	WeightedNormalizeStage,
)
from lftomo.pipeline.planning import _read_spec
from lftomo.storage import ManagedBuffer
from lftomo.runner import run_reconstruction
from lftomo.util.mem import SharedNPManager
from lftomo.util.shapes import ProjOrder, SinoOrder, X
from tests.fixtures.tier1 import shepp_logan
from tests.pipeline.test_streaming import _write_yaml
from tests.sample_data import attach_source_geometry


def _scan(*, projections: int = 10, rows: int = 2, columns: int = 4) -> Scan:
	flats = np.stack([
		np.full((rows, columns), 20, dtype=np.float32),
		np.full((rows, columns), 20, dtype=np.float32),
		np.zeros((rows, columns), dtype=np.float32),
		np.zeros((rows, columns), dtype=np.float32),
	])
	return Scan(
		energy_keV=20,
		pixel_size_um=1,
		ssd_cm=10,
		n_projections=projections,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1, rows=rows, columns=columns),
		acquisition=Acquisition(n_projections=projections, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		metadata={"flat_fields": flats},
	)


def _recon(*, aliasing: bool = True, engine_aliasing: bool = True) -> Recon:
	return Recon(
		pipeline=[StageConfig(kind="test")],
		buffer_aliasing=aliasing,
		tomopy_input_aliasing=engine_aliasing,
	)


def test_retirement_occurs_only_after_normalization_handoff():
	with PipelineRunContext() as context:
		raw = context.storage.allocate(
			ProjOrder(Theta=10, Y=2, X=4), np.uint16, role="raw_staging",
		)
		raw.array()[...] = 10
		state = attach_source_geometry(PipelineState(
			raw, _scan(), _recon(), storage=context.storage,
			domain=DataDomain.RAW,
		))
		result = WeightedNormalizeStage().run(state)

		assert isinstance(result.working_array, ManagedBuffer)
		assert isinstance(result.artifacts["flat_fields"], ManagedBuffer)
		assert result.working_array.name != raw.name
		retire = [event for event in context.storage.allocation_events if event["action"] == "retire"]
		assert retire[-1]["name"] == raw.name
		np.testing.assert_allclose(np.asarray(result.working_array), 0.5)


def test_failed_normalization_does_not_retire_current_input():
	scan = _scan()
	scan.metadata["flat_fields"] = np.zeros((4, 1, 4), dtype=np.float32)
	with PipelineRunContext() as context:
		raw = context.storage.allocate(
			ProjOrder(Theta=10, Y=2, X=4), np.uint16, role="raw_staging",
		)
		state = attach_source_geometry(PipelineState(
			raw, scan, _recon(), storage=context.storage,
		))
		with pytest.raises(ValueError, match="flat_fields"):
			WeightedNormalizeStage().run(state)
		assert not any(
			event["action"] == "retire" and event["name"] == raw.name
			for event in context.storage.allocation_events
		)


def test_external_pin_blocks_named_alias_until_released():
	with PipelineRunContext() as context:
		raw = context.storage.allocate(X(128), np.uint16, role="raw_staging")
		context.storage.pin(raw, owner="trial-a")
		context.storage.retire(raw, role="raw_staging")
		pinned = context.storage.allocate(
			X(32), np.float32, role="reconstruction", reuse_from=("raw_staging",),
		)
		assert pinned.name != raw.name

		context.storage.unpin(raw, owner="trial-a")
		reused = context.storage.allocate(
			X(32), np.float32, role="reconstruction:retry", reuse_from=("raw_staging",),
		)
		assert reused.name == raw.name


def test_layout_and_super_resolution_outputs_are_manager_issued():
	with PipelineRunContext() as context:
		working = context.storage.allocate(
			ProjOrder(Theta=3, Y=2, X=4), np.float32, role="normalized",
		)
		working.array()[...] = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
		state = attach_source_geometry(PipelineState(
			working, _scan(projections=3), _recon(), storage=context.storage,
			domain=DataDomain.TRANSMISSION,
		))
		converted = LayoutConvertStage("sinogram", "bounded sinogram kernels").run(state)
		assert isinstance(converted.working_array, ManagedBuffer)
		assert converted.working_array.name != working.name

		restored = DRHTStage(allow_bicubic_fallback=True).run(converted)
		assert isinstance(restored.working_array, ManagedBuffer)
		assert restored.working_array.name != converted.working_array.name


def test_fused_geometry_and_helical_outputs_are_manager_issued():
	with PipelineRunContext() as context:
		working = context.storage.allocate(
			ProjOrder(Theta=3, Y=2, X=4), np.float32, role="normalized",
		)
		working.array()[...] = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
		state = attach_source_geometry(PipelineState(
			working, _scan(projections=3), _recon(), storage=context.storage,
			domain=DataDomain.TRANSMISSION,
		))
		geometry = AffineSkewMovementStage(
			transition_to="sinogram", pad_x=1,
		).run(state)
		assert isinstance(geometry.working_array, ManagedBuffer)
		assert geometry.working_array.name != working.name

		remapped = HelicalIndexRemapStage(
			pitch_px_per_projection=0, crop_valid=True,
		).run(geometry)
		assert isinstance(remapped.working_array, ManagedBuffer)
		assert remapped.working_array.name != geometry.working_array.name


def test_phase_filter_is_a_manager_issued_producer_artifact():
	with PipelineRunContext() as context:
		working = context.storage.allocate(
			ProjOrder(Theta=3, Y=2, X=4), np.float32, role="normalized",
		)
		working.array()[...] = 0.8
		state = PipelineState(
			working, _scan(projections=3), _recon(), storage=context.storage,
			domain=DataDomain.TRANSMISSION,
		)
		result = PaganinFilterStage().run(state)
		assert isinstance(result.artifacts["phase_filter"], ManagedBuffer)
		assert sum(
			event["action"] == "allocate" and event["role"] == "phase_filter"
			for event in context.storage.allocation_events
		) == 1


def test_planner_emits_liveness_alias_and_managed_artifact_allocations():
	scan = _scan()
	stages = [
		TIFFStackReadStage(order="projection"),
		WeightedNormalizeStage(),
		PaganinFilterStage(),
		NegativeLogStage(),
		GridrecStage(center=2, circle_mask=False),
	]
	plan = validate_pipeline(stages, scan, mode="complete")

	recon_alias = next(
		event for event in plan.allocation_schedule
		if event.role == "reconstruction" and event.action is BufferAction.ALIAS
	)
	assert recon_alias.physical_role == "raw_staging"
	assert recon_alias.proof == "internal_liveness"
	assert any(event.role == "flat_fields" for event in plan.allocation_schedule)
	assert any(event.role == "phase_filter" for event in plan.allocation_schedule)
	allocated = sum(
		event.bytes for event in plan.allocation_schedule
		if event.action is BufferAction.ALLOCATE
	)
	assert allocated > 0
	assert plan.shared_memory_peak_bytes == allocated

	unaliased = validate_pipeline(stages, scan, mode="complete", enable_aliasing=False)
	unaliased_recon = next(
		event for event in unaliased.allocation_schedule if event.role == "reconstruction"
	)
	assert unaliased_recon.action is BufferAction.ALLOCATE
	unaliased_allocated = sum(
		event.bytes for event in unaliased.allocation_schedule
		if event.action is BufferAction.ALLOCATE
	)
	assert unaliased_allocated > allocated
	assert unaliased.shared_memory_peak_bytes == unaliased_allocated

	pinned = validate_pipeline(
		stages, scan, mode="complete", external_pins=("raw_staging",),
	)
	pinned_recon = next(event for event in pinned.allocation_schedule if event.role == "reconstruction")
	assert pinned_recon.proof == "tomopy_input_materialization"

	fully_pinned = validate_pipeline(
		stages, scan, mode="complete", external_pins=("raw_staging", "normalized"),
	)
	fully_pinned_recon = next(
		event for event in fully_pinned.allocation_schedule if event.role == "reconstruction"
	)
	assert fully_pinned_recon.action is BufferAction.ALLOCATE


@pytest.mark.parametrize("sizes", [(3, 7), (7, 3)])
def test_bounded_chain_reserves_shared_work_at_maximum_member_capacity(sizes):
	scan = _scan(projections=12, rows=2, columns=16)
	prefix = [
		TIFFStackReadStage(order="projection"), WeightedNormalizeStage(), NegativeLogStage(),
	]
	baseline = validate_pipeline(prefix, scan, enable_aliasing=False)
	plan = validate_pipeline(prefix + [
		VoSortStripeStage(size=sizes[0]), VoLargeStripeStage(size=sizes[1]),
	], scan, enable_aliasing=False)

	work_bytes = {
		"vo_float_work": 2 * 12 * 16 * 4,
		"vo_index_work": 2 * 12 * 16 * 4,
		"vo_median_pad_work": 2 * 12 * (16 + 6) * 4,
	}
	allocations = [
		event for event in plan.allocation_schedule
		if event.action is BufferAction.ALLOCATE and event.role in work_bytes
	]
	assert len(allocations) == 3
	assert {event.role: event.bytes for event in allocations} == work_bytes
	assert plan.shared_memory_peak_bytes - baseline.shared_memory_peak_bytes == sum(work_bytes.values())


@pytest.mark.parametrize("order", ["projection", "sinogram"])
@pytest.mark.parametrize("aliasing", [False, True])
def test_separate_vo_dispatches_account_only_managed_work(order, aliasing):
	scan = _scan(projections=12, rows=2, columns=16)
	prefix = [TIFFStackReadStage(order=order), WeightedNormalizeStage(), NegativeLogStage()]
	baseline = validate_pipeline(prefix, scan, enable_aliasing=aliasing)
	plan = validate_pipeline(prefix + [
		VoSortStripeStage(size=7), ManualCenterStage(value_px=8), VoLargeStripeStage(size=3),
	], scan, enable_aliasing=aliasing)

	larger_work = 2 * 12 * (16 + 16 + 22) * 4
	smaller_work = 2 * 12 * (16 + 16 + 18) * 4
	expected = larger_work if aliasing else larger_work + smaller_work
	if order == "sinogram":
		expected = 0
	assert plan.shared_memory_peak_bytes - baseline.shared_memory_peak_bytes == expected


def test_read_planning_uses_canonical_strided_window_geometry():
	spec = _read_spec(
		TIFFStackReadStage(
			order="sinogram", y_start=1, y_stop=5,
			theta_start=1, theta_stop=10, theta_stride=3,
		),
		_scan(projections=10, rows=6, columns=4),
	)

	assert spec.shape == SinoOrder(Y=4, Theta=3, X=4)


def test_reconstruction_crop_keeps_full_physical_allocation_in_plan():
	scan = _scan()
	stages = [
		TIFFStackReadStage(order="projection"),
		WeightedNormalizeStage(),
		NegativeLogStage(),
		GridrecStage(
			center=2, circle_mask=False, edge_buffer_left=1,
			edge_buffer_right=1, output_width=1,
		),
	]
	plan = validate_pipeline(stages, scan, mode="complete", enable_aliasing=False)
	recon_step = plan.steps[-1]
	recon_event = next(event for event in plan.allocation_schedule if event.role == "reconstruction")

	assert recon_step.output_buffer.shape.X == 1
	assert recon_step.allocation_buffer.shape.X == 4
	assert recon_event.bytes == 2 * 4 * 4 * np.dtype(np.float32).itemsize


def test_engine_alias_is_refused_when_normalized_capacity_is_too_small():
	"""Planner and runtime enforce the same capacity rule for engine aliases."""

	scan = _scan(projections=6, rows=1, columns=16)
	stages = [
		TIFFStackReadStage(order="projection"),
		WeightedNormalizeStage(),
		NegativeLogStage(),
		GridrecStage(center=8, circle_mask=False),
	]
	plan = validate_pipeline(
		stages,
		scan,
		mode="complete",
		external_pins=("raw_staging",),
	)
	reconstruction = next(
		event for event in plan.allocation_schedule
		if event.role == "reconstruction"
	)
	normalized = next(
		event for event in plan.allocation_schedule
		if event.role == "normalized" and event.action is BufferAction.ALLOCATE
	)

	assert normalized.bytes < reconstruction.bytes
	assert reconstruction.action is BufferAction.ALLOCATE
	assert reconstruction.physical_role == "reconstruction"
	assert sum(
		event.bytes for event in plan.allocation_schedule
		if event.action is BufferAction.ALLOCATE
	) > reconstruction.bytes


def _actual_gridrec(*, engine_aliasing: bool):
	shape = ProjOrder(Theta=16, Y=1, X=8)
	shape_tuple = (shape.Theta, shape.Y, shape.X)
	input_values = np.exp(-np.linspace(0.05, 0.5, np.prod(shape_tuple), dtype=np.float32)).reshape(shape_tuple)
	with PipelineRunContext(enable_engine_aliasing=engine_aliasing) as context:
		working = context.storage.allocate(shape, np.float32, role="normalized")
		working.array()[...] = input_values
		state = PipelineState(
			working, _scan(projections=shape.Theta, rows=shape.Y, columns=shape.X),
			_recon(engine_aliasing=engine_aliasing), storage=context.storage,
			domain=DataDomain.ATTENUATION,
		)
		result = GridrecStage(center=4, circle_mask=False, ncore=1).run(state)
		return working.name, result.working_array.name, np.asarray(result.working_array).copy()


def test_installed_tomopy_runtime_proves_projection_input_alias_parity():
	assert tuple(int(part) for part in version("tomopy").split(".")[:3]) >= (1, 15, 3)
	input_name, aliased_name, aliased = _actual_gridrec(engine_aliasing=True)
	unaliased_input_name, unaliased_name, unaliased = _actual_gridrec(engine_aliasing=False)

	assert aliased_name == input_name
	assert unaliased_name != unaliased_input_name
	np.testing.assert_allclose(aliased, unaliased, rtol=1e-5, atol=1e-5)


def _normalization_to_recon_pipeline(*, aliasing: bool):
	shape = ProjOrder(Theta=16, Y=1, X=8)
	scan = _scan(projections=shape.Theta, rows=shape.Y, columns=shape.X)
	with PipelineRunContext(
		enable_aliasing=aliasing, enable_engine_aliasing=False,
	) as context:
		raw = context.storage.allocate(shape, np.uint16, role="raw_staging")
		raw.array()[...] = 10
		state = attach_source_geometry(PipelineState(
			raw, scan, _recon(aliasing=aliasing, engine_aliasing=False),
			storage=context.storage, domain=DataDomain.RAW,
		))
		state = WeightedNormalizeStage().run(state)
		state.domain = DataDomain.TRANSMISSION
		state = NegativeLogStage().run(state)
		state.domain = DataDomain.ATTENUATION
		state = GridrecStage(center=4, circle_mask=False, ncore=1).run(state)
		return raw.name, state.working_array.name, np.asarray(state.working_array).copy()


def test_raw_staging_alias_matches_unaliased_full_numeric_pipeline():
	raw_name, aliased_name, aliased = _normalization_to_recon_pipeline(aliasing=True)
	unaliased_raw, unaliased_name, unaliased = _normalization_to_recon_pipeline(aliasing=False)

	assert aliased_name == raw_name
	assert unaliased_name != unaliased_raw
	np.testing.assert_allclose(aliased, unaliased, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("aliasing", [False, True])
@pytest.mark.parametrize("center_prepass", [False, True])
def test_streaming_capacity_accounts_for_reservations_shared_between_segments(
	tmp_path, monkeypatch, aliasing, center_prepass,
):
	import tomopy

	rows, columns, theta = 32, 8, 3
	volume = np.repeat(shepp_logan(columns)[None, ...], rows, axis=0)
	angles = np.linspace(0, np.pi, theta, endpoint=False, dtype=np.float32)
	attenuation = tomopy.project(volume, angles, pad=False, ncore=1)
	projections = np.rint(1000 * np.exp(-0.05 * attenuation)).astype(np.uint16)
	data_path = tmp_path / "scan.h5"
	with h5py.File(data_path, "w") as handle:
		handle["exchange/data"] = projections
		handle["exchange/data_white"] = np.full((2, rows, columns), 1000, dtype=np.uint16)
		handle["exchange/data_dark"] = np.zeros((2, rows, columns), dtype=np.uint16)
	scan_path = tmp_path / "scan.yaml"
	_write_yaml(scan_path, {
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": theta, "degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": rows, "columns": columns},
		"acquisition": {"n_projections": theta, "rotation_end_deg": 180},
		"projections": {
			"format": "hdf5_dxchange", "path": str(data_path),
			"data_path": "exchange/data", "white_path": "exchange/data_white",
			"dark_path": "exchange/data_dark",
		},
	})
	pipeline_path = tmp_path / "pipeline.yaml"
	_write_yaml(pipeline_path, {"process_workers": 1, "buffer_aliasing": aliasing, "pipeline": [
		{"kind": "read", "algorithm": "hdf5_dxchange"},
		{"kind": "normalize", "algorithm": "weighted"},
		{"kind": "phase", "algorithm": "paganin_filter", "params": {
			"slope": 1e-10, "fft_pad": False,
		}},
		{"kind": "phase", "algorithm": "paganin", "params": {
			"slope": 1e-10, "fft_pad": True,
		}},
		*([{"kind": "center", "algorithm": "pc"}] if center_prepass else []),
		{"kind": "recon", "algorithm": "gridrec", "params": {"center": 4, "ncore": 1}},
	]})
	centers = []
	def fixed_center(*args, **kwargs):
		centers.append(4.0)
		return 4.0
	monkeypatch.setattr(tomopy, "find_center_pc", fixed_center)
	capacity_checks = []
	monkeypatch.setattr(SharedNPManager, "check_capacity", capacity_checks.append)
	result, = run_reconstruction(
		pipeline_path, scan_path, window_config=WindowConfig(rows=2, theta=1),
	)
	assert len(capacity_checks) == 1
	assert bool(centers) is center_prepass
	assert 0 < result.artifacts["measured_shared_memory_bytes"] <= capacity_checks[0]
