from importlib.metadata import version

import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	BufferAction,
	DataDomain,
	PipelineRunContext,
	PipelineState,
	working_view,
	validate_pipeline,
)
from lftomo.pipeline.stages import (
	AffineSkewMovementStage,
	DRHTStage,
	GridrecStage,
	HelicalIndexRemapStage,
	LayoutConvertStage,
	NegativeLogStage,
	PaganinFilterStage,
	TIFFStackReadStage,
	WeightedNormalizeStage,
)
from lftomo.storage import ManagedBuffer
from lftomo.util.shapes import ProjOrder, SinoOrder


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
		state = PipelineState(
			raw, _scan(), _recon(), storage=context.storage, layout=ProjOrder,
			domain=DataDomain.RAW,
		)
		result = WeightedNormalizeStage(order="projection").run(state)

		assert isinstance(result.working_array, ManagedBuffer)
		assert isinstance(result.artifacts["flat_fields"], ManagedBuffer)
		assert result.working_array.name != raw.name
		retire = [event for event in context.storage.allocation_events if event["action"] == "retire"]
		assert retire[-1]["name"] == raw.name
		np.testing.assert_allclose(working_view(result), 0.5)


def test_failed_normalization_does_not_retire_current_input():
	scan = _scan()
	scan.metadata["flat_fields"] = np.zeros((4, 1, 4), dtype=np.float32)
	with PipelineRunContext() as context:
		raw = context.storage.allocate(
			ProjOrder(Theta=10, Y=2, X=4), np.uint16, role="raw_staging",
		)
		state = PipelineState(raw, scan, _recon(), storage=context.storage, layout=ProjOrder)
		with pytest.raises(ValueError, match="flat_fields"):
			WeightedNormalizeStage(order="projection").run(state)
		assert not any(
			event["action"] == "retire" and event["name"] == raw.name
			for event in context.storage.allocation_events
		)


def test_external_pin_blocks_named_alias_until_released():
	with PipelineRunContext() as context:
		raw = context.storage.allocate((128,), np.uint16, role="raw_staging")
		context.storage.pin(raw, owner="trial-a")
		context.storage.retire(raw, role="raw_staging")
		pinned = context.storage.allocate(
			(32,), np.float32, role="reconstruction", reuse_from=("raw_staging",),
		)
		assert pinned.name != raw.name

		context.storage.unpin(raw, owner="trial-a")
		reused = context.storage.allocate(
			(32,), np.float32, role="reconstruction:retry", reuse_from=("raw_staging",),
		)
		assert reused.name == raw.name


def test_layout_and_super_resolution_outputs_are_manager_issued():
	with PipelineRunContext() as context:
		working = context.storage.allocate(
			ProjOrder(Theta=3, Y=2, X=4), np.float32, role="normalized",
		)
		working.array()[...] = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
		state = PipelineState(
			working, _scan(projections=3), _recon(), storage=context.storage,
			layout=ProjOrder, domain=DataDomain.TRANSMISSION,
		)
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
		state = PipelineState(
			working, _scan(projections=3), _recon(), storage=context.storage,
			layout=ProjOrder, domain=DataDomain.TRANSMISSION,
		)
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
			layout=ProjOrder, domain=DataDomain.TRANSMISSION,
		)
		result = PaganinFilterStage(order="projection").run(state)
		assert isinstance(result.artifacts["phase_filter"], ManagedBuffer)


def test_planner_emits_liveness_alias_and_three_memory_components():
	scan = _scan()
	stages = [
		TIFFStackReadStage(order="projection"),
		WeightedNormalizeStage(order="projection"),
		PaganinFilterStage(order="projection"),
		NegativeLogStage(order="projection"),
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
	assert plan.peak_memory.shared_memory_bytes > 0
	assert plan.peak_memory.per_worker_private_scratch_bytes > 0
	assert plan.peak_memory.external_engine_temporary_bytes > 0

	unaliased = validate_pipeline(stages, scan, mode="complete", enable_aliasing=False)
	unaliased_recon = next(
		event for event in unaliased.allocation_schedule if event.role == "reconstruction"
	)
	assert unaliased_recon.action is BufferAction.ALLOCATE
	assert unaliased.peak_memory.shared_memory_bytes > plan.peak_memory.shared_memory_bytes

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


def test_reconstruction_crop_keeps_full_physical_allocation_in_plan():
	scan = _scan()
	stages = [
		TIFFStackReadStage(order="projection"),
		WeightedNormalizeStage(order="projection"),
		NegativeLogStage(order="projection"),
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
		WeightedNormalizeStage(order="projection"),
		NegativeLogStage(order="projection"),
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
	assert plan.peak_memory.shared_memory_bytes == sum(
		event.bytes for event in plan.allocation_schedule
		if event.action is BufferAction.ALLOCATE
	)


def _actual_gridrec(*, engine_aliasing: bool):
	shape = ProjOrder(Theta=16, Y=1, X=8)
	angles = np.linspace(0, np.pi, shape.Theta, endpoint=False, dtype=np.float32)
	shape_tuple = (shape.Theta, shape.Y, shape.X)
	input_values = np.exp(-np.linspace(0.05, 0.5, np.prod(shape_tuple), dtype=np.float32)).reshape(shape_tuple)
	with PipelineRunContext(enable_engine_aliasing=engine_aliasing) as context:
		working = context.storage.allocate(shape, np.float32, role="normalized")
		working.array()[...] = input_values
		state = PipelineState(
			working, _scan(projections=shape.Theta, rows=shape.Y, columns=shape.X),
			_recon(engine_aliasing=engine_aliasing), storage=context.storage,
			layout=ProjOrder, domain=DataDomain.ATTENUATION,
			artifacts={"angles": angles},
		)
		result = GridrecStage(center=4, circle_mask=False, ncore=1).run(state)
		return working.name, result.working_array.name, working_view(result).copy()


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
		state = PipelineState(
			raw, scan, _recon(aliasing=aliasing, engine_aliasing=False),
			storage=context.storage, layout=ProjOrder, domain=DataDomain.RAW,
		)
		state = WeightedNormalizeStage(order="projection").run(state)
		state.domain = DataDomain.TRANSMISSION
		state = NegativeLogStage(order="projection").run(state)
		state.domain = DataDomain.ATTENUATION
		state = GridrecStage(center=4, circle_mask=False, ncore=1).run(state)
		return raw.name, state.working_array.name, working_view(state).copy()


def test_raw_staging_alias_matches_unaliased_full_numeric_pipeline():
	raw_name, aliased_name, aliased = _normalization_to_recon_pipeline(aliasing=True)
	unaliased_raw, unaliased_name, unaliased = _normalization_to_recon_pipeline(aliasing=False)

	assert aliased_name == raw_name
	assert unaliased_name != unaliased_raw
	np.testing.assert_allclose(aliased, unaliased, rtol=1e-5, atol=1e-5)
