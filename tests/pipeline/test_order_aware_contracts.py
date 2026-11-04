import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	BufferSpec,
	CompletePipelineError,
	DataDomain,
	DomainCompatibilityError,
	MaterializationKind,
	PRESERVE,
	Pipeline,
	PipelineState,
	RowIdentityError,
	ShapeCompatibilityError,
	Stage,
	StageContract,
	validate_pipeline,
)
from lftomo.pipeline.stages import LayoutConvertStage, create_stage
from lftomo.util.shapes import ProjOrder, ReconOrder, SinoOrder


def scan_config():
	return Scan(
		energy_keV=24,
		pixel_size_um=1.3,
		n_projections=3,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1.3, rows=2, columns=4),
		acquisition=Acquisition(n_projections=3, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


def recon_config():
	return Recon(pipeline=[StageConfig(kind="echo")])


class ContractStage(Stage):
	def __init__(self, name, contract):
		self.name = name
		self.contract = contract
		if name.startswith("recon."):
			self.edge_buffer_input = False
			self.edge_buffer_left = 0
			self.edge_buffer_right = 0
			self.output_width = None

	def run(self, state):
		return state


def complete_stages(*middle):
	return [
		ContractStage("read.test", StageContract(
			produces=ProjOrder, produces_domain=DataDomain.RAW,
		)),
		ContractStage("normalize.test", StageContract(
			accepts=(ProjOrder,), produces=PRESERVE,
			accepts_domain=(DataDomain.RAW,), produces_domain=DataDomain.TRANSMISSION,
		)),
		*middle,
		ContractStage("phase.test", StageContract(
			accepts=(ProjOrder,), produces=PRESERVE,
			accepts_domain=(DataDomain.TRANSMISSION,), produces_domain=DataDomain.ATTENUATION,
		)),
		ContractStage("recon.test", StageContract(
			accepts=(ProjOrder, SinoOrder), prefers=SinoOrder, produces=ReconOrder,
			accepts_domain=(DataDomain.ATTENUATION,), produces_domain=DataDomain.RECON,
		)),
	]


def test_layout_contract_resolves_canonical_declarations():
	contract = StageContract(accepts=(ProjOrder,), produces=SinoOrder)

	assert contract.accepted_layouts == (ProjOrder,)
	assert contract.preferred_layout is ProjOrder
	assert contract.output_layout(ProjOrder) is SinoOrder


def test_with_array_binds_new_spec_and_preserves_domain():
	array = np.zeros((3, 2, 4))
	state = PipelineState(
		working_array=array, scan=scan_config(), recon=recon_config(),
		buffer_spec=BufferSpec(ProjOrder(*array.shape), array.dtype),
		domain=DataDomain.TRANSMISSION,
	)

	replacement = np.ones((3, 2, 4))
	result = state.with_array(
		replacement, buffer_spec=BufferSpec(ProjOrder(*replacement.shape), replacement.dtype),
	)

	assert result.layout is ProjOrder
	assert "layout" not in result.__dict__
	assert not hasattr(result, "sync_buffer_spec")
	assert result.domain is DataDomain.TRANSMISSION


def test_layout_convert_is_explicit_contiguous_and_visible_in_plan():
	stage = create_stage(StageConfig(
		kind="layout", algorithm="convert",
		params={"target": "sinogram", "reason": "make Vo-family input contiguous"},
	))
	plan = validate_pipeline(
		[stage], scan_config(), initial_layout=ProjOrder,
		initial_domain=DataDomain.TRANSMISSION,
	)
	array = np.arange(24).reshape(3, 2, 4)
	state = PipelineState(
		working_array=array, scan=scan_config(),
		recon=recon_config(),
		buffer_spec=BufferSpec(ProjOrder(*array.shape), array.dtype),
		domain=DataDomain.TRANSMISSION,
	)

	result = Pipeline([stage], validate=False).run(state)

	assert isinstance(stage, LayoutConvertStage)
	assert result.layout is SinoOrder
	assert result.domain is DataDomain.TRANSMISSION
	assert result.working_array.flags.c_contiguous
	assert np.array_equal(result.working_array, np.arange(24).reshape(3, 2, 4).transpose(1, 0, 2))
	assert plan.full_volume_materializations[0].kind is MaterializationKind.LAYOUT_CONVERSION


def test_layout_convert_requires_reason_and_real_transition():
	with pytest.raises(ValueError, match="non-empty reason"):
		LayoutConvertStage(target="sinogram", reason=" ")
	with pytest.raises(ShapeCompatibilityError, match="must change"):
		validate_pipeline(
			[LayoutConvertStage(target="sinogram", reason="test")], scan_config(),
			initial_layout=SinoOrder,
		)


def test_layout_mismatch_never_auto_converts():
	stage = ContractStage("sino-only", StageContract(
		accepts=(SinoOrder,), produces=PRESERVE,
	))

	with pytest.raises(ShapeCompatibilityError, match="explicit layout.convert"):
		validate_pipeline([stage], scan_config(), initial_layout=ProjOrder)


def test_only_explicit_or_fused_stages_may_flip_tomography_layout():
	stage = ContractStage("ordinary", StageContract(
		accepts=(ProjOrder,), produces=SinoOrder,
	))
	with pytest.raises(ShapeCompatibilityError, match="fused_transition"):
		validate_pipeline([stage], scan_config(), initial_layout=ProjOrder)

	fused = ContractStage("geometry.fused", StageContract(
		accepts=(ProjOrder,), produces=SinoOrder, fused_transition=True,
	))
	validate_pipeline([fused], scan_config(), initial_layout=ProjOrder)


def test_complete_and_fragment_validation_have_distinct_skeleton_rules():
	fragment = ContractStage("phase.fragment", StageContract(
		accepts=(ProjOrder,), produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,), produces_domain=DataDomain.ATTENUATION,
	))
	validate_pipeline(
		[fragment], scan_config(), mode="fragment",
		initial_layout=ProjOrder, initial_domain=DataDomain.TRANSMISSION,
	)

	with pytest.raises(CompletePipelineError, match="must begin"):
		validate_pipeline([fragment], scan_config(), mode="complete")
	validate_pipeline(complete_stages(), scan_config(), mode="complete")


def test_complete_validation_counts_concrete_domain_transitions():
	disabled_phase = ContractStage("phase.disabled", StageContract(
		accepts=(ProjOrder,), produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,), produces_domain=PRESERVE,
	))
	stages = complete_stages()
	stages[2] = disabled_phase

	with pytest.raises(DomainCompatibilityError, match="recon.test.*got transmission"):
		validate_pipeline(stages, scan_config(), mode="complete")


def test_transmission_preserving_operations_chain_before_one_terminal_conversion():
	def correction(name):
		return ContractStage(name, StageContract(
			accepts=(ProjOrder,), produces=PRESERVE,
			accepts_domain=(DataDomain.TRANSMISSION,), produces_domain=PRESERVE,
		))

	stages = complete_stages(correction("correction.one"), correction("correction.two"))

	plan = validate_pipeline(stages, scan_config(), mode="complete")

	assert plan.steps[2].input_domain is DataDomain.TRANSMISSION
	assert plan.steps[2].output_domain is DataDomain.TRANSMISSION
	assert plan.steps[3].input_domain is DataDomain.TRANSMISSION
	assert plan.steps[3].output_domain is DataDomain.TRANSMISSION
	assert plan.steps[4].output_domain is DataDomain.ATTENUATION


def test_terminal_conversion_is_the_ordering_boundary_for_same_domain_operations():
	correction = ContractStage("correction", StageContract(
		accepts=(ProjOrder,), produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,), produces_domain=PRESERVE,
	))
	terminal = ContractStage("phase.negative_log", StageContract(
		accepts=(ProjOrder,), produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,),
		produces_domain=DataDomain.ATTENUATION,
	))
	correction.occurrence_id = "post-log-correction"
	terminal.occurrence_id = "terminal-log"

	validate_pipeline(
		[correction, terminal], scan_config(), initial_layout=ProjOrder,
		initial_domain=DataDomain.TRANSMISSION,
	)
	with pytest.raises(
		DomainCompatibilityError, match="post-log-correction:correction.*got attenuation",
	):
		validate_pipeline(
			[terminal, correction], scan_config(), initial_layout=ProjOrder,
			initial_domain=DataDomain.TRANSMISSION,
		)
	with pytest.raises(
		DomainCompatibilityError, match="terminal-log:phase.negative_log.*got attenuation",
	):
		validate_pipeline(
			[terminal, terminal], scan_config(), initial_layout=ProjOrder,
			initial_domain=DataDomain.TRANSMISSION,
		)


def test_domain_cannot_move_backward():
	stage = ContractStage("bad", StageContract(produces_domain=DataDomain.RAW))
	with pytest.raises(DomainCompatibilityError, match="cannot move backward"):
		validate_pipeline(
			[stage], scan_config(), initial_domain=DataDomain.ATTENUATION,
		)


def test_detector_resampling_invalidates_later_row_identity_requirement():
	resample = ContractStage("geometry.skew", StageContract(
		produces=PRESERVE, produces_domain=PRESERVE, resamples_detector=True,
	))
	remap = ContractStage("helical.index_remap", StageContract(
		produces=PRESERVE, produces_domain=PRESERVE, requires_row_identity=True,
	))

	with pytest.raises(RowIdentityError, match="row identity"):
		validate_pipeline([resample, remap], scan_config())


def test_plan_records_preference_warning_and_hidden_tomopy_copy():
	plan = validate_pipeline(complete_stages(), scan_config(), mode="complete")
	recon_step = plan.steps[-1]

	assert "prefers SinoOrder" in recon_step.warnings[0]
	assert recon_step.materializations[0].kind is MaterializationKind.TOMOPY_BOUNDARY
	assert recon_step.materializations[0].hidden is True
