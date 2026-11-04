from pathlib import Path

import numpy as np
import yaml

from lftomo.config import Acquisition, Detector, Scan, TIFFStackProjections
from lftomo.pipeline import DataDomain, MaterializationKind, validate_pipeline
from lftomo.pipeline.stages import create_stage
from lftomo.runner import load_recon
from lftomo.util.shapes import ProjOrder, SinoOrder


ROOT = Path(__file__).parents[2]


def scan_config(theta=1969, rows=140, columns=64):
	return Scan(
		energy_keV=24, pixel_size_um=0.5, ssd_cm=3,
		n_projections=theta, degrees_per_fov=180,
		detector=Detector(pixel_size_um=0.5, rows=rows, columns=columns),
		acquisition=Acquisition(n_projections=theta, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
		metadata={"flat_fields": np.ones((1,), dtype=np.float32)},
	)


def test_recon_minimal_canonical_pins_fused_sino_to_projection_boundary():
	recon, _ = load_recon(ROOT / "lftomo" / "pipelines" / "recon-minimal-canonical.yaml")
	stages = [create_stage(config) for config in recon.pipeline]
	plan = validate_pipeline(stages, scan_config(3937), mode="complete")
	index = next(
		index for index, stage in enumerate(stages)
		if stage.name == "geometry.affine_skew_movement"
	)
	step = plan.steps[index]

	assert stages[index].contract.fused_transition is True
	assert stages[index].contract.resamples_detector is False
	assert step.input_layout is SinoOrder
	assert step.output_layout is ProjOrder
	assert plan.steps[index + 1].stage_name == "phase.paganin"


def test_aaa523_keeps_nonzero_warp_in_projection_order_before_stripe():
	recon, _ = load_recon(ROOT / "lftomo" / "pipelines" / "phase-contrast-canonical.yaml")
	stages = [create_stage(config) for config in recon.pipeline]
	plan = validate_pipeline(stages, scan_config(columns=5008), mode="complete")
	skew_index = next(index for index, stage in enumerate(stages) if stage.name == "skew_correct.cv2_affine")
	recon_step = next(step for step in plan.steps if step.stage_name == "recon.gridrec")

	assert stages[skew_index].params.degrees == 0.149
	assert stages[skew_index].contract.fused_transition is False
	assert stages[skew_index].contract.resamples_detector is True
	assert plan.steps[skew_index].input_layout is ProjOrder
	assert plan.steps[skew_index].output_layout is ProjOrder
	assert plan.steps[skew_index + 2].stage_name == "stripe.fw"
	assert not any(step.stage_name == "layout.convert" for step in plan.steps)
	assert all(
		step.output_layout is ProjOrder
		for step in plan.steps
		if step.index < recon_step.index
	)
	assert recon_step.input_layout is ProjOrder
	assert recon_step.materializations[0].kind is MaterializationKind.TOMOPY_BOUNDARY


def test_aaa592_reference_records_nonzero_warp_before_paganin():
	path = (
		ROOT / "tests" / "tier2_metadata" / "references" / "striping_tiff"
		/ "recon_minimal" / "pipeline.yaml"
	)
	pipeline = yaml.safe_load(path.read_text(encoding="utf-8"))["pipeline"]
	geometry_index = next(index for index, stage in enumerate(pipeline) if stage["kind"] == "geometry")
	phase_index = next(index for index, stage in enumerate(pipeline) if stage["kind"] == "phase")

	assert pipeline[geometry_index]["algorithm"] == "projection_rotation"
	assert pipeline[geometry_index]["params"]["tilt_radians"] == -0.003134796238244514
	assert geometry_index < phase_index
