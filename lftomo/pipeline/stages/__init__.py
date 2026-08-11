"""Built-in reconstruction Pipeline stages."""

from lftomo.config import StageConfig
from lftomo.evaluation.stage import EvaluationStage
from lftomo.pipeline.dispatch import DispatchKind, DispatchSpec

from .center import EntropyLocalCenterStage, ManualCenterStage, PhaseCorrelationCenterStage, VoCenterStage
from .filter import BilateralFilterStage
from .geometry import AffineSkewMovementStage, MovementCorrectStage, SkewCorrectStage
from .helical import HelicalIndexRemapStage
from .layout import LayoutConvertStage
from .normalize import (
	BeamBasisFitNormalizeStage,
	BeamTrackingNormalizeStage,
	BeamKeyframesNormalizeStage,
	TaperBlendedNormalizeStage,
	ThetaSwitchNormalizeStage,
	WeightedNormalizeStage,
)
from .phase import (
	BetaScaleStage,
	BronnikovAidedCorrectionFilterStage,
	BronnikovAidedCorrectionStage,
	NegativeLogStage,
	PaganinFilterStage,
	PaganinPhaseAbsorptionStage,
	PaganinStage,
)
from .read import HDF5DxchangeReadStage, TIFFStackReadStage
from .recon import (
	AstraCudaStage,
	BPCudaStage,
	CGLSCudaStage,
	EMCudaStage,
	FBPCudaStage,
	FPCudaStage,
	GridrecStage,
	SARTCudaStage,
	SIRTCudaStage,
)
from .stripe import (
	AllOfTheStripeStage,
	FourierWaveletStripeStage,
	VoDeadStripeStage,
	VoFamilyStripeStage,
	VoLargeStripeStage,
	VoSortStripeStage,
)
from .super_resolve import DRHTStage, LDMStage
from .write import HDF5VolumeWriteStage, NeuroglancerWriteStage, TIFFSlicesWriteStage


STAGE_REGISTRY = {
	"center.entropy_local": EntropyLocalCenterStage,
	"center.manual": ManualCenterStage,
	"center.pc": PhaseCorrelationCenterStage,
	"center.vo": VoCenterStage,
	"evaluate.metrics": EvaluationStage,
	"filter.bilateral": BilateralFilterStage,
	"geometry.affine_skew": SkewCorrectStage,
	"geometry.affine_skew_movement": AffineSkewMovementStage,
	"geometry.movement": MovementCorrectStage,
	"geometry.helical_pitch": HelicalIndexRemapStage,
	"helical_correct.index_remap": HelicalIndexRemapStage,
	"layout.convert": LayoutConvertStage,
	"movement_correct.linear_shift": MovementCorrectStage,
	"normalize.phase_contrast": TaperBlendedNormalizeStage,
	"normalize.beam_basis_fit": BeamBasisFitNormalizeStage,
	"normalize.beam_tracking": BeamTrackingNormalizeStage,
	"normalize.beam_keyframes": BeamKeyframesNormalizeStage,
	"normalize.taper_blended": TaperBlendedNormalizeStage,
	"normalize.theta_switch": ThetaSwitchNormalizeStage,
	"normalize.weighted": WeightedNormalizeStage,
	"phase.bac": BronnikovAidedCorrectionStage,
	"phase.bac_correction": BronnikovAidedCorrectionFilterStage,
	"phase.beta_scale": BetaScaleStage,
	"phase.bronnikov_aided": BronnikovAidedCorrectionStage,
	"phase.negative_log": NegativeLogStage,
	"phase.paganin": PaganinStage,
	"phase.paganin_filter": PaganinFilterStage,
	"phase.paganin_phase_absorption": PaganinPhaseAbsorptionStage,
	"read.hdf5_dxchange": HDF5DxchangeReadStage,
	"read.tiff_stack": TIFFStackReadStage,
	"recon.bp_cuda": BPCudaStage,
	"recon.cgls_cuda": CGLSCudaStage,
	"recon.em_cuda": EMCudaStage,
	"recon.fbp_cuda": FBPCudaStage,
	"recon.fp_cuda": FPCudaStage,
	"recon.gridrec": GridrecStage,
	"recon.sart_cuda": SARTCudaStage,
	"recon.sirt_cuda": SIRTCudaStage,
	"stripe.all_of_the_stripe": AllOfTheStripeStage,
	"stripe.fw": FourierWaveletStripeStage,
	"stripe.vo_dead": VoDeadStripeStage,
	"stripe.vo_family": VoFamilyStripeStage,
	"stripe.vo_large": VoLargeStripeStage,
	"stripe.vo_sort": VoSortStripeStage,
	"super_resolve.drht": DRHTStage,
	"super_resolve.ldm": LDMStage,
	"write.hdf5_volume": HDF5VolumeWriteStage,
	"write.neuroglancer": NeuroglancerWriteStage,
	"skew_correct.cv2_affine": SkewCorrectStage,
	"write.tiff_slices": TIFFSlicesWriteStage,
}


def _dispatch(kind: DispatchKind, kernel: str, *, projection_units: bool = False,
		reason: str | None = None) -> DispatchSpec:
	return DispatchSpec(
		kind=kind, kernel=kernel, projection_units=projection_units, reason=reason,
	)


_PROJECTION_PROCESS = {
	"filter.bilateral",
	"geometry.affine_skew",
	"geometry.movement",
	"movement_correct.linear_shift",
	"normalize.phase_contrast",
	"normalize.beam_basis_fit",
	"normalize.beam_tracking",
	"normalize.beam_keyframes",
	"normalize.taper_blended",
	"normalize.theta_switch",
	"normalize.weighted",
	"phase.bac",
	"phase.bac_correction",
	"phase.beta_scale",
	"phase.bronnikov_aided",
	"phase.negative_log",
	"phase.paganin",
	"phase.paganin_filter",
	"phase.paganin_phase_absorption",
}
_BOUNDED_PROCESS = {
	"stripe.fw", "stripe.vo_dead", "stripe.vo_family", "stripe.vo_large", "stripe.vo_sort",
}
_OUT_OF_PLACE_PROCESS = {
	"geometry.affine_skew_movement", "geometry.helical_pitch",
	"helical_correct.index_remap", "layout.convert",
}
_THREAD_IO = {"read.hdf5_dxchange", "read.tiff_stack", "write.hdf5_volume", "write.neuroglancer"}
_PROCESS_WRITERS = {"write.tiff_slices"}
_ENGINE = {
	"center.entropy_local", "center.pc", "center.vo",
	"recon.bp_cuda", "recon.cgls_cuda", "recon.em_cuda", "recon.fbp_cuda",
	"recon.fp_cuda", "recon.gridrec", "recon.sart_cuda", "recon.sirt_cuda",
	"stripe.all_of_the_stripe", "super_resolve.drht", "super_resolve.ldm",
}
_SERIAL = {"center.manual", "evaluate.metrics", "skew_correct.cv2_affine"}


def _build_stage_dispatch(registry: dict[str, type]) -> dict[str, DispatchSpec]:
	"""Build dispatch metadata, rejecting every unclassified registration."""

	classified = (
		_PROJECTION_PROCESS | _BOUNDED_PROCESS | _OUT_OF_PLACE_PROCESS
		| _THREAD_IO | _PROCESS_WRITERS | _ENGINE | _SERIAL
	)
	stale = classified - set(registry)
	if stale:
		raise ValueError(f"dispatch classifications have no registered Stage: {sorted(stale)}")
	result: dict[str, DispatchSpec] = {}
	for key in registry:
		if key in _PROJECTION_PROCESS:
			result[key] = _dispatch(
				DispatchKind.PROCESS, "run_projection_kernel", projection_units=True,
			)
		elif key in _BOUNDED_PROCESS:
			result[key] = _dispatch(DispatchKind.PROCESS, "run_sinogram_chain_kernel")
		elif key in _OUT_OF_PLACE_PROCESS:
			result[key] = _dispatch(
				DispatchKind.PROCESS, "manager_owned_parallel_fill",
				reason="shape-changing fill remains parent-orchestrated until its unit adapter is selected",
			)
		elif key in _PROCESS_WRITERS:
			result[key] = _dispatch(DispatchKind.PROCESS, "write_tiff_slice_kernel")
		elif key in _THREAD_IO:
			result[key] = _dispatch(DispatchKind.THREAD, "stage_io_chunks")
		elif key in _ENGINE:
			result[key] = _dispatch(DispatchKind.ENGINE, "internal_core_budget")
		elif key in _SERIAL:
			result[key] = _dispatch(DispatchKind.SERIAL, "stage.run")
		else:
			raise ValueError(
				f"registered Stage {key!r} requires an explicit dispatch classification"
			)
	return result


STAGE_DISPATCH = _build_stage_dispatch(STAGE_REGISTRY)


def dispatch_for_stage(stage) -> DispatchSpec:
	"""Return the registry classification for a concrete Stage instance."""

	key = next((name for name, stage_type in STAGE_REGISTRY.items()
		if stage.name == name and isinstance(stage, stage_type)), None)
	if key is None:
		key = next((name for name, stage_type in STAGE_REGISTRY.items()
			if isinstance(stage, stage_type)), None)
	return (
		STAGE_DISPATCH[key] if key is not None else
		DispatchSpec(
			DispatchKind.EXEMPT,
			reason="unregistered test or extension Stage uses serial hermetic execution",
		)
	)


def create_stage(config: StageConfig):
	"""Create a built-in Stage from its canonical ``kind.algorithm`` key."""
	key = f"{config.kind}.{config.algorithm}" if config.algorithm else config.kind
	try:
		stage_type = STAGE_REGISTRY[key]
	except KeyError as error:
		location = f" {config.id!r}" if config.id is not None else ""
		raise ValueError(f"unknown pipeline Stage{location} {key!r}") from error
	from_config = getattr(stage_type, "from_config", None)
	stage = from_config(config) if from_config is not None else stage_type(**config.params)
	stage.occurrence_id = config.id
	stage.dispatch = STAGE_DISPATCH[key]
	return stage


__all__ = [
	"STAGE_REGISTRY",
	"STAGE_DISPATCH",
	"dispatch_for_stage",
	"AstraCudaStage",
	"AllOfTheStripeStage",
	"AffineSkewMovementStage",
	"BPCudaStage",
	"BeamBasisFitNormalizeStage",
	"BeamTrackingNormalizeStage",
	"BeamKeyframesNormalizeStage",
	"BilateralFilterStage",
	"BetaScaleStage",
	"BronnikovAidedCorrectionFilterStage",
	"BronnikovAidedCorrectionStage",
	"CGLSCudaStage",
	"DRHTStage",
	"EMCudaStage",
	"EntropyLocalCenterStage",
	"EvaluationStage",
	"HDF5DxchangeReadStage",
	"FourierWaveletStripeStage",
	"FBPCudaStage",
	"FPCudaStage",
	"GridrecStage",
	"HelicalIndexRemapStage",
	"HDF5VolumeWriteStage",
	"LDMStage",
	"LayoutConvertStage",
	"ManualCenterStage",
	"MovementCorrectStage",
	"NeuroglancerWriteStage",
	"NegativeLogStage",
	"PaganinFilterStage",
	"PaganinPhaseAbsorptionStage",
	"PaganinStage",
	"PhaseCorrelationCenterStage",
	"SARTCudaStage",
	"SIRTCudaStage",
	"SkewCorrectStage",
	"TIFFStackReadStage",
	"TIFFSlicesWriteStage",
	"TaperBlendedNormalizeStage",
	"ThetaSwitchNormalizeStage",
	"VoDeadStripeStage",
	"VoFamilyStripeStage",
	"VoLargeStripeStage",
	"VoSortStripeStage",
	"VoCenterStage",
	"WeightedNormalizeStage",
	"create_stage",
]
