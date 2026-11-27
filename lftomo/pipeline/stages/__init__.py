"""Built-in reconstruction Pipeline stages."""

from lftomo.config import StageConfig
from lftomo.evaluation.stage import EvaluationStage
from lftomo.pipeline.dispatch import STAGE_DISPATCH, dispatch_for_stage

from .center import EntropyLocalCenterStage, ManualCenterStage, PhaseCorrelationCenterStage, VoCenterStage
from .filter import BilateralFilterStage, ZingerFilterStage
from .beam_band_shift import BeamBandShiftNormalizeStage
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
from .multimaterial import MultimaterialStage
from .write import HDF5VolumeWriteStage, NeuroglancerWriteStage, TIFFSlicesWriteStage


STAGE_REGISTRY = {
	"center.entropy_local": EntropyLocalCenterStage,
	"center.manual": ManualCenterStage,
	"center.pc": PhaseCorrelationCenterStage,
	"center.vo": VoCenterStage,
	"evaluate.metrics": EvaluationStage,
	"filter.bilateral": BilateralFilterStage,
	"filter.zinger": ZingerFilterStage,
	"geometry.affine_skew": SkewCorrectStage,
	"geometry.affine_skew_movement": AffineSkewMovementStage,
	"geometry.movement": MovementCorrectStage,
	"geometry.helical_pitch": HelicalIndexRemapStage,
	"helical_correct.index_remap": HelicalIndexRemapStage,
	"layout.convert": LayoutConvertStage,
	"movement_correct.linear_shift": MovementCorrectStage,
	"normalize.phase_contrast": TaperBlendedNormalizeStage,
	"normalize.beam_basis_fit": BeamBasisFitNormalizeStage,
	"normalize.beam_band_shift": BeamBandShiftNormalizeStage,
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
	"phase.multimaterial": MultimaterialStage,
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


def _build_stage_dispatch(registry: dict[str, type]):
	"""Validate the heavy Stage registry against the canonical dispatch table."""

	unclassified = set(registry) - set(STAGE_DISPATCH)
	if unclassified:
		raise ValueError(
			f"registered Stage {sorted(unclassified)[0]!r} requires an explicit "
			"dispatch classification"
		)
	stale = set(STAGE_DISPATCH) - set(registry)
	if stale:
		raise ValueError(f"dispatch classifications have no registered Stage: {sorted(stale)}")
	return {key: STAGE_DISPATCH[key] for key in registry}


_build_stage_dispatch(STAGE_REGISTRY)


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
	"BeamBandShiftNormalizeStage",
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
	"MultimaterialStage",
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
	"ZingerFilterStage",
	"create_stage",
]
