"""Built-in reconstruction Pipeline stages."""

from lftomo.config import StageConfig

from .center import EntropyLocalCenterStage, ManualCenterStage, PhaseCorrelationCenterStage, VoCenterStage
from .filter import BilateralFilterStage
from .geometry import AffineSkewMovementStage, MovementCorrectStage, SkewCorrectStage
from .helical import HelicalIndexRemapStage
from .layout import LayoutConvertStage
from .normalize import TaperBlendedNormalizeStage, ThetaSwitchNormalizeStage, WeightedNormalizeStage
from .phase import BronnikovAidedCorrectionStage, PaganinPhaseAbsorptionStage, PaganinStage
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
	"filter.bilateral": BilateralFilterStage,
	"geometry.affine_skew": SkewCorrectStage,
	"geometry.affine_skew_movement": AffineSkewMovementStage,
	"geometry.movement": MovementCorrectStage,
	"geometry.helical_pitch": HelicalIndexRemapStage,
	"helical_correct.index_remap": HelicalIndexRemapStage,
	"layout.convert": LayoutConvertStage,
	"movement_correct.linear_shift": MovementCorrectStage,
	"normalize.phase_contrast": TaperBlendedNormalizeStage,
	"normalize.taper_blended": TaperBlendedNormalizeStage,
	"normalize.theta_switch": ThetaSwitchNormalizeStage,
	"normalize.weighted": WeightedNormalizeStage,
	"phase.bac": BronnikovAidedCorrectionStage,
	"phase.bronnikov_aided": BronnikovAidedCorrectionStage,
	"phase.paganin": PaganinStage,
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


def create_stage(config: StageConfig):
	"""Create a built-in Stage from its canonical ``kind.algorithm`` key."""
	key = f"{config.kind}.{config.algorithm}" if config.algorithm else config.kind
	try:
		stage_type = STAGE_REGISTRY[key]
	except KeyError as error:
		raise ValueError(f"unknown pipeline Stage {key!r}") from error
	from_config = getattr(stage_type, "from_config", None)
	return from_config(config) if from_config is not None else stage_type(**config.params)


__all__ = [
	"STAGE_REGISTRY",
	"AstraCudaStage",
	"AllOfTheStripeStage",
	"AffineSkewMovementStage",
	"BPCudaStage",
	"BilateralFilterStage",
	"BronnikovAidedCorrectionStage",
	"CGLSCudaStage",
	"DRHTStage",
	"EMCudaStage",
	"EntropyLocalCenterStage",
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
