"""Built-in reconstruction Pipeline stages."""

from lftomo.config import StageConfig

from .center import EntropyLocalCenterStage, ManualCenterStage, PhaseCorrelationCenterStage, VoCenterStage
from .normalize import WeightedNormalizeStage
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
from .stripe import AllOfTheStripeStage, FourierWaveletStripeStage, VoDeadStripeStage, VoLargeStripeStage, VoSortStripeStage
from .write import TIFFSlicesWriteStage


STAGE_REGISTRY = {
	"center.entropy_local": EntropyLocalCenterStage,
	"center.manual": ManualCenterStage,
	"center.pc": PhaseCorrelationCenterStage,
	"center.vo": VoCenterStage,
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
	"stripe.vo_large": VoLargeStripeStage,
	"stripe.vo_sort": VoSortStripeStage,
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
	"BPCudaStage",
	"BronnikovAidedCorrectionStage",
	"CGLSCudaStage",
	"EMCudaStage",
	"EntropyLocalCenterStage",
	"HDF5DxchangeReadStage",
	"FourierWaveletStripeStage",
	"FBPCudaStage",
	"FPCudaStage",
	"GridrecStage",
	"ManualCenterStage",
	"PaganinPhaseAbsorptionStage",
	"PaganinStage",
	"PhaseCorrelationCenterStage",
	"SARTCudaStage",
	"SIRTCudaStage",
	"TIFFStackReadStage",
	"TIFFSlicesWriteStage",
	"VoDeadStripeStage",
	"VoLargeStripeStage",
	"VoSortStripeStage",
	"VoCenterStage",
	"WeightedNormalizeStage",
	"create_stage",
]
