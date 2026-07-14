"""Built-in reconstruction Pipeline stages."""

from lftomo.config import StageConfig

from .normalize import WeightedNormalizeStage
from .phase import PaganinPhaseAbsorptionStage, PaganinStage
from .read import HDF5DxchangeReadStage, TIFFStackReadStage
from .write import TIFFSlicesWriteStage


STAGE_REGISTRY = {
	"normalize.weighted": WeightedNormalizeStage,
	"phase.paganin": PaganinStage,
	"phase.paganin_phase_absorption": PaganinPhaseAbsorptionStage,
	"read.hdf5_dxchange": HDF5DxchangeReadStage,
	"read.tiff_stack": TIFFStackReadStage,
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
	"HDF5DxchangeReadStage",
	"PaganinPhaseAbsorptionStage",
	"PaganinStage",
	"TIFFStackReadStage",
	"TIFFSlicesWriteStage",
	"WeightedNormalizeStage",
	"create_stage",
]
