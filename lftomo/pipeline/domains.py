"""Processing-domain values carried independently from array layout."""

from __future__ import annotations

from enum import Enum, auto


class ContractEffect(Enum):
	"""Sentinel effects shared by the layout and domain contract axes."""

	PRESERVE = auto()


PRESERVE = ContractEffect.PRESERVE


class DataDomain(Enum):
	"""Monotonic processing domain of the pipeline working array."""

	RAW = auto()
	TRANSMISSION = auto()
	ATTENUATION = auto()
	RECON = auto()
