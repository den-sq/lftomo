"""Serializable records describing validated pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from lftomo.pipeline.domains import DataDomain


class MaterializationKind(Enum):
	"""Kinds of full-volume materialization visible in a plan."""

	LAYOUT_CONVERSION = auto()
	TOMOPY_BOUNDARY = auto()


@dataclass(frozen=True)
class Materialization:
	"""One explicit or library-boundary full-volume copy."""

	kind: MaterializationKind
	reason: str
	hidden: bool = False


@dataclass(frozen=True)
class PlanStep:
	"""Resolved axes and copy behavior for one Stage."""

	index: int
	stage_name: str
	input_layout: type[Any] | None
	output_layout: type[Any] | None
	input_domain: DataDomain | None
	output_domain: DataDomain | None
	materializations: tuple[Materialization, ...] = ()
	warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelinePlan:
	"""Validated ordered Stage plan; later phases may add grouped segments."""

	steps: tuple[PlanStep, ...]
	warnings: tuple[str, ...] = ()

	@property
	def full_volume_materializations(self) -> tuple[Materialization, ...]:
		return tuple(item for step in self.steps for item in step.materializations)
