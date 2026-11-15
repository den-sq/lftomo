"""Pydantic models for canonical lftomo configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
	BaseModel, ConfigDict, Field, TypeAdapter, ValidationError,
	field_validator, model_serializer, model_validator,
)

from lftomo.util.shapes import Proj, ProjOrder


# Normalization configuration keys consumed by the runner before Stage construction.
CALIBRATION_RESOLUTION_FIELDS = {
	"calibration_manifest",
	"calibration_components",
	"compatibility_policy",
	"compatibility_override_reason",
}


class ConfigModel(BaseModel):
	"""Base class for YAML-friendly lftomo configuration models."""

	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	def to_yaml(self) -> str:
		"""Serialize the model to a YAML string."""
		return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

	@classmethod
	def from_yaml(cls, text: str):
		"""Parse a model instance from a YAML string."""
		return cls.model_validate(yaml.safe_load(text))


class Machine(ConfigModel):
	"""Per-deployment machine, storage, and resource-limit configuration."""

	name: str = Field(description="Human-readable machine or deployment name.")
	storage_roots: dict[str, Path] = Field(default_factory=dict, description="Named root paths available on this machine.")
	scratch_root: Path | None = Field(default=None, description="Optional scratch/work directory for intermediate data.")
	threads: int | None = Field(default=None, gt=0, description="Optional maximum local worker/thread count.")
	memory_gb: float | None = Field(default=None, gt=0, description="Optional memory budget in GiB.")
	runner_labels: list[str] = Field(default_factory=list, description="Optional CI or scheduler labels for this machine.")


class Session(ConfigModel):
	"""Per-session acquisition context shared by scans from the same session."""

	sample_id: str = Field(description="Sample identifier for the session.")
	operator: str | None = Field(default=None, description="Operator or beamline user who ran the session.")
	location: str | None = Field(default=None, description="Beamline, lab, or instrument location.")
	project: str | None = Field(default=None, description="Project, proposal, or experiment grouping.")
	metadata: dict[str, Any] = Field(default_factory=dict, description="Additional session-level metadata.")


class Detector(ConfigModel):
	"""Detector geometry and identity for a scan."""

	pixel_size_um: float = Field(gt=0, description="Detector pixel size in micrometers.")
	rows: int | None = Field(default=None, gt=0, description="Detector image height in pixels.")
	columns: int | None = Field(default=None, gt=0, description="Detector image width in pixels.")
	name: str | None = Field(default=None, description="Detector or camera model/name.")

	def projection_shape(self) -> Proj:
		"""Return canonical detector image geometry once metadata is concrete."""

		if self.rows is None or self.columns is None:
			raise ValueError("window planning requires detector rows and columns")
		return Proj(Y=self.rows, X=self.columns)


class ParallelGeometry(ConfigModel):
	"""Measured effective geometry for an existing parallel-beam scan."""

	kind: Literal["parallel"] = "parallel"
	pixel_size_um: float = Field(gt=0, description="Effective sample pixel size in micrometers.")
	ssd_cm: float | None = Field(default=None, gt=0, description="Propagation distance in centimeters.")

	@property
	def magnification(self) -> float:
		return 1.0

	@property
	def effective_distance_cm(self) -> float | None:
		return self.ssd_cm

	def effective_pixel_size_um(self, detector: Detector) -> float:
		return self.pixel_size_um


class ConeGeometry(ConfigModel):
	"""Native cone distances; derive equivalent parallel geometry by Fresnel scaling."""

	kind: Literal["cone"] = "cone"
	source_to_sample_m: float = Field(gt=0, allow_inf_nan=False, description="Source-to-sample distance in metres.")
	sample_to_detector_m: float = Field(gt=0, allow_inf_nan=False, description="Sample-to-detector distance in metres.")

	@property
	def magnification(self) -> float:
		return (self.source_to_sample_m + self.sample_to_detector_m) / self.source_to_sample_m

	@property
	def effective_distance_cm(self) -> float:
		return 100 * self.source_to_sample_m * self.sample_to_detector_m / (
			self.source_to_sample_m + self.sample_to_detector_m
		)

	def effective_pixel_size_um(self, detector: Detector) -> float:
		return detector.pixel_size_um / self.magnification


ScanGeometry = Annotated[ParallelGeometry | ConeGeometry, Field(discriminator="kind")]


class Helical(ConfigModel):
	"""Helical acquisition metadata."""

	enabled: bool = Field(default=False, description="Whether the scan used helical acquisition.")
	start_um: float | None = Field(default=None, description="Starting vertical position in micrometers.")
	stop_um: float | None = Field(default=None, description="Ending vertical position in micrometers.")
	pitch_um_per_projection: float | None = Field(default=None, description="Vertical pitch per projection in micrometers.")


class UniformAngleSource(ConfigModel):
	"""Uniform endpoint-exclusive angles with no missing acquisition slots."""

	kind: Literal["uniform"] = "uniform"


class ExplicitAngleSource(ConfigModel):
	"""Per-recorded-projection acquisition indices and/or angles."""

	kind: Literal["explicit"] = "explicit"
	acquisition_indices: list[int] | None = None
	angles_degrees: list[float] | None = None

	@model_validator(mode="after")
	def _has_explicit_values(self):
		if self.acquisition_indices is None and self.angles_degrees is None:
			raise ValueError("explicit angle source requires indices or angles")
		return self


class HardwareTimestampAngleSource(ConfigModel):
	"""Derive acquisition slots cumulatively from hardware timestamps."""

	kind: Literal["hardware_timestamps"] = "hardware_timestamps"
	path: Path
	timestamp_column: int = Field(default=3, ge=0)
	delimiter: str = Field(default="\t", min_length=1)
	header_rows: int = Field(default=1, ge=0)


class HDF5AngleSource(ConfigModel):
	"""Read a per-projection angle dataset from the projection HDF5 file."""

	kind: Literal["hdf5"] = "hdf5"
	path: str = Field(default="exchange/theta", min_length=1)
	units: Literal["degrees", "radians"] = "degrees"


AngleSource = Annotated[
	UniformAngleSource | ExplicitAngleSource | HardwareTimestampAngleSource | HDF5AngleSource,
	Field(discriminator="kind"),
]


class Acquisition(ConfigModel):
	"""Projection-count and angle metadata for a scan."""

	n_projections: int = Field(
		gt=0,
		description="Number of planned acquisition/trigger slots, including dropped slots.",
	)
	rotation_start_deg: float = Field(default=0.0, description="Start angle in degrees.")
	rotation_end_deg: float = Field(description="End angle in degrees.")
	exposure_ms: float | None = Field(default=None, gt=0, description="Exposure time in milliseconds.")
	angle_source: AngleSource = Field(
		default_factory=UniformAngleSource,
		exclude_if=lambda value: isinstance(value, UniformAngleSource),
	)


class TIFFStackProjections(ConfigModel):
	"""TIFF-stack projection storage description."""

	format: Literal["tiff_stack"] = Field(description="Projection storage format discriminator.")
	path: Path = Field(description="Directory containing projection TIFF files.")
	pattern: str = Field(default="*.tif*", description="Glob pattern for projection files.")


class HDF5DxchangeProjections(ConfigModel):
	"""HDF5 dxchange projection storage description."""

	format: Literal["hdf5_dxchange"] = Field(description="Projection storage format discriminator.")
	path: Path = Field(description="HDF5 file path.")
	data_path: str = Field(default="exchange/data", description="Dataset path for projections.")
	white_path: str = Field(default="exchange/data_white", description="Dataset path for flat/white fields.")
	dark_path: str = Field(default="exchange/data_dark", description="Dataset path for dark fields.")


ProjectionSource = Annotated[TIFFStackProjections | HDF5DxchangeProjections, Field(discriminator="format")]


class Scan(ConfigModel):
	"""Per-dataset immutable acquisition metadata."""

	energy_keV: float = Field(gt=0, description="Beam energy in keV.")
	geometry: ScanGeometry = Field(description="Authoritative native or effective physical scan geometry.")
	n_projections: int = Field(
		gt=0, description="Number of recorded projection arrays/files present in the source."
	)
	degrees_per_fov: float = Field(gt=0, description="Angular coverage per field of view.")
	n_fov: int = Field(default=1, gt=0, description="Number of fields of view in the scan.")
	helical: Helical = Field(default_factory=Helical, description="Helical acquisition metadata.")
	detector: Detector = Field(description="Detector geometry and identity.")
	acquisition: Acquisition = Field(description="Projection-count and angle metadata.")
	projections: ProjectionSource = Field(description="Projection storage description.")
	metadata: dict[str, Any] = Field(default_factory=dict, description="Additional lineage-specific metadata.")

	@model_validator(mode="before")
	@classmethod
	def _normalize_geometry(cls, values):
		if not isinstance(values, dict):
			return values
		values = dict(values)
		legacy = {name: values.pop(name) for name in ("pixel_size_um", "ssd_cm") if name in values}
		if "geometry" in values:
			if legacy:
				raise ValueError("provide geometry or legacy pixel_size_um/ssd_cm, not both")
		else:
			values["geometry"] = {"kind": "parallel", **legacy}
		return values

	@model_serializer(mode="wrap")
	def _serialize_geometry(self, handler):
		values = handler(self)
		if isinstance(self.geometry, ParallelGeometry) and "geometry" in values:
			geometry = values.pop("geometry")
			geometry.pop("kind", None)
			values.update(geometry)
		return values

	@property
	def pixel_size_um(self) -> float:
		"""Effective sample pitch, derived from the authoritative geometry."""
		return self.geometry.effective_pixel_size_um(self.detector)

	@property
	def ssd_cm(self) -> float | None:
		"""Equivalent parallel propagation distance for phase retrieval."""
		return self.geometry.effective_distance_cm

	@property
	def magnification(self) -> float:
		return self.geometry.magnification

	def projection_shape(self) -> ProjOrder:
		"""Return canonical recorded-projection geometry for this scan."""

		detector = self.detector.projection_shape()
		return ProjOrder(Theta=self.n_projections, Y=detector.Y, X=detector.X)

	@field_validator("acquisition")
	@classmethod
	def _projection_counts_match(cls, acquisition: Acquisition, info):
		recorded = info.data.get("n_projections")
		if recorded is not None and acquisition.n_projections < recorded:
			raise ValueError(
				"acquisition.n_projections planned slots cannot be fewer than "
				"scan.n_projections recorded projections"
			)
		if (
			recorded is not None
			and acquisition.n_projections != recorded
			and isinstance(acquisition.angle_source, UniformAngleSource)
		):
			raise ValueError(
				"differing planned and recorded projection counts require an explicit "
				"angle_source"
			)
		return acquisition


class DiscreteDomain(ConfigModel):
	"""Finite ordered values for a numeric or otherwise discrete parameter."""

	kind: Literal["discrete"] = "discrete"
	values: list[Any] = Field(min_length=1)


class CategoricalDomain(ConfigModel):
	"""Finite ordered categorical choices."""

	kind: Literal["categorical"] = "categorical"
	values: list[str | int | float | bool] = Field(min_length=1)


class LinearDomain(ConfigModel):
	"""Linear numeric domain; ``count=None`` denotes a continuous domain."""

	kind: Literal["linear"] = "linear"
	start: float
	stop: float
	count: int | None = Field(default=None, ge=2)
	endpoint: bool = True


class LogDomain(ConfigModel):
	"""Log-spaced positive domain; ``count=None`` denotes continuity."""

	kind: Literal["log"] = "log"
	start: float = Field(gt=0)
	stop: float = Field(gt=0)
	count: int | None = Field(default=None, ge=2)
	base: float = Field(default=10.0, gt=0)

	@field_validator("base")
	@classmethod
	def _base_is_not_one(cls, value: float) -> float:
		if value == 1:
			raise ValueError("log-domain base must not be 1")
		return value


ParameterDomain = Annotated[
	DiscreteDomain | CategoricalDomain | LinearDomain | LogDomain,
	Field(discriminator="kind"),
]
_PARAMETER_DOMAIN_ADAPTER = TypeAdapter(ParameterDomain)


class StageCase(ConfigModel):
	"""One named algorithm choice and only the parameters valid for it."""

	name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
	algorithm: str = Field(min_length=1)
	params: dict[str, Any] = Field(default_factory=dict)
	domains: dict[str, ParameterDomain] = Field(default_factory=dict)

	@model_validator(mode="after")
	def _parameters_are_unambiguous(self):
		overlap = set(self.params) & set(self.domains)
		if overlap:
			raise ValueError(f"case params and domains overlap: {sorted(overlap)}")
		return self


class SweepConfig(ConfigModel):
	"""The canonical parameter/algorithm candidate-space description.

	The original list/grid/linspace/zip forms remain supported. ``domain``
	adds typed finite or continuous domains, while ``cases`` keeps
	algorithm-specific parameters together.
	"""

	kind: Literal["list", "grid", "linspace", "zip", "domain", "cases"] = Field(
		description="Candidate expansion strategy.",
	)
	params: dict[str, Any] = Field(
		default_factory=dict, description="Parameter values or typed domains.",
	)
	cases: list[StageCase] = Field(
		default_factory=list, description="Named conditional algorithm cases.",
	)

	@model_validator(mode="after")
	def _shape_matches_kind(self):
		if self.kind == "cases":
			if not self.cases:
				raise ValueError("cases sweep requires at least one case")
			if self.params:
				raise ValueError("cases sweep uses 'cases', not 'params'")
			names = [case.name for case in self.cases]
			if len(names) != len(set(names)):
				raise ValueError("case names must be unique within a Stage")
		else:
			if self.cases:
				raise ValueError(f"{self.kind} sweep does not accept algorithm cases")
			if self.kind == "domain":
				if not self.params:
					raise ValueError("domain sweep requires at least one parameter")
				for name, spec in self.params.items():
					try:
						_PARAMETER_DOMAIN_ADAPTER.validate_python(spec)
					except ValueError as error:
						raise ValueError(f"invalid domain for parameter {name!r}: {error}") from error
		return self


class StageConfig(ConfigModel):
	"""Single pipeline stage configuration entry."""

	id: str | None = Field(
		default=None,
		pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
		description="Stable identity for this ordered Stage occurrence.",
	)
	kind: str = Field(description="Stage kind, such as read, normalize, phase, recon, or write.")
	algorithm: str | None = Field(default=None, description="Optional algorithm choice within the stage kind.")
	params: dict[str, Any] = Field(default_factory=dict, description="Stage-specific parameters.")
	sweep: SweepConfig | None = Field(default=None, description="Optional sweep applied at this stage.")

	@model_validator(mode="after")
	def _validate_declared_params(self):
		if not self.params:
			return self
		# Resolve through the construction registry, without constructing a Stage
		# or importing its optional numerical engine. The serialized dictionary
		# stays unchanged for sweep expansion and prefix-cache identity.
		from lftomo.pipeline.stages import STAGE_REGISTRY

		key = f"{self.kind}.{self.algorithm}" if self.algorithm else self.kind
		stage_type = STAGE_REGISTRY.get(key)
		if stage_type is None:
			return self
		params = self.params
		if self.kind == "normalize":
			params = {
				key: value for key, value in params.items()
				if key not in CALIBRATION_RESOLUTION_FIELDS
			}
		try:
			stage_type.Params.model_validate(params)
		except ValidationError as error:
			# A StageConfig can be an unresolved template. Required inputs are
			# checked when create_stage materializes it; supplied inputs are
			# always checked here, including when another input is absent.
			failures = [item for item in error.errors() if item["type"] != "missing"]
			if failures:
				fields = "; ".join(
					f"{'.'.join(map(str, item['loc'])) or 'params'}: {item['msg']}"
					for item in failures
				)
				raise ValueError(f"Stage {self.id or key!r} ({key}) parameters: {fields}") from error
		return self


class PipelineVariant(ConfigModel):
	"""One explicitly permitted named ordering/subset of Stage IDs."""

	name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
	stages: list[str] = Field(min_length=1, description="Ordered stable Stage IDs to include.")

	@field_validator("stages")
	@classmethod
	def _stage_ids_are_unique(cls, values: list[str]) -> list[str]:
		if len(values) != len(set(values)):
			raise ValueError("a structural variant cannot repeat a Stage ID")
		return values


class CandidateLimits(ConfigModel):
	"""Preflight limits protecting ordinary exhaustive candidate runs."""

	max_candidates: int = Field(default=10_000, gt=0)
	max_estimated_output_bytes: int = Field(default=500 * 1024**3, gt=0)


class Recon(ConfigModel):
	"""Per-run reconstruction pipeline configuration."""

	pipeline: list[StageConfig] = Field(description="Ordered reconstruction pipeline stages.")
	pipeline_template: str | None = Field(default=None, description="Optional named pipeline template.")
	output_path: Path | None = Field(default=None, description="Optional output directory for the run.")
	buffer_aliasing: bool = Field(
		default=True, description="Allow planner-proved named shared-buffer aliases.",
	)
	tomopy_input_aliasing: bool = Field(
		default=True,
		description="Allow the TomoPy input/output alias when its runtime boundary proves it safe.",
	)
	process_workers: int | None = Field(
		default=None, gt=0,
		description="Persistent process workers; defaults to a bounded machine-derived count.",
	)
	worker_timeout_seconds: float | None = Field(
		default=None, gt=0, description="Optional timeout for each dispatched worker task.",
	)
	metadata: dict[str, Any] = Field(default_factory=dict, description="Additional run metadata.")
	variants: list[PipelineVariant] = Field(
		default_factory=list,
		description="Explicitly permitted named Stage-ID structures.",
	)
	candidate_limits: CandidateLimits = Field(default_factory=CandidateLimits)

	@model_validator(mode="after")
	def _validate_stage_occurrence_ids(self):
		ids = [stage.id for stage in self.pipeline if stage.id is not None]
		duplicates = sorted({stage_id for stage_id in ids if ids.count(stage_id) > 1})
		if duplicates:
			raise ValueError(f"duplicate pipeline Stage IDs: {duplicates}")
		variant_names = [variant.name for variant in self.variants]
		if len(variant_names) != len(set(variant_names)):
			raise ValueError("pipeline variant names must be unique")
		if self.variants:
			missing = [index for index, stage in enumerate(self.pipeline) if stage.id is None]
			if missing:
				raise ValueError(
					"structural variants require stable IDs on every pipeline Stage; "
					+ f"missing at indexes {missing}"
				)
			known = set(ids)
			for variant in self.variants:
				unknown = sorted(set(variant.stages) - known)
				if unknown:
					raise ValueError(
						f"pipeline variant {variant.name!r} references unknown Stage IDs: {unknown}"
					)
		return self
