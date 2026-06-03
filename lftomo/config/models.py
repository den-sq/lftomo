"""Pydantic models for canonical lftomo configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class Helical(ConfigModel):
	"""Helical acquisition metadata."""

	enabled: bool = Field(default=False, description="Whether the scan used helical acquisition.")
	start_um: float | None = Field(default=None, description="Starting vertical position in micrometers.")
	stop_um: float | None = Field(default=None, description="Ending vertical position in micrometers.")
	pitch_um_per_projection: float | None = Field(default=None, description="Vertical pitch per projection in micrometers.")


class Acquisition(ConfigModel):
	"""Projection-count and angle metadata for a scan."""

	n_projections: int = Field(gt=0, description="Number of acquired projections.")
	rotation_start_deg: float = Field(default=0.0, description="Start angle in degrees.")
	rotation_end_deg: float = Field(description="End angle in degrees.")
	exposure_ms: float | None = Field(default=None, gt=0, description="Exposure time in milliseconds.")


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
	pixel_size_um: float = Field(gt=0, description="Effective reconstructed pixel size in micrometers.")
	ssd_cm: float | None = Field(default=None, gt=0, description="Sample-to-scintillator distance in centimeters.")
	n_projections: int = Field(gt=0, description="Number of acquired projections.")
	degrees_per_fov: float = Field(gt=0, description="Angular coverage per field of view.")
	n_fov: int = Field(default=1, gt=0, description="Number of fields of view in the scan.")
	helical: Helical = Field(default_factory=Helical, description="Helical acquisition metadata.")
	detector: Detector = Field(description="Detector geometry and identity.")
	acquisition: Acquisition = Field(description="Projection-count and angle metadata.")
	projections: ProjectionSource = Field(description="Projection storage description.")
	metadata: dict[str, Any] = Field(default_factory=dict, description="Additional lineage-specific metadata.")

	@field_validator("acquisition")
	@classmethod
	def _projection_counts_match(cls, acquisition: Acquisition, info):
		if "n_projections" in info.data and acquisition.n_projections != info.data["n_projections"]:
			raise ValueError("acquisition.n_projections must match scan.n_projections")
		return acquisition


class SweepConfig(ConfigModel):
	"""Parameter sweep description for a pipeline stage."""

	kind: Literal["list", "grid", "linspace", "zip"] = Field(description="Sweep expansion strategy.")
	params: dict[str, Any] = Field(default_factory=dict, description="Sweep parameter values.")


class StageConfig(ConfigModel):
	"""Single pipeline stage configuration entry."""

	kind: str = Field(description="Stage kind, such as read, normalize, phase, recon, or write.")
	algorithm: str | None = Field(default=None, description="Optional algorithm choice within the stage kind.")
	params: dict[str, Any] = Field(default_factory=dict, description="Stage-specific parameters.")
	sweep: SweepConfig | None = Field(default=None, description="Optional sweep applied at this stage.")


class Recon(ConfigModel):
	"""Per-run reconstruction pipeline configuration."""

	pipeline: list[StageConfig] = Field(description="Ordered reconstruction pipeline stages.")
	pipeline_template: str | None = Field(default=None, description="Optional named pipeline template.")
	output_path: Path | None = Field(default=None, description="Optional output directory for the run.")
	metadata: dict[str, Any] = Field(default_factory=dict, description="Additional run metadata.")
