"""Versioned models for persistent tomography calibration bundles."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from lftomo.config.models import ConfigModel


SCHEMA_VERSION = "1.0"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

ComponentKind = Literal[
	"dark",
	"detector_static",
	"dynamic_beam",
	"beam_keyframes",
	"beam_schedule",
	"beam_residual",
	"beam_basis",
	"camera_registration",
]
CoordinateFrame = Literal["detector", "beam", "camera", "motor", "time"]
PayloadFormat = Literal["npy", "tiff", "hdf5"]
Interpolation = Literal["nearest", "linear", "log_linear"]


class DetectorRegion(ConfigModel):
	"""Detector-coordinate extent and orientation covered by a component."""

	y_start: int = Field(default=0, ge=0)
	x_start: int = Field(default=0, ge=0)
	height: int = Field(gt=0)
	width: int = Field(gt=0)
	binning_y: int = Field(default=1, gt=0)
	binning_x: int = Field(default=1, gt=0)
	orientation: Literal[
		"identity", "flip_x", "flip_y", "rotate_180", "transpose",
	] = "identity"

	def contains(self, other: "DetectorRegion") -> bool:
		"""Return whether ``other`` is a compatible subregion of this region."""

		return (
			self.binning_y == other.binning_y
			and self.binning_x == other.binning_x
			and self.orientation == other.orientation
			and self.y_start <= other.y_start
			and self.x_start <= other.x_start
			and self.y_start + self.height >= other.y_start + other.height
			and self.x_start + self.width >= other.x_start + other.width
		)


class MotorAxisCalibration(ConfigModel):
	"""Signed conversion from one recorded motor axis to detector pixels."""

	unit: str = Field(min_length=1)
	pixels_per_unit: float
	sign: Literal[-1, 1]


class CompatibilityMetadata(ConfigModel):
	"""Compatibility facts whose relevance depends on component kind."""

	detector_id: str | None = None
	beamline: str | None = None
	optics_id: str | None = None
	energy_keV: float | None = Field(default=None, gt=0)
	exposure_ms: float | None = Field(default=None, gt=0)
	temperature_c: float | None = None
	region: DetectorRegion | None = None
	static_artifact_epoch: str | None = None
	camera_segment: str | None = None
	beam_state: str | None = None
	training_coverage: list[str] = Field(default_factory=list)
	motor_axes: dict[str, MotorAxisCalibration] = Field(default_factory=dict)
	valid_from: datetime | None = None
	valid_until: datetime | None = None

	@model_validator(mode="after")
	def _validity_order(self):
		if self.valid_from is not None and self.valid_until is not None:
			if self.valid_until < self.valid_from:
				raise ValueError("valid_until must not precede valid_from")
		return self


class CalibrationTarget(CompatibilityMetadata):
	"""Facts about a scan against which calibration compatibility is checked."""

	acquisition_time: datetime | None = None


class TimeAxis(ConfigModel):
	"""Physical or projection-index time coordinates for a dynamic component."""

	kind: Literal["projection_index", "timestamp_seconds"]
	unit: str = Field(default="index", min_length=1)
	origin: datetime | None = None
	values: list[float] | None = None

	@field_validator("values")
	@classmethod
	def _strictly_increasing(cls, values: list[float] | None):
		if values is None:
			return values
		array = np.asarray(values, dtype=np.float64)
		if not np.isfinite(array).all():
			raise ValueError("time-axis values must be finite")
		if len(array) > 1 and not np.all(np.diff(array) > 0):
			raise ValueError("time-axis values must be strictly increasing")
		return values


class ArrayPayload(ConfigModel):
	"""One externally stored array with enough metadata for bounded access."""

	path: Path
	format: PayloadFormat
	dataset: str | None = None
	sha256: str
	shape: tuple[int, ...]
	dtype: str
	axes: tuple[str, ...]

	@field_validator("path")
	@classmethod
	def _bundle_relative_path(cls, path: Path):
		if path.is_absolute() or ".." in path.parts:
			raise ValueError("calibration payload paths must remain inside the bundle")
		return path

	@field_validator("sha256")
	@classmethod
	def _sha256(cls, value: str):
		value = value.lower()
		if not SHA256_PATTERN.fullmatch(value):
			raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")
		return value

	@field_validator("dtype")
	@classmethod
	def _dtype(cls, value: str):
		try:
			return np.dtype(value).name
		except (TypeError, ValueError) as error:
			raise ValueError(f"unsupported NumPy dtype {value!r}") from error

	@model_validator(mode="after")
	def _shape_axes_and_format(self):
		if not self.shape or any(size < 1 for size in self.shape):
			raise ValueError("payload shape dimensions must be positive")
		if len(self.shape) != len(self.axes):
			raise ValueError("payload axes must name every shape dimension")
		if len(set(self.axes)) != len(self.axes):
			raise ValueError("payload axes must be unique")
		if self.format == "hdf5" and not self.dataset:
			raise ValueError("HDF5 payloads require a dataset path")
		if self.format != "hdf5" and self.dataset is not None:
			raise ValueError("dataset is only valid for HDF5 payloads")
		return self


class ValidationResult(ConfigModel):
	"""One persisted or freshly computed calibration validation result."""

	name: str = Field(min_length=1)
	status: Literal["pass", "warning", "fail"]
	message: str
	component: str | None = None
	metrics: dict[str, Any] = Field(default_factory=dict)


class BundleValidationReport(ConfigModel):
	"""Integrity and descriptor-validation results for one bundle."""

	calibration_id: str
	valid: bool
	results: list[ValidationResult]


class BeamBandShiftModel(ConfigModel):
	"""Calibration-owned shift/stretch convention and its detector maps.

	The associated schedule has columns ``dy, m`` in detector pixels and
	magnification. These facts travel with that measurement, not Stage params.
	"""

	yc: float = Field(allow_inf_nan=False)
	dy_end: float | None = Field(default=None, allow_inf_nan=False)
	sigma_x: float = Field(default=60, gt=0, allow_inf_nan=False)
	s_max: float = Field(default=1.2, ge=0, allow_inf_nan=False)
	white_component: str = Field(default="pre_gain", min_length=1)
	dark_component: str = Field(default="pre_dark", min_length=1)
	residual_component: str | None = Field(default=None, min_length=1)

	def source_y(self, rows, dy, magnification):
		"""Map shifted detector rows back to the reference beam coordinates."""
		return self.yc + (rows - self.yc - dy) / magnification

	@model_validator(mode="after")
	def _residual_scale(self):
		if self.residual_component is not None and self.dy_end in (None, 0):
			raise ValueError("a beam residual requires a nonzero dy_end")
		return self


class CalibrationComponent(ConfigModel):
	"""Descriptor for one named component of a calibration bundle."""

	kind: ComponentKind
	payload: ArrayPayload
	coordinate_frame: CoordinateFrame
	compatibility: CompatibilityMetadata = Field(default_factory=CompatibilityMetadata)
	time_axis: TimeAxis | None = None
	interpolation: Interpolation = "nearest"
	validation_results: list[ValidationResult] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)
	band_shift: BeamBandShiftModel | None = None

	@model_validator(mode="after")
	def _dynamic_axes(self):
		if self.band_shift is not None:
			if (self.kind != "beam_schedule" or self.coordinate_frame != "beam"
					or self.payload.axes != ("Theta", "Parameter")
					or self.payload.shape[1] != 2 or self.time_axis is None
					or self.time_axis.kind != "projection_index"
					or self.interpolation != "nearest"):
				raise ValueError("band-shift schedules require beam coordinates, (Theta, Parameter) dy/m columns, and nearest projection-index sampling")
		if self.kind == "beam_residual" and (
			self.coordinate_frame != "beam" or self.payload.axes != ("Y", "X")
		):
			raise ValueError("beam_residual requires a static (Y, X) field in beam coordinates")
		temporal_axes = [axis for axis in ("Theta", "T") if axis in self.payload.axes]
		if len(temporal_axes) > 1:
			raise ValueError("components may have only one temporal axis")
		if temporal_axes and self.time_axis is None:
			raise ValueError("dynamic components require an explicit time axis")
		if not temporal_axes and self.time_axis is not None:
			raise ValueError("time_axis requires a Theta or T payload axis")
		if self.kind in ("dynamic_beam", "beam_keyframes", "beam_schedule") and not temporal_axes:
			raise ValueError(f"{self.kind} components require a Theta or T payload axis")
		if self.time_axis is not None and self.time_axis.values is not None:
			axis = self.payload.axes.index(temporal_axes[0])
			if len(self.time_axis.values) != self.payload.shape[axis]:
				raise ValueError("time-axis length must match the temporal payload dimension")
		return self


class SourceRecord(ConfigModel):
	"""One scan or file used to create a calibration component."""

	path: Path
	uuid: str | None = None
	sha256: str | None = None
	acquisition_time: datetime | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)

	@field_validator("sha256")
	@classmethod
	def _optional_sha256(cls, value: str | None):
		if value is not None and not SHA256_PATTERN.fullmatch(value.lower()):
			raise ValueError("source sha256 must contain 64 hexadecimal characters")
		return None if value is None else value.lower()


class ProducerInfo(ConfigModel):
	"""Software identity responsible for creating a calibration bundle."""

	package: str = Field(min_length=1)
	version: str = Field(min_length=1)
	git_commit: str | None = None
	parameters: dict[str, Any] = Field(default_factory=dict)


class CalibrationManifest(ConfigModel):
	"""Versioned manifest for persistent, cross-scan calibration artifacts."""

	schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
	calibration_id: str = Field(min_length=1)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	producer: ProducerInfo
	sources: list[SourceRecord] = Field(default_factory=list)
	compatibility: CompatibilityMetadata
	components: dict[str, CalibrationComponent] = Field(min_length=1)
	validation_results: list[ValidationResult] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)

	@model_validator(mode="after")
	def _detector_and_component_names(self):
		if self.compatibility.detector_id is None:
			raise ValueError("bundle compatibility requires detector_id")
		for name in self.components:
			if not name or "/" in name or "\\" in name:
				raise ValueError(f"invalid calibration component name {name!r}")
		return self


class CalibrationBundleRef(ConfigModel):
	"""Lightweight reference passed through runner and pipeline artifacts."""

	manifest_path: Path
	calibration_id: str
	schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION

	@property
	def root(self) -> Path:
		return self.manifest_path.parent


class CompatibilityDiagnostic(ConfigModel):
	"""One component-specific compatibility mismatch."""

	component: str
	field: str
	expected: Any = None
	actual: Any = None
	message: str


class CompatibilityOverride(ConfigModel):
	"""Explicit acknowledgement of otherwise-fatal compatibility mismatches."""

	reason: str = Field(min_length=1)
	diagnostics: list[CompatibilityDiagnostic]


class CompatibilityReport(ConfigModel):
	"""Serializable compatibility decision suitable for run provenance."""

	calibration_id: str
	components: list[str]
	compatible: bool
	diagnostics: list[CompatibilityDiagnostic] = Field(default_factory=list)
	override: CompatibilityOverride | None = None
