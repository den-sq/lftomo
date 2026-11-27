"""Memory-bounded and atomic construction of calibration bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from .bundle import CalibrationBundleValidationError, stream_sha256, validate_bundle
from .models import (
	ArrayPayload,
	BeamBandShiftModel,
	CalibrationBundleRef,
	CalibrationComponent,
	CalibrationManifest,
	CompatibilityMetadata,
	CoordinateFrame,
	ProducerInfo,
	SCHEMA_VERSION,
	SourceRecord,
	TimeAxis,
	ValidationResult,
)


class CalibrationBuildError(ValueError):
	"""Raised when a bundle build cannot be completed safely."""

	def __init__(self, message: str, staging_path: Path | None = None):
		self.staging_path = staging_path
		suffix = "" if staging_path is None else f" (partial build: {staging_path})"
		super().__init__(message + suffix)


@dataclass(frozen=True)
class BuildInspection:
	"""Status of a finalized, interrupted, failed, or corrupt build directory."""

	path: Path
	state: Literal["complete", "building", "failed", "corrupt", "unknown"]
	message: str
	calibration_id: str | None = None
	payloads: tuple[Path, ...] = ()


def _atomic_text(path: Path, value: str) -> None:
	temporary = path.with_name(f".{path.name}.tmp")
	temporary.write_text(value, encoding="utf-8")
	os.replace(temporary, path)


def inspect_build(path: str | Path, *, verify_hashes: bool = False) -> BuildInspection:
	"""Inspect a build without mistaking a partial directory for a valid bundle."""

	path = Path(path).expanduser().resolve()
	payloads = tuple(sorted((path / "components").glob("*"))) if path.is_dir() else ()
	failed = path / "BUILD_FAILED.json"
	if failed.is_file():
		try:
			metadata = json.loads(failed.read_text(encoding="utf-8"))
		except (OSError, ValueError):
			metadata = {}
		return BuildInspection(
			path, "failed", str(metadata.get("reason", "build failed")),
			calibration_id=metadata.get("calibration_id"), payloads=payloads,
		)
	manifest_path = path / "calibration.yaml"
	if manifest_path.is_file():
		try:
			report = validate_bundle(manifest_path, verify_hashes=verify_hashes)
		except (OSError, ValueError) as error:
			return BuildInspection(path, "corrupt", str(error), payloads=payloads)
		return BuildInspection(
			path,
			"complete" if report.valid else "corrupt",
			"bundle validation passed" if report.valid else "bundle validation failed",
			calibration_id=report.calibration_id,
			payloads=payloads,
		)
	building = path / "BUILDING.json"
	if building.is_file():
		try:
			metadata = json.loads(building.read_text(encoding="utf-8"))
		except (OSError, ValueError):
			metadata = {}
		return BuildInspection(
			path, "building", "bundle construction is incomplete",
			calibration_id=metadata.get("calibration_id"), payloads=payloads,
		)
	return BuildInspection(path, "unknown", "no calibration manifest or build marker found", payloads=payloads)


class IncrementalArrayWriter:
	"""Sequential-axis NPY writer that retains at most the caller's current chunk."""

	def __init__(
		self,
		session: "CalibrationBuildSession",
		name: str,
		*,
		shape: Sequence[int],
		dtype: str | np.dtype,
		axes: Sequence[str],
		kind: str,
		coordinate_frame: CoordinateFrame,
		append_axis: int = 0,
		compatibility: CompatibilityMetadata | dict[str, Any] | None = None,
		time_axis: TimeAxis | dict[str, Any] | None = None,
		interpolation: str = "nearest",
		validation_results: Iterable[ValidationResult | dict[str, Any]] = (),
		metadata: dict[str, Any] | None = None,
		band_shift: BeamBandShiftModel | dict[str, Any] | None = None,
	):
		if not name or "/" in name or "\\" in name:
			raise ValueError(f"invalid calibration component name {name!r}")
		self.session = session
		self.name = name
		self.shape = tuple(int(value) for value in shape)
		if not self.shape or any(value < 1 for value in self.shape):
			raise ValueError("component shape dimensions must be positive")
		self.dtype = np.dtype(dtype)
		self.axes = tuple(axes)
		if len(self.axes) != len(self.shape):
			raise ValueError("axes must name every component dimension")
		if not -len(self.shape) <= append_axis < len(self.shape):
			raise ValueError("append_axis is outside the component dimensions")
		self.append_axis = append_axis % len(self.shape)
		self.kind = kind
		self.coordinate_frame = coordinate_frame
		self.compatibility = CompatibilityMetadata.model_validate(compatibility or {})
		self.time_axis = None if time_axis is None else TimeAxis.model_validate(time_axis)
		self.interpolation = interpolation
		self.validation_results = [
			ValidationResult.model_validate(result) for result in validation_results
		]
		self.metadata = dict(metadata or {})
		self.band_shift = None if band_shift is None else BeamBandShiftModel.model_validate(band_shift)
		self.path = session.staging_path / "components" / f"{name}.npy"
		self._array = np.lib.format.open_memmap(
			self.path, mode="w+", dtype=self.dtype, shape=self.shape,
		)
		self._cursor = 0
		self._closed = False
		self.max_buffer_bytes = 0

	@property
	def complete(self) -> bool:
		return self._cursor == self.shape[self.append_axis]

	def append(self, values: Any) -> None:
		"""Append one full cross-section or bounded batch along ``append_axis``."""

		if self._closed:
			raise CalibrationBuildError(f"component {self.name!r} is already closed")
		array = np.asarray(values)
		if array.ndim == len(self.shape) - 1:
			array = np.expand_dims(array, axis=self.append_axis)
		if array.ndim != len(self.shape):
			raise ValueError(f"chunk for {self.name!r} has {array.ndim} dimensions; expected {len(self.shape)}")
		for axis, (actual, expected) in enumerate(zip(array.shape, self.shape, strict=True)):
			if axis != self.append_axis and actual != expected:
				raise ValueError(
					f"chunk for {self.name!r} has shape {array.shape}; axis {axis} must be {expected}"
				)
		stop = self._cursor + array.shape[self.append_axis]
		if stop > self.shape[self.append_axis]:
			raise ValueError(f"component {self.name!r} received more values than declared")
		selection = [slice(None)] * len(self.shape)
		selection[self.append_axis] = slice(self._cursor, stop)
		self._array[tuple(selection)] = array
		self._cursor = stop
		self.max_buffer_bytes = max(self.max_buffer_bytes, int(array.nbytes))

	def add_validation_result(self, result: ValidationResult | dict[str, Any]) -> None:
		self.validation_results.append(ValidationResult.model_validate(result))

	def close(self) -> CalibrationComponent:
		"""Flush, hash, describe, and register the completed component."""

		if self._closed:
			return self.session.components[self.name]
		if not self.complete:
			raise CalibrationBuildError(
				f"component {self.name!r} is incomplete: wrote {self._cursor} of "
				f"{self.shape[self.append_axis]} append-axis values",
				self.session.staging_path,
			)
		self._array.flush()
		self._array = None
		payload = ArrayPayload(
			path=self.path.relative_to(self.session.staging_path),
			format="npy",
			sha256=stream_sha256(self.path),
			shape=self.shape,
			dtype=self.dtype.name,
			axes=self.axes,
		)
		component = CalibrationComponent(
			kind=self.kind,
			payload=payload,
			coordinate_frame=self.coordinate_frame,
			compatibility=self.compatibility,
			time_axis=self.time_axis,
			interpolation=self.interpolation,
			validation_results=self.validation_results,
			metadata=self.metadata,
			band_shift=self.band_shift,
		)
		self.session._register_component(self.name, component)
		self._closed = True
		return component


class CalibrationBuildSession:
	"""Build a calibration bundle in a sibling directory and publish it atomically."""

	def __init__(
		self,
		target: str | Path,
		*,
		calibration_id: str,
		producer: ProducerInfo | dict[str, Any],
		compatibility: CompatibilityMetadata | dict[str, Any],
		sources: Iterable[SourceRecord | dict[str, Any]] = (),
		created_at: datetime | None = None,
		metadata: dict[str, Any] | None = None,
		schema_version: str = SCHEMA_VERSION,
	):
		if schema_version != SCHEMA_VERSION:
			raise ValueError(
				f"unsupported calibration build schema {schema_version!r}; expected {SCHEMA_VERSION!r}"
			)
		self.target = Path(target).expanduser().resolve()
		if self.target.exists():
			raise FileExistsError(f"calibration bundle target already exists: {self.target}")
		self.target.parent.mkdir(parents=True, exist_ok=True)
		self.staging_path = Path(tempfile.mkdtemp(
			prefix=f".{self.target.name}.building-", dir=self.target.parent,
		))
		(self.staging_path / "components").mkdir()
		self.calibration_id = calibration_id
		self.schema_version = schema_version
		self.producer = ProducerInfo.model_validate(producer)
		self.compatibility = CompatibilityMetadata.model_validate(compatibility)
		self.sources = [SourceRecord.model_validate(source) for source in sources]
		self.created_at = created_at or datetime.now(timezone.utc)
		self.metadata = dict(metadata or {})
		self.components: dict[str, CalibrationComponent] = {}
		self.validation_results: list[ValidationResult] = []
		self._writers: dict[str, IncrementalArrayWriter] = {}
		self._finalized = False
		self._failed = False
		_atomic_text(self.staging_path / "BUILDING.json", json.dumps({
			"schema_version": self.schema_version,
			"calibration_id": calibration_id,
			"target": str(self.target),
			"created_at": self.created_at.isoformat(),
		}, indent=2))

	def __enter__(self) -> "CalibrationBuildSession":
		return self

	def __exit__(self, exc_type, exc, traceback) -> bool:
		if exc is not None:
			self.abort(str(exc))
		elif not self._finalized and not self._failed:
			self.abort("build context exited without finalize()")
		return False

	def add_source(
		self,
		path: str | Path,
		*,
		uuid: str | None = None,
		sha256: str | None = None,
		hash_source: bool = False,
		acquisition_time: datetime | None = None,
		metadata: dict[str, Any] | None = None,
	) -> SourceRecord:
		"""Record source identity, optionally computing its hash with bounded reads."""

		path = Path(path).expanduser().resolve()
		if hash_source:
			actual = stream_sha256(path)
			if sha256 is not None and actual != sha256.lower():
				raise CalibrationBuildError(f"source hash does not match: {path}", self.staging_path)
			sha256 = actual
		record = SourceRecord(
			path=path, uuid=uuid, sha256=sha256,
			acquisition_time=acquisition_time, metadata=metadata or {},
		)
		self.sources.append(record)
		return record

	def create_array_component(self, name: str, **kwargs: Any) -> IncrementalArrayWriter:
		"""Create a sequential, memory-mapped NPY component writer."""

		if self._finalized or self._failed:
			raise CalibrationBuildError("build session is no longer writable", self.staging_path)
		if name in self._writers or name in self.components:
			raise ValueError(f"duplicate calibration component {name!r}")
		writer = IncrementalArrayWriter(self, name, **kwargs)
		self._writers[name] = writer
		return writer

	def write_array_component(
		self, name: str, chunks: Iterable[Any], **kwargs: Any,
	) -> CalibrationComponent:
		"""Consume a bounded chunk iterator into one component."""

		writer = self.create_array_component(name, **kwargs)
		for chunk in chunks:
			writer.append(chunk)
		return writer.close()

	def write_table_component(
		self,
		name: str,
		row_chunks: Iterable[Any],
		*,
		n_rows: int,
		columns: Sequence[str],
		dtype: str | np.dtype = "float64",
		metadata: dict[str, Any] | None = None,
		**kwargs: Any,
	) -> CalibrationComponent:
		"""Write a dense numeric table with explicit column names as bounded row chunks."""

		if not columns or len(set(columns)) != len(columns):
			raise ValueError("table column names must be non-empty and unique")
		component_metadata = {**(metadata or {}), "columns": list(columns)}
		return self.write_array_component(
			name,
			row_chunks,
			shape=(n_rows, len(columns)),
			dtype=dtype,
			axes=("T", "Feature"),
			append_axis=0,
			metadata=component_metadata,
			**kwargs,
		)

	def add_validation_result(self, result: ValidationResult | dict[str, Any]) -> None:
		self.validation_results.append(ValidationResult.model_validate(result))

	def add_component_validation(
		self, name: str, result: ValidationResult | dict[str, Any],
	) -> None:
		result = ValidationResult.model_validate(result)
		if name in self.components:
			self.components[name].validation_results.append(result)
		elif name in self._writers:
			self._writers[name].add_validation_result(result)
		else:
			raise KeyError(f"unknown calibration component {name!r}")

	def _register_component(self, name: str, component: CalibrationComponent) -> None:
		if name in self.components:
			raise ValueError(f"duplicate calibration component {name!r}")
		self.components[name] = component

	def abort(self, reason: str) -> None:
		"""Mark a partial build as failed while preserving it for diagnosis."""

		if self._finalized or self._failed:
			return
		for writer in self._writers.values():
			if writer._array is not None:
				writer._array.flush()
		_atomic_text(self.staging_path / "BUILD_FAILED.json", json.dumps({
			"schema_version": self.schema_version,
			"calibration_id": self.calibration_id,
			"target": str(self.target),
			"reason": reason,
			"failed_at": datetime.now(timezone.utc).isoformat(),
		}, indent=2))
		(self.staging_path / "BUILDING.json").unlink(missing_ok=True)
		self._failed = True

	def finalize(self, *, verify_hashes: bool = True) -> CalibrationBundleRef:
		"""Validate all descriptors and atomically publish the completed bundle."""

		if self._finalized:
			return CalibrationBundleRef(
				manifest_path=self.target / "calibration.yaml",
				calibration_id=self.calibration_id,
			)
		if self._failed:
			raise CalibrationBuildError("cannot finalize a failed build", self.staging_path)
		try:
			for writer in self._writers.values():
				writer.close()
			if not self.components:
				raise CalibrationBuildError("bundle has no completed components", self.staging_path)
			manifest = CalibrationManifest(
				calibration_id=self.calibration_id,
				created_at=self.created_at,
				producer=self.producer,
				sources=self.sources,
				compatibility=self.compatibility,
				components=self.components,
				validation_results=self.validation_results,
				metadata=self.metadata,
			)
			manifest_path = self.staging_path / "calibration.yaml"
			_atomic_text(manifest_path, manifest.to_yaml())
			validate_bundle(
				manifest_path, verify_hashes=verify_hashes, raise_on_error=True,
			)
			(self.staging_path / "BUILDING.json").unlink(missing_ok=True)
			if self.target.exists():
				raise FileExistsError(f"calibration bundle target appeared during build: {self.target}")
			os.replace(self.staging_path, self.target)
		except Exception as error:
			self.abort(str(error))
			if isinstance(error, CalibrationBuildError):
				raise
			if isinstance(error, CalibrationBundleValidationError):
				raise CalibrationBuildError(str(error), self.staging_path) from error
			raise CalibrationBuildError(f"calibration build failed: {error}", self.staging_path) from error
		self._finalized = True
		return CalibrationBundleRef(
			manifest_path=self.target / "calibration.yaml",
			calibration_id=self.calibration_id,
		)
