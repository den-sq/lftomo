"""Loading, validation, and bounded reads for calibration bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tifffile as tf

from .models import (
	ArrayPayload,
	BundleValidationReport,
	CalibrationBundleRef,
	CalibrationComponent,
	CalibrationManifest,
	CompatibilityMetadata,
	ValidationResult,
)


class CalibrationBundleError(ValueError):
	"""Base error for malformed or unusable calibration bundles."""


class CalibrationBundleValidationError(CalibrationBundleError):
	"""Raised when payload integrity does not match the manifest."""

	def __init__(self, report: BundleValidationReport):
		self.report = report
		failures = [result.message for result in report.results if result.status == "fail"]
		super().__init__("calibration bundle validation failed: " + "; ".join(failures))


@dataclass(frozen=True)
class TimeBracket:
	"""Two bounded keyframes and the interpolation weight between them."""

	before_time: float
	before: np.ndarray
	after_time: float
	after: np.ndarray
	weight: float


def stream_sha256(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
	"""Hash one payload without reading it all into memory."""

	digest = hashlib.sha256()
	with Path(path).open("rb") as stream:
		for block in iter(lambda: stream.read(block_size), b""):
			digest.update(block)
	return digest.hexdigest()


def resolve_manifest_path(path: str | Path) -> Path:
	"""Resolve either a bundle directory or its calibration manifest path."""

	path = Path(path).expanduser().resolve()
	return path / "calibration.yaml" if path.is_dir() else path


def load_manifest(path: str | Path) -> tuple[CalibrationManifest, CalibrationBundleRef]:
	"""Load a manifest and return it with a lightweight typed reference."""

	manifest_path = resolve_manifest_path(path)
	if not manifest_path.is_file():
		raise FileNotFoundError(f"calibration manifest not found: {manifest_path}")
	manifest = CalibrationManifest.from_yaml(manifest_path.read_text(encoding="utf-8"))
	return manifest, CalibrationBundleRef(
		manifest_path=manifest_path,
		calibration_id=manifest.calibration_id,
		schema_version=manifest.schema_version,
	)


def _effective_compatibility(
	manifest: CalibrationManifest, component: CalibrationComponent,
) -> CompatibilityMetadata:
	base = manifest.compatibility.model_dump()
	override = component.compatibility.model_dump(exclude_none=True)
	for key in ("training_coverage", "motor_axes"):
		if not override.get(key):
			override.pop(key, None)
	return CompatibilityMetadata.model_validate({**base, **override})


def _payload_path(ref: CalibrationBundleRef, payload: ArrayPayload) -> Path:
	root = ref.root.resolve()
	path = (root / payload.path).resolve()
	if not path.is_relative_to(root):
		raise CalibrationBundleError(f"payload escapes calibration bundle: {payload.path}")
	return path


def _payload_metadata(path: Path, payload: ArrayPayload) -> tuple[tuple[int, ...], str]:
	if payload.format == "npy":
		array = np.load(path, mmap_mode="r", allow_pickle=False)
		return tuple(array.shape), np.dtype(array.dtype).name
	if payload.format == "hdf5":
		with h5py.File(path, "r") as handle:
			if payload.dataset not in handle:
				raise CalibrationBundleError(
					f"HDF5 dataset {payload.dataset!r} is missing from {path}"
				)
			dataset = handle[payload.dataset]
			return tuple(dataset.shape), np.dtype(dataset.dtype).name
	with tf.TiffFile(path) as tif:
		series = tif.series[0]
		return tuple(series.shape), np.dtype(series.dtype).name


def validate_bundle(
	path: str | Path,
	*,
	verify_hashes: bool = True,
	raise_on_error: bool = False,
) -> BundleValidationReport:
	"""Validate descriptors, payload metadata, and optionally streaming hashes."""

	manifest, ref = load_manifest(path)
	results: list[ValidationResult] = []
	for name, component in manifest.components.items():
		payload = component.payload
		try:
			payload_path = _payload_path(ref, payload)
		except CalibrationBundleError as error:
			results.append(ValidationResult(
				name="payload_path", status="fail", component=name, message=str(error),
			))
			continue
		if not payload_path.is_file():
			results.append(ValidationResult(
				name="payload_exists", status="fail", component=name,
				message=f"calibration payload is missing: {payload_path}",
			))
			continue
		try:
			shape, dtype = _payload_metadata(payload_path, payload)
		except (OSError, ValueError, CalibrationBundleError) as error:
			results.append(ValidationResult(
				name="payload_open", status="fail", component=name, message=str(error),
			))
			continue
		if shape != payload.shape:
			results.append(ValidationResult(
				name="payload_shape", status="fail", component=name,
				message=f"{name} declares shape {payload.shape} but payload has {shape}",
			))
		if dtype != payload.dtype:
			results.append(ValidationResult(
				name="payload_dtype", status="fail", component=name,
				message=f"{name} declares dtype {payload.dtype} but payload has {dtype}",
			))
		compatibility = _effective_compatibility(manifest, component)
		recommended_scopes = {
			"detector_static": ("static_artifact_epoch",),
			"dynamic_beam": ("camera_segment",),
			"beam_keyframes": ("camera_segment",),
		}
		missing_scopes = [
			field for field in recommended_scopes.get(component.kind, ())
			if getattr(compatibility, field) is None
		]
		if missing_scopes:
			results.append(ValidationResult(
				name="compatibility_scope", status="warning", component=name,
				message=(
					f"{name} does not declare recommended compatibility scope(s): "
					+ ", ".join(missing_scopes)
				),
				metrics={"missing": missing_scopes},
			))
		if compatibility.region is not None and "Y" in payload.axes and "X" in payload.axes:
			y_size = payload.shape[payload.axes.index("Y")]
			x_size = payload.shape[payload.axes.index("X")]
			if (y_size, x_size) != (compatibility.region.height, compatibility.region.width):
				results.append(ValidationResult(
					name="payload_region", status="fail", component=name,
					message=(
						f"{name} detector axes {(y_size, x_size)} do not match declared "
						f"region {(compatibility.region.height, compatibility.region.width)}"
					),
				))
		if verify_hashes:
			actual_hash = stream_sha256(payload_path)
			if actual_hash != payload.sha256:
				results.append(ValidationResult(
					name="payload_sha256", status="fail", component=name,
					message=f"{name} payload hash does not match calibration.yaml",
					metrics={"expected": payload.sha256, "actual": actual_hash},
				))
		if not any(result.component == name and result.status == "fail" for result in results):
			results.append(ValidationResult(
				name="payload", status="pass", component=name,
				message=f"{name} payload metadata and integrity are valid",
			))
	report = BundleValidationReport(
		calibration_id=manifest.calibration_id,
		valid=not any(result.status == "fail" for result in results),
		results=results,
	)
	if raise_on_error and not report.valid:
		raise CalibrationBundleValidationError(report)
	return report


class CalibrationBundle:
	"""Loaded manifest with bounded readers for its external payloads."""

	def __init__(self, manifest: CalibrationManifest, ref: CalibrationBundleRef):
		self.manifest = manifest
		self.ref = ref

	@classmethod
	def open(
		cls, path: str | Path, *, verify_hashes: bool = False,
	) -> "CalibrationBundle":
		manifest, ref = load_manifest(path)
		if verify_hashes:
			validate_bundle(ref.manifest_path, verify_hashes=True, raise_on_error=True)
		return cls(manifest, ref)

	def component(self, name: str) -> CalibrationComponent:
		try:
			return self.manifest.components[name]
		except KeyError as error:
			raise KeyError(
				f"calibration {self.manifest.calibration_id!r} has no component {name!r}"
			) from error

	def compatibility(self, name: str) -> CompatibilityMetadata:
		return _effective_compatibility(self.manifest, self.component(name))

	def _y_slice(
		self, component: CalibrationComponent, y_start: int | None, y_stop: int | None,
	) -> slice:
		if "Y" not in component.payload.axes:
			if y_start is not None or y_stop is not None:
				raise CalibrationBundleError("Y window requested for a component without a Y axis")
			return slice(None)
		region = self.compatibility_for_component(component).region
		y_size = component.payload.shape[component.payload.axes.index("Y")]
		base = 0 if region is None else region.y_start
		start = base if y_start is None else y_start
		stop = base + y_size if y_stop is None else y_stop
		if not base <= start < stop <= base + y_size:
			raise CalibrationBundleError(
				f"requested detector Y window [{start}, {stop}) is outside [{base}, {base + y_size})"
			)
		return slice(start - base, stop - base)

	def compatibility_for_component(
		self, component: CalibrationComponent,
	) -> CompatibilityMetadata:
		return _effective_compatibility(self.manifest, component)

	def _read_selection(self, payload: ArrayPayload, selection: tuple[Any, ...]) -> np.ndarray:
		path = _payload_path(self.ref, payload)
		if payload.format == "npy":
			array = np.load(path, mmap_mode="r", allow_pickle=False)
			return np.asarray(array[selection])
		if payload.format == "hdf5":
			with h5py.File(path, "r") as handle:
				return np.asarray(handle[payload.dataset][selection])
		try:
			array = tf.memmap(path)
		except (ValueError, OSError) as error:
			raise CalibrationBundleError(
				f"TIFF calibration payload must be memory-mappable for bounded reads: {path}"
			) from error
		return np.asarray(array[selection])

	def _selection(
		self,
		component: CalibrationComponent,
		y_start: int | None,
		y_stop: int | None,
		temporal_index: int | None,
	) -> tuple[Any, ...]:
		selection: list[Any] = [slice(None)] * len(component.payload.axes)
		if "Y" in component.payload.axes:
			selection[component.payload.axes.index("Y")] = self._y_slice(component, y_start, y_stop)
		if temporal_index is not None:
			temporal = "Theta" if "Theta" in component.payload.axes else "T"
			selection[component.payload.axes.index(temporal)] = temporal_index
		return tuple(selection)

	def _time_values(self, component: CalibrationComponent) -> np.ndarray:
		if component.time_axis is None:
			raise CalibrationBundleError("component has no time axis")
		temporal = "Theta" if "Theta" in component.payload.axes else "T"
		count = component.payload.shape[component.payload.axes.index(temporal)]
		values = component.time_axis.values
		return np.arange(count, dtype=np.float64) if values is None else np.asarray(values)

	def read_time_bracket(
		self,
		name: str,
		time_value: float,
		*,
		y_start: int | None = None,
		y_stop: int | None = None,
	) -> TimeBracket:
		"""Read at most two Y-windowed frames bracketing a physical time."""

		component = self.component(name)
		values = self._time_values(component)
		if time_value < values[0] or time_value > values[-1]:
			raise CalibrationBundleError(
				f"requested time {time_value} is outside component range "
				f"[{values[0]}, {values[-1]}]"
			)
		if time_value == values[0]:
			before = after = 0
		elif time_value == values[-1]:
			before = after = len(values) - 1
		else:
			after = int(np.searchsorted(values, time_value, side="right"))
			before = after - 1
		before_array = self._read_selection(
			component.payload, self._selection(component, y_start, y_stop, before),
		)
		after_array = before_array if before == after else self._read_selection(
			component.payload, self._selection(component, y_start, y_stop, after),
		)
		span = values[after] - values[before]
		weight = 0.0 if span == 0 else float((time_value - values[before]) / span)
		return TimeBracket(
			before_time=float(values[before]), before=before_array,
			after_time=float(values[after]), after=after_array, weight=weight,
		)

	def read_component(
		self,
		name: str,
		*,
		y_start: int | None = None,
		y_stop: int | None = None,
		projection_index: int | None = None,
		time_value: float | None = None,
	) -> np.ndarray:
		"""Read a static Y window or evaluate one dynamic projection-time window."""

		component = self.component(name)
		temporal = "Theta" if "Theta" in component.payload.axes else (
			"T" if "T" in component.payload.axes else None
		)
		if temporal is None:
			if projection_index is not None or time_value is not None:
				raise CalibrationBundleError("time selection requested for a static component")
			return self._read_selection(
				component.payload, self._selection(component, y_start, y_stop, None),
			)
		query = time_value
		if query is None and projection_index is not None:
			if component.time_axis.kind != "projection_index":
				raise CalibrationBundleError(
					f"dynamic component {name!r} uses physical timestamps; provide time_value"
				)
			query = float(projection_index)
		if query is None:
			raise CalibrationBundleError(
				f"dynamic component {name!r} requires projection_index or time_value"
			)
		bracket = self.read_time_bracket(
			name, query, y_start=y_start, y_stop=y_stop,
		)
		if component.interpolation == "nearest":
			return bracket.before if bracket.weight < 0.5 else bracket.after
		if component.interpolation == "linear":
			return (1 - bracket.weight) * bracket.before + bracket.weight * bracket.after
		if np.any(bracket.before <= 0) or np.any(bracket.after <= 0):
			raise CalibrationBundleError("log-linear interpolation requires positive fields")
		return np.exp(
			(1 - bracket.weight) * np.log(bracket.before)
			+ bracket.weight * np.log(bracket.after)
		)
