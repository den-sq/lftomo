"""Component-specific compatibility decisions for calibration bundles."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Literal

import numpy as np

from .bundle import CalibrationBundle
from .models import (
	CalibrationTarget,
	CompatibilityDiagnostic,
	CompatibilityOverride,
	CompatibilityReport,
	DetectorRegion,
)


class CalibrationCompatibilityError(ValueError):
	"""Raised when a selected bundle is incompatible with a target scan."""

	def __init__(self, report: CompatibilityReport):
		self.report = report
		messages = "; ".join(item.message for item in report.diagnostics)
		super().__init__(f"calibration {report.calibration_id!r} is incompatible: {messages}")


_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
	"dark": ("detector_id", "region", "exposure_ms", "temperature_c"),
	"detector_static": ("detector_id", "optics_id", "region", "static_artifact_epoch"),
	"dynamic_beam": (
		"detector_id", "beamline", "optics_id", "energy_keV", "region",
		"camera_segment", "beam_state",
	),
	"beam_keyframes": (
		"detector_id", "beamline", "optics_id", "energy_keV", "region",
		"camera_segment", "beam_state",
	),
	"beam_schedule": ("beamline", "energy_keV", "camera_segment", "beam_state"),
	"beam_basis": (
		"detector_id", "beamline", "optics_id", "energy_keV", "region",
	),
	"camera_registration": ("detector_id", "optics_id", "region", "motor_axes"),
}


def _plain(value: Any) -> Any:
	if hasattr(value, "model_dump"):
		return value.model_dump(mode="json")
	if isinstance(value, datetime):
		return value.isoformat()
	return value


def _matches(field: str, expected: Any, actual: Any) -> bool:
	if isinstance(expected, DetectorRegion):
		return isinstance(actual, DetectorRegion) and expected.contains(actual)
	if field in ("energy_keV", "exposure_ms", "temperature_c"):
		return actual is not None and bool(np.isclose(expected, actual, rtol=0.01, atol=1e-6))
	return expected == actual


def _diagnostic(component: str, field: str, expected: Any, actual: Any) -> CompatibilityDiagnostic:
	return CompatibilityDiagnostic(
		component=component,
		field=field,
		expected=_plain(expected),
		actual=_plain(actual),
		message=(
			f"component {component!r} requires {field}={_plain(expected)!r}; "
			f"target provides {_plain(actual)!r}"
		),
	)


def check_compatibility(
	bundle: CalibrationBundle,
	target: CalibrationTarget,
	*,
	components: Iterable[str] | None = None,
	policy: Literal["error", "report", "override"] = "error",
	override_reason: str | None = None,
) -> CompatibilityReport:
	"""Apply component-specific matching rules with a default-error policy."""

	names = list(bundle.manifest.components if components is None else components)
	diagnostics: list[CompatibilityDiagnostic] = []
	for name in names:
		component = bundle.component(name)
		expected = bundle.compatibility(name)
		for field in _FIELDS_BY_KIND[component.kind]:
			expected_value = getattr(expected, field)
			if expected_value is None or expected_value == [] or expected_value == {}:
				continue
			actual_value = getattr(target, field)
			if not _matches(field, expected_value, actual_value):
				diagnostics.append(_diagnostic(name, field, expected_value, actual_value))
		if expected.valid_from is not None:
			if target.acquisition_time is None or target.acquisition_time < expected.valid_from:
				diagnostics.append(_diagnostic(
					name, "valid_from", expected.valid_from, target.acquisition_time,
				))
		if expected.valid_until is not None:
			if target.acquisition_time is None or target.acquisition_time > expected.valid_until:
				diagnostics.append(_diagnostic(
					name, "valid_until", expected.valid_until, target.acquisition_time,
				))
		if component.kind == "beam_basis" and expected.training_coverage:
			if target.beam_state is None or target.beam_state not in expected.training_coverage:
				diagnostics.append(CompatibilityDiagnostic(
					component=name,
					field="training_coverage",
					expected=expected.training_coverage,
					actual=target.beam_state,
					message=(
						f"component {name!r} training library does not cover target beam state "
						f"{target.beam_state!r}"
					),
				))

	override = None
	if diagnostics and policy == "override":
		if not override_reason:
			raise ValueError("compatibility override requires a non-empty reason")
		override = CompatibilityOverride(reason=override_reason, diagnostics=diagnostics)
	report = CompatibilityReport(
		calibration_id=bundle.manifest.calibration_id,
		components=names,
		compatible=not diagnostics or override is not None,
		diagnostics=diagnostics,
		override=override,
	)
	if diagnostics and policy == "error":
		raise CalibrationCompatibilityError(report)
	return report

