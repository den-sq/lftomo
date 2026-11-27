"""Active-runtime compatibility checks for experiment execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import os
import platform
from typing import Any, Callable, Literal, Mapping

from pydantic import Field

from lftomo.config import ConfigModel
from lftomo.pipeline.core import RuntimeRequirement
from lftomo.pipeline.runtime import (
	LibTomoIdentity, collect_libtomo_identity, tomopy_numpy_incompatible,
)

from .models import EnvironmentIdentity, ExperimentConfig


RuntimeSeverity = Literal["info", "warning", "error"]
SmokeStatus = Literal["passed", "failed", "not_required"]
ModuleLoader = Callable[[str], Any]


class RuntimeCompatibilityDiagnostic(ConfigModel):
	"""One actionable fact produced by active-runtime validation."""

	capability: str = Field(min_length=1)
	severity: RuntimeSeverity
	message: str = Field(min_length=1)
	package: str | None = None
	installed_version: str | None = None
	requirement: str | None = None
	evidence: str = Field(min_length=1)
	corrective_action: str | None = None
	stage_ids: tuple[str, ...] = ()


class RuntimeSmokeCheck(ConfigModel):
	"""A bounded import/API probe for one enabled runtime capability."""

	capability: str = Field(min_length=1)
	module: str = Field(min_length=1)
	status: SmokeStatus
	checked_attributes: tuple[str, ...] = ()
	message: str = Field(min_length=1)


class RuntimeCompatibilityReport(ConfigModel):
	"""Versioned, machine-readable result of active-runtime preflight."""

	schema_version: Literal[1] = 1
	compatible: bool
	environment: EnvironmentIdentity
	enabled_features: tuple[str, ...]
	diagnostics: tuple[RuntimeCompatibilityDiagnostic, ...] = ()
	smoke_checks: tuple[RuntimeSmokeCheck, ...] = ()

	def human_summary(self) -> str:
		"""Return compact actionable text suitable for CLI failure output."""

		if self.compatible:
			features = ", ".join(self.enabled_features) or "core"
			return f"runtime compatibility passed for {features}"
		lines = ["runtime compatibility failed before experiment execution:"]
		for diagnostic in self.diagnostics:
			if diagnostic.severity != "error":
				continue
			line = f"- {diagnostic.message}"
			if diagnostic.corrective_action:
				line += f" Corrective action: {diagnostic.corrective_action}"
			lines.append(line)
		return "\n".join(lines)


class RuntimeCompatibilityError(ValueError):
	"""Raised when active-runtime preflight rejects an experiment."""

	def __init__(self, report: RuntimeCompatibilityReport):
		self.report = report
		super().__init__(report.human_summary())


@dataclass
class _EnabledRequirement:
	declaration: RuntimeRequirement
	stage_ids: set[str] = field(default_factory=set)
	attribute_stage_ids: dict[str, set[str]] = field(default_factory=dict)

	@property
	def checked_attributes(self) -> tuple[str, ...]:
		return tuple(sorted(self.attribute_stage_ids))


def _possible_stage_keys(config: ExperimentConfig) -> dict[str, set[str]]:
	"""Return every Stage implementation selectable by the candidate space."""

	recon = config.candidate_space
	included_ids = None
	if recon.variants:
		included_ids = {stage_id for variant in recon.variants for stage_id in variant.stages}
	keys: dict[str, set[str]] = {}
	for index, stage in enumerate(recon.pipeline):
		if included_ids is not None and stage.id not in included_ids:
			continue
		stage_id = stage.id or f"stage-{index}"
		algorithms = [] if stage.algorithm is None else [stage.algorithm]
		if stage.sweep is not None and stage.sweep.kind == "cases":
			algorithms.extend(case.algorithm for case in stage.sweep.cases)
		for algorithm in algorithms:
			key = f"{stage.kind}.{algorithm}"
			keys.setdefault(key, set()).add(stage_id)
	return keys


def _enabled_runtime_requirements(
	stage_keys: Mapping[str, set[str]],
) -> tuple[_EnabledRequirement, ...]:
	"""Collect and deduplicate runtime requirements owned by selectable Stages."""

	from lftomo.pipeline.stages import STAGE_REGISTRY

	enabled: dict[tuple[str, str, str], _EnabledRequirement] = {}
	for stage_key, stage_ids in sorted(stage_keys.items()):
		stage_type = STAGE_REGISTRY.get(stage_key)
		if stage_type is None:
			continue
		for declaration in stage_type.contract.runtime_requirements:
			identity = (
				declaration.capability,
				declaration.package,
				declaration.module,
			)
			item = enabled.get(identity)
			if item is None:
				item = _EnabledRequirement(declaration=declaration)
				enabled[identity] = item
			elif item.declaration.corrective_action != declaration.corrective_action:
				raise ValueError(
					f"conflicting corrective actions declared for {declaration.capability!r}"
				)
			item.stage_ids.update(stage_ids)
			for attribute in declaration.probe_attributes:
				item.attribute_stage_ids.setdefault(attribute, set()).update(stage_ids)
	return tuple(enabled[identity] for identity in sorted(enabled))


def _metadata_version(package: str) -> str | None:
	try:
		return version(package)
	except PackageNotFoundError:
		return None


def _normalized_versions(
	packages: set[str], supplied: Mapping[str, str | None] | None,
) -> dict[str, str | None]:
	if supplied is None:
		return {package: _metadata_version(package) for package in sorted(packages)}
	lookup = {name.lower().replace("_", "-"): value for name, value in supplied.items()}
	return {
		package: lookup.get(package.lower().replace("_", "-"))
		for package in sorted(packages)
	}


def collect_environment_identity(
	*,
	package_versions: Mapping[str, str | None] | None = None,
	package_names: set[str] | None = None,
	backend: str = "numpy",
	libtomo: LibTomoIdentity | None = None,
) -> EnvironmentIdentity:
	"""Collect identity; supplied package versions require supplied native provenance."""

	if libtomo is None and package_versions is None:
		libtomo = collect_libtomo_identity()
	versions = _normalized_versions(
		package_names or {"numpy", "scipy", "pydantic"}, package_versions,
	)
	return EnvironmentIdentity(
		python=platform.python_version(),
		platform=platform.platform(),
		backend=backend,
		dependencies={name: value for name, value in versions.items() if value is not None},
		libtomo=libtomo,
		hardware={"cpu_count": os.cpu_count() or 1},
	)


def check_runtime_compatibility(
	config: ExperimentConfig,
	*,
	package_versions: Mapping[str, str | None] | None = None,
	module_loader: ModuleLoader = import_module,
	backend: str = "numpy",
	libtomo: LibTomoIdentity | None = None,
) -> RuntimeCompatibilityReport:
	"""Validate active packages and bounded APIs required by ``config``.

	``package_versions`` and ``module_loader`` are injectable so callers can
	qualify a prospective environment without mutating or importing from it.
	Supply its ``libtomo`` identity too, or native provenance remains unknown.
	The default probes the active interpreter, including source/PYTHONPATH
	imports that may not have installed distribution metadata.
	"""

	if libtomo is None and package_versions is None and module_loader is import_module:
		libtomo = collect_libtomo_identity()
	stage_keys = _possible_stage_keys(config)
	enabled = _enabled_runtime_requirements(stage_keys)
	package_names = {"numpy", "scipy", "pydantic"}
	package_names.update(item.declaration.package for item in enabled)
	versions = _normalized_versions(package_names, package_versions)
	diagnostics: list[RuntimeCompatibilityDiagnostic] = []
	smoke_checks: list[RuntimeSmokeCheck] = []

	for item in enabled:
		declaration = item.declaration
		stage_ids = tuple(sorted(item.stage_ids))
		checked_attributes = item.checked_attributes
		installed = versions.get(declaration.package)
		module = None
		try:
			module = module_loader(declaration.module)
		except Exception as error:
			smoke_checks.append(RuntimeSmokeCheck(
				capability=declaration.capability,
				module=declaration.module,
				status="failed",
				checked_attributes=checked_attributes,
				message=f"{type(error).__name__}: {error}",
			))
			diagnostics.append(RuntimeCompatibilityDiagnostic(
				capability=declaration.capability,
				severity="error",
				message=(
					f"enabled capability {declaration.capability!r} cannot import "
					f"{declaration.module!r}"
				),
				package=declaration.package,
				installed_version=installed,
				evidence=f"bounded import failed: {type(error).__name__}: {error}",
				corrective_action=declaration.corrective_action,
				stage_ids=stage_ids,
			))
			continue
		if installed is None:
			module_version = getattr(module, "__version__", None)
			if isinstance(module_version, str) and module_version:
				installed = module_version
				versions[declaration.package] = module_version
		missing_attributes = tuple(
			attribute for attribute in checked_attributes if not hasattr(module, attribute)
		)
		if missing_attributes:
			missing_stage_ids = tuple(sorted({
				stage_id
				for attribute in missing_attributes
				for stage_id in item.attribute_stage_ids[attribute]
			}))
			smoke_checks.append(RuntimeSmokeCheck(
				capability=declaration.capability,
				module=declaration.module,
				status="failed",
				checked_attributes=checked_attributes,
				message=f"missing required API attributes: {', '.join(missing_attributes)}",
			))
			diagnostics.append(RuntimeCompatibilityDiagnostic(
				capability=declaration.capability,
				severity="error",
				message=(
					f"{declaration.module!r} is importable but lacks required APIs: "
					f"{', '.join(missing_attributes)}"
				),
				package=declaration.package,
				installed_version=installed,
				evidence="bounded API smoke check",
				corrective_action=declaration.corrective_action,
				stage_ids=missing_stage_ids,
			))
		else:
			smoke_checks.append(RuntimeSmokeCheck(
				capability=declaration.capability,
				module=declaration.module,
				status="passed",
				checked_attributes=checked_attributes,
				message="module import and required API attributes passed",
			))

	# TomoPy 1.15.3 fixed the NumPy 2 copy-semantics incompatibility.
	tomopy_runtime = next(
		(item for item in enabled if item.declaration.capability == "tomopy"),
		None,
	)
	if tomopy_runtime is not None:
		numpy_version = versions.get("numpy")
		tomopy_version = versions.get("tomopy")
		if tomopy_numpy_incompatible(numpy_version, tomopy_version):
			diagnostics.append(RuntimeCompatibilityDiagnostic(
				capability="tomopy",
				severity="error",
				message=(
					f"NumPy {numpy_version} requires TomoPy >=1.15.3; "
					f"found TomoPy {tomopy_version}"
				),
				package="tomopy",
				installed_version=tomopy_version,
				requirement=">=1.15.3",
				evidence="TomoPy releases before 1.15.3 are incompatible with NumPy 2",
				corrective_action=(
					"install TomoPy 1.15.3 or newer in this interpreter and verify that "
					"this same interpreter imports it"
				),
				stage_ids=tuple(sorted(tomopy_runtime.stage_ids)),
			))

	environment = collect_environment_identity(
		package_versions=versions,
		package_names=package_names,
		backend=backend,
		libtomo=libtomo,
	)
	return RuntimeCompatibilityReport(
		compatible=not any(item.severity == "error" for item in diagnostics),
		environment=environment,
		enabled_features=tuple(sorted({
			item.declaration.capability for item in enabled
		})),
		diagnostics=tuple(diagnostics),
		smoke_checks=tuple(smoke_checks),
	)


def require_runtime_compatibility(
	config: ExperimentConfig,
	**kwargs: Any,
) -> RuntimeCompatibilityReport:
	"""Return a compatible report or raise with the complete structured result."""

	report = check_runtime_compatibility(config, **kwargs)
	if not report.compatible:
		raise RuntimeCompatibilityError(report)
	return report
