"""Active-runtime compatibility checks for experiment execution."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import os
import platform
from typing import Any, Callable, Literal, Mapping

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import Field

from lftomo.config import ConfigModel

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


@dataclass(frozen=True)
class _FeatureRule:
	name: str
	stage_keys: frozenset[str]
	package: str
	module: str
	requirement: str | None
	attributes: tuple[str, ...]
	corrective_action: str


_TOMOPY_KEYS = frozenset({
	"center.entropy_local",
	"center.pc",
	"center.vo",
	"recon.gridrec",
	"recon.bp_cuda",
	"recon.cgls_cuda",
	"recon.em_cuda",
	"recon.fbp_cuda",
	"recon.fp_cuda",
	"recon.sart_cuda",
	"recon.sirt_cuda",
	"stripe.all_of_the_stripe",
})
_ASTRA_KEYS = frozenset(key for key in _TOMOPY_KEYS if key.endswith("_cuda"))
_TOMOPY_APIS = {
	"center.entropy_local": ("recon",),
	"center.pc": ("find_center_pc",),
	"center.vo": ("find_center_vo",),
	"recon.gridrec": ("recon", "circ_mask"),
	"stripe.all_of_the_stripe": ("remove_all_stripe",),
}

_FEATURE_RULES = (
	_FeatureRule(
		name="tomopy",
		stage_keys=_TOMOPY_KEYS,
		package="tomopy",
		module="tomopy",
		requirement=">=1,<2",
		attributes=(),
		corrective_action=(
			"install the lftomo tomopy extra in this interpreter with numpy<2 "
			"(for example: python -m pip install 'lftomo[tomopy]' 'numpy<2')"
		),
	),
	_FeatureRule(
		name="astra-cuda",
		stage_keys=_ASTRA_KEYS,
		package="astra-toolbox",
		module="astra",
		requirement=None,
		attributes=(),
		corrective_action="install a compatible ASTRA Toolbox build in this interpreter",
	),
	_FeatureRule(
		name="bac",
		stage_keys=frozenset({"phase.bac", "phase.bronnikov_aided"}),
		package="hotopy",
		module="hotopy.holo",
		requirement=None,
		attributes=("BronnikovAidedCorrection",),
		corrective_action="install lftomo with the bac extra in this interpreter",
	),
	_FeatureRule(
		name="super-resolution",
		stage_keys=frozenset({"super_resolve.drht"}),
		package="torch",
		module="torch",
		requirement=None,
		attributes=("load", "no_grad"),
		corrective_action="install lftomo with the super-resolution extra",
	),
	_FeatureRule(
		name="neuroglancer",
		stage_keys=frozenset({"write.neuroglancer"}),
		package="cloud-volume",
		module="cloudvolume",
		requirement=None,
		attributes=("CloudVolume",),
		corrective_action="install lftomo with the neuroglancer extra",
	),
)


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


def _version_satisfies(installed: str, requirement: str) -> bool:
	try:
		return Version(installed) in SpecifierSet(requirement)
	except (InvalidVersion, InvalidSpecifier):
		return False


def collect_environment_identity(
	*,
	package_versions: Mapping[str, str | None] | None = None,
	package_names: set[str] | None = None,
	backend: str = "numpy",
) -> EnvironmentIdentity:
	"""Collect active interpreter/package identity without assuming an env file."""

	versions = _normalized_versions(
		package_names or {"numpy", "scipy", "pydantic"}, package_versions,
	)
	return EnvironmentIdentity(
		python=platform.python_version(),
		platform=platform.platform(),
		backend=backend,
		dependencies={name: value for name, value in versions.items() if value is not None},
		hardware={"cpu_count": os.cpu_count() or 1},
	)


def check_runtime_compatibility(
	config: ExperimentConfig,
	*,
	package_versions: Mapping[str, str | None] | None = None,
	module_loader: ModuleLoader = import_module,
	backend: str = "numpy",
) -> RuntimeCompatibilityReport:
	"""Validate active packages and bounded APIs required by ``config``.

	``package_versions`` and ``module_loader`` are injectable so callers can
	qualify a prospective environment without mutating or importing from it.
	The default probes the active interpreter, including source/PYTHONPATH
	imports that may not have installed distribution metadata.
	"""

	stage_keys = _possible_stage_keys(config)
	enabled = [rule for rule in _FEATURE_RULES if rule.stage_keys & set(stage_keys)]
	package_names = {"numpy", "scipy", "pydantic"}
	package_names.update(rule.package for rule in enabled)
	versions = _normalized_versions(package_names, package_versions)
	diagnostics: list[RuntimeCompatibilityDiagnostic] = []
	smoke_checks: list[RuntimeSmokeCheck] = []

	for rule in enabled:
		active_keys = rule.stage_keys & set(stage_keys)
		stage_ids = tuple(sorted({
			stage_id for key in active_keys for stage_id in stage_keys[key]
		}))
		attributes = set(rule.attributes)
		if rule.name == "tomopy":
			for key in active_keys:
				attributes.update(_TOMOPY_APIS.get(key, ("recon", "astra") if key in _ASTRA_KEYS else ()))
		checked_attributes = tuple(sorted(attributes))
		installed = versions.get(rule.package)
		module = None
		try:
			module = module_loader(rule.module)
		except Exception as error:
			smoke_checks.append(RuntimeSmokeCheck(
				capability=rule.name,
				module=rule.module,
				status="failed",
				checked_attributes=checked_attributes,
				message=f"{type(error).__name__}: {error}",
			))
			diagnostics.append(RuntimeCompatibilityDiagnostic(
				capability=rule.name,
				severity="error",
				message=f"enabled capability {rule.name!r} cannot import {rule.module!r}",
				package=rule.package,
				installed_version=installed,
				requirement=rule.requirement,
				evidence=f"bounded import failed: {type(error).__name__}: {error}",
				corrective_action=rule.corrective_action,
				stage_ids=stage_ids,
			))
			continue
		if installed is None:
			module_version = getattr(module, "__version__", None)
			if isinstance(module_version, str) and module_version:
				installed = module_version
				versions[rule.package] = module_version
		missing_attributes = tuple(
			attribute for attribute in checked_attributes if not hasattr(module, attribute)
		)
		if missing_attributes:
			smoke_checks.append(RuntimeSmokeCheck(
				capability=rule.name,
				module=rule.module,
				status="failed",
				checked_attributes=checked_attributes,
				message=f"missing required API attributes: {', '.join(missing_attributes)}",
			))
			diagnostics.append(RuntimeCompatibilityDiagnostic(
				capability=rule.name,
				severity="error",
				message=(
					f"{rule.module!r} is importable but lacks required APIs: "
					f"{', '.join(missing_attributes)}"
				),
				package=rule.package,
				installed_version=installed,
				requirement=rule.requirement,
				evidence="bounded API smoke check",
				corrective_action=rule.corrective_action,
				stage_ids=stage_ids,
			))
		else:
			smoke_checks.append(RuntimeSmokeCheck(
				capability=rule.name,
				module=rule.module,
				status="passed",
				checked_attributes=checked_attributes,
				message="module import and required API attributes passed",
			))
		if rule.requirement is not None:
			if installed is None:
				diagnostics.append(RuntimeCompatibilityDiagnostic(
					capability=rule.name,
					severity="error",
					message=f"{rule.package!r} is importable but its version cannot be determined",
					package=rule.package,
					requirement=rule.requirement,
					evidence="distribution metadata and module __version__ are both absent",
					corrective_action=rule.corrective_action,
					stage_ids=stage_ids,
				))
			elif not _version_satisfies(installed, rule.requirement):
				diagnostics.append(RuntimeCompatibilityDiagnostic(
					capability=rule.name,
					severity="error",
					message=f"{rule.package} {installed} does not satisfy {rule.requirement}",
					package=rule.package,
					installed_version=installed,
					requirement=rule.requirement,
					evidence="active package version",
					corrective_action=rule.corrective_action,
					stage_ids=stage_ids,
				))

	# TomoPy 1.x uses NumPy APIs whose copy semantics changed in NumPy 2.
	if any(rule.name == "tomopy" for rule in enabled):
		numpy_version = versions.get("numpy")
		tomopy_version = versions.get("tomopy")
		try:
			incompatible_pair = (
				numpy_version is not None
				and tomopy_version is not None
				and Version(numpy_version).major >= 2
				and Version(tomopy_version).major < 2
			)
		except InvalidVersion:
			incompatible_pair = True
		if incompatible_pair:
			diagnostics.append(RuntimeCompatibilityDiagnostic(
				capability="tomopy",
				severity="error",
				message=(
					f"NumPy {numpy_version} is incompatible with TomoPy {tomopy_version}; "
					"the current TomoPy 1.x runtime contract requires numpy<2"
				),
				package="numpy",
				installed_version=numpy_version,
				requirement="<2",
				evidence="known TomoPy 1.x / NumPy 2.x incompatibility",
				corrective_action=(
					"activate/rebuild the lftomo environment with numpy<2 and verify that "
					"this same interpreter imports TomoPy"
				),
				stage_ids=tuple(sorted({
					stage_id for key in _TOMOPY_KEYS & set(stage_keys)
					for stage_id in stage_keys[key]
				})),
			))

	environment = collect_environment_identity(
		package_versions=versions,
		package_names=package_names,
		backend=backend,
	)
	return RuntimeCompatibilityReport(
		compatible=not any(item.severity == "error" for item in diagnostics),
		environment=environment,
		enabled_features=tuple(sorted(rule.name for rule in enabled)),
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
