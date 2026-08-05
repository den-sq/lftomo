"""Filesystem and capacity preflight for durable experiment resources."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Callable, Literal

from pydantic import Field

from lftomo.config import ConfigModel


PathRole = Literal["result", "scratch", "cache"]
PathSeverity = Literal["info", "warning", "error"]
PathRelationshipKind = Literal["distinct", "same", "contains", "inside"]
ContextFlag = Literal[
	"inside_cwd", "inside_repository", "inside_session", "inside_config_directory",
]
AccessProbe = Callable[[Path, int], bool]
DiskUsageProbe = Callable[[Path], object]


class PathContextRoots(ConfigModel):
	"""Optional caller context used only to classify path placement."""

	cwd: Path | None = None
	repository: Path | None = None
	session: Path | None = None
	config_directory: Path | None = None


class PathResourceRequest(ConfigModel):
	"""Explicit roots, estimates, and guards for one experiment invocation."""

	schema_version: Literal[1] = 1
	result_root: Path
	scratch_root: Path | None = None
	cache_root: Path | None = None
	estimated_result_bytes: int | None = Field(default=None, ge=0)
	estimated_scratch_bytes: int | None = Field(default=None, ge=0)
	estimated_cache_bytes: int | None = Field(default=None, ge=0)
	result_guard_bytes: int | None = Field(default=None, gt=0)
	scratch_guard_bytes: int | None = Field(default=None, gt=0)
	cache_guard_bytes: int | None = Field(default=None, gt=0)
	reserve_free_bytes: int = Field(default=0, ge=0)
	contexts: PathContextRoots = Field(default_factory=PathContextRoots)


class PathRootReport(ConfigModel):
	"""Resolved objective facts for one storage role."""

	role: PathRole
	requested_path: Path
	resolved_path: Path
	exists: bool
	will_create: bool
	writable: bool
	probe_path: Path | None = None
	device_id: int | None = None
	total_bytes: int | None = Field(default=None, ge=0)
	free_bytes: int | None = Field(default=None, ge=0)
	estimated_bytes: int | None = Field(default=None, ge=0)
	guard_bytes: int | None = Field(default=None, gt=0)
	context_flags: tuple[ContextFlag, ...] = ()


class PathRelationship(ConfigModel):
	"""Canonical overlap/containment relationship between two roots."""

	left: PathRole
	right: PathRole
	relationship: PathRelationshipKind
	left_path: Path
	right_path: Path


class PathResourceDiagnostic(ConfigModel):
	"""One stable machine-readable path or capacity diagnostic."""

	code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
	severity: PathSeverity
	message: str = Field(min_length=1)
	roles: tuple[PathRole, ...] = ()


class PathResourceReport(ConfigModel):
	"""Versioned preflight report; callers decide contextual placement policy."""

	schema_version: Literal[1] = 1
	accepted: bool
	roots: tuple[PathRootReport, ...]
	relationships: tuple[PathRelationship, ...]
	diagnostics: tuple[PathResourceDiagnostic, ...] = ()

	def root(self, role: PathRole) -> PathRootReport | None:
		return next((item for item in self.roots if item.role == role), None)

	def human_summary(self) -> str:
		if self.accepted:
			return "path/resource preflight passed"
		lines = ["path/resource preflight failed before experiment execution:"]
		lines.extend(
			f"- {item.message}" for item in self.diagnostics if item.severity == "error"
		)
		return "\n".join(lines)


class PathResourceError(ValueError):
	"""Raised when objective root/capacity invariants reject an invocation."""

	def __init__(self, report: PathResourceReport):
		self.report = report
		super().__init__(report.human_summary())


class PathRelocationCompatibility(ConfigModel):
	"""Compatibility of new operational roots with an existing result store."""

	compatible: bool
	cache_reusable: bool
	diagnostics: tuple[PathResourceDiagnostic, ...] = ()


def _resolved(path: Path) -> Path:
	return path.expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
	try:
		path.relative_to(root)
	except ValueError:
		return False
	return True


def _context_flags(path: Path, contexts: PathContextRoots) -> tuple[ContextFlag, ...]:
	flags = []
	for field, flag in (
		("cwd", "inside_cwd"),
		("repository", "inside_repository"),
		("session", "inside_session"),
		("config_directory", "inside_config_directory"),
	):
		context = getattr(contexts, field)
		if context is not None and _is_within(path, _resolved(context)):
			flags.append(flag)
	return tuple(flags)


def _nearest_existing(path: Path) -> Path | None:
	probe = path
	while not probe.exists() and probe.parent != probe:
		probe = probe.parent
	return probe if probe.exists() else None


def _root_report(
	role: PathRole,
	path: Path,
	*,
	estimated_bytes: int | None,
	guard_bytes: int | None,
	contexts: PathContextRoots,
	access: AccessProbe,
	disk_usage: DiskUsageProbe,
) -> tuple[PathRootReport, list[PathResourceDiagnostic]]:
	resolved = _resolved(path)
	exists = resolved.exists()
	probe = _nearest_existing(resolved)
	diagnostics: list[PathResourceDiagnostic] = []
	writable = False
	device_id = None
	total_bytes = None
	free_bytes = None
	if probe is None:
		diagnostics.append(PathResourceDiagnostic(
			code="missing_parent",
			severity="error",
			message=f"{role} root {resolved} has no existing parent to create from",
			roles=(role,),
		))
	elif not probe.is_dir():
		diagnostics.append(PathResourceDiagnostic(
			code="not_a_directory",
			severity="error",
			message=f"{role} root {resolved} resolves through non-directory {probe}",
			roles=(role,),
		))
	else:
		writable = access(probe, os.W_OK | os.X_OK)
		if not writable:
			diagnostics.append(PathResourceDiagnostic(
				code="not_writable",
				severity="error",
				message=f"{role} root {resolved} is not writable via {probe}",
				roles=(role,),
			))
		try:
			device_id = probe.stat().st_dev
		except OSError as error:
			diagnostics.append(PathResourceDiagnostic(
				code="stat_failed",
				severity="error",
				message=f"cannot stat {role} root probe {probe}: {error}",
				roles=(role,),
			))
		try:
			usage = disk_usage(probe)
			total_bytes = int(getattr(usage, "total"))
			free_bytes = int(getattr(usage, "free"))
		except (OSError, TypeError, ValueError, AttributeError) as error:
			diagnostics.append(PathResourceDiagnostic(
				code="capacity_unavailable",
				severity="error",
				message=f"cannot determine capacity for {role} root {resolved}: {error}",
				roles=(role,),
			))
	if exists and not resolved.is_dir() and probe == resolved.parent:
		diagnostics.append(PathResourceDiagnostic(
			code="not_a_directory",
			severity="error",
			message=f"{role} root already exists and is not a directory: {resolved}",
			roles=(role,),
		))
	if estimated_bytes is not None and guard_bytes is not None and estimated_bytes > guard_bytes:
		diagnostics.append(PathResourceDiagnostic(
			code="guard_exceeded",
			severity="error",
			message=(
				f"estimated {role} use {estimated_bytes} bytes exceeds configured "
				f"guard {guard_bytes} bytes; reduce the estimate or explicitly "
				"override the configured guard after reviewing it"
			),
			roles=(role,),
		))
	return PathRootReport(
		role=role,
		requested_path=path,
		resolved_path=resolved,
		exists=exists,
		will_create=not exists,
		writable=writable,
		probe_path=probe,
		device_id=device_id,
		total_bytes=total_bytes,
		free_bytes=free_bytes,
		estimated_bytes=estimated_bytes,
		guard_bytes=guard_bytes,
		context_flags=_context_flags(resolved, contexts),
	), diagnostics


def _relationship(left: PathRootReport, right: PathRootReport) -> PathRelationship:
	if left.resolved_path == right.resolved_path:
		relationship: PathRelationshipKind = "same"
	elif _is_within(right.resolved_path, left.resolved_path):
		relationship = "contains"
	elif _is_within(left.resolved_path, right.resolved_path):
		relationship = "inside"
	else:
		relationship = "distinct"
	return PathRelationship(
		left=left.role,
		right=right.role,
		relationship=relationship,
		left_path=left.resolved_path,
		right_path=right.resolved_path,
	)


def preflight_path_resources(
	request: PathResourceRequest,
	*,
	access: AccessProbe = os.access,
	disk_usage: DiskUsageProbe = shutil.disk_usage,
) -> PathResourceReport:
	"""Resolve roots and validate objective durability/capacity invariants.

	Repository/session/cwd placement is reported as context flags, not rejected;
	operator-facing callers own that policy.
	"""

	specs = [(
		"result", request.result_root,
		request.estimated_result_bytes, request.result_guard_bytes,
	)]
	if request.scratch_root is not None:
		specs.append((
			"scratch", request.scratch_root,
			request.estimated_scratch_bytes, request.scratch_guard_bytes,
		))
	elif request.estimated_scratch_bytes is not None or request.scratch_guard_bytes is not None:
		specs.append(("scratch", None, None, None))
	if request.cache_root is not None:
		specs.append((
			"cache", request.cache_root,
			request.estimated_cache_bytes, request.cache_guard_bytes,
		))
	elif request.estimated_cache_bytes is not None or request.cache_guard_bytes is not None:
		specs.append(("cache", None, None, None))

	roots = []
	diagnostics: list[PathResourceDiagnostic] = []
	for role, path, estimate, guard in specs:
		if path is None:
			diagnostics.append(PathResourceDiagnostic(
				code="root_required",
				severity="error",
				message=f"{role} estimates/guards require an explicit {role} root",
				roles=(role,),
			))
			continue
		report, root_diagnostics = _root_report(
			role,
			path,
			estimated_bytes=estimate,
			guard_bytes=guard,
			contexts=request.contexts,
			access=access,
			disk_usage=disk_usage,
		)
		roots.append(report)
		diagnostics.extend(root_diagnostics)

	relationships = []
	for index, left in enumerate(roots):
		for right in roots[index + 1:]:
			relation = _relationship(left, right)
			relationships.append(relation)
			if relation.relationship == "distinct":
				continue
			roles = {left.role, right.role}
			if "result" in roles or relation.relationship == "same":
				diagnostics.append(PathResourceDiagnostic(
					code="ambiguous_root_overlap",
					severity="error",
					message=(
						f"{left.role} root {left.resolved_path} and {right.role} root "
						f"{right.resolved_path} are {relation.relationship}; durable result "
						"and disposable cleanup boundaries must not overlap"
					),
					roles=(left.role, right.role),
				))
			else:
				diagnostics.append(PathResourceDiagnostic(
					code="nested_disposable_roots",
					severity="warning",
					message=(
						f"{left.role} and {right.role} roots are {relation.relationship}; "
						"both are disposable, but callers should present the shared cleanup boundary"
					),
					roles=(left.role, right.role),
				))

	# Capacity is shared by device; do not compare each root independently and
	# accidentally over-commit one filesystem through several distinct paths.
	by_device: dict[int, list[PathRootReport]] = {}
	for root in roots:
		if root.device_id is not None and root.estimated_bytes is not None:
			by_device.setdefault(root.device_id, []).append(root)
	for device_id, device_roots in by_device.items():
		available = min(
			(root.free_bytes for root in device_roots if root.free_bytes is not None),
			default=None,
		)
		required = sum(root.estimated_bytes or 0 for root in device_roots)
		if available is not None and required + request.reserve_free_bytes > available:
			diagnostics.append(PathResourceDiagnostic(
				code="insufficient_capacity",
				severity="error",
				message=(
					f"filesystem device {device_id} has {available} free bytes but roles "
					f"{[root.role for root in device_roots]} require {required} bytes plus "
					f"{request.reserve_free_bytes} bytes reserve"
				),
				roles=tuple(root.role for root in device_roots),
			))

	return PathResourceReport(
		accepted=not any(item.severity == "error" for item in diagnostics),
		roots=tuple(roots),
		relationships=tuple(relationships),
		diagnostics=tuple(diagnostics),
	)


def require_path_resources(
	request: PathResourceRequest,
	**kwargs,
) -> PathResourceReport:
	"""Return an accepted report or raise with the complete structured result."""

	report = preflight_path_resources(request, **kwargs)
	if not report.accepted:
		raise PathResourceError(report)
	return report


def compare_path_resource_reports(
	previous: PathResourceReport,
	current: PathResourceReport,
) -> PathRelocationCompatibility:
	"""Validate operational root relocation for resume/rerun."""

	diagnostics = []
	prior_result = previous.root("result")
	new_result = current.root("result")
	if prior_result is None or new_result is None:
		diagnostics.append(PathResourceDiagnostic(
			code="result_identity_missing",
			severity="error",
			message="both path reports must contain the persistent result root",
			roles=("result",),
		))
	elif prior_result.resolved_path != new_result.resolved_path:
		diagnostics.append(PathResourceDiagnostic(
			code="result_identity_changed",
			severity="error",
			message=(
				f"persistent result root changed from {prior_result.resolved_path} "
				f"to {new_result.resolved_path}; use a new experiment store"
			),
			roles=("result",),
		))
	cache_reusable = True
	for role in ("scratch", "cache"):
		prior = previous.root(role)
		new = current.root(role)
		prior_path = None if prior is None else prior.resolved_path
		new_path = None if new is None else new.resolved_path
		if prior_path == new_path:
			continue
		if role == "cache":
			cache_reusable = False
		diagnostics.append(PathResourceDiagnostic(
			code=f"{role}_relocated",
			severity="info",
			message=(
				f"operational {role} root changed from {prior_path} to {new_path}; "
				+ ("cached prefixes must be rebuilt" if role == "cache" else "scientific identity is unchanged")
			),
			roles=(role,),
		))
	return PathRelocationCompatibility(
		compatible=not any(item.severity == "error" for item in diagnostics),
		cache_reusable=cache_reusable,
		diagnostics=tuple(diagnostics),
	)
