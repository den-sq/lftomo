"""Opt-in, content-addressed common-prefix snapshots for experiment trials."""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import fcntl
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable, Literal, Sequence
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator

from lftomo.config import ConfigModel, StageConfig
from lftomo.pipeline import (
	BufferSpec,
	ExecutionSpanObserver,
	Pipeline,
	PipelineState,
	Stage,
	StageObserver,
)
from lftomo.pipeline.domains import DataDomain
from lftomo.storage import FileBackedNPArray, ManagedBuffer, buffer_view
from lftomo.util.converter import CONVERTER
from lftomo.util.shapes import DataRange, DataShape, Interval
from lftomo.util.window import DataRef, TomoDataSet, TomoWindow

from .models import canonical_json


class PrefixCacheError(RuntimeError):
	"""Base error for safe prefix-cache fallback decisions."""


class PrefixCacheIntegrityError(PrefixCacheError):
	"""A published snapshot failed integrity or equivalence checks."""


class PrefixCacheCapacityError(PrefixCacheError):
	"""A snapshot cannot fit inside the declared cache bounds."""


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


def _sha256_bytes(value: bytes) -> str:
	return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(chunk_bytes), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _fingerprint(value: Any) -> Any:
	"""Return a deterministic identity document or reject an unsafe value."""

	if isinstance(value, ManagedBuffer):
		return _fingerprint(value.array())
	if isinstance(value, type):
		return {"kind": "type", "value": f"{value.__module__}.{value.__qualname__}"}
	if hasattr(value, "to_document"):
		return {
			"kind": "document",
			"type": f"{type(value).__module__}.{type(value).__qualname__}",
			"value": _fingerprint(value.to_document()),
		}
	if hasattr(value, "model_dump"):
		return {
			"kind": "model",
			"type": f"{type(value).__module__}.{type(value).__qualname__}",
			"value": _fingerprint(value.model_dump(mode="python")),
		}
	if is_dataclass(value) and not isinstance(value, type):
		return {
			"kind": "dataclass",
			"type": f"{type(value).__module__}.{type(value).__qualname__}",
			"value": _fingerprint(CONVERTER.unstructure(value)),
		}
	if isinstance(value, np.ndarray):
		if value.dtype.hasobject:
			raise TypeError("object arrays do not have a safe cache identity")
		array = np.ascontiguousarray(value)
		return {
			"kind": "ndarray",
			"dtype": array.dtype.str,
			"shape": list(array.shape),
			"sha256": _sha256_bytes(memoryview(array).cast("B")),
		}
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, Path):
		return {"kind": "path", "value": str(value)}
	if isinstance(value, Enum):
		return {
			"kind": "enum",
			"type": f"{type(value).__module__}.{type(value).__qualname__}",
			"value": value.name,
		}
	if isinstance(value, dict):
		return {str(key): _fingerprint(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_fingerprint(item) for item in value]
	if isinstance(value, (set, frozenset)):
		items = [_fingerprint(item) for item in value]
		return sorted(items, key=canonical_json)
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	raise TypeError(
		f"{type(value).__module__}.{type(value).__qualname__} lacks a cache identity contract"
	)


def _hash_document(value: Any) -> str:
	return _sha256_bytes(canonical_json(_fingerprint(value)).encode("utf-8"))


def _state_value_to_document(value: Any) -> Any:
	"""Encode an ordinary snapshot value without persisting an object graph."""

	if isinstance(value, np.ndarray):
		return {"kind": "ndarray", "value": CONVERTER.unstructure(value)}
	if isinstance(value, np.generic):
		return _state_value_to_document(value.item())
	if isinstance(value, Path):
		return {"kind": "path", "value": str(value)}
	if isinstance(value, Enum):
		return {
			"kind": "enum", "module": type(value).__module__,
			"qualname": type(value).__qualname__, "name": value.name,
		}
	if hasattr(value, "model_dump"):
		return {
			"kind": "model", "module": type(value).__module__,
			"qualname": type(value).__qualname__,
			"value": _state_value_to_document(value.model_dump(mode="python")),
		}
	if isinstance(value, dict):
		if any(not isinstance(key, str) for key in value):
			raise TypeError("snapshot document dictionaries require string keys")
		return {
			"kind": "dict",
			"items": {key: _state_value_to_document(item) for key, item in value.items()},
		}
	if isinstance(value, list):
		return {"kind": "list", "items": [_state_value_to_document(item) for item in value]}
	if isinstance(value, tuple):
		return {"kind": "tuple", "items": [_state_value_to_document(item) for item in value]}
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	raise TypeError(f"{type(value).__module__}.{type(value).__qualname__} is not documentable")


def _document_type(module_name: str, qualname: str) -> type:
	if not module_name.startswith("lftomo.") or "<locals>" in qualname:
		raise TypeError(f"snapshot document type {module_name}.{qualname} is not allowed")
	value = import_module(module_name)
	for name in qualname.split("."):
		value = getattr(value, name)
	if not isinstance(value, type):
		raise TypeError(f"snapshot document target {module_name}.{qualname} is not a type")
	return value


def _state_value_from_document(document: Any) -> Any:
	if document is None or isinstance(document, (str, int, float, bool)):
		return document
	if not isinstance(document, dict) or not isinstance(document.get("kind"), str):
		raise TypeError("snapshot value document is invalid")
	kind = document["kind"]
	if kind == "ndarray":
		return CONVERTER.structure(document["value"], np.ndarray)
	if kind == "path":
		return Path(document["value"])
	if kind in {"enum", "model"}:
		target = _document_type(document["module"], document["qualname"])
		if kind == "enum":
			return target[document["name"]]
		return target.model_validate(_state_value_from_document(document["value"]))
	if kind == "dict":
		return {key: _state_value_from_document(item) for key, item in document["items"].items()}
	if kind in {"list", "tuple"}:
		items = [_state_value_from_document(item) for item in document["items"]]
		return items if kind == "list" else tuple(items)
	raise ValueError(f"unknown snapshot value document kind {kind!r}")


def _buffer_spec_to_document(value: BufferSpec | None) -> dict[str, Any] | None:
	return None if value is None else CONVERTER.unstructure(value)


def _buffer_spec_from_document(document: dict[str, Any] | None) -> BufferSpec | None:
	return None if document is None else CONVERTER.structure(document, BufferSpec)


def _clone_state_value(value: Any) -> Any:
	"""Clone ordinary state while preserving run-owned managed handles."""

	if isinstance(value, ManagedBuffer):
		return value
	if isinstance(value, dict):
		return {key: _clone_state_value(item) for key, item in value.items()}
	if isinstance(value, list):
		return [_clone_state_value(item) for item in value]
	if isinstance(value, tuple):
		return tuple(_clone_state_value(item) for item in value)
	if isinstance(value, (BufferSpec, DataRange, DataRef, DataShape, Interval)):
		return value
	return copy.deepcopy(value)


def _managed_buffers(value: Any) -> tuple[ManagedBuffer, ...]:
	result = []
	seen = set()

	def collect(item):
		if isinstance(item, ManagedBuffer):
			identity = getattr(item, "name", id(item))
			if identity not in seen:
				seen.add(identity)
				result.append(item)
		elif isinstance(item, dict):
			for nested in item.values():
				collect(nested)
		elif isinstance(item, (list, tuple)):
			for nested in item:
				collect(nested)

	collect(value)
	return tuple(result)


@contextmanager
def _pinned_state(state: PipelineState, *, owner: str):
	"""Keep shared inputs live while a cache/fork operation retains them."""

	storage = state.storage
	buffers = _managed_buffers((state.working_array, state.artifacts))
	if storage is None:
		yield
		return
	for buffer in buffers:
		storage.pin(buffer, owner=owner)
	try:
		yield
	finally:
		for buffer in reversed(buffers):
			storage.unpin(buffer, owner=owner)


def _identity_differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
	if isinstance(expected, dict) and isinstance(actual, dict):
		fields = []
		for key in sorted(set(expected) | set(actual)):
			field = f"{prefix}.{key}" if prefix else str(key)
			if key not in expected or key not in actual:
				fields.append(field)
			else:
				fields.extend(_identity_differences(expected[key], actual[key], field))
		return fields
	if isinstance(expected, list) and isinstance(actual, list):
		fields = []
		if len(expected) != len(actual):
			fields.append(f"{prefix}.length")
		for index, (left, right) in enumerate(zip(expected, actual)):
			fields.extend(_identity_differences(left, right, f"{prefix}[{index}]"))
		return fields
	return [] if expected == actual else [prefix]


class PrefixCachePolicy(ConfigModel):
	"""Explicit opt-in controls and hard resource bounds for prefix snapshots."""

	enabled: bool = False
	scratch: Path | None = None
	max_disk_bytes: int | None = Field(default=None, gt=0)
	max_memory_bytes: int | None = Field(default=None, gt=0)
	max_entries: int = Field(default=64, gt=0)
	on_error: Literal["rebuild", "bypass", "fail"] = "rebuild"
	verify_fraction: float = Field(default=0.05, ge=0, le=1)
	verify_rtol: float = Field(default=1e-6, ge=0)
	verify_atol: float = Field(default=1e-7, ge=0)

	@model_validator(mode="after")
	def _enabled_cache_is_explicitly_bounded(self):
		if self.enabled and (
			self.scratch is None
			or self.max_disk_bytes is None
			or self.max_memory_bytes is None
		):
			raise ValueError(
				"enabled prefix caching requires explicit scratch, max_disk_bytes, "
				"and max_memory_bytes"
			)
		return self


class PrefixCacheDecision(ConfigModel):
	"""Dry-run-visible decision for one trial/window pipeline."""

	candidate_id: str
	action: Literal["cache", "independent", "bypass"]
	prefix_id: str | None = None
	stage_count: int = Field(default=0, ge=0)
	stage_ids: tuple[str, ...] = ()
	estimated_snapshot_bytes: int | None = Field(default=None, ge=0)
	requires_array_copy: bool = False
	reason: str


class PrefixCachePlan(ConfigModel):
	"""Deterministic prefix plan embedded in experiment preflight output."""

	schema_version: Literal[1] = 1
	enabled: bool
	scratch: Path | None
	estimated_unique_snapshot_bytes: int | None = Field(default=None, ge=0)
	decisions: tuple[PrefixCacheDecision, ...]


class PrefixCacheEvent(ConfigModel):
	"""Per-window cache provenance and lightweight benchmark measurements."""

	status: Literal["hit", "miss", "rebuilt", "bypass", "independent"]
	prefix_id: str | None = None
	cache_key: str | None = None
	stage_count: int = Field(default=0, ge=0)
	snapshot_bytes: int | None = Field(default=None, ge=0)
	fork_strategy: Literal["explicit_copy", "read_only", "independent"]
	invalidation_fields: tuple[str, ...] = ()
	evicted_keys: tuple[str, ...] = ()
	recovered_partials: int = Field(default=0, ge=0)
	build_seconds: float = Field(default=0.0, ge=0)
	load_seconds: float = Field(default=0.0, ge=0)
	read_bytes: int = Field(default=0, ge=0)
	written_bytes: int = Field(default=0, ge=0)
	cache_bytes_after: int | None = Field(default=None, ge=0)
	verified: bool = False
	reason: str | None = None

	def progress_projection(self) -> tuple[str, int | None, dict[str, Any]]:
		"""Return the lifecycle kind, top-level byte counter, and typed details."""

		kind = "rebuild" if self.status == "rebuilt" else self.status
		return kind, self.cache_bytes_after, {
			"cache_status": self.status,
			"prefix_id": self.prefix_id,
			"cache_key": self.cache_key,
			"stage_count": self.stage_count,
			"snapshot_bytes": self.snapshot_bytes,
			"fork_strategy": self.fork_strategy,
			"invalidation_fields": self.invalidation_fields,
			"evicted_keys": self.evicted_keys,
			"recovered_partials": self.recovered_partials,
			"build_seconds": self.build_seconds,
			"load_seconds": self.load_seconds,
			"read_bytes": self.read_bytes,
			"written_bytes": self.written_bytes,
			"verified": self.verified,
			"reason": self.reason,
		}


@dataclass(frozen=True)
class PrefixPlanningCandidate:
	"""Runtime Stage/config facts used by deterministic prefix planning."""

	candidate_id: str
	configs: tuple[StageConfig, ...]
	stages: tuple[Stage, ...]
	share_identity: Any
	estimated_snapshot_bytes: int | None = None


def _stage_identity(config: StageConfig, stage: Stage) -> dict[str, Any]:
	return {
		"config": _fingerprint(config),
		"contract": _fingerprint(stage.contract),
	}


def _stage_id(config: StageConfig, index: int) -> str:
	return config.id or f"{config.kind}.{config.algorithm or 'default'}[{index}]"


def _cacheable(stage: Stage) -> bool:
	contract = stage.contract
	return (
		contract.cacheable_prefix
		and not contract.forks_pipeline
		and not contract.resources.requires_gpu
	)


def plan_common_prefixes(
	candidates: Sequence[PrefixPlanningCandidate],
	policy: PrefixCachePolicy,
) -> PrefixCachePlan:
	"""Find each candidate's longest safe prefix shared with another candidate."""

	ordered = sorted(candidates, key=lambda item: item.candidate_id)
	signatures = [
		[_stage_identity(config, stage) for config, stage in zip(item.configs, item.stages)]
		for item in ordered
	]
	share_identities = [_fingerprint(item.share_identity) for item in ordered]
	decisions = []
	unique_estimates: dict[str, int | None] = {}
	for index, candidate in enumerate(ordered):
		safe_limit = 0
		for stage in candidate.stages:
			if not _cacheable(stage):
				break
			safe_limit += 1
		best = 0
		for other_index, other in enumerate(ordered):
			if index == other_index or share_identities[index] != share_identities[other_index]:
				continue
			other_safe = 0
			for stage in other.stages:
				if not _cacheable(stage):
					break
				other_safe += 1
			common = 0
			for left, right in zip(signatures[index], signatures[other_index]):
				if common >= min(safe_limit, other_safe) or left != right:
					break
				common += 1
			best = max(best, common)
		stage_ids = tuple(
			_stage_id(config, offset) for offset, config in enumerate(candidate.configs[:best])
		)
		prefix_id = None if best == 0 else _hash_document({"stages": signatures[index][:best]})
		requires_copy = any(stage.contract.mutates_input for stage in candidate.stages[best:])
		action: Literal["cache", "independent", "bypass"] = "independent"
		if best == 0:
			if candidate.stages and not _cacheable(candidate.stages[0]):
				reason = (
					f"stage {_stage_id(candidate.configs[0], 0)} is not explicitly "
					"prefix-cacheable"
				)
			else:
				reason = "no semantically identical safe prefix is shared"
		elif not policy.enabled:
			reason = "prefix caching is disabled; independent execution remains the default"
		elif (
			candidate.estimated_snapshot_bytes is not None
			and policy.max_disk_bytes is not None
			and candidate.estimated_snapshot_bytes > policy.max_disk_bytes
		):
			action = "bypass"
			reason = "estimated snapshot exceeds the explicit disk bound"
		elif (
			requires_copy
			and candidate.estimated_snapshot_bytes is not None
			and policy.max_memory_bytes is not None
			and candidate.estimated_snapshot_bytes > policy.max_memory_bytes
		):
			action = "bypass"
			reason = "required mutation-isolation copy exceeds the explicit memory bound"
		else:
			action = "cache"
			if best < len(candidate.stages) and not _cacheable(candidate.stages[best]):
				reason = (
					f"stopped before insufficiently contracted stage "
					f"{_stage_id(candidate.configs[best], best)}"
				)
			elif best < len(candidate.stages):
				reason = f"stopped before differing stage {_stage_id(candidate.configs[best], best)}"
			else:
				reason = "complete pipeline is a shared, explicitly cacheable prefix"
		if prefix_id is not None:
			unique_estimates.setdefault(prefix_id, candidate.estimated_snapshot_bytes)
		decisions.append(PrefixCacheDecision(
			candidate_id=candidate.candidate_id,
			action=action,
			prefix_id=prefix_id,
			stage_count=best,
			stage_ids=stage_ids,
			estimated_snapshot_bytes=candidate.estimated_snapshot_bytes,
			requires_array_copy=requires_copy,
			reason=reason,
		))
	total = None if any(value is None for value in unique_estimates.values()) else sum(
		value or 0 for value in unique_estimates.values()
	)
	return PrefixCachePlan(
		enabled=policy.enabled,
		scratch=policy.scratch,
		estimated_unique_snapshot_bytes=total,
		decisions=tuple(decisions),
	)


class PrefixSnapshotManifest(ConfigModel):
	"""Immutable integrity manifest published last inside one snapshot directory."""

	schema_version: Literal[2] = 2
	prefix_id: str
	cache_key: str
	stage_count: int = Field(gt=0)
	identity: dict[str, Any]
	array_sha256: str
	state_sha256: str
	array_bytes: int = Field(ge=0)
	state_bytes: int = Field(ge=0)
	artifact_files: dict[str, str] = Field(default_factory=dict)
	artifact_sha256: dict[str, str] = Field(default_factory=dict)
	artifact_bytes: dict[str, int] = Field(default_factory=dict)
	snapshot_bytes: int = Field(ge=0)
	created_at: datetime


def _write_bytes(path: Path, payload: bytes) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("xb") as stream:
		stream.write(payload)
		stream.flush()
		os.fsync(stream.fileno())


def _write_json(path: Path, payload: Any) -> None:
	_write_bytes(
		path,
		(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n").encode("utf-8"),
	)


def _directory_bytes(path: Path) -> int:
	return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _run_stages(
	state: PipelineState,
	stages: Sequence[Stage],
	observer: StageObserver | None = None,
	checkpoint: Callable[[], None] | None = None,
	*,
	execution_observer: ExecutionSpanObserver | None = None,
	stage_index_offset: int = 0,
) -> PipelineState:
	if not stages:
		return state
	return Pipeline(
		list(stages),
		backend=state.backend,
		validation_mode="fragment",
		initial_layout=state.layout,
		initial_domain=state.domain,
		observer=observer,
		execution_observer=execution_observer,
		stage_indices=tuple(range(stage_index_offset, stage_index_offset + len(stages))),
		checkpoint=checkpoint,
	).run(state)


def _clone_initial_state(state: PipelineState) -> PipelineState:
	working = np.asarray(state.working_array)
	working_copy = (
		working.copy() if state.storage is None or working.size == 0 else
		state.storage.fork(
			state.working_array, shape=state.buffer_spec.shape,
			label="experiment.prefix-fork",
		)
	)
	return PipelineState(
		working_array=working_copy,
		scan=state.scan.model_copy(deep=True),
		recon=state.recon.model_copy(deep=True),
		artifacts=_clone_state_value(state.artifacts),
		backend=state.backend,
		storage=state.storage,
		dataset=state.dataset,
		source_window=state.source_window,
		buffer_spec=state.buffer_spec,
		domain=state.domain,
	)


def _equivalent(left: Any, right: Any, *, rtol: float, atol: float) -> bool:
	if isinstance(left, ManagedBuffer) or isinstance(right, ManagedBuffer):
		try:
			left = buffer_view(left)
			right = buffer_view(right)
		except (OSError, TypeError, ValueError):
			return False
	if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
		try:
			np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=rtol, atol=atol)
		except (AssertionError, TypeError, ValueError):
			return False
		return True
	if hasattr(left, "model_dump") and hasattr(right, "model_dump"):
		return _equivalent(
			left.model_dump(mode="python"), right.model_dump(mode="python"),
			rtol=rtol, atol=atol,
		)
	if isinstance(left, dict) and isinstance(right, dict):
		return set(left) == set(right) and all(
			_equivalent(left[key], right[key], rtol=rtol, atol=atol) for key in left
		)
	if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
		return len(left) == len(right) and all(
			_equivalent(a, b, rtol=rtol, atol=atol) for a, b in zip(left, right)
		)
	return left == right


class PrefixCacheStore:
	"""Atomic, bounded filesystem store for immutable NumPy prefix snapshots."""

	def __init__(self, policy: PrefixCachePolicy):
		if not policy.enabled or policy.scratch is None:
			raise ValueError("PrefixCacheStore requires an enabled policy with scratch")
		self.policy = policy
		self.root = policy.scratch
		self._thread_lock = threading.RLock()

	@contextmanager
	def _locked(self):
		self.root.mkdir(parents=True, exist_ok=True)
		lock_path = self.root / ".cache.lock"
		with self._thread_lock, lock_path.open("a+", encoding="utf-8") as stream:
			fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
			stream.seek(0)
			prior = stream.read().strip()
			if prior:
				quarantine = self.root / "quarantine" / "locks"
				quarantine.mkdir(parents=True, exist_ok=True)
				(quarantine / f"stale-{time.time_ns()}.json").write_text(
					prior + "\n", encoding="utf-8",
				)
			stream.seek(0)
			stream.truncate()
			json.dump({"pid": os.getpid(), "claimed_at": _utcnow().isoformat()}, stream)
			stream.flush()
			os.fsync(stream.fileno())
			try:
				yield bool(prior)
			finally:
				stream.seek(0)
				stream.truncate()
				stream.flush()
				fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

	def _entry(self, prefix_id: str, cache_key: str) -> Path:
		return self.root / "entries" / prefix_id / cache_key

	def _quarantine(self, path: Path, category: str) -> None:
		if not path.exists():
			return
		destination = self.root / "quarantine" / category / f"{path.name}-{time.time_ns()}"
		destination.parent.mkdir(parents=True, exist_ok=True)
		os.replace(path, destination)
		self._trim_quarantine()

	def _trim_quarantine(self, reserve: int = 0) -> None:
		root = self.root / "quarantine"
		if not root.exists() or self.policy.max_disk_bytes is None:
			return
		items = [item for category in root.iterdir() for item in category.iterdir()]
		active = _directory_bytes(self.root / "entries")
		quarantine = _directory_bytes(root)
		for item in sorted(items, key=lambda value: value.stat().st_mtime):
			if active + quarantine + reserve <= self.policy.max_disk_bytes:
				break
			size = _directory_bytes(item) if item.is_dir() else item.stat().st_size
			if item.is_dir():
				shutil.rmtree(item)
			else:
				item.unlink()
			quarantine -= size

	def _recover_partials(self) -> int:
		builds = self.root / "builds"
		partials = list(builds.iterdir()) if builds.exists() else []
		for path in partials:
			self._quarantine(path, "partial")
		return len(partials)

	def _read_manifest(self, entry: Path) -> PrefixSnapshotManifest:
		manifest_path = entry / "manifest.json"
		try:
			manifest = PrefixSnapshotManifest.model_validate_json(
				manifest_path.read_text(encoding="utf-8")
			)
		except Exception as error:
			raise PrefixCacheIntegrityError(f"invalid snapshot manifest: {error}") from error
		array_path = entry / "working.npy"
		state_path = entry / "state.json"
		try:
			array_bytes = array_path.stat().st_size
			state_bytes = state_path.stat().st_size
		except OSError as error:
			raise PrefixCacheIntegrityError(f"snapshot payload is missing: {error}") from error
		if _sha256_file(array_path) != manifest.array_sha256:
			raise PrefixCacheIntegrityError("working-array integrity hash mismatch")
		if _sha256_file(state_path) != manifest.state_sha256:
			raise PrefixCacheIntegrityError("snapshot-state integrity hash mismatch")
		if array_bytes != manifest.array_bytes or state_bytes != manifest.state_bytes:
			raise PrefixCacheIntegrityError("snapshot byte count mismatch")
		if set(manifest.artifact_files) != set(manifest.artifact_sha256) \
				or set(manifest.artifact_files) != set(manifest.artifact_bytes):
			raise PrefixCacheIntegrityError("managed-artifact manifest is incomplete")
		for name, relative in manifest.artifact_files.items():
			path = entry / relative
			try:
				artifact_bytes = path.stat().st_size
			except OSError as error:
				raise PrefixCacheIntegrityError(
					f"managed artifact {name!r} is missing: {error}"
				) from error
			if artifact_bytes != manifest.artifact_bytes[name]:
				raise PrefixCacheIntegrityError(
					f"managed artifact {name!r} byte count mismatch"
				)
			if _sha256_file(path) != manifest.artifact_sha256[name]:
				raise PrefixCacheIntegrityError(
					f"managed artifact {name!r} integrity hash mismatch"
				)
		if _directory_bytes(entry) != manifest.snapshot_bytes:
			raise PrefixCacheIntegrityError("snapshot directory byte count mismatch")
		return manifest

	def _load(
		self,
		entry: Path,
		current: PipelineState,
		*,
		requires_copy: bool,
	) -> tuple[PipelineState, PrefixSnapshotManifest, float]:
		started = time.perf_counter()
		manifest = self._read_manifest(entry)
		if requires_copy and manifest.snapshot_bytes > (self.policy.max_memory_bytes or 0):
			raise PrefixCacheCapacityError(
				f"snapshot fork requires up to {manifest.snapshot_bytes} bytes, exceeding "
				f"cache memory bound {self.policy.max_memory_bytes}"
			)
		try:
			payload = json.loads((entry / "state.json").read_text(encoding="utf-8"))
		except Exception as error:
			raise PrefixCacheIntegrityError(f"snapshot cannot be loaded: {error}") from error
		if not isinstance(payload, dict) or not {
			"artifacts", "scan", "recon_metadata", "buffer_spec",
			"managed_artifact_specs", "domain",
		}.issubset(payload):
			raise PrefixCacheIntegrityError("snapshot state payload is incomplete")
		try:
			artifacts = _state_value_from_document(payload["artifacts"])
			scan = type(current.scan).model_validate(
				_state_value_from_document(payload["scan"])
			)
			recon_metadata = _state_value_from_document(payload["recon_metadata"])
			dataset = (
				None if payload.get("dataset") is None
				else CONVERTER.structure(payload["dataset"], DataRef)
			)
			source_window = (
				None if payload.get("source_window") is None
				else CONVERTER.structure(payload["source_window"], DataRef)
			)
			if dataset is not None and not isinstance(dataset, TomoDataSet):
				raise TypeError("snapshot dataset is not a TomoDataSet")
			if source_window is not None and not isinstance(source_window, TomoWindow):
				raise TypeError("snapshot source window is not a TomoWindow")
			buffer_spec = _buffer_spec_from_document(payload.get("buffer_spec"))
			artifact_specs = {
				name: _buffer_spec_from_document(payload["managed_artifact_specs"][name])
				for name in manifest.artifact_files
			}
			domain = None if payload["domain"] is None else DataDomain[payload["domain"]]
		except Exception as error:
			raise PrefixCacheIntegrityError(f"snapshot state is invalid: {error}") from error
		try:
			mapped = FileBackedNPArray(entry / "working.npy")
		except Exception as error:
			raise PrefixCacheIntegrityError(f"snapshot array cannot be loaded: {error}") from error
		if requires_copy:
			working = (
				np.array(mapped.array(), copy=True) if current.storage is None else
				current.storage.fork(
					mapped, shape=buffer_spec.shape,
					label="experiment.cache-fork",
				)
			)
			mapped.close()
		else:
			working = mapped
		for name, relative in manifest.artifact_files.items():
			artifact_map = FileBackedNPArray(entry / relative)
			if requires_copy:
				artifact_spec = artifact_specs[name]
				artifacts[name] = (
					np.array(artifact_map.array(), copy=True) if current.storage is None else
					current.storage.fork(
						artifact_map, shape=artifact_spec.shape,
						label=f"experiment.cache-artifact:{name}",
					)
				)
				artifact_map.close()
			else:
				artifacts[name] = artifact_map
		recon = current.recon.model_copy(deep=True)
		recon.metadata = recon_metadata
		state = PipelineState(
			working_array=working,
			scan=scan,
			recon=recon,
			artifacts=artifacts,
			backend=current.backend,
			storage=current.storage,
			dataset=dataset,
			source_window=source_window,
			buffer_spec=buffer_spec,
			domain=domain,
		)
		return state, manifest, time.perf_counter() - started

	def _publish(
		self,
		prefix_id: str,
		cache_key: str,
		identity: dict[str, Any],
		stage_count: int,
		state: PipelineState,
		managed_artifacts: tuple[str, ...] = (),
	) -> tuple[PrefixSnapshotManifest, tuple[str, ...]]:
		build = self.root / "builds" / f"{cache_key}-{uuid4().hex}"
		build.mkdir(parents=True)
		try:
			array_path = build / "working.npy"
			with array_path.open("xb") as stream:
				np.save(stream, np.asarray(state.working_array), allow_pickle=False)
				stream.flush()
				os.fsync(stream.fileno())
			artifact_files = {}
			artifact_sha256 = {}
			artifact_bytes = {}
			for index, name in enumerate(managed_artifacts):
				if name not in state.artifacts:
					raise PrefixCacheError(
						f"Stage contract declares managed artifact {name!r}, but it was not produced"
					)
				value = buffer_view(state.artifacts[name])
				if value.dtype.hasobject:
					raise PrefixCacheError(
						f"managed artifact {name!r} cannot have object dtype"
					)
				relative = f"artifacts/{index:04d}-{hashlib.sha256(name.encode()).hexdigest()[:16]}.npy"
				path = build / relative
				path.parent.mkdir(parents=True, exist_ok=True)
				with path.open("xb") as stream:
					np.save(stream, value, allow_pickle=False)
					stream.flush()
					os.fsync(stream.fileno())
				artifact_files[name] = relative
				artifact_sha256[name] = _sha256_file(path)
				artifact_bytes[name] = path.stat().st_size
			ordinary_artifacts = {
				key: value for key, value in state.artifacts.items()
				if key not in artifact_files
			}
			uncontracted = _managed_buffers(ordinary_artifacts)
			if uncontracted:
				raise PrefixCacheError(
					"managed cache artifacts require an explicit producer contract: "
					+ ", ".join(sorted({
						str(getattr(value, "role", None) or getattr(value, "name", "unknown"))
						for value in uncontracted
					})),
				)
			state_path = build / "state.json"
			managed_artifact_specs = {
				name: _buffer_spec_to_document(BufferSpec(
					state.artifacts[name].shape, state.artifacts[name].dtype,
				))
				for name in managed_artifacts
			}
			_write_json(state_path, {
				"artifacts": _state_value_to_document(ordinary_artifacts),
				"scan": _state_value_to_document(state.scan.model_dump(mode="python")),
				"recon_metadata": _state_value_to_document(state.recon.metadata),
				"dataset": None if state.dataset is None else CONVERTER.unstructure(state.dataset),
				"source_window": (
					None if state.source_window is None else CONVERTER.unstructure(state.source_window)
				),
				"buffer_spec": _buffer_spec_to_document(state.buffer_spec),
				"managed_artifact_specs": managed_artifact_specs,
				"domain": None if state.domain is None else state.domain.name,
			})
			array_bytes = array_path.stat().st_size
			state_bytes = state_path.stat().st_size
			manifest = PrefixSnapshotManifest(
				prefix_id=prefix_id,
				cache_key=cache_key,
				stage_count=stage_count,
				identity=identity,
				array_sha256=_sha256_file(array_path),
				state_sha256=_sha256_file(state_path),
				array_bytes=array_bytes,
				state_bytes=state_bytes,
				artifact_files=artifact_files,
				artifact_sha256=artifact_sha256,
				artifact_bytes=artifact_bytes,
				snapshot_bytes=array_bytes + state_bytes + sum(artifact_bytes.values()),
				created_at=_utcnow(),
			)
			manifest_payload = b""
			for _attempt in range(10):
				manifest_payload = (
					json.dumps(
						manifest.model_dump(mode="json"), sort_keys=True, indent=2,
					) + "\n"
				).encode("utf-8")
				snapshot_bytes = (
					array_bytes + state_bytes + sum(artifact_bytes.values())
					+ len(manifest_payload)
				)
				if snapshot_bytes == manifest.snapshot_bytes:
					break
				manifest = manifest.model_copy(update={"snapshot_bytes": snapshot_bytes})
			else:
				raise PrefixCacheError("snapshot manifest byte count did not stabilize")
			_write_bytes(build / "manifest.json", manifest_payload)
			if manifest.snapshot_bytes > (self.policy.max_disk_bytes or 0):
				raise PrefixCacheCapacityError(
					f"snapshot is {manifest.snapshot_bytes} bytes, exceeding cache disk bound "
					f"{self.policy.max_disk_bytes}"
				)
			evicted = self._evict_for(manifest.snapshot_bytes, exclude=cache_key)
			entry = self._entry(prefix_id, cache_key)
			entry.parent.mkdir(parents=True, exist_ok=True)
			if entry.exists():
				self._quarantine(entry, "superseded")
			os.replace(build, entry)
			return manifest, evicted
		except PrefixCacheError:
			if build.exists():
				self._quarantine(build, "partial")
			raise
		except Exception as error:
			if build.exists():
				self._quarantine(build, "partial")
			raise PrefixCacheError(f"snapshot publication failed: {error}") from error
		except BaseException:
			if build.exists():
				self._quarantine(build, "partial")
			raise

	def _entry_records(self) -> list[tuple[float, str, Path, int]]:
		records = []
		entries = self.root / "entries"
		if not entries.exists():
			return records
		for manifest_path in entries.glob("*/*/manifest.json"):
			entry = manifest_path.parent
			key = entry.name
			access = self.root / "access" / f"{key}.json"
			stamp = access.stat().st_mtime if access.exists() else manifest_path.stat().st_mtime
			records.append((stamp, key, entry, _directory_bytes(entry)))
		return records

	def _cache_bytes(self) -> int:
		return _directory_bytes(self.root / "entries") + _directory_bytes(
			self.root / "quarantine"
		)

	def _evict_for(self, incoming: int, *, exclude: str) -> tuple[str, ...]:
		self._trim_quarantine(reserve=incoming)
		records = self._entry_records()
		total = sum(item[3] for item in records) + _directory_bytes(
			self.root / "quarantine"
		)
		count = len(records)
		evicted = []
		for _stamp, key, entry, size in sorted(records):
			if (
				total + incoming <= (self.policy.max_disk_bytes or 0)
				and count + 1 <= self.policy.max_entries
			):
				break
			if key == exclude:
				continue
			shutil.rmtree(entry)
			total -= size
			count -= 1
			evicted.append(key)
		if (
			total + incoming > (self.policy.max_disk_bytes or 0)
			or count + 1 > self.policy.max_entries
		):
			raise PrefixCacheCapacityError("cache bounds cannot admit the new snapshot")
		return tuple(evicted)

	def _touch(self, cache_key: str) -> None:
		path = self.root / "access" / f"{cache_key}.json"
		path.parent.mkdir(parents=True, exist_ok=True)
		temporary = path.with_name(f".{path.name}-{uuid4().hex}")
		_write_json(temporary, {"accessed_at": _utcnow().isoformat()})
		os.replace(temporary, path)

	def _invalidation_fields(self, prefix_id: str, identity: dict[str, Any]) -> tuple[str, ...]:
		root = self.root / "entries" / prefix_id
		if not root.exists():
			return ()
		fields = set()
		for path in root.glob("*/manifest.json"):
			try:
				prior = PrefixSnapshotManifest.model_validate_json(path.read_text(encoding="utf-8"))
			except Exception:
				continue
			fields.update(_identity_differences(prior.identity, identity))
		return tuple(sorted(fields))

	def execute(
		self,
		decision: PrefixCacheDecision,
		state: PipelineState,
		stages: Sequence[Stage],
		configs: Sequence[StageConfig],
		*,
		runtime_identity: Any,
		observer: StageObserver | None = None,
		execution_observer: ExecutionSpanObserver | None = None,
		cache_build_execution_observer: ExecutionSpanObserver | None = None,
		cache_verify_execution_observer: ExecutionSpanObserver | None = None,
		cache_observer: Callable[[PrefixCacheEvent], None] | None = None,
		checkpoint: Callable[[], None] | None = None,
	) -> tuple[PipelineState, PrefixCacheEvent]:
		"""Load/build one prefix, isolate a fork, then execute the suffix."""

		def publish(event: PrefixCacheEvent) -> PrefixCacheEvent:
			if cache_observer is not None:
				cache_observer(event)
			return event

		if decision.action != "cache" or decision.stage_count == 0:
			event = publish(PrefixCacheEvent(
				status="independent" if decision.action == "independent" else "bypass",
				stage_count=decision.stage_count,
				fork_strategy="independent",
				reason=decision.reason,
			))
			result = _run_stages(
				state, stages, observer, checkpoint,
				execution_observer=execution_observer,
			)
			return result, event
		if state.backend.name != "numpy":
			event = publish(PrefixCacheEvent(
				status="bypass",
				stage_count=decision.stage_count,
				fork_strategy="independent",
				reason=f"backend {state.backend.name!r} lacks a prefix snapshot contract",
			))
			result = _run_stages(
				state, stages, observer, checkpoint,
				execution_observer=execution_observer,
			)
			return result, event
		prefix_stages = tuple(stages[:decision.stage_count])
		suffix_stages = tuple(stages[decision.stage_count:])
		prefix_configs = tuple(configs[:decision.stage_count])
		managed_artifacts = tuple(dict.fromkeys(
			name for stage in prefix_stages for name in stage.contract.managed_artifacts
		))
		if len(prefix_stages) != decision.stage_count or any(
			not _cacheable(stage) for stage in prefix_stages
		):
			event = publish(PrefixCacheEvent(
				status="bypass",
				stage_count=decision.stage_count,
				fork_strategy="independent",
				reason="runtime pipeline no longer matches a safe planned prefix",
			))
			result = _run_stages(
				state, stages, observer, checkpoint,
				execution_observer=execution_observer,
			)
			return result, event
		requires_copy = any(stage.contract.mutates_input for stage in suffix_stages)
		try:
			stage_identity = [
				_stage_identity(config, stage)
				for config, stage in zip(prefix_configs, prefix_stages)
			]
			prefix_id = _hash_document({"stages": stage_identity})
			identity = _fingerprint({
				"schema_version": 2,
				"runtime": runtime_identity,
				"backend": state.backend,
				"initial_working_array": state.working_array,
				"initial_layout": state.layout,
				"initial_domain": state.domain,
				"scan": state.scan,
				"recon_metadata": state.recon.metadata,
				"artifacts": state.artifacts,
				"dataset": state.dataset,
				"source_window": state.source_window,
				"stages": stage_identity,
			})
			cache_key = _hash_document(identity)
		except Exception as error:
			if self.policy.on_error == "fail":
				raise PrefixCacheError(f"cache identity is insufficient: {error}") from error
			event = publish(PrefixCacheEvent(
				status="bypass",
				stage_count=decision.stage_count,
				fork_strategy="independent",
				reason=f"cache identity is insufficient: {error}",
			))
			result = _run_stages(
				state, stages, observer, checkpoint,
				execution_observer=execution_observer,
			)
			return result, event
		entry = self._entry(prefix_id, cache_key)
		build_seconds = 0.0
		load_seconds = 0.0
		invalidation = ()
		evicted: tuple[str, ...] = ()
		recovered = 0
		status: Literal["hit", "miss", "rebuilt"] = "hit"
		verified = False
		cache_hit = False
		cache_bytes_after = None
		try:
			with self._locked() as recovered_lock:
				recovered = self._recover_partials() + int(recovered_lock)
				if entry.exists():
					try:
						fork, manifest, load_seconds = self._load(
							entry, state, requires_copy=requires_copy,
						)
					except PrefixCacheIntegrityError:
						if self.policy.on_error == "fail":
							raise
						self._quarantine(entry, "corrupt")
						if self.policy.on_error == "bypass":
							raise PrefixCacheIntegrityError(
								"corrupt snapshot was quarantined and cache bypass was requested"
							)
						status = "rebuilt"
					else:
						self._touch(cache_key)
						cache_hit = True
				else:
					status = "miss"
				if not cache_hit:
					invalidation = self._invalidation_fields(prefix_id, identity)
					started = time.perf_counter()
					with _pinned_state(
						state, owner=f"cache-build:{prefix_id}:{uuid4().hex}",
					):
						built = _run_stages(
							_clone_initial_state(state), prefix_stages, observer, checkpoint,
							execution_observer=(
								cache_build_execution_observer or execution_observer
							),
						)
						build_seconds = time.perf_counter() - started
						with _pinned_state(
							built, owner=f"cache-publish:{prefix_id}:{uuid4().hex}",
						):
							manifest, evicted = self._publish(
								prefix_id, cache_key, identity, decision.stage_count, built,
								managed_artifacts,
							)
					fork, manifest, load_seconds = self._load(
						self._entry(prefix_id, cache_key), state, requires_copy=requires_copy,
					)
					self._touch(cache_key)
				cache_bytes_after = self._cache_bytes()
		except PrefixCacheError as error:
			if self.policy.on_error == "fail":
				raise
			event = publish(PrefixCacheEvent(
				status="bypass",
				prefix_id=prefix_id,
				cache_key=cache_key,
				stage_count=decision.stage_count,
				fork_strategy="independent",
				invalidation_fields=invalidation,
				recovered_partials=recovered,
				reason=str(error),
			))
			result = _run_stages(
				state, stages, observer, checkpoint,
				execution_observer=execution_observer,
			)
			return result, event
		if cache_hit and self._sampled(decision.candidate_id):
			with _pinned_state(
				state, owner=f"cache-verify:{prefix_id}:{uuid4().hex}",
			):
				check = _run_stages(
					_clone_initial_state(state), prefix_stages, observer, checkpoint,
					execution_observer=(
						cache_verify_execution_observer or execution_observer
					),
				)
			if not (
				_equivalent(
					np.asarray(check.working_array), np.asarray(fork.working_array),
					rtol=self.policy.verify_rtol, atol=self.policy.verify_atol,
				)
				and _equivalent(
					check.artifacts, fork.artifacts,
					rtol=self.policy.verify_rtol, atol=self.policy.verify_atol,
				)
			):
				with self._locked():
					if entry.exists():
						self._quarantine(entry, "mismatch")
				if self.policy.on_error == "fail":
					raise PrefixCacheIntegrityError(
						"sampled independent prefix differs from cached snapshot"
					)
				event = publish(PrefixCacheEvent(
					status="bypass",
					prefix_id=prefix_id,
					cache_key=cache_key,
					stage_count=decision.stage_count,
					fork_strategy="independent",
					load_seconds=load_seconds,
					read_bytes=manifest.snapshot_bytes,
					cache_bytes_after=cache_bytes_after,
					reason="sampled independent prefix differs from cached snapshot",
				))
				return _run_stages(
					check, suffix_stages, observer, checkpoint,
					execution_observer=execution_observer,
					stage_index_offset=decision.stage_count,
				), event
			verified = True
		event = publish(PrefixCacheEvent(
			status=status,
			prefix_id=prefix_id,
			cache_key=cache_key,
			stage_count=decision.stage_count,
			snapshot_bytes=manifest.snapshot_bytes,
			fork_strategy="explicit_copy" if requires_copy else "read_only",
			invalidation_fields=invalidation,
			evicted_keys=evicted,
			recovered_partials=recovered,
			build_seconds=build_seconds,
			load_seconds=load_seconds,
			read_bytes=manifest.snapshot_bytes,
			written_bytes=0 if cache_hit else manifest.snapshot_bytes,
			cache_bytes_after=cache_bytes_after,
			verified=verified,
		))
		return _run_stages(
			fork, suffix_stages, observer, checkpoint,
			execution_observer=execution_observer,
			stage_index_offset=decision.stage_count,
		), event

	def _sampled(self, candidate_id: str) -> bool:
		if self.policy.verify_fraction <= 0:
			return False
		value = int(hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:16], 16)
		return value / float(0xFFFFFFFFFFFFFFFF) < self.policy.verify_fraction


class PrefixCachingPipelineExecutor:
	"""Execute one planned pipeline through an optional shared prefix store."""

	def __init__(self, policy: PrefixCachePolicy):
		self.policy = policy
		self.store = PrefixCacheStore(policy) if policy.enabled else None

	def execute(
		self,
		decision: PrefixCacheDecision,
		state: PipelineState,
		stages: Sequence[Stage],
		configs: Sequence[StageConfig],
		*,
		runtime_identity: Any,
		observer: StageObserver | None = None,
		execution_observer: ExecutionSpanObserver | None = None,
		cache_build_execution_observer: ExecutionSpanObserver | None = None,
		cache_verify_execution_observer: ExecutionSpanObserver | None = None,
		cache_observer: Callable[[PrefixCacheEvent], None] | None = None,
		checkpoint: Callable[[], None] | None = None,
	) -> tuple[PipelineState, PrefixCacheEvent]:
		if self.store is None:
			event = PrefixCacheEvent(
				status="independent",
				stage_count=decision.stage_count,
				fork_strategy="independent",
				reason=decision.reason,
			)
			if cache_observer is not None:
				cache_observer(event)
			result = _run_stages(
				state, stages, observer, checkpoint,
				execution_observer=execution_observer,
			)
			return result, event
		return self.store.execute(
			decision, state, stages, configs, runtime_identity=runtime_identity,
			observer=observer,
			execution_observer=execution_observer,
			cache_build_execution_observer=cache_build_execution_observer,
			cache_verify_execution_observer=cache_verify_execution_observer,
			cache_observer=cache_observer,
			checkpoint=checkpoint,
		)
