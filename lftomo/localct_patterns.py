"""Infrastructure patterns adapted from ChengLab localCT.

The structures in this module are not a direct code copy. They adapt localCT's
useful request/result objects, scan-folder inspection, JSON run summaries, and
runtime bootstrap checks into lftomo-owned interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReconstructionRequest:
	"""Typed reconstruction invocation request, adapted from localCT's request pattern."""

	session_path: Path
	recon_config: Path
	output_path: Path
	scan_config: Path | None = None
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_document(self) -> dict[str, Any]:
		return {
			"session_path": str(self.session_path), "recon_config": str(self.recon_config),
			"output_path": str(self.output_path),
			"scan_config": None if self.scan_config is None else str(self.scan_config),
			"metadata": self.metadata,
		}

	@classmethod
	def from_document(cls, document: dict[str, Any]) -> "ReconstructionRequest":
		return cls(
			session_path=Path(document["session_path"]),
			recon_config=Path(document["recon_config"]),
			output_path=Path(document["output_path"]),
			scan_config=None if document["scan_config"] is None else Path(document["scan_config"]),
			metadata=document["metadata"],
		)


@dataclass(frozen=True)
class ReconstructionResult:
	"""Typed reconstruction result payload, adapted from localCT's result pattern."""

	output_path: Path
	artifacts: dict[str, Path] = field(default_factory=dict)
	metrics: dict[str, float] = field(default_factory=dict)
	status: str = "success"

	def to_document(self) -> dict[str, Any]:
		return {
			"output_path": str(self.output_path),
			"artifacts": {key: str(value) for key, value in self.artifacts.items()},
			"metrics": self.metrics, "status": self.status,
		}

	@classmethod
	def from_document(cls, document: dict[str, Any]) -> "ReconstructionResult":
		return cls(
			output_path=Path(document["output_path"]),
			artifacts={key: Path(value) for key, value in document["artifacts"].items()},
			metrics=document["metrics"], status=document["status"],
		)


@dataclass(frozen=True)
class ScanFolderInspection:
	"""Structured scan-folder discovery result, adapted from localCT scan inspection."""

	root: Path
	projections: list[Path]
	flats: list[Path]
	darks: list[Path]
	metadata_files: list[Path]


@dataclass(frozen=True)
class TomopyRuntime:
	"""TomoPy runtime availability check result, adapted from localCT bootstrap checks."""

	available: bool
	module: str = "tomopy"
	message: str = ""


@dataclass(frozen=True)
class RunSummary:
	"""JSON-serializable run summary, adapted from localCT's run-record pattern."""

	request: ReconstructionRequest
	result: ReconstructionResult
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_document(self) -> dict[str, Any]:
		return {
			"request": self.request.to_document(),
			"result": self.result.to_document(),
			"metadata": self.metadata,
		}

	@classmethod
	def from_document(cls, document: dict[str, Any]) -> "RunSummary":
		return cls(
			request=ReconstructionRequest.from_document(document["request"]),
			result=ReconstructionResult.from_document(document["result"]),
			metadata=document["metadata"],
		)

	def to_json(self) -> str:
		"""Serialize the stable run-summary document to JSON."""
		return json.dumps(self.to_document(), indent=2, sort_keys=True)

	@classmethod
	def from_json(cls, text: str) -> "RunSummary":
		"""Deserialize a run summary from JSON."""
		return cls.from_document(json.loads(text))


def inspect_scan_folder(root: Path) -> ScanFolderInspection:
	"""Inspect a canonical-ish scan folder for projections, flats, darks, and metadata.

	This follows localCT's pattern of turning loose folder discovery into a typed
	object before reconstruction code makes decisions.
	"""
	root = Path(root)
	projection_roots = [root / "projections", root]
	flat_roots = [root / "flats", root]

	projections = _collect_tiffs(projection_roots, include=("proj", "projection"), fallback_all=True)
	flats = _collect_tiffs(flat_roots, include=("flat", "gain", "white"))
	darks = _collect_tiffs(flat_roots, include=("dark",))
	metadata_files = sorted(path for path in root.glob("*.y*ml") if path.is_file())

	return ScanFolderInspection(root=root, projections=projections, flats=flats, darks=darks, metadata_files=metadata_files)


def check_tomopy_runtime() -> TomopyRuntime:
	"""Check whether TomoPy is importable before launching reconstruction work."""
	if find_spec("tomopy") is None:
		return TomopyRuntime(available=False, message="tomopy is not importable in this environment")
	return TomopyRuntime(available=True, message="tomopy is importable")


def write_run_summary(summary: RunSummary, path: Path):
	"""Write a JSON run summary."""
	Path(path).write_text(summary.to_json())


def read_run_summary(path: Path) -> RunSummary:
	"""Read a JSON run summary."""
	return RunSummary.from_json(Path(path).read_text())


def _collect_tiffs(roots: list[Path], include: tuple[str, ...], fallback_all: bool = False) -> list[Path]:
	matches: list[Path] = []
	for root in roots:
		if not root.exists():
			continue
		for path in root.glob("*.tif*"):
			if any(token in path.name.lower() for token in include):
				matches.append(path)
	if matches or not fallback_all:
		return sorted(dict.fromkeys(matches))
	for root in roots:
		if root.exists():
			matches.extend(path for path in root.glob("*.tif*") if path.is_file())
	return sorted(dict.fromkeys(matches))
