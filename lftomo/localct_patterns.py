"""Infrastructure patterns adapted from ChengLab localCT.

The structures in this module are not a direct code copy. They adapt localCT's
useful request/result objects, scan-folder inspection, JSON run summaries, and
runtime bootstrap checks into lftomo-owned interfaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class ReconstructionResult:
	"""Typed reconstruction result payload, adapted from localCT's result pattern."""

	output_path: Path
	artifacts: dict[str, Path] = field(default_factory=dict)
	metrics: dict[str, float] = field(default_factory=dict)
	status: str = "success"


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

	def to_json(self) -> str:
		"""Serialize the run summary to JSON."""
		return json.dumps(_json_ready(asdict(self)), indent=2, sort_keys=True)

	@classmethod
	def from_json(cls, text: str) -> "RunSummary":
		"""Deserialize a run summary from JSON."""
		data = json.loads(text)
		return cls(
			request=ReconstructionRequest(
				session_path=Path(data["request"]["session_path"]),
				recon_config=Path(data["request"]["recon_config"]),
				output_path=Path(data["request"]["output_path"]),
				scan_config=None if data["request"]["scan_config"] is None else Path(data["request"]["scan_config"]),
				metadata=data["request"]["metadata"],
			),
			result=ReconstructionResult(
				output_path=Path(data["result"]["output_path"]),
				artifacts={key: Path(value) for key, value in data["result"]["artifacts"].items()},
				metrics=data["result"]["metrics"],
				status=data["result"]["status"],
			),
			metadata=data["metadata"],
		)


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


def _json_ready(value):
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {key: _json_ready(item) for key, item in value.items()}
	if isinstance(value, list):
		return [_json_ready(item) for item in value]
	return value
