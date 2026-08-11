"""Opt-in assembled acceptance run for the #173 pipeline-core epic.

This module deliberately is not a pytest promotion gate.  It records the two
Tier-2 modes required by issue #185 and leaves their TIFFs, experiment records,
cache, measurements, and proof summary under an operator-selected result root.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable

import click
import numpy as np
import psutil
import tifffile as tf
import yaml

from lftomo.experiment import (
	ExperimentConfig,
	ExperimentLifecycle,
	ExperimentRunPolicy,
	IndependentExperimentRunner,
	PathContextRoots,
	PathResourceRequest,
	PrefixCachePolicy,
	collect_environment_identity,
)
from lftomo.experiment.cli import PipelineTrialExecutor, build_experiment_inputs
from lftomo.pipeline import AxisPad, WindowConfig
from lftomo.pipeline.dispatch import DispatchKind
from lftomo.pipeline.stages import dispatch_for_stage
from lftomo.prep.flat import FLAT
from lftomo.reference import NumericalTolerance, checksum_manifest, compare_slice, sha256_file
from lftomo.runner import load_recon, load_scan, run_reconstruction
from lftomo.util.log import log as legacy_log
from lftomo.util.mem import SharedNPManager


_SCHEMA_VERSION = 1
_ISSUES = {"epic": 173, "acceptance": 185, "deferred_window_cache": 184}
_DEFAULT_SELECTION = (1068, 1092)
_DEFAULT_EXPERIMENT_SELECTION = (1076, 1080)
_DEFAULT_WINDOW_ROWS = 12


def _flat_path(root: Path, flat: FLAT) -> Path:
	matches = sorted(root.glob(f"{flat}*.tif*"))
	if not matches:
		raise FileNotFoundError(f"missing {flat} flat under {root}")
	return matches[0]


def _load_b10_current_module():
	"""Load the named source-tree oracle without pretending it is wheel API."""

	path = Path(__file__).resolve().parents[1] / "examples" / "b10_recon_current.py"
	if not path.is_file():
		raise FileNotFoundError(
			"#185 mode 1 must run from an lftomo source checkout containing "
			f"the named oracle {path}"
		)
	spec = importlib.util.spec_from_file_location("lftomo_b10_acceptance_oracle", path)
	if spec is None or spec.loader is None:
		raise ImportError(f"cannot load B10 acceptance oracle: {path}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
	if hasattr(value, "model_dump"):
		return _json_value(value.model_dump(mode="json"))
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {str(key): _json_value(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_value(item) for item in value]
	if isinstance(value, np.generic):
		return value.item()
	return value


def _write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
	temporary.write_text(
		json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	os.replace(temporary, path)


class AcceptanceLog:
	"""Durable JSONL progress plus a concise terminal projection."""

	def __init__(self, path: Path):
		self.path = path
		self.path.parent.mkdir(parents=True, exist_ok=True)
		self._lock = threading.Lock()

	def emit(self, event: str, message: str, **details: Any) -> None:
		record = {
			"at": _utcnow(), "event": event, "message": message,
			"details": _json_value(details),
		}
		with self._lock, self.path.open("a", encoding="utf-8") as stream:
			stream.write(json.dumps(record, sort_keys=True) + "\n")
			stream.flush()
			os.fsync(stream.fileno())
		click.echo(f"[{record['at']}] {message}", err=True)


def _private_bytes(process: psutil.Process) -> int:
	"""Return private mapped bytes while excluding managed /dev/shm pages."""

	try:
		maps = process.memory_maps(grouped=True)
	except (psutil.AccessDenied, psutil.NoSuchProcess, ProcessLookupError):
		return 0
	result = 0
	for mapping in maps:
		path = str(getattr(mapping, "path", ""))
		if path.startswith("/dev/shm/") or path == "[shmem]":
			continue
		private = int(getattr(mapping, "private_clean", 0)) + int(
			getattr(mapping, "private_dirty", 0)
		)
		result += private if private else int(getattr(mapping, "rss", 0))
	return result


class ProcessTreeMemorySampler(AbstractContextManager):
	"""Measure transient private memory around scratch and engine Stages.

	The sampler compares each process against the synchronous Stage-start map and
	excludes ``/dev/shm`` mappings.  It therefore measures private scratch/engine
	materialization without counting the run-owned shared buffers a second time.
	"""

	def __init__(self, interval_seconds: float = 0.05):
		self.interval_seconds = interval_seconds
		self._root = psutil.Process()
		self._active: str | None = None
		self._baseline: dict[int, int] = {}
		self._peaks = {
			"per_worker_private_scratch_bytes": 0,
			"external_engine_temporary_bytes": 0,
		}
		self._lock = threading.Lock()
		self._stop = threading.Event()
		self._thread: threading.Thread | None = None

	def _snapshot(self) -> dict[int, int]:
		try:
			processes = [self._root, *self._root.children(recursive=True)]
		except (psutil.NoSuchProcess, ProcessLookupError):
			return {}
		return {process.pid: _private_bytes(process) for process in processes}

	def _sample(self) -> None:
		values = self._snapshot()
		with self._lock:
			component = self._active
			baseline = dict(self._baseline)
		if component is None:
			return
		deltas = [
			max(0, value - baseline.get(pid, value))
			for pid, value in values.items()
		]
		# #177's private component is explicitly per-worker, not the sum of
		# simultaneous workers. Engine work is parent-owned in the v1 dispatch
		# hierarchy, so the largest process delta is the applicable boundary too.
		delta = max(deltas, default=0)
		with self._lock:
			if self._active == component:
				self._peaks[component] = max(self._peaks[component], delta)

	def _loop(self) -> None:
		while not self._stop.wait(self.interval_seconds):
			self._sample()

	def __enter__(self):
		self._thread = threading.Thread(target=self._loop, daemon=True)
		self._thread.start()
		return self

	def __exit__(self, *args):
		self._sample()
		self._stop.set()
		if self._thread is not None:
			self._thread.join(timeout=2)
		return False

	def observer(self, phase, _index, stage, _error) -> None:
		if phase == "started":
			dispatch = dispatch_for_stage(stage)
			if dispatch.kind is DispatchKind.ENGINE:
				component = "external_engine_temporary_bytes"
			elif stage.contract.resources.memory_multiplier > 1.0:
				component = "per_worker_private_scratch_bytes"
			else:
				component = None
			baseline = self._snapshot() if component is not None else {}
			with self._lock:
				self._active = component
				self._baseline = baseline
		elif phase in ("completed", "failed"):
			self._sample()
			with self._lock:
				self._active = None
				self._baseline = {}

	@property
	def peaks(self) -> dict[str, int]:
		with self._lock:
			return dict(self._peaks)


def memory_proof(planned: dict[str, int], measured: dict[str, int]) -> dict[str, Any]:
	"""Compare all three independent #177 memory components."""

	components = {}
	for name in (
		"shared_memory_bytes",
		"per_worker_private_scratch_bytes",
		"external_engine_temporary_bytes",
	):
		estimate = int(planned.get(name, 0))
		actual = int(measured.get(name, 0))
		components[name] = {
			"planned_bytes": estimate,
			"measured_bytes": actual,
			"within_bound": actual <= estimate,
		}
	return {
		"passed": all(item["within_bound"] for item in components.values()),
		"measurement": {
			"shared_memory_bytes": "unique physical allocations in the manager ledger",
			"per_worker_private_scratch_bytes": (
				"largest per-process private mapped-byte delta during non-engine scratch Stages; "
				"/dev/shm excluded"
			),
			"external_engine_temporary_bytes": (
				"largest per-process private mapped-byte delta during engine Stages; /dev/shm excluded"
			),
		},
		"components": components,
	}


def audit_pin_events(events: list[dict[str, Any]]) -> dict[str, Any]:
	"""Prove balanced pins and reject alias reuse while any owner retains a name."""

	pins: dict[str, set[str]] = {}
	violations = []
	seen_pin = False
	for index, event in enumerate(events):
		name = str(event.get("name"))
		action = event.get("action")
		owner = str(event.get("alias"))
		if action == "pin":
			seen_pin = True
			pins.setdefault(name, set()).add(owner)
		elif action == "unpin":
			if owner not in pins.get(name, set()):
				violations.append({"event": index, "reason": "unbalanced unpin", "name": name})
			else:
				pins[name].remove(owner)
		elif action == "alias" and pins.get(name):
			violations.append({
				"event": index, "reason": "alias reused pinned storage", "name": name,
				"owners": sorted(pins[name]),
			})
	unreleased = {name: sorted(owners) for name, owners in pins.items() if owners}
	return {
		"passed": seen_pin and not violations and not unreleased,
		"pin_events_present": seen_pin,
		"violations": violations,
		"unreleased": unreleased,
	}


def inspect_segments(names: list[str] | tuple[str, ...]) -> dict[str, Any]:
	"""Inspect Linux shared-memory names without attaching/registering them."""

	unique = sorted(set(names))
	if Path("/dev/shm").is_dir():
		leaked = [name for name in unique if (Path("/dev/shm") / name).exists()]
		method = "/dev/shm directory inspection"
	else:
		leaked = []
		method = "unsupported: /dev/shm is unavailable"
	return {
		"passed": not leaked and method != "unsupported: /dev/shm is unavailable",
		"method": method,
		"inspected": unique,
		"leaked": leaked,
	}


def _allocation_names(artifacts: dict[str, Any]) -> list[str]:
	return [
		str(event["name"]) for event in artifacts.get("allocation_events", ())
		if event.get("action") == "allocate"
	]


def _strict_tiff_parity(expected_root: Path, actual_root: Path,
		indices: range, *, rtol: float = 2e-6, atol: float = 2e-7) -> dict[str, Any]:
	comparisons = []
	for index in indices:
		expected_path = expected_root / f"{index:05d}.tif"
		actual_path = actual_root / f"{index:05d}.tif"
		expected = tf.imread(expected_path)
		actual = tf.imread(actual_path)
		difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
		passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False))
		comparisons.append({
			"slice": index,
			"expected": str(expected_path),
			"actual": str(actual_path),
			"max_absolute_error": float(difference.max(initial=0)),
			"passed": passed,
		})
	return {
		"passed": all(item["passed"] for item in comparisons),
		"rtol": rtol, "atol": atol, "comparisons": comparisons,
	}


def _reference_file(archive: Path, index: int) -> Path:
	matches = [path for path in (archive / "recon").glob("*.tif*") if str(index) in path.stem]
	if len(matches) != 1:
		raise ValueError(f"expected one frozen reference for slice {index}, found {matches}")
	return matches[0]


def _frozen_b10_parity(generated_root: Path, archive: Path,
		tolerance: NumericalTolerance | None = None) -> dict[str, Any]:
	metadata = json.loads(json.dumps(yaml.safe_load(
		(archive / "metadata.yaml").read_text(encoding="utf-8")
	)))
	indices = [int(value) for value in metadata["reference_selection"]["slices"]]
	manifest = checksum_manifest(archive / "recon.sha256")
	tolerance = tolerance or NumericalTolerance()
	comparisons = []
	for index in indices:
		reference_path = _reference_file(archive, index)
		relative = str(reference_path.relative_to(archive))
		digest = sha256_file(reference_path)
		if manifest.get(relative) != digest:
			raise ValueError(f"frozen reference checksum mismatch: {relative}")
		comparison = compare_slice(
			tf.memmap(generated_root / f"{index:05d}.tif", mode="r"), reference_path, tolerance,
		)
		comparisons.append({
			**comparison.__dict__, "slice": index, "sha256": digest,
			"reference": str(reference_path),
		})
	return {
		"passed": all(item["passed"] for item in comparisons),
		"producer": "b10_recon",
		"producer_snapshot": metadata["source"]["lftomo_snapshot"],
		"producer_window_rows": 12,
		"comparisons": comparisons,
	}


def _discover_flat_root(dataset_root: Path, supplied: Path | None) -> Path:
	candidates = [supplied] if supplied is not None else [
		dataset_root / "gains", dataset_root / "flats",
	]
	for candidate in candidates:
		if candidate is None or not candidate.is_dir():
			continue
		try:
			[_flat_path(candidate, flat) for flat in FLAT]
		except FileNotFoundError:
			continue
		return candidate.resolve()
	raise FileNotFoundError(
		"could not find pregain/postgain/predark/postdark median TIFFs; pass --flat-root"
	)


def _prepare_scan(dataset_root: Path, config_root: Path, flat_root: Path,
		dataset_content_hash: str | None, log: AcceptanceLog) -> tuple[Path, Path, str]:
	scan, _ = load_scan(dataset_root / "scan.yaml")
	source = Path(scan.projections.path).resolve()
	if not source.is_file():
		raise FileNotFoundError(f"projection source is unavailable: {source}")
	content_hash = dataset_content_hash or scan.metadata.get("dataset_content_hash")
	if content_hash is None:
		log.emit("dataset.hash.started", f"hashing {source.name} for stable experiment identity")
		content_hash = sha256_file(source)
		log.emit("dataset.hash.completed", f"dataset SHA-256 completed: {content_hash}")
	if not isinstance(content_hash, str) or len(content_hash) != 64 or any(
		character not in "0123456789abcdef" for character in content_hash.lower()
	):
		raise ValueError("dataset content hash must be 64 hexadecimal characters")
	# The named oracle subtracts each dark median from its corresponding gain
	# before invoking weighted normalization with ``pre_sub=True``. Materialize
	# that exact four-map contract once so both pipeline modes consume identical
	# calibration values rather than merely identical source filenames.
	original_fields = {
		flat: tf.imread(_flat_path(flat_root, flat)).astype(np.float32)
		for flat in FLAT
	}
	calibration_root = config_root / "pre-subtracted-flats"
	calibration_root.mkdir(parents=True, exist_ok=True)
	resolved_fields = {
		FLAT.PREGAIN: original_fields[FLAT.PREGAIN] - original_fields[FLAT.PREDARK],
		FLAT.POSTGAIN: original_fields[FLAT.POSTGAIN] - original_fields[FLAT.POSTDARK],
		FLAT.PREDARK: original_fields[FLAT.PREDARK],
		FLAT.POSTDARK: original_fields[FLAT.POSTDARK],
	}
	flat_paths = []
	for flat in FLAT:
		path = calibration_root / f"{flat}.tif"
		tf.imwrite(path, resolved_fields[flat])
		flat_paths.append(str(path))
	scan.metadata = {
		**scan.metadata,
		"dataset_id": scan.metadata.get("dataset_id", "phase_contrast_hdf5"),
		"dataset_content_hash": content_hash.lower(),
		"flat_fields": flat_paths,
		"flat_fields_provenance": {
			"source_root": str(flat_root),
			"pre_subtracted": True,
			"operation": "pre_gain -= pre_dark; post_gain -= post_dark",
		},
	}
	scan.projections.path = source
	config_root.mkdir(parents=True, exist_ok=True)
	scan_path = config_root / "scan.yaml"
	scan_path.write_text(scan.to_yaml(), encoding="utf-8")
	return scan_path, source, content_hash.lower()


def _current_pipeline(path: Path, *, y_start: int, y_stop: int, workers: int) -> Path:
	document = {
		"process_workers": workers,
		"pipeline": [
			{"id": "input", "kind": "read", "algorithm": "hdf5_dxchange", "params": {
				"order": "projection", "calibration_policy": "required",
				"y_start": y_start, "y_stop": y_stop,
			}},
			{"id": "normalize", "kind": "normalize", "algorithm": "weighted", "params": {
				"pre_subtracted": True,
			}},
			{"id": "bac-filter", "kind": "phase", "algorithm": "bac_correction", "params": {
				"alpha": 1.6, "gamma": 1.2, "processes": workers,
			}},
			{"id": "negative-log", "kind": "phase", "algorithm": "negative_log", "params": {
				"epsilon": None,
			}},
			{"id": "center", "kind": "center", "algorithm": "manual", "params": {
				"value_px": 1282.75, "tilt": 0.0,
			}},
			{"id": "reconstruct", "kind": "recon", "algorithm": "gridrec", "params": {
				"ncore": workers, "sinogram_order": False, "circle_mask": False,
			}},
			{"id": "write", "kind": "write", "algorithm": "tiff_slices", "params": {
				"dtype": "float32", "naming_template": "{slice:05d}.tif",
			}},
		],
	}
	path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
	return path


def _historical_pipeline(path: Path, *, y_start: int, y_stop: int, workers: int) -> Path:
	recon, _ = load_recon("b10-recon-canonical")
	recon.process_workers = workers
	for stage in recon.pipeline:
		if stage.kind == "read":
			stage.params.update({"y_start": y_start, "y_stop": y_stop})
		elif isinstance(stage.params.get("processes"), int):
			stage.params["processes"] = min(stage.params["processes"], workers)
		elif isinstance(stage.params.get("ncore"), int):
			stage.params["ncore"] = min(stage.params["ncore"], workers)
	path.write_text(recon.to_yaml(), encoding="utf-8")
	return path


def _compose_observers(*observers: Callable | None) -> Callable:
	def observe(phase, index, stage, error):
		for observer in observers:
			if observer is not None:
				observer(phase, index, stage, error)
	return observe


def _progress_observer(log: AcceptanceLog, mode: str) -> Callable:
	def observe(phase, index, stage, error):
		if phase == "started":
			log.emit(
				f"{mode}.stage.started", f"{mode}: starting {stage.name}",
				stage=stage.name, stage_index=index,
			)
		elif phase == "completed":
			log.emit(
				f"{mode}.stage.completed", f"{mode}: completed {stage.name}",
				stage=stage.name, stage_index=index,
			)
		else:
			log.emit(
				f"{mode}.stage.failed", f"{mode}: {stage.name} failed: {error}",
				stage=stage.name, stage_index=index,
			)
	return observe


def _interrupt_child(
		pipeline_path: Path, scan_path: Path, output_path: Path, trace_path: Path,
		ready_path: Path, window_rows: int) -> None:
	"""Allocate one real window then wait to be killed by the acceptance parent."""

	original = SharedNPManager.allocate

	def traced(manager, shape, dtype, **kwargs):
		result = original(manager, shape, dtype, **kwargs)
		buffers = result if isinstance(result, list) else [result]
		trace_path.parent.mkdir(parents=True, exist_ok=True)
		with trace_path.open("a", encoding="utf-8") as stream:
			for buffer in buffers:
				stream.write(json.dumps({
					"name": buffer.name, "role": kwargs.get("role"),
				}) + "\n")
			stream.flush()
			os.fsync(stream.fileno())
		if not ready_path.exists():
			ready_path.write_text("allocated\n", encoding="utf-8")
			# Deterministically hold the owner inside its first window.  The parent
			# kills the complete process group while the manager death-watch remains
			# outside it and unlinks the recorded allocation.
			time.sleep(3600)
		return result

	SharedNPManager.allocate = traced
	run_reconstruction(
		pipeline_path, scan_path, output_path,
		window_config=WindowConfig(rows=window_rows, extra_pad=AxisPad(y=0)),
	)


def _run_interrupted_cleanup(
		pipeline_path: Path, scan_path: Path, output_path: Path, proof_root: Path,
		window_rows: int, *, startup_timeout: float = 60, cleanup_timeout: float = 20,
) -> dict[str, Any]:
	trace_path = proof_root / "interrupted-segments.jsonl"
	ready_path = proof_root / "interrupted-ready"
	command = [
		sys.executable, "-m", "lftomo.acceptance", "_interrupt-child",
		str(pipeline_path), str(scan_path), str(output_path), str(trace_path),
		str(ready_path), str(window_rows),
	]
	repository = Path(__file__).resolve().parents[1]
	environment = dict(os.environ)
	environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
		str(repository), environment.get("PYTHONPATH"),
	)))
	proof_root.mkdir(parents=True, exist_ok=True)
	process = subprocess.Popen(
		command,
		cwd=proof_root,
		env=environment,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		start_new_session=True,
	)
	deadline = time.monotonic() + startup_timeout
	while not ready_path.exists() and process.poll() is None and time.monotonic() < deadline:
		time.sleep(0.05)
	if not ready_path.exists():
		stdout, stderr = process.communicate(timeout=5) if process.poll() is not None else ("", "")
		if process.poll() is None:
			os.killpg(process.pid, signal.SIGKILL)
			process.wait(timeout=5)
		raise RuntimeError(
			"interrupted cleanup child did not reach its first managed allocation; "
			f"stdout={stdout!r}, stderr={stderr!r}"
		)
	os.killpg(process.pid, signal.SIGKILL)
	process.wait(timeout=10)
	names = [
		json.loads(line)["name"]
		for line in trace_path.read_text(encoding="utf-8").splitlines()
		if line.strip()
	]
	deadline = time.monotonic() + cleanup_timeout
	inspection = inspect_segments(names)
	while not inspection["passed"] and time.monotonic() < deadline:
		time.sleep(0.1)
		inspection = inspect_segments(names)
	return {
		**inspection,
		"child_pid": process.pid,
		"child_returncode": process.returncode,
		"termination": "SIGKILL to isolated owner process group",
		"trace": str(trace_path),
	}


def run_windowed_acceptance(
		dataset_root: Path, result_root: Path, *, scan_path: Path, source: Path,
		flat_root: Path, workers: int, log: AcceptanceLog,
		y_start: int = _DEFAULT_SELECTION[0], y_stop: int = _DEFAULT_SELECTION[1],
		window_rows: int = _DEFAULT_WINDOW_ROWS,
) -> dict[str, Any]:
	"""Execute and record #185 mode 1."""

	started = _utcnow()
	root = result_root / "windowed"
	configs = root / "configs"
	configs.mkdir(parents=True, exist_ok=True)
	current_pipeline = _current_pipeline(
		configs / "current-oracle-pipeline.yaml",
		y_start=y_start, y_stop=y_stop, workers=workers,
	)
	historical_pipeline = _historical_pipeline(
		configs / "historical-b10-pipeline.yaml",
		y_start=y_start, y_stop=y_stop, workers=workers,
	)
	legacy_output = root / "outputs" / "b10-current-oracle"
	current_output = root / "outputs" / "current-pipeline"
	historical_output = root / "outputs" / "historical-b10"
	log.emit(
		"windowed.oracle.started",
		f"windowed: running b10_recon_current.py over rows [{y_start}, {y_stop})",
	)
	oracle = _load_b10_current_module()
	oracle.run(oracle.B10ReconConfig(
		source=source,
		flats=flat_root,
		output=legacy_output,
		center=1282.75,
		window_size=window_rows,
		alpha=1.6,
		gamma=1.2,
		apply_bac=True,
		processes=workers,
		y_start=y_start,
		y_stop=y_stop,
	))
	log.emit("windowed.oracle.completed", "windowed: named current oracle completed")
	current_result = run_reconstruction(
		current_pipeline,
		scan_path,
		current_output,
		window_config=WindowConfig(rows=window_rows, extra_pad=AxisPad(y=0)),
		stage_observer=_progress_observer(log, "windowed.current"),
	)[0]
	current_parity = _strict_tiff_parity(
		legacy_output, current_output, range(y_start, y_stop),
	)
	log.emit(
		"windowed.current.parity",
		f"windowed: current oracle parity {'passed' if current_parity['passed'] else 'failed'}",
	)

	log.emit("windowed.historical.started", "windowed: running historical B10 family pipeline")
	with ProcessTreeMemorySampler() as sampler:
		historical_result = run_reconstruction(
			historical_pipeline,
			scan_path,
			historical_output,
			window_config=WindowConfig(rows=window_rows, extra_pad=AxisPad(y=0)),
			stage_observer=_compose_observers(
				sampler.observer, _progress_observer(log, "windowed.historical"),
			),
		)[0]
	measured = {
		"shared_memory_bytes": int(
			historical_result.artifacts["measured_shared_memory_bytes"]
		),
		**sampler.peaks,
	}
	planned = {
		key: int(value)
		for key, value in historical_result.artifacts["planned_peak_memory"].items()
	}
	memory = memory_proof(planned, measured)
	archive = dataset_root / "reference" / "b10_recon"
	frozen = _frozen_b10_parity(historical_output, archive)
	recon_minimal = {
		"producer": "recon_minimal",
		"status": "not_applicable",
		"passed": True,
		"reason": (
			"the frozen recon_minimal reference belongs to striping_tiff and its "
			"64-row producer extent; this phase_contrast_hdf5 12-row configuration "
			"matches only the b10_recon frozen producer"
		),
	}
	normal_names = [
		*_allocation_names(current_result.artifacts),
		*_allocation_names(historical_result.artifacts),
	]
	normal_cleanup = inspect_segments(normal_names)
	log.emit("windowed.interruption.started", "windowed: starting isolated interrupted-run proof")
	interrupted_cleanup = _run_interrupted_cleanup(
		historical_pipeline,
		scan_path,
		root / "outputs" / "interrupted",
		root / "proofs",
		window_rows,
	)
	log.emit(
		"windowed.interruption.completed",
		f"windowed: interrupted cleanup {'passed' if interrupted_cleanup['passed'] else 'failed'}",
	)
	proofs = {
		"current_oracle_parity": current_parity,
		"frozen_references": {
			"passed": frozen["passed"] and recon_minimal["passed"],
			"b10_recon": frozen,
			"recon_minimal": recon_minimal,
		},
		"peak_memory": memory,
		"normal_shared_memory_cleanup": normal_cleanup,
		"interrupted_shared_memory_cleanup": interrupted_cleanup,
	}
	return {
		"schema_version": _SCHEMA_VERSION,
		"mode": "windowed_pipeline",
		"started_at": started,
		"finished_at": _utcnow(),
		"passed": all(item["passed"] for item in proofs.values()),
		"dataset": {
			"id": "phase_contrast_hdf5", "root": str(dataset_root), "source": str(source),
		},
		"configuration": {
			"selection": [y_start, y_stop],
			"window_rows": window_rows,
			"extra_pad": AxisPad(y=0).as_dict(),
			"workers": workers,
			"current_pipeline": str(current_pipeline),
			"historical_pipeline": str(historical_pipeline),
		},
		"outputs": {
			"named_oracle": str(legacy_output),
			"current_pipeline": str(current_output),
			"historical_pipeline": str(historical_output),
		},
		"proofs": proofs,
	}


def _experiment_config(*, y_start: int, y_stop: int, workers: int) -> ExperimentConfig:
	return ExperimentConfig.model_validate({
		"schema_version": 1,
		"name": "pipeline-core-epic-acceptance",
		"candidate_space": {
			"process_workers": workers,
			"candidate_limits": {
				"max_candidates": 2,
				"max_estimated_output_bytes": 4 * 1024**3,
			},
			"pipeline": [
				{"id": "input", "kind": "read", "algorithm": "hdf5_dxchange", "params": {
					"order": "projection", "calibration_policy": "required",
				}},
				{"id": "normalize", "kind": "normalize", "algorithm": "weighted", "params": {
					"pre_subtracted": True,
				}},
				{"id": "bac-filter", "kind": "phase", "algorithm": "bac_correction", "params": {
					"alpha": 1.6, "gamma": 1.2, "processes": workers,
				}, "sweep": {
					"kind": "list", "params": {"gamma": [1.1, 1.2]},
				}},
				{"id": "negative-log", "kind": "phase", "algorithm": "negative_log", "params": {
					"epsilon": None,
				}},
				{"id": "center", "kind": "center", "algorithm": "manual", "params": {
					"value_px": 1282.75, "tilt": 0.0,
				}},
				{"id": "reconstruct", "kind": "recon", "algorithm": "gridrec", "params": {
					"ncore": workers, "sinogram_order": False,
				}},
				{"id": "evaluate", "kind": "evaluate", "algorithm": "metrics", "params": {
					"metrics": [{"kind": "ring", "inner_radius": 2, "smoothing_sigma": 2}],
					"aggregation": {"method": "mean", "invalid": "fail"},
				}},
				{"id": "write", "kind": "write", "algorithm": "tiff_slices", "params": {
					"dtype": "float32", "naming_template": "{slice:05d}.tif",
				}},
			],
		},
		"sampler": {"name": "exhaustive", "seed": 185},
		"resources": {
			"max_concurrency": 2, "memory_gb": 64, "max_output_bytes": 4 * 1024**3,
		},
		"metrics": [{
			"id": "rings", "metric": "artifact.ring_score", "protocol_version": "1",
			"direction": "minimize", "units": "ratio",
		}],
		"objectives": [{"metric_id": "rings"}],
		"windows": [{
			"id": "volume", "role": "tuning", "y_start": y_start, "y_stop": y_stop,
			"halo": 0,
		}],
		"fidelities": [{
			"id": "volume", "windows": ["volume"], "projection_stride": 1,
			"resource_fraction": 1.0, "memory_gb": 64,
			"max_output_bytes": 4 * 1024**3,
		}],
		"aggregation": {"name": "mean", "metric_ids": ["rings"], "invalid": "fail"},
		"output": {
			"screening": "configured_writers", "final_fidelity_id": "volume",
			"write_full_finalists": False, "max_bytes": 4 * 1024**3,
		},
		"metadata": {
			"purpose": "issue #185 whole-volume concurrent prefix-cache acceptance",
			"windowed_prefix_caching": "deferred to issue #184",
		},
	})


def _run_experiment(
		config: ExperimentConfig, scan_path: Path, result_root: Path, *,
		cache_policy: PrefixCachePolicy,
) -> Any:
	executor = PipelineTrialExecutor(config, scan_path, cache_policy)
	path_resources = PathResourceRequest(
		result_root=result_root,
		cache_root=cache_policy.scratch if cache_policy.enabled else None,
		cache_guard_bytes=cache_policy.max_disk_bytes if cache_policy.enabled else None,
		contexts=PathContextRoots(
			cwd=Path.cwd(),
			repository=Path(__file__).resolve().parents[1],
			config_directory=scan_path.parent,
		),
	)
	runner = IndependentExperimentRunner(
		config,
		build_experiment_inputs(scan_path),
		result_root,
		executor,
		validator=executor.validate,
		identity_validator=executor.validate_identity,
		policy=ExperimentRunPolicy(failure="fail_fast"),
		path_resources=path_resources,
	)
	return ExperimentLifecycle(runner).run()


def _attempt_events(attempt) -> list[dict[str, Any]]:
	artifacts = attempt.observations.get("artifacts", {})
	windows = artifacts.get("windows", {}) if isinstance(artifacts, dict) else {}
	result = []
	for window in windows.values():
		if isinstance(window, dict):
			result.extend(window.get("allocation_events", ()))
	return result


def _attempt_cache_events(attempt) -> list[dict[str, Any]]:
	artifacts = attempt.observations.get("artifacts", {})
	windows = artifacts.get("windows", {}) if isinstance(artifacts, dict) else {}
	return [
		window["prefix_cache"] for window in windows.values()
		if isinstance(window, dict) and isinstance(window.get("prefix_cache"), dict)
	]


def _attempt_output_fingerprints(attempt) -> dict[str, str]:
	if attempt.result is None:
		return {}
	files = {}
	for output in attempt.result.outputs:
		path = Path(output.path)
		if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
			files[path.name] = sha256_file(path)
	return dict(sorted(files.items()))


def _experiment_fingerprints(summary) -> dict[str, dict[str, str]]:
	return {
		attempt.identity.candidate.value: _attempt_output_fingerprints(attempt)
		for attempt in summary.attempts
	}


def _intervals_overlap(summary) -> bool:
	succeeded = [attempt for attempt in summary.attempts if attempt.status == "succeeded"]
	if len(succeeded) < 2:
		return False
	latest_start = max(attempt.started_at for attempt in succeeded)
	earliest_finish = min(attempt.finished_at for attempt in succeeded)
	return latest_start < earliest_finish


def run_experiment_acceptance(
		result_root: Path, *, scan_path: Path, source: Path, workers: int,
		log: AcceptanceLog,
		y_start: int = _DEFAULT_EXPERIMENT_SELECTION[0],
		y_stop: int = _DEFAULT_EXPERIMENT_SELECTION[1],
) -> dict[str, Any]:
	"""Execute and record #185 mode 2 without enabling windowed cache semantics."""

	started = _utcnow()
	root = result_root / "whole-volume-experiment"
	config = _experiment_config(y_start=y_start, y_stop=y_stop, workers=workers)
	(root / "configs").mkdir(parents=True, exist_ok=True)
	config_path = root / "configs" / "experiment.yaml"
	config_path.write_text(config.to_yaml(), encoding="utf-8")
	cache_root = root / "prefix-cache"
	cache_policy = PrefixCachePolicy(
		enabled=True,
		scratch=cache_root,
		max_disk_bytes=4 * 1024**3,
		max_memory_bytes=2 * 1024**3,
		max_entries=4,
		on_error="fail",
		verify_fraction=1.0,
	)
	log.emit(
		"experiment.cached.started",
		"whole-volume experiment: starting two concurrent cached trials",
	)
	cached = _run_experiment(
		config, scan_path, root / "cached-results", cache_policy=cache_policy,
	)
	log.emit("experiment.cached.completed", "whole-volume experiment: cached run completed")
	log.emit(
		"experiment.independent.started",
		"whole-volume experiment: starting independent contamination oracle",
	)
	independent = _run_experiment(
		config,
		scan_path,
		root / "independent-results",
		cache_policy=PrefixCachePolicy(enabled=False),
	)
	log.emit(
		"experiment.independent.completed",
		"whole-volume experiment: independent oracle completed",
	)

	statuses = []
	prefix_ids = set()
	cache_keys = set()
	all_events = []
	for attempt in cached.attempts:
		all_events.extend(_attempt_events(attempt))
		for event in _attempt_cache_events(attempt):
			statuses.append(event.get("status"))
			prefix_ids.add(event.get("prefix_id"))
			cache_keys.add(event.get("cache_key"))
	pin_proof = audit_pin_events(all_events)
	cache_proof = {
		"passed": (
			len(cached.attempts) > 1
			and all(attempt.status == "succeeded" for attempt in cached.attempts)
			and "miss" in statuses
			and "hit" in statuses
			and len(prefix_ids - {None}) == 1
			and len(cache_keys - {None}) == 1
		),
		"trial_count": len(cached.attempts),
		"statuses": statuses,
		"prefix_ids": sorted(value for value in prefix_ids if value is not None),
		"cache_keys": sorted(value for value in cache_keys if value is not None),
	}
	concurrency = {
		"passed": cached.preflight.max_concurrency > 1 and _intervals_overlap(cached),
		"configured_max_concurrency": cached.preflight.max_concurrency,
		"attempt_intervals": [{
			"trial_id": attempt.trial_id.value,
			"started_at": attempt.started_at,
			"finished_at": attempt.finished_at,
		} for attempt in cached.attempts],
	}
	cached_fingerprints = _experiment_fingerprints(cached)
	independent_fingerprints = _experiment_fingerprints(independent)
	fingerprints = {
		"passed": (
			cached_fingerprints == independent_fingerprints
			and len(cached_fingerprints) == 2
			and len({tuple(value.items()) for value in cached_fingerprints.values()}) == 2
		),
		"cached": cached_fingerprints,
		"independent": independent_fingerprints,
		"candidate_keys_stable": set(cached_fingerprints) == set(independent_fingerprints),
		"candidates_remain_distinct": (
			len({tuple(value.items()) for value in cached_fingerprints.values()}) == 2
		),
	}
	all_independent_events = [
		event for attempt in independent.attempts for event in _attempt_events(attempt)
	]
	cleanup = inspect_segments([
		str(event["name"])
		for event in [*all_events, *all_independent_events]
		if event.get("action") == "allocate"
	])
	whole_volume = {
		"passed": all(
			window.get("window_plan", {}).get("whole_volume") is True
			for attempt in cached.attempts
			for window in attempt.observations.get("artifacts", {}).get("windows", {}).values()
			if isinstance(window, dict)
		),
		"windowed_prefix_cache_deferred_to": 184,
	}
	proofs = {
		"whole_volume_only": whole_volume,
		"concurrent_trials": concurrency,
		"shared_prefix_cache": cache_proof,
		"pins_honored": pin_proof,
		"stable_fingerprints_and_no_cross_trial_contamination": fingerprints,
		"shared_memory_cleanup": cleanup,
	}
	return {
		"schema_version": _SCHEMA_VERSION,
		"mode": "whole_volume_experiment",
		"started_at": started,
		"finished_at": _utcnow(),
		"passed": all(item["passed"] for item in proofs.values()),
		"dataset": {
			"id": "phase_contrast_hdf5", "source": str(source),
		},
		"configuration": {
			"selection": [y_start, y_stop],
			"workers_per_trial": workers,
			"experiment": str(config_path),
			"cached_result_root": str(root / "cached-results"),
			"independent_result_root": str(root / "independent-results"),
			"prefix_cache_root": str(cache_root),
			"runner_window_config": None,
		},
		"proofs": proofs,
	}


def _summary_markdown(record: dict[str, Any]) -> str:
	lines = [
		"## #185 assembled acceptance record",
		"",
		f"- Overall: **{'PASS' if record['passed'] else 'FAIL'}**",
		f"- Dataset: `{record['dataset']['id']}`",
		f"- Dataset SHA-256: `{record['dataset']['content_hash']}`",
		f"- Started: `{record['started_at']}`",
		f"- Finished: `{record['finished_at']}`",
		f"- Full JSON record: `{record['record_path']}`",
		f"- Durable progress log: `{record['log_path']}`",
		"",
		"| mode | result | proofs |",
		"| --- | --- | --- |",
	]
	for mode in record["modes"]:
		proofs = ", ".join(
			f"{name}={'pass' if proof.get('passed') else 'FAIL'}"
			for name, proof in mode.get("proofs", {}).items()
		)
		lines.append(
			f"| `{mode['mode']}` | **{'PASS' if mode.get('passed') else 'FAIL'}** | {proofs} |"
		)
	lines.extend([
		"",
		"Windowed TIFF outputs, experiment attempt records, cache manifests, and measurements "
		"remain under the result root named in the JSON record. Prefix caching was exercised "
		"only in whole-volume runner mode; #184 remains deferred.",
	])
	return "\n".join(lines) + "\n"


def run_acceptance(
		dataset_root: Path, result_root: Path, *, flat_root: Path | None = None,
		dataset_content_hash: str | None = None, workers: int = 2,
		modes: tuple[str, ...] = ("windowed", "experiment"),
) -> dict[str, Any]:
	"""Run selected #185 modes and always publish a terminal record."""

	dataset_root = dataset_root.expanduser().resolve()
	result_root = result_root.expanduser().resolve()
	modes = tuple(dict.fromkeys(modes))
	unknown_modes = sorted(set(modes) - {"windowed", "experiment"})
	if not modes or unknown_modes:
		raise ValueError(f"acceptance modes must be windowed and/or experiment: {unknown_modes}")
	if not (dataset_root / "scan.yaml").is_file():
		raise FileNotFoundError(f"missing canonical scan metadata: {dataset_root / 'scan.yaml'}")
	if (
		result_root == dataset_root
		or result_root in dataset_root.parents
		or dataset_root in result_root.parents
	):
		raise ValueError("acceptance result root and Tier-2 dataset root must not overlap")
	if result_root.exists() and any(result_root.iterdir()):
		raise ValueError("acceptance result root must be new or empty")
	result_root.mkdir(parents=True, exist_ok=True)
	legacy_log.set_logdir(result_root / "legacy-logs")
	log = AcceptanceLog(result_root / "acceptance-progress.jsonl")
	started = _utcnow()
	log.emit("acceptance.started", "#185 assembled acceptance started", modes=modes)
	resolved_flats = _discover_flat_root(dataset_root, flat_root)
	scan_path, source, content_hash = _prepare_scan(
		dataset_root, result_root / "invocation", resolved_flats,
		dataset_content_hash, log,
	)
	record_path = result_root / "acceptance-record.json"
	record: dict[str, Any] = {
		"schema_version": _SCHEMA_VERSION,
		"issues": _ISSUES,
		"started_at": started,
		"finished_at": None,
		"passed": False,
		"dataset": {
			"id": "phase_contrast_hdf5",
			"root": str(dataset_root),
			"source": str(source),
			"source_bytes": source.stat().st_size,
			"source_mtime_ns": source.stat().st_mtime_ns,
			"content_hash": content_hash,
			"flat_root": str(resolved_flats),
			"normalized_scan": str(scan_path),
		},
		"environment": collect_environment_identity().model_dump(mode="json"),
		"record_path": str(record_path),
		"log_path": str(log.path),
		"modes": [],
	}
	try:
		if "windowed" in modes:
			mode = run_windowed_acceptance(
				dataset_root,
				result_root,
				scan_path=scan_path,
				source=source,
				flat_root=resolved_flats,
				workers=workers,
				log=log,
			)
			record["modes"].append(mode)
			_write_json(record_path, record)
		if "experiment" in modes:
			mode = run_experiment_acceptance(
				result_root,
				scan_path=scan_path,
				source=source,
				workers=workers,
				log=log,
			)
			record["modes"].append(mode)
	except BaseException as error:
		record["failure"] = {
			"type": type(error).__name__,
			"message": str(error),
			"traceback": traceback.format_exc(),
		}
		log.emit("acceptance.failed", f"#185 assembled acceptance failed: {error}")
	finally:
		record["finished_at"] = _utcnow()
		record["passed"] = (
			len(record["modes"]) == len(modes)
			and all(mode.get("passed") for mode in record["modes"])
			and "failure" not in record
		)
		_write_json(record_path, record)
		(result_root / "acceptance-summary.md").write_text(
			_summary_markdown(record), encoding="utf-8",
		)
		log.emit(
			"acceptance.completed",
			f"#185 assembled acceptance {'passed' if record['passed'] else 'failed'}; "
			f"record: {record_path}",
		)
	return record


@click.group()
def main() -> None:
	"""Run the opt-in, real-data integration gate for pipeline-core epic #173."""


@main.command("run")
@click.argument("dataset_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("result_root", type=click.Path(path_type=Path))
@click.option("--flat-root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--dataset-content-hash", type=str)
@click.option("--workers", type=click.IntRange(min=1), default=2, show_default=True)
@click.option(
	"--mode", "modes", type=click.Choice(["windowed", "experiment"]),
	multiple=True, help="Run only a selected mode; repeat for both. Defaults to both.",
)
def run_command(
		dataset_root: Path, result_root: Path, flat_root: Path | None,
		dataset_content_hash: str | None, workers: int, modes: tuple[str, ...],
) -> None:
	"""Execute #185 against DATASET_ROOT and retain every artifact in RESULT_ROOT."""

	selected = modes or ("windowed", "experiment")
	record = run_acceptance(
		dataset_root,
		result_root,
		flat_root=flat_root,
		dataset_content_hash=dataset_content_hash,
		workers=workers,
		modes=selected,
	)
	click.echo(json.dumps({
		"passed": record["passed"],
		"record": record["record_path"],
		"summary": str(Path(record["record_path"]).with_name("acceptance-summary.md")),
	}, sort_keys=True))
	if not record["passed"]:
		raise click.ClickException(
			f"acceptance proofs failed; inspect {record['record_path']} and retained artifacts"
		)


@main.command("_interrupt-child", hidden=True)
@click.argument("pipeline_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("scan_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.argument("trace_path", type=click.Path(path_type=Path))
@click.argument("ready_path", type=click.Path(path_type=Path))
@click.argument("window_rows", type=int)
def interrupt_child_command(
		pipeline_path: Path, scan_path: Path, output_path: Path, trace_path: Path,
		ready_path: Path, window_rows: int,
) -> None:
	_interrupt_child(
		pipeline_path.resolve(), scan_path.resolve(), output_path.resolve(),
		trace_path.resolve(), ready_path.resolve(), window_rows,
	)


if __name__ == "__main__":
	main()
