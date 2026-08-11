"""Pickleable executors used to exercise spawned experiment trials."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from lftomo.experiment.runner import TrialExecution


def candidate_alpha(identity) -> float:
	return float(identity.resolved_recon.pipeline[0].params["alpha"])


@dataclass
class AlphaExecutor:
	quality: float | None = None
	fail_on: float | None = None
	interrupt: bool = False
	delay: float = 0
	mutate_identity: bool = False
	call_root: Path | None = None

	def __call__(self, identity, _output_root):
		value = candidate_alpha(identity)
		started = time.monotonic()
		if self.call_root is not None:
			self.call_root.mkdir(parents=True, exist_ok=True)
			(self.call_root / f"{uuid4().hex}.json").write_text(json.dumps({
				"pid": os.getpid(),
				"fidelity": identity.fidelity.id,
				"alpha": value,
			}), encoding="utf-8")
		if self.fail_on == value:
			raise RuntimeError("candidate exploded")
		if self.interrupt:
			raise InterruptedError("stopped")
		if self.mutate_identity:
			identity.resolved_recon.pipeline[0].params["alpha"] = 999
		if self.delay:
			time.sleep(self.delay)
		return TrialExecution(
			metrics={"quality": value if self.quality is None else self.quality},
			artifacts={
				"pid": os.getpid(),
				"started_monotonic": started,
				"finished_monotonic": time.monotonic(),
			},
		)


@dataclass
class WindowedAlphaExecutor:
	call_root: Path | None = None

	def __call__(self, identity, _output_root):
		value = candidate_alpha(identity)
		if self.call_root is not None:
			self.call_root.mkdir(parents=True, exist_ok=True)
			(self.call_root / f"{uuid4().hex}.json").write_text(json.dumps({
				"fidelity": identity.fidelity.id,
				"alpha": value,
			}), encoding="utf-8")
		return TrialExecution(
			metrics={"quality": value},
			window_metrics={
				"tune-a": {"quality": [value]},
				"tune-b": {"quality": [value]},
				"hold": {"quality": [100 - value]},
			},
		)


@dataclass
class DisabledCacheExecutor:
	scratch: Path

	def plan_prefix_cache(self, _trials):
		return {
			"enabled": False,
			"scratch": str(self.scratch),
			"estimated_unique_snapshot_bytes": 10**18,
			"decisions": [],
		}

	def __call__(self, *_args):
		return TrialExecution(metrics={"quality": 1})


class CrashOneExecutor:
	def __call__(self, identity, _output_root):
		value = candidate_alpha(identity)
		if value == 0.01:
			os._exit(17)
		return TrialExecution(metrics={"quality": value})


class HangingProcessTreeExecutor:
	def __call__(self, _identity, output_root):
		output_root.mkdir(parents=True, exist_ok=True)
		descendant = subprocess.Popen(["sleep", "60"])
		(output_root / "process-tree.json").write_text(json.dumps({
			"trial_pid": os.getpid(),
			"descendant_pid": descendant.pid,
		}), encoding="utf-8")
		while True:
			time.sleep(1)
