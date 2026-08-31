"""Pickleable executors used to exercise spawned experiment trials."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import numpy as np

from lftomo.experiment.runner import TrialExecution
from lftomo.experiment.events import current_execution_context
from lftomo.pipeline import PipelineRunContext
from lftomo.util.shapes import ImgSlice


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
	rendezvous_participants: int | None = None

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
		if self.rendezvous_participants is not None:
			if self.call_root is None:
				raise ValueError("rendezvous_participants requires call_root")
			deadline = time.monotonic() + 15
			while (
				len(tuple(self.call_root.glob("*.json"))) < self.rendezvous_participants
				and time.monotonic() < deadline
			):
				time.sleep(0.01)
			if len(tuple(self.call_root.glob("*.json"))) < self.rendezvous_participants:
				raise TimeoutError("candidate rendezvous did not reach all participants")
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


@dataclass
class HangingProcessTreeExecutor:
	detached_descendant: bool = False

	def __call__(self, _identity, output_root):
		output_root.mkdir(parents=True, exist_ok=True)
		descendant = subprocess.Popen(
			["sleep", "60"], start_new_session=self.detached_descendant,
		)
		(output_root / "process-tree.json").write_text(json.dumps({
			"trial_pid": os.getpid(),
			"descendant_pid": descendant.pid,
		}), encoding="utf-8")
		while True:
			time.sleep(1)


class HangingManagedExecutor:
	def __call__(self, _identity, output_root):
		output_root.mkdir(parents=True, exist_ok=True)
		with PipelineRunContext(max_workers=1) as context:
			buffer = context.storage.allocate(ImgSlice(8, 8), np.float32, role="test")
			buffer.array()[...] = 1
			(output_root / "managed-process-tree.json").write_text(json.dumps({
				"trial_pid": os.getpid(),
				"segment_name": buffer.name,
			}), encoding="utf-8")
			while True:
				time.sleep(1)


@dataclass
class TimeoutOneExecutor:
	"""Hang one candidate after publishing context and partial attempt output."""

	hang_on: float = 0.01
	detached_descendant: bool = True

	def __call__(self, identity, output_root):
		value = candidate_alpha(identity)
		output_root.mkdir(parents=True, exist_ok=True)
		if value != self.hang_on:
			completed = output_root / "windows" / "completed" / "output"
			completed.mkdir(parents=True, exist_ok=True)
			(completed / "completed.txt").write_text("complete\n", encoding="utf-8")
			return TrialExecution(metrics={"quality": value})
		context = current_execution_context()
		if context is not None:
			context.window_started("timeout-window", position=1, total=1)
			context.stage_started(
				"timeout-stage", "blocking-output", position=2, total=3,
			)
		descendant = subprocess.Popen(
			["sleep", "60"], start_new_session=self.detached_descendant,
		)
		partial = output_root / "windows" / "timeout-window" / "output"
		partial.mkdir(parents=True, exist_ok=True)
		(partial / "partial-output.txt").write_text("partial\n", encoding="utf-8")
		(output_root / "timeout-process-tree.json").write_text(json.dumps({
			"trial_pid": os.getpid(),
			"descendant_pid": descendant.pid,
		}), encoding="utf-8")
		while True:
			time.sleep(1)
