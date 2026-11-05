"""Standalone owner used to exercise fatal-exit cleanup of a complete run."""

import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from lftomo.storage import PipelineRunContext
from lftomo.util.shapes import ImgSlice


def _busy_worker(buffer, ready_path):
	values = buffer.array()
	values[0, 0] = 1
	_publish(Path(ready_path), {"pid": os.getpid()})
	while True:
		values[0, 0] += 1
		time.sleep(0.01)


def _publish(path, details):
	temporary = path.with_suffix(".tmp")
	temporary.write_text(json.dumps(details), encoding="utf-8")
	temporary.replace(path)


def run_crash_owner(start_method, directory):
	resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
	if start_method != "default":
		multiprocessing.set_start_method(start_method)
	directory = Path(directory)
	with PipelineRunContext(max_workers=2) as context:
		buffer = context.storage.allocate(ImgSlice(2, 3), np.float32)
		details = {
			"segment": buffer.name,
			"manager_pid": context.storage._process.pid,
			"manager_start_method": context.storage._process._start_method,
		}
		_publish(directory / "resources.json", details)
		pool = context.executor._ensure_pool()
		details["semaphores"] = [
			lock._semlock.name
			for queue, attributes in (
				(pool._call_queue, ("_rlock", "_wlock", "_sem")),
				(pool._result_queue, ("_rlock", "_wlock")),
			)
			for attribute in attributes
			if (lock := getattr(queue, attribute)) is not None
			and lock._semlock.name is not None
		]
		_publish(directory / "resources.json", details)
		busy_ready = directory / "busy-worker.json"
		pool.submit(_busy_worker, buffer, str(busy_ready))
		idle_pid = pool.submit(os.getpid).result(timeout=15)
		deadline = time.monotonic() + 15
		while not busy_ready.exists() and time.monotonic() < deadline:
			time.sleep(0.01)
		busy_pid = json.loads(busy_ready.read_text(encoding="utf-8"))["pid"]
		details["worker_pids"] = [busy_pid, idle_pid]
		_publish(directory / "resources.json", details)
		while True:
			time.sleep(1)


if __name__ == "__main__":
	run_crash_owner(*sys.argv[1:])
