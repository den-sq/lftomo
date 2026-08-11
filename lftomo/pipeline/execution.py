"""Bounded-memory executors for grouped PipelinePlan segments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Callable, Sequence

import numpy as np

from lftomo.pipeline.core import PipelineState, Stage, working_view
from lftomo.pipeline.stages._layout import n_rows, sinogram_scratch, sinogram_view
from lftomo.util.shapes import ProjOrder


def run_bounded_sinogram_segment(
	state: PipelineState,
	stages: Sequence[Stage],
	*,
	max_workers: int = 1,
	failure_observer: Callable[[Stage, BaseException], None] | None = None,
	checkpoint: Callable[[], None] | None = None,
) -> PipelineState:
	"""Gather, process, and scatter each row once for an entire Stage chain."""

	if state.backend.name != "numpy":
		raise TypeError("bounded sinogram segments currently require the NumPy backend")
	if state.layout is not ProjOrder:
		raise ValueError("bounded sinogram execution requires ProjOrder input")
	if max_workers < 1:
		raise ValueError("bounded sinogram workers must be positive")
	data = working_view(state)
	if data.ndim != 3:
		raise ValueError("bounded sinogram execution expects a three-dimensional array")
	for stage in stages:
		if not callable(getattr(stage, "run_sinogram", None)):
			raise TypeError(f"{stage.name} declares bounded views but has no run_sinogram kernel")

	reported_failure = threading.Event()
	failure_lock = threading.Lock()

	def process(index: int) -> None:
		if checkpoint is not None:
			checkpoint()
		scratch = sinogram_scratch(data, state.layout, index)
		for stage in stages:
			if checkpoint is not None:
				checkpoint()
			try:
				result = stage.run_sinogram(scratch, state)
			except BaseException as error:
				if failure_observer is not None:
					with failure_lock:
						if not reported_failure.is_set():
							reported_failure.set()
							failure_observer(stage, error)
				raise
			if result is not None and result is not scratch:
				scratch = np.asarray(result)
		sinogram_view(data, state.layout, index)[:] = scratch

	row_count = n_rows(data, state.layout)
	if max_workers == 1:
		for index in range(row_count):
			process(index)
	else:
		# Each worker owns at most one row-sized scratch buffer. Results are written
		# back inside the worker, so completed buffers never queue in the parent.
		with ThreadPoolExecutor(max_workers=min(max_workers, row_count)) as pool:
			list(pool.map(process, range(row_count)))
	return state
