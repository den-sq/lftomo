import numpy as np

import benchmarks.order_layout as order_layout


def test_direct_benchmark_arms_apply_requested_row_concurrency(monkeypatch):
	worker_counts = []

	class RecordingExecutor:
		def __init__(self, max_workers):
			worker_counts.append(max_workers)

		def __enter__(self):
			return self

		def __exit__(self, *_args):
			pass

		def map(self, function, values):
			return map(function, values)

	monkeypatch.setattr(order_layout, "ThreadPoolExecutor", RecordingExecutor)
	transition = order_layout.run("transition", (12, 3, 16), workers=2)
	strided = order_layout.run("strided", (12, 3, 16), workers=2)

	assert worker_counts == [2, 2]
	assert np.allclose(transition.fingerprint, strided.fingerprint)


def test_transition_releases_obsolete_projection_mapping(monkeypatch, tmp_path):
	closed = []
	original = order_layout._close_memmap

	def record_close(array):
		closed.append(array.filename)
		original(array)

	monkeypatch.setattr(order_layout, "_close_memmap", record_close)
	order_layout.run("transition", (12, 3, 16), workers=2, scratch=tmp_path)

	assert closed == [tmp_path / "transition-projection.float32"]
