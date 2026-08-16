"""Pickleable synthetic executor for review-contract tests."""

from __future__ import annotations

from lftomo.experiment import OutputRecord
from lftomo.experiment.runner import TrialExecution


class ReviewFixtureExecutor:
	def __call__(self, identity, output_root):
		alpha = float(identity.resolved_recon.pipeline[0].params["alpha"])
		path = output_root / f"{identity.fidelity.id}-retained.tif"
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(f"alpha={alpha}".encode("ascii"))
		return TrialExecution(
			metrics={"quality": alpha},
			outputs=(OutputRecord(
				role="retained_reconstruction",
				path=path,
				bytes=path.stat().st_size,
			),),
			peak_memory_bytes=4096,
		)
