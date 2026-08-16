from __future__ import annotations

import numpy as np
import pytest

from lftomo.evaluation import PreviewContract
from lftomo.experiment import (
	CodeIdentity,
	ConfigurationIdentity,
	DatasetIdentity,
	EnvironmentIdentity,
	ExperimentConfig,
	FidelityExperimentRunner,
	export_blinded_previews,
)
from lftomo.experiment.runner import ExperimentInputs

from .helpers import ReviewFixtureExecutor


HASHES = [f"{index:064x}" for index in range(1, 8)]
BLINDING_KEY = "review-fixture-private-key"


def review_experiment() -> ExperimentConfig:
	return ExperimentConfig.model_validate({
		"name": "review-contract-fixture",
		"candidate_space": {"pipeline": [{
			"id": "phase",
			"kind": "phase",
			"algorithm": "paganin",
			"sweep": {"kind": "list", "params": {"alpha": [0.01, 0.02, 0.03]}},
		}]},
		"sampler": {"name": "exhaustive", "seed": 7},
		"resources": {"max_concurrency": 1},
		"metrics": [{
			"id": "quality",
			"metric": "quality",
			"protocol_version": "1",
			"direction": "minimize",
			"units": "ratio",
		}],
		"objectives": [{"metric_id": "quality"}],
		"windows": [
			{"id": "feature", "role": "tuning", "y_start": 0, "y_stop": 2},
			{"id": "held", "role": "held_out", "y_start": 2, "y_stop": 3},
		],
		"fidelities": [
			{"id": "screen", "windows": ["feature", "held"]},
			{"id": "final", "windows": ["feature", "held"]},
		],
		"aggregation": {"metric_ids": ["quality"]},
		"output": {
			"screening": "metrics_and_previews",
			"final_fidelity_id": "final",
			"write_full_finalists": True,
		},
		"promotion": {"reduction_factor": 3, "min_survivors": 1},
	})


def review_inputs() -> ExperimentInputs:
	return ExperimentInputs(
		dataset=DatasetIdentity(dataset_id="review-fixture", content_hash=HASHES[0]),
		configuration=ConfigurationIdentity(scan_config_hash=HASHES[1]),
		code=CodeIdentity(version="test", git_commit="abcdef123456", dirty=False),
		environment=EnvironmentIdentity(
			python="3.12", platform="test", backend="numpy",
		),
	)


@pytest.fixture(scope="module")
def experiment_root(tmp_path_factory):
	root = tmp_path_factory.mktemp("review-results")
	summary = FidelityExperimentRunner(
		review_experiment(), review_inputs(), root, ReviewFixtureExecutor(),
	).run()
	screening = summary.levels[0].attempts
	images = {
		attempt.identity.candidate.value: {
			"homogeneous": np.full((4, 4), 0.2 + index * 0.1, dtype=np.float32),
			"feature": np.eye(4, dtype=np.float32) * (index + 1),
			"held-out": np.fliplr(np.eye(4, dtype=np.float32)) * (index + 1),
		}
		for index, attempt in enumerate(screening)
	}
	export_blinded_previews(
		images,
		root,
		PreviewContract(scale_min=0, scale_max=3, blinded=True, format="png"),
		blinding_key=BLINDING_KEY,
	)
	return root
