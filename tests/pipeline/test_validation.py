import numpy as np
import pytest

from lftomo.config import Acquisition, Detector, Recon, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	Backend,
	BackendCompatibilityError,
	DaskStubBackend,
	MissingArtifactError,
	MissingMetadataError,
	MutationForkError,
	NumpyBackend,
	Pipeline,
	PipelineState,
	ResourceHints,
	ShapeCompatibilityError,
	Stage,
	StageContract,
	validate_pipeline,
)
from lftomo.util.shapes import ProjOrder, ReconOrder, SinoOrder


def scan_config():
	return Scan(
		energy_keV=24,
		pixel_size_um=1.3,
		n_projections=3,
		degrees_per_fov=180,
		detector=Detector(pixel_size_um=1.3),
		acquisition=Acquisition(n_projections=3, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


def recon_config():
	return Recon(pipeline=[StageConfig(kind="echo")])


class ContractStage(Stage):
	def __init__(self, name, contract):
		self.name = name
		self.contract = contract

	def run(self, state):
		for artifact in self.contract.produces_artifacts:
			state.artifacts[artifact] = True
		return state


def state_for(backend=None):
	return PipelineState(
		working_array=np.zeros((3, 4, 5), dtype=np.float32),
		scan=scan_config(),
		recon=recon_config(),
		backend=backend or NumpyBackend(),
	)


def test_valid_noop_pipeline_builds_for_numpy_and_dask_stub():
	stages = [
		ContractStage("read", StageContract(expected_input=ProjOrder, produced_output=ProjOrder)),
		ContractStage("echo", StageContract(expected_input=ProjOrder, produced_output=ProjOrder)),
	]

	for backend in (NumpyBackend(), DaskStubBackend()):
		pipeline = Pipeline(stages, backend=backend, scan=scan_config())
		result = pipeline.run(state_for(backend))
		assert result.backend.name == backend.name


def test_shape_mismatch_identifies_stage_and_field():
	stages = [
		ContractStage("read", StageContract(expected_input=ProjOrder, produced_output=SinoOrder)),
		ContractStage("recon", StageContract(expected_input=ReconOrder, produced_output=ReconOrder)),
	]

	with pytest.raises(ShapeCompatibilityError, match="1:recon: expected_input"):
		validate_pipeline(stages, scan_config())


def test_shape_mismatch_uses_stable_stage_occurrence_id_when_available():
	stages = [
		ContractStage("read", StageContract(expected_input=ProjOrder, produced_output=SinoOrder)),
		ContractStage("recon", StageContract(expected_input=ReconOrder, produced_output=ReconOrder)),
	]
	stages[1].occurrence_id = "final-recon"

	with pytest.raises(ShapeCompatibilityError, match="final-recon:recon: expected_input"):
		validate_pipeline(stages, scan_config())


def test_missing_required_metadata_identifies_stage_and_field():
	stages = [
		ContractStage("normalize", StageContract(required_metadata=("metadata.flat_fields",))),
	]

	with pytest.raises(MissingMetadataError, match="metadata.flat_fields"):
		validate_pipeline(stages, scan_config())


def test_artifact_consumer_requires_upstream_producer():
	stages = [
		ContractStage("write", StageContract(consumes_artifacts=("center",))),
	]

	with pytest.raises(MissingArtifactError, match="center"):
		validate_pipeline(stages, scan_config())


def test_artifact_producer_allows_downstream_consumer():
	stages = [
		ContractStage("center", StageContract(produces_artifacts=("center",))),
		ContractStage("recon", StageContract(consumes_artifacts=("center",))),
	]

	validate_pipeline(stages, scan_config())


def test_initial_pipeline_state_artifact_allows_consumer():
	stages = [
		ContractStage("recon", StageContract(consumes_artifacts=("center",))),
	]

	validate_pipeline(stages, scan_config(), available_artifacts={"center": 2.5})


def test_gpu_stage_rejects_cpu_backend():
	stages = [
		ContractStage("astra", StageContract(resources=ResourceHints(requires_gpu=True))),
	]

	with pytest.raises(BackendCompatibilityError, match="requires GPU"):
		validate_pipeline(stages, scan_config(), backend=NumpyBackend())


def test_gpu_stage_accepts_gpu_backend():
	stages = [
		ContractStage("astra", StageContract(resources=ResourceHints(requires_gpu=True))),
	]

	validate_pipeline(stages, scan_config(), backend=Backend(name="cuda", supports_gpu=True))


def test_mutating_prefix_rejects_later_fork():
	stages = [
		ContractStage("normalize", StageContract(mutates_input=True)),
		ContractStage("bac-sweep", StageContract(forks_pipeline=True)),
	]

	with pytest.raises(MutationForkError, match="forks_pipeline"):
		validate_pipeline(stages, scan_config())
