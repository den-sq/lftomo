import pytest
from pydantic import ValidationError

from lftomo.config import Recon, StageConfig
from lftomo.pipeline.stages import BilateralFilterStage, STAGE_REGISTRY, create_stage


@pytest.mark.parametrize("parameter,value", [
	("sigam_color", 1),
	("sigma_color", 0),
])
def test_yaml_rejects_invalid_stage_parameter_with_occurrence_and_field(parameter, value):
	with pytest.raises(ValidationError, match=rf"denoise.*{parameter}"):
		Recon.from_yaml(f"""
pipeline:
  - id: denoise
    kind: filter
    algorithm: bilateral
    params:
      {parameter}: {value}
""")


def test_runtime_params_are_canonical_and_aliases_do_not_rewrite_serialized_config():
	config = StageConfig(kind="stripe", algorithm="fw", params={"wavelet": "db2"})
	stage = create_stage(config)
	assert stage.params.wname == "db2"
	assert config.params == {"wavelet": "db2"}
	assert not hasattr(stage, "wname")
	assert not hasattr(stage.params, "wavelet")
	with pytest.raises(ValidationError, match="frozen"):
		stage.params.wname = "db3"


def test_registered_parameter_schemas_expose_fields_and_forbid_unknown_keys():
	for stage_type in set(STAGE_REGISTRY.values()):
		schema = stage_type.Params.model_json_schema()
		assert schema["additionalProperties"] is False, stage_type.name
		with pytest.raises(ValidationError, match="not_a_stage_parameter"):
			stage_type.Params.model_validate({"not_a_stage_parameter": True})
	schema = BilateralFilterStage.Params.model_json_schema()
	assert schema["properties"]["sigma_color"]["exclusiveMinimum"] == 0
