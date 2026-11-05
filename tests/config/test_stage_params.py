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


@pytest.mark.parametrize("kind,algorithm,parameters,alias,field,expected", [
	("stripe", "fw", {"wavelet": "db2"}, "wavelet", "wname", "db2"),
	("normalize", "weighted", {"pre_subtracted": True, "timing_source": "acquisition"},
		"pre_subtracted", "pre_sub", True),
])
def test_runtime_params_are_canonical_and_aliases_do_not_rewrite_serialized_config(
	kind, algorithm, parameters, alias, field, expected,
):
	config = StageConfig(kind=kind, algorithm=algorithm, params=parameters)
	restored = StageConfig.model_validate_json(config.model_dump_json())
	assert restored.model_dump() == config.model_dump()
	stage = create_stage(restored)
	assert getattr(stage.params, field) == expected
	assert restored.params == parameters
	assert not hasattr(stage, field)
	assert not hasattr(stage.params, alias)
	with pytest.raises(ValidationError, match="frozen"):
		setattr(stage.params, field, expected)


def test_registered_parameter_schemas_expose_fields_and_forbid_unknown_keys():
	for stage_type in set(STAGE_REGISTRY.values()):
		schema = stage_type.Params.model_json_schema()
		assert schema["additionalProperties"] is False, stage_type.name
		with pytest.raises(ValidationError, match="not_a_stage_parameter"):
			stage_type.Params.model_validate({"not_a_stage_parameter": True})
	schema = BilateralFilterStage.Params.model_json_schema()
	assert schema["properties"]["sigma_color"]["exclusiveMinimum"] == 0
