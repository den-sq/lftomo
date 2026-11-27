from pathlib import Path
import tomllib

import yaml


ROOT = Path(__file__).parents[1]


def test_runtime_test_environment_has_bounded_dependency_ownership():
	project = tomllib.loads(ROOT.joinpath("pyproject.toml").read_text())
	environment = yaml.safe_load(ROOT.joinpath("environment.yml").read_text())
	dependencies = set(environment["dependencies"])

	assert project["project"]["requires-python"] == ">=3.11,<3.15"
	assert "python>=3.11,<3.15" in dependencies
	assert "tomopy=1.15.3=pyhd8ed1ab_2" in dependencies
	assert "libtomo=1.15.3=cuda12.9h1234567_densq1_103" in dependencies
	assert environment["channels"][0] == (
		"https://raw.githubusercontent.com/den-sq/tomopy/"
		"716c0d2b2826e54b99e8041dc7bdf01fefdd9acd/conda/channel"
	)
	assert "numpy" in dependencies
	assert "packaging" in dependencies
	assert "nodefaults" in environment["channels"]
	assert not any(
		dependency.startswith(("mkdocs", "mkdocstrings", "pymdown"))
		for dependency in dependencies
	)
	legacy = yaml.safe_load(ROOT.joinpath("environment-legacy.yml").read_text())
	assert legacy["name"] != environment["name"]
	assert "tomopy>=1.15.3" in legacy["dependencies"]
	assert legacy["channels"][0] == "conda-forge"


def test_documentation_and_optional_algorithms_remain_explicit_extras():
	project = tomllib.loads(ROOT.joinpath("pyproject.toml").read_text())["project"]
	extras = project["optional-dependencies"]

	assert "hotopy" in extras["bac"]
	assert "tomopy>=1.15.3" in extras["tomopy"]
	assert "pymdown-extensions" in extras["docs"]
	assert any(requirement.startswith("mkdocs-jupyter>=") for requirement in extras["docs"])


def test_ci_workflows_do_not_restore_the_obsolete_numpy_ceiling():
	workflows = "\n".join(
		path.read_text() for path in ROOT.joinpath(".github", "workflows").glob("*.yml")
	)

	assert "numpy<2" not in workflows
