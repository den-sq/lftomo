"""Regression gates for the canonical window-geometry ownership seam."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import lftomo.pipeline as pipeline
from lftomo.util.shapes import Y


_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "lftomo"
_REMOVED_TYPES = {"AxisPad", "WindowRegion"}
_REMOVED_UNIT_HELPERS = {
	"_buffer_units", "_buffer_ref", "_unit_states", "_unit_index", "_unit_handle",
	"_scope_bounded_work_to_unit",
}

# Counts are AST reads, including multiple reads on one line and negative
# indices. Lower these ceilings when removing reads; new modules start at zero.
# Remaining reads inspect numerical-library inputs/results at these boundaries.
_POSITIONAL_SHAPE_CEILINGS = {
	"pipeline/stages/geometry.py": 5,  # OpenCV affine image and target dimensions.
	"pipeline/stages/multimaterial.py": 1,  # NumPy air-reference image kernel.
	"pipeline/stages/normalize.py": 1,  # Calibration bundle's external basis array.
	"pipeline/stages/read.py": 3,  # HDF5 calibration dataset validation/read.
	"pipeline/stages/super_resolve.py": 8,  # NumPy interpolation/Torch patch adapters.
}

# Exact modules, rather than package exemptions: these adapt public selection
# fields or derive arguments for an external API. Pipeline execution consumes
# TomoWindow/TypedBuffer geometry instead of carrying those scalar fields.
_Y_BOUNDS_BOUNDARIES = {
	"acceptance.py": "external oracle and reconstruction configuration",
	"calibration/beam_basis.py": "cropped calibration-frame input API",
	"calibration/bundle.py": "calibration component row-selection API",
	"calibration/keyframes.py": "cropped calibration-frame input API",
	"calibration/models.py": "public detector-region model",
	"config/models.py": "public acquisition/configuration models",
	"evaluation/metrics.py": "evaluation ROI to numerical image slice",
	"evaluation/models.py": "public evaluation ROI model",
	"experiment/fidelity.py": "requested fidelity crop to numerical image slice",
	"experiment/models.py": "public experiment selection to TomoWindow",
	"pipeline/stages/read.py": "public read-Stage selection parameters",
	"pipeline/streaming.py": "read-Stage parameters to TomoWindow",
	"prep/flat_median.py": "detector-region and cropped flat-frame input API",
	"runner.py": "canonical selection to flat-field input APIs",
	"util/window.py": "canonical from_bounds input conversion",
}
_Y_BOUNDS_READ_CEILINGS: dict[str, int] = {}


def _production_trees():
	for path in _SOURCE.rglob("*.py"):
		yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_removed_padding_and_window_types_are_not_reintroduced():
	found = []
	for path, tree in _production_trees():
		for node in ast.walk(tree):
			name = getattr(node, "id", getattr(node, "attr", getattr(node, "name", None)))
			if name in _REMOVED_UNIT_HELPERS or (
				isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "units"
			) or (
				isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
				and node.func.attr == "units"
			):
				found.append(f"{path.relative_to(_ROOT)}:{node.lineno} retired unit geometry")
			if isinstance(node, ast.ClassDef) and node.name in _REMOVED_TYPES:
				found.append(f"{path.relative_to(_ROOT)}:{node.lineno} class {node.name}")
			if isinstance(node, ast.ImportFrom):
				for alias in node.names:
					if alias.name in _REMOVED_TYPES:
						found.append(
							f"{path.relative_to(_ROOT)}:{node.lineno} import {alias.name}"
						)

	assert found == []
	for name in _REMOVED_TYPES:
		assert not hasattr(pipeline, name)
	assert pipeline.Y is Y


def test_affine_pad_x_shape_arithmetic_has_one_canonical_owner():
	found = []
	for path, tree in _production_trees():
		for node in ast.walk(tree):
			if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
				continue
			operands = (node.left, node.right)
			has_two = any(
				isinstance(item, ast.Constant) and item.value == 2 for item in operands
			)
			has_pad_x = any(
				isinstance(item, ast.Attribute) and item.attr == "pad_x" for item in operands
			)
			if has_two and has_pad_x:
				found.append(f"{path.relative_to(_ROOT)}:{node.lineno}")

	assert found == []


def test_positional_shape_and_scalar_y_bounds_reads_do_not_increase():
	shape_reads = Counter()
	y_reads = Counter()
	for path, tree in _production_trees():
		module = path.relative_to(_SOURCE).as_posix()
		for node in ast.walk(tree):
			if module.startswith("pipeline/") and isinstance(node, ast.Subscript) \
					and isinstance(node.ctx, ast.Load) \
					and isinstance(node.value, ast.Attribute) and node.value.attr == "shape":
				index = node.slice
				if isinstance(index, ast.UnaryOp) and isinstance(index.op, (ast.USub, ast.UAdd)):
					index = index.operand
				if isinstance(index, ast.Constant) and type(index.value) is int:
					shape_reads[module] += 1
			if module in _Y_BOUNDS_BOUNDARIES:
				continue
			name = None
			if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
				name = node.id if isinstance(node, ast.Name) else node.attr
			elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) \
					and isinstance(node.slice, ast.Constant):
				name = node.slice.value
			elif isinstance(node, ast.Call):
				if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) > 1:
					name = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
				elif isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
					name = node.args[0].value if isinstance(node.args[0], ast.Constant) else None
			if name in ("y_start", "y_stop"):
				y_reads[module] += 1

	failures = [
		f"{module}: {kind} reads {count} exceed ceiling {ceilings.get(module, 0)}"
		for kind, counts, ceilings in (
			("positional shape", shape_reads, _POSITIONAL_SHAPE_CEILINGS),
			("scalar Y bounds", y_reads, _Y_BOUNDS_READ_CEILINGS),
		)
		for module, count in sorted(counts.items())
		if count > ceilings.get(module, 0)
	]
	assert failures == []
