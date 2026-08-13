"""Regression gates for canonical window/padding ownership."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2] / "lftomo"
REMOVED_GEOMETRY_TYPES = {"AxisPad", "WindowRegion"}


def _python_sources():
	return sorted(SOURCE_ROOT.rglob("*.py"))


def test_removed_parallel_geometry_types_do_not_return():
	found = []
	for path in _python_sources():
		tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
		for node in ast.walk(tree):
			name = None
			if isinstance(node, (ast.ClassDef, ast.Name)):
				name = node.name if isinstance(node, ast.ClassDef) else node.id
			elif isinstance(node, ast.Attribute):
				name = node.attr
			elif isinstance(node, ast.alias):
				name = node.asname or node.name.rsplit(".", 1)[-1]
			if name in REMOVED_GEOMETRY_TYPES:
				found.append(f"{path.relative_to(SOURCE_ROOT.parent)}:{node.lineno}:{name}")
	assert not found, "parallel window geometry returned:\n" + "\n".join(found)


def test_affine_pad_x_is_not_used_for_independent_geometry_arithmetic():
	found = []
	for path in _python_sources():
		tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
		for node in ast.walk(tree):
			if not isinstance(node, ast.BinOp):
				continue
			uses_pad_x = any(
				(isinstance(child, ast.Name) and child.id == "pad_x")
				or (isinstance(child, ast.Attribute) and child.attr == "pad_x")
				for child in ast.walk(node)
			)
			if uses_pad_x:
				found.append(f"{path.relative_to(SOURCE_ROOT.parent)}:{node.lineno}")
	assert not found, (
		"affine pad_x arithmetic must be expressed by TomoWindow.expanded():\n"
		+ "\n".join(found)
	)
