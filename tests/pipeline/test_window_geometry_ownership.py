"""Regression gates for the canonical window-geometry ownership seam."""

from __future__ import annotations

import ast
from pathlib import Path

import lftomo.pipeline as pipeline
from lftomo.util.shapes import Y


_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "lftomo"
_REMOVED_TYPES = {"AxisPad", "WindowRegion"}


def _production_trees():
	for path in _SOURCE.rglob("*.py"):
		yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_removed_padding_and_window_types_are_not_reintroduced():
	found = []
	for path, tree in _production_trees():
		for node in ast.walk(tree):
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
