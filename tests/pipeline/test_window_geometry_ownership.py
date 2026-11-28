"""Regression gates for the canonical window-geometry ownership seam."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import lftomo.pipeline as pipeline
from lftomo.pipeline.core import PadRequirement, ProjectionStage, SinogramStage
from lftomo.pipeline.streaming import resolve_stage_windows
from lftomo.storage import LocalArray
from lftomo.util.shapes import ProjOrder, SinoOrder, Y
from lftomo.util.window import TomoWindow
from tests.pipeline.test_core import recon_config, scan_config


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
	"pipeline/stages/geometry.py": 4,  # OpenCV affine image and target dimensions.
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


def test_runtime_source_bounds_are_not_written_back_to_stage_configuration():
	selection_fields = {"y_start", "y_stop", "theta_start", "theta_stop", "theta_stride"}
	found = []
	for path, tree in _production_trees():
		for node in ast.walk(tree):
			if (
				isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
				and isinstance(node.value, ast.Attribute) and node.value.attr == "params"
				and isinstance(node.slice, ast.Constant) and node.slice.value in selection_fields
			):
				found.append(f"{path.relative_to(_ROOT)}:{node.lineno} {node.slice.value}")

	assert found == []


class _ContextProjection(ProjectionStage):
	def run_projection(self, image, state):
		state.artifacts.setdefault("reads", []).append(image.shape)
		return image + 1


class _ProducedSinogram(SinogramStage):
	def run_sinogram(self, sinogram, state):
		state.artifacts.setdefault("rows", []).extend(state.work_indices("Y"))
		sinogram += 1


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
@pytest.mark.parametrize("bounds", [(0, 4), (5, 9), (12, 15)])
@pytest.mark.parametrize("sensitive", [False, True])
@pytest.mark.parametrize("fragmented", [False, True])
def test_stage_windows_own_dispatch_extents_and_consume_context(layout, bounds, sensitive, fragmented):
	"""Every projection/sinogram adapter must honor canonical computed coverage."""
	physical = ProjOrder(3, 15, 4)
	held = TomoWindow.from_bounds(physical, y=bounds, halo=5, dtype=np.float32)
	stages = [_ContextProjection(), _ContextProjection(), _ProducedSinogram()]
	stages[0].contract = replace(stages[0].contract, required_pad=PadRequirement(fixed=Y(2)))
	stages[1].contract = replace(
		stages[1].contract, required_pad=PadRequirement(fixed=Y(3)),
		extent_sensitive_axes=("Y",) if sensitive else (),
	)
	windows = resolve_stage_windows(stages, scan_config(), held)
	shape = held.with_pad().frange.length.to(layout)
	values = np.zeros(shape.as_tuple(), np.float32)
	state = pipeline.PipelineState(LocalArray(values, shape), scan_config(), recon_config(), source_window=held)
	if fragmented:
		state = pipeline.Pipeline(
			stages[:1], validate=False, continuation_stages=stages[1:],
		).run(state)
		result = pipeline.Pipeline(stages[1:], validate=False).run(state)
	else:
		result = pipeline.Pipeline(stages, validate=False).run(state)
	expected = np.zeros_like(values)
	for window in windows:
		expected[held.with_pad().packed_index(window, layout)] += 1
	np.testing.assert_array_equal(values, expected)
	assert result.artifacts["reads"] == [
		(window.with_pad().shape.Y, physical.X) for window in windows[:2] for _ in range(physical.Theta)
	]
	assert result.artifacts["rows"] == list(range(
		held.pad_before.Y, held.pad_before.Y + bounds[1] - bounds[0],
	))
	assert windows[0].shape.Y == (
		held.with_pad().shape.Y if sensitive else
		min(physical.Y, bounds[1] + 3) - max(0, bounds[0] - 3)
	)
	if sensitive:
		assert windows[1].with_pad().frange == held.with_pad().frange
	assert windows[1].shape.Y == windows[2].shape.Y == bounds[1] - bounds[0]


def test_assigned_ranges_and_compact_workspace_share_the_stage_window_frame():
	held = TomoWindow.from_bounds(ProjOrder(4, 20, 6), y=(7, 11), halo=(Y(3), Y(4)), dtype=np.float32)
	computed = held.for_stage(Y(1), Y(2), Y(2), Y(3))
	shape = held.with_pad().frange.length
	state = pipeline.PipelineState(
		LocalArray(np.zeros(shape.as_tuple(), np.float32), shape), scan_config(), recon_config(),
		source_window=held, stage_window=computed, work_range=Y.drange(0, shape.Y, 2),
	)
	assert list(state.work_indices("Y")) == [2, 4, 6, 8]
	state.work_range = Y.drange(4, 5, 1)
	assert state.scratch_index(computed.frange.length.to(SinoOrder))[0] == slice(2, 3, 1)
	assert state.work_index(SinoOrder)[0] == slice(4, 5, 1)
