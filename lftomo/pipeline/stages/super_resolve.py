"""Sinogram super-resolution Stages.

The DRHT adapter and patch-merging helpers are adapted, with permission and
attribution, from ``ReconChengLab/localCT`` commit 14a233a. The source tree did
not contain the LDM implementation or checkpoints; ``LDMStage`` therefore
defines an adapter to a restored external repository instead of fabricating a
model implementation.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from pydantic import Field

from lftomo.pipeline.core import (
	PipelineState,
	RuntimeRequirement,
	Stage,
	StageContract,
	StageParams,
	allocate_output,
	commit_output,
)
from lftomo.util.shapes import ChunkGrid, ProjOrder, SinoOrder, YSlice


_TORCH_RUNTIME_REQUIREMENT = RuntimeRequirement(
	capability="super-resolution",
	package="torch",
	module="torch",
	probe_attributes=("load", "no_grad"),
	corrective_action="install lftomo with the super-resolution extra",
)

_DRHT_SCRATCH_NAMES = (
	"output", "count", "l1", "l1_count", "kl_row", "kl_row_count",
	"kl_col", "kl_col_count",
)


def _contract(
	runtime_requirements: tuple[RuntimeRequirement, ...] = (),
) -> StageContract:
	return StageContract(
		accepts=(SinoOrder,),
		produces=SinoOrder,
		window_axes=("Y", "Theta"),
		produces_artifacts=("super_resolution",),
		runtime_requirements=runtime_requirements,
	)


def _dense_inputs(state: PipelineState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if state.backend.name != "numpy":
		raise TypeError("super-resolution Stages currently require the NumPy backend")
	sinograms = np.asarray(state.working_array).astype(np.float32, copy=False)
	if sinograms.ndim != 3:
		raise ValueError("super-resolution Stages expect a (Y, Theta, X) sinogram array")
	volume = np.transpose(sinograms, (1, 0, 2))
	mask = np.asarray(state.artifacts.get("availability_mask", np.isfinite(volume)), dtype=bool)
	if mask.ndim == 1:
		mask = np.broadcast_to(mask[:, None, None], volume.shape)
	else:
		mask = np.broadcast_to(mask, volume.shape)
	angles = None
	if state.dataset is not None and state.source_window is not None:
		angles = state.dataset.angles[state.source_window.axis_slice("Theta")]
	if angles is None:
		angles = np.linspace(
			state.scan.acquisition.rotation_start_deg,
			state.scan.acquisition.rotation_end_deg,
			state.working_array.shape.Theta,
			endpoint=False,
			dtype=np.float32,
		)
	else:
		angles = np.rad2deg(angles) if np.max(np.abs(angles)) <= 2 * np.pi + 0.1 else angles
	angles = np.asarray(angles, dtype=np.float32)
	if angles.shape != (state.working_array.shape.Theta,):
		raise ValueError("super-resolution angles must match the projection count")
	return volume, mask, angles


def interpolate_missing_angles(volume: np.ndarray, mask: np.ndarray, theta: np.ndarray) -> np.ndarray:
	"""Deterministically interpolate missing angular samples column-by-column."""

	dense = np.asarray(volume, dtype=np.float32)
	flat = dense.reshape(dense.shape[0], -1)
	flat_mask = np.asarray(mask, dtype=bool).reshape(dense.shape[0], -1)
	out = np.empty_like(flat)
	for column in range(flat.shape[1]):
		known = flat_mask[:, column]
		if not np.any(known):
			out[:, column] = 0
		elif np.count_nonzero(known) == 1:
			out[:, column] = flat[known, column][0]
		else:
			values = flat[known, column]
			out[:, column] = np.interp(theta, theta[known], values, left=values[0], right=values[-1])
	return out.reshape(dense.shape)


def _store_result(state: PipelineState, volume: np.ndarray, metadata: dict) -> PipelineState:
	values = np.asarray(volume, dtype=np.float32)
	shape = ProjOrder(*values.shape)
	output, target = allocate_output(state, shape.to(SinoOrder), np.float32, role="super_resolution")
	target[...] = shape.transposed_view(values, SinoOrder)
	result = commit_output(
		state, output, retire_as=None,
	)
	result.artifacts["super_resolution"] = metadata
	return result


class DRHTStage(Stage):
	"""Dense residual hierarchical transformer sinogram restoration.

	``repo_path`` is imported as Python code and ``checkpoint_path`` is loaded
	through ``torch.load``. Both inputs must come from a trusted operator source;
	see ``docs/super-resolution-security.md``.
	"""

	name = "super_resolve.drht"
	contract = _contract(runtime_requirements=(_TORCH_RUNTIME_REQUIREMENT,))

	class Params(StageParams):
		repo_path: Path | None = None
		checkpoint_path: Path | None = None
		device: str = "cuda"
		arch: str = "DRHT"
		embed_dim: int = 32
		patch_size: int = Field(default=128, ge=1)
		patch_overlap: float = Field(default=0.25, ge=0, lt=1)
		dd_in: int = 1
		allow_bicubic_fallback: bool = False

	def _issues(self) -> list[str]:
		issues = []
		if self.params.repo_path is None or not self.params.repo_path.exists():
			issues.append("DRHT repo path is not set or does not exist")
		if self.params.checkpoint_path is None or not self.params.checkpoint_path.exists():
			issues.append("DRHT checkpoint path is not set or does not exist")
		return issues

	def _load_model(self, torch, device):
		repo = str(self.params.repo_path.resolve())
		old_path = list(sys.path)
		try:
			if repo not in sys.path:
				sys.path.insert(0, repo)
			importlib.invalidate_caches()
			util = importlib.import_module("util")
			options = SimpleNamespace(
				arch=self.params.arch,
				embed_dim=self.params.embed_dim,
				train_ps=self.params.patch_size,
				dd_in=self.params.dd_in,
			)
			model = util.get_arch(options)
			checkpoint = torch.load(str(self.params.checkpoint_path), map_location=device)
			state_dict = checkpoint.get("state_dict", checkpoint)
			try:
				model.load_state_dict(state_dict)
			except Exception:
				model.load_state_dict({
					key[7:] if key.startswith("module.") else key: value
					for key, value in state_dict.items()
				})
		finally:
			sys.path[:] = old_path
		return model.to(device).eval()

	def _infer(
		self, model, image: np.ndarray, torch, device,
		scratch: dict[str, np.ndarray],
	) -> tuple[np.ndarray, dict]:
		normalized, offset, scale = normalize01(image)
		stride = max(1, round(self.params.patch_size * (1 - self.params.patch_overlap)))
		y_grid = ChunkGrid(self.params.patch_size, stride, normalized.shape[0], "reanchor")
		x_grid = ChunkGrid(self.params.patch_size, stride, normalized.shape[1], "reanchor")
		for values in scratch.values():
			values.fill(0)
		output, count = scratch["output"], scratch["count"]
		attention = {name: scratch[name] for name in ("l1", "kl_row", "kl_col")}
		attention_count = {name: scratch[f"{name}_count"] for name in attention}
		with torch.no_grad():
			for y_window in y_grid:
				for x_window in x_grid:
					chunk = normalized[y_window, x_window]
					patch = np.pad(
						chunk,
						((0, self.params.patch_size - chunk.shape[0]), (0, self.params.patch_size - chunk.shape[1])),
						mode="edge",
					).astype(np.float32)
					tensor = torch.from_numpy(patch[None, None]).to(
						device=device, dtype=torch.float32,
					)
					prediction, l1, kl_row, kl_col = model(tensor)
					pred = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
					merge_patch(output, count, pred, y_window, x_window)
					for name, values in zip(attention, (l1, kl_row, kl_col)):
						expanded = expand_attention(
							values[0, 0].detach().cpu().numpy(), pred.shape,
						)
						merge_patch(
							attention[name], attention_count[name], expanded,
							y_window, x_window,
						)
		np.maximum(count, 1, out=count)
		np.divide(output, count, out=output)
		output *= scale
		output += offset
		for name in attention:
			np.maximum(attention_count[name], 1, out=attention_count[name])
			np.divide(attention[name], attention_count[name], out=attention[name])
		return output, attention

	def run(self, state: PipelineState) -> PipelineState:
		volume, mask, theta = _dense_inputs(state)
		seed = interpolate_missing_angles(volume, mask, theta)
		issues = self._issues()
		if issues:
			if not self.params.allow_bicubic_fallback:
				raise RuntimeError("; ".join(issues))
			return _store_result(state, seed, {"method": "drht", "fallback": "interpolation", "issues": issues})

		import torch

		device = torch.device("cuda" if self.params.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
		model = self._load_model(torch, device)
		restored = np.empty_like(seed)
		attention_volumes = {name: np.empty_like(seed) for name in ("l1", "kl_row", "kl_col")}
		scratch_shape = state.working_array.shape.extract(YSlice)
		scratch_handles = []
		scratch = {}
		try:
			for name in _DRHT_SCRATCH_NAMES:
				if state.storage is None:
					scratch[name] = np.empty(scratch_shape.as_tuple(), dtype=np.float32)
				else:
					handle = state.storage.allocate(
						scratch_shape, np.float32, role=f"drht_{name}_scratch",
					)
					scratch_handles.append(handle)
					scratch[name] = handle.array()
			for index in range(state.working_array.shape.Y):
				restored[:, index], maps = self._infer(
					model, seed[:, index], torch, device, scratch,
				)
				for name, values in maps.items():
					attention_volumes[name][:, index] = values
		finally:
			if state.storage is not None:
				for handle in scratch_handles:
					state.storage.retire(handle)
		result = _store_result(state, restored, {"method": "drht", "device": str(device)})
		result.artifacts["drht_attention"] = attention_volumes
		return result


class LDMStage(Stage):
	"""Adapter for the separately restored physics-guided LDM repository.

	The configured repository module is arbitrary Python code. Its repository,
	configuration, and checkpoint must be treated as trusted executable inputs;
	see ``docs/super-resolution-security.md``.
	"""

	name = "super_resolve.ldm"
	contract = _contract()

	class Params(StageParams):
		repo_path: Path | None = None
		config_path: Path | None = None
		checkpoint_path: Path | None = None
		module: str = "inference"
		callable_name: str = "restore_sinograms"
		device: str = "cuda"
		allow_interpolation_fallback: bool = False

	def _issues(self) -> list[str]:
		paths = {
			"LDM repo": self.params.repo_path,
			"LDM config": self.params.config_path,
			"LDM checkpoint": self.params.checkpoint_path,
		}
		return [f"{name} path is not set or does not exist" for name, path in paths.items() if path is None or not path.exists()]

	def run(self, state: PipelineState) -> PipelineState:
		volume, mask, theta = _dense_inputs(state)
		issues = self._issues()
		if issues:
			if not self.params.allow_interpolation_fallback:
				raise RuntimeError("; ".join(issues))
			seed = interpolate_missing_angles(volume, mask, theta)
			return _store_result(state, seed, {"method": "ldm", "fallback": "interpolation", "issues": issues})
		old_path = list(sys.path)
		try:
			sys.path.insert(0, str(self.params.repo_path.resolve()))
			module = importlib.import_module(self.params.module)
			restore = getattr(module, self.params.callable_name)
			restored = restore(
				volume=volume,
				availability_mask=mask,
				theta_degrees=theta,
				config_path=self.params.config_path,
				checkpoint_path=self.params.checkpoint_path,
				device=self.params.device,
			)
		finally:
			sys.path[:] = old_path
		return _store_result(state, restored, {"method": "ldm", "device": self.params.device})


def normalize01(image: np.ndarray) -> tuple[np.ndarray, float, float]:
	array = np.asarray(image, dtype=np.float32)
	low, high = float(array.min()), float(array.max())
	scale = high - low or 1.0
	return (array - low) / scale, low, scale


def merge_patch(target, count, patch, y: slice, x: slice):
	target[y, x] += patch[:y.stop - y.start, :x.stop - x.start]
	count[y, x] += 1


def expand_attention(attention: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
	array = np.asarray(attention, dtype=np.float32)
	if array.shape == target_shape:
		return array
	if array.ndim == 1:
		axis = 1 if array.shape[0] == target_shape[0] else 0
		return np.repeat(array[:, None] if axis == 1 else array[None, :], target_shape[axis], axis=axis)
	return np.resize(array, target_shape)
