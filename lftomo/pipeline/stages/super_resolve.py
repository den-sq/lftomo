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

from lftomo.config import StageConfig
from lftomo.pipeline.core import (
	PipelineState,
	RuntimeRequirement,
	Stage,
	StageContract,
	allocate_output,
	commit_output,
)
from lftomo.storage import ManagedBuffer
from lftomo.util.shapes import ProjOrder, SinoOrder


_TORCH_RUNTIME_REQUIREMENT = RuntimeRequirement(
	capability="super-resolution",
	package="torch",
	module="torch",
	probe_attributes=("load", "no_grad"),
	corrective_action="install lftomo with the super-resolution extra",
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
	angles = state.artifacts.get("angles")
	if angles is None:
		angles = np.linspace(
			state.scan.acquisition.rotation_start_deg,
			state.scan.acquisition.rotation_end_deg,
			volume.shape[0],
			endpoint=False,
			dtype=np.float32,
		)
	else:
		angles = np.rad2deg(angles) if np.max(np.abs(angles)) <= 2 * np.pi + 0.1 else angles
	angles = np.asarray(angles, dtype=np.float32)
	if angles.shape != (volume.shape[0],):
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
	shape = SinoOrder(Y=values.shape[1], Theta=values.shape[0], X=values.shape[2])
	output, target, buffer_spec = allocate_output(
		state, shape, np.float32, role="super_resolution",
	)
	if (
		isinstance(output, ManagedBuffer)
		and state.storage is not None
		and state.executor is not None
	):
		from lftomo.pipeline.execution import run_layout_fill_kernel

		source = state.storage.allocate(
			ProjOrder(*values.shape), np.float32, role="super_resolution_assembly",
		)
		try:
			source.array()[...] = values
			state.executor.submit_all(run_layout_fill_kernel, (
				(source, output, ProjOrder, SinoOrder, index)
				for index in range(values.shape[0])
			))
		finally:
			state.storage.retire(source, role=None)
	else:
		target[...] = np.transpose(values, (1, 0, 2))
	result = commit_output(
		state, output, retire_as=None, buffer_spec=buffer_spec,
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

	def __init__(self, repo_path: str | Path | None = None,
				checkpoint_path: str | Path | None = None, device: str = "cuda",
				arch: str = "DRHT", embed_dim: int = 32, patch_size: int = 128,
				patch_overlap: float = 0.25, dd_in: int = 1,
				allow_bicubic_fallback: bool = False):
		if patch_size < 1 or not 0 <= patch_overlap < 1:
			raise ValueError("invalid DRHT patch size or overlap")
		self.repo_path = None if repo_path is None else Path(repo_path)
		self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
		self.device = device
		self.arch = arch
		self.embed_dim = embed_dim
		self.patch_size = patch_size
		self.patch_overlap = patch_overlap
		self.dd_in = dd_in
		self.allow_bicubic_fallback = allow_bicubic_fallback

	@classmethod
	def from_config(cls, config: StageConfig) -> "DRHTStage":
		return cls(**config.params)

	def _issues(self) -> list[str]:
		issues = []
		if self.repo_path is None or not self.repo_path.exists():
			issues.append("DRHT repo path is not set or does not exist")
		if self.checkpoint_path is None or not self.checkpoint_path.exists():
			issues.append("DRHT checkpoint path is not set or does not exist")
		return issues

	def _load_model(self, torch, device):
		repo = str(self.repo_path.resolve())
		old_path = list(sys.path)
		try:
			if repo not in sys.path:
				sys.path.insert(0, repo)
			importlib.invalidate_caches()
			util = importlib.import_module("util")
			options = SimpleNamespace(
				arch=self.arch,
				embed_dim=self.embed_dim,
				train_ps=self.patch_size,
				dd_in=self.dd_in,
			)
			model = util.get_arch(options)
			checkpoint = torch.load(str(self.checkpoint_path), map_location=device)
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

	def _infer(self, model, image: np.ndarray, torch, device) -> tuple[np.ndarray, dict]:
		normalized, offset, scale = normalize01(image)
		stride = max(1, round(self.patch_size * (1 - self.patch_overlap)))
		patches, locations = extract_patches(normalized, self.patch_size, stride)
		output = np.zeros_like(normalized)
		count = np.zeros_like(normalized)
		attention = {name: np.zeros_like(normalized) for name in ("l1", "kl_row", "kl_col")}
		attention_count = {name: np.zeros_like(normalized) for name in attention}
		with torch.no_grad():
			for patch, location in zip(patches, locations):
				tensor = torch.from_numpy(patch[None, None]).to(device=device, dtype=torch.float32)
				prediction, l1, kl_row, kl_col = model(tensor)
				pred = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
				merge_patch(output, count, pred, *location)
				for name, values in zip(attention, (l1, kl_row, kl_col)):
					expanded = expand_attention(values[0, 0].detach().cpu().numpy(), pred.shape)
					merge_patch(attention[name], attention_count[name], expanded, *location)
		output = np.divide(output, np.clip(count, 1, None), where=count > 0)
		for name in attention:
			attention[name] = np.divide(
				attention[name], np.clip(attention_count[name], 1, None), where=attention_count[name] > 0
			)
		return output * scale + offset, attention

	def run(self, state: PipelineState) -> PipelineState:
		volume, mask, theta = _dense_inputs(state)
		seed = interpolate_missing_angles(volume, mask, theta)
		issues = self._issues()
		if issues:
			if not self.allow_bicubic_fallback:
				raise RuntimeError("; ".join(issues))
			return _store_result(state, seed, {"method": "drht", "fallback": "interpolation", "issues": issues})

		import torch

		device = torch.device("cuda" if self.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
		model = self._load_model(torch, device)
		restored = np.empty_like(seed)
		attention_volumes = {name: np.empty_like(seed) for name in ("l1", "kl_row", "kl_col")}
		for index in range(seed.shape[1]):
			restored[:, index], maps = self._infer(model, seed[:, index], torch, device)
			for name, values in maps.items():
				attention_volumes[name][:, index] = values
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

	def __init__(self, repo_path: str | Path | None = None,
				config_path: str | Path | None = None,
				checkpoint_path: str | Path | None = None,
				module: str = "inference", callable_name: str = "restore_sinograms",
				device: str = "cuda", allow_interpolation_fallback: bool = False):
		self.repo_path = None if repo_path is None else Path(repo_path)
		self.config_path = None if config_path is None else Path(config_path)
		self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
		self.module = module
		self.callable_name = callable_name
		self.device = device
		self.allow_interpolation_fallback = allow_interpolation_fallback

	@classmethod
	def from_config(cls, config: StageConfig) -> "LDMStage":
		return cls(**config.params)

	def _issues(self) -> list[str]:
		paths = {
			"LDM repo": self.repo_path,
			"LDM config": self.config_path,
			"LDM checkpoint": self.checkpoint_path,
		}
		return [f"{name} path is not set or does not exist" for name, path in paths.items() if path is None or not path.exists()]

	def run(self, state: PipelineState) -> PipelineState:
		volume, mask, theta = _dense_inputs(state)
		issues = self._issues()
		if issues:
			if not self.allow_interpolation_fallback:
				raise RuntimeError("; ".join(issues))
			seed = interpolate_missing_angles(volume, mask, theta)
			return _store_result(state, seed, {"method": "ldm", "fallback": "interpolation", "issues": issues})
		old_path = list(sys.path)
		try:
			sys.path.insert(0, str(self.repo_path.resolve()))
			module = importlib.import_module(self.module)
			restore = getattr(module, self.callable_name)
			restored = restore(
				volume=volume,
				availability_mask=mask,
				theta_degrees=theta,
				config_path=self.config_path,
				checkpoint_path=self.checkpoint_path,
				device=self.device,
			)
		finally:
			sys.path[:] = old_path
		return _store_result(state, restored, {"method": "ldm", "device": self.device})


def normalize01(image: np.ndarray) -> tuple[np.ndarray, float, float]:
	array = np.asarray(image, dtype=np.float32)
	low, high = float(array.min()), float(array.max())
	scale = high - low or 1.0
	return (array - low) / scale, low, scale


def patch_positions(length: int, patch_size: int, stride: int) -> list[int]:
	if length <= patch_size:
		return [0]
	positions = list(range(0, length - patch_size + 1, stride))
	if positions[-1] != length - patch_size:
		positions.append(length - patch_size)
	return positions


def extract_patches(image: np.ndarray, patch_size: int, stride: int):
	patches, locations = [], []
	for y in patch_positions(image.shape[0], patch_size, stride):
		for x in patch_positions(image.shape[1], patch_size, stride):
			chunk = image[y:y + patch_size, x:x + patch_size]
			patch = np.pad(
				chunk,
				((0, patch_size - chunk.shape[0]), (0, patch_size - chunk.shape[1])),
				mode="edge",
			)
			patches.append(patch.astype(np.float32))
			locations.append((y, x, chunk.shape[0], chunk.shape[1]))
	return patches, locations


def merge_patch(target, count, patch, y, x, height, width):
	target[y:y + height, x:x + width] += patch[:height, :width]
	count[y:y + height, x:x + width] += 1


def expand_attention(attention: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
	array = np.asarray(attention, dtype=np.float32)
	if array.shape == target_shape:
		return array
	if array.ndim == 1:
		axis = 1 if array.shape[0] == target_shape[0] else 0
		return np.repeat(array[:, None] if axis == 1 else array[None, :], target_shape[axis], axis=axis)
	return np.resize(array, target_shape)
