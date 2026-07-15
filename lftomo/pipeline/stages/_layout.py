"""Axis-safe views used by order-aware Stage implementations."""

from __future__ import annotations

from typing import Any

import numpy as np

from lftomo.util.shapes import ProjOrder, SinoOrder


def _tomography_layout(layout: type[Any]) -> None:
	if layout not in (ProjOrder, SinoOrder):
		raise TypeError(f"expected ProjOrder or SinoOrder, got {layout.__name__}")


def configured_layout(order: str | None) -> type[Any] | None:
	"""Translate a legacy textual order flag into a layout type."""

	if order is None:
		return None
	if order == "projection":
		return ProjOrder
	if order == "sinogram":
		return SinoOrder
	raise ValueError("array order must be 'projection' or 'sinogram'")


def resolve_layout(
	state: Any,
	configured: str | None,
	stage_name: str,
	*,
	legacy_default: type[Any] = SinoOrder,
) -> type[Any]:
	"""Use state as truth and check, rather than apply, a compatibility flag."""

	compatibility = configured_layout(configured)
	if state.layout is None:
		return compatibility or legacy_default
	_tomography_layout(state.layout)
	if compatibility is not None and compatibility is not state.layout:
		raise ValueError(
			f"{stage_name} order compatibility flag disagrees with PipelineState.layout"
		)
	return state.layout


def projection_view(data: np.ndarray, layout: type[Any], index: int) -> np.ndarray:
	"""Return projection ``index`` as a potentially strided ``(Y, X)`` view."""

	_tomography_layout(layout)
	return data[index, :, :] if layout is ProjOrder else data[:, index, :]


def sinogram_view(data: np.ndarray, layout: type[Any], index: int) -> np.ndarray:
	"""Return detector-row ``index`` as a potentially strided ``(Theta, X)`` view."""

	_tomography_layout(layout)
	return data[:, index, :] if layout is ProjOrder else data[index, :, :]


def sinogram_scratch(data: np.ndarray, layout: type[Any], index: int) -> np.ndarray:
	"""Gather one sinogram into a writable C-contiguous scratch buffer."""

	return np.ascontiguousarray(sinogram_view(data, layout, index))


def n_projections(data: np.ndarray, layout: type[Any]) -> int:
	"""Return the Theta extent for either tomography layout."""

	_tomography_layout(layout)
	return data.shape[0] if layout is ProjOrder else data.shape[1]


def n_rows(data: np.ndarray, layout: type[Any]) -> int:
	"""Return the detector-row extent for either tomography layout."""

	_tomography_layout(layout)
	return data.shape[1] if layout is ProjOrder else data.shape[0]
