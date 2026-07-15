"""Axis-safe views used by order-aware Stage implementations."""

from __future__ import annotations

from typing import Any

import numpy as np

from lftomo.util.shapes import ProjOrder, SinoOrder


def _tomography_layout(layout: type[Any]) -> None:
	if layout not in (ProjOrder, SinoOrder):
		raise TypeError(f"expected ProjOrder or SinoOrder, got {layout.__name__}")


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
