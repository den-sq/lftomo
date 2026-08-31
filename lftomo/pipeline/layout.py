"""Deprecated compatibility shims for named-axis tomography operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from lftomo.util.shapes import ProjOrder


def projection_view(data: np.ndarray, layout: type[Any], index: int) -> np.ndarray:
	"""Return projection ``index`` as a potentially strided ``(Y, X)`` view."""

	return data[layout.point_index(Theta=index)]


def sinogram_view(data: np.ndarray, layout: type[Any], index: int) -> np.ndarray:
	"""Return detector-row ``index`` as a potentially strided ``(Theta, X)`` view."""

	return data[layout.point_index(Y=index)]


def sinogram_scratch(data: np.ndarray, layout: type[Any], index: int) -> np.ndarray:
	"""Gather one sinogram into a writable C-contiguous scratch buffer."""

	return np.ascontiguousarray(sinogram_view(data, layout, index))


def n_projections(data: np.ndarray, layout: type[Any]) -> int:
	"""Return the Theta extent for either tomography layout."""

	return data.shape[layout.axis_of("Theta")]


def n_rows(data: np.ndarray, layout: type[Any]) -> int:
	"""Return the detector-row extent for either tomography layout."""

	return data.shape[layout.axis_of("Y")]
