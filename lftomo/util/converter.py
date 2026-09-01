"""Shared cattrs converter for geometry documents."""

from __future__ import annotations

from functools import partial
from typing import Any

from cattrs import Converter
from cattrs.strategies import configure_tagged_union, include_subclasses
import numpy as np

from .shapes import (
	DataShape, GenericOrder, ImageStack, ImgSlice, Proj, ProjOrder, ReconOrder,
	ReverseReconOrder, ReverseSinoOrder, SinoOrder, Theta, X, XSlice, Y, YSlice, Z,
)
from .window import CTFov, DataRef, TomoDataSet, TomoWindow


CONVERTER = Converter(forbid_extra_keys=True, unstruct_collection_overrides={tuple: list})


def _array_to_document(value: np.ndarray) -> dict[str, Any]:
	array = np.ascontiguousarray(value)
	if array.dtype.hasobject:
		raise TypeError("object arrays are not documentable")
	return {
		"dtype": array.dtype.str,
		"shape": list(array.shape),
		"data": array.tobytes().hex(),
	}


def _array_from_document(document: Any, _target: Any) -> np.ndarray:
	if not isinstance(document, dict) or set(document) != {"dtype", "shape", "data"}:
		raise ValueError("array document is incomplete")
	array = np.frombuffer(bytes.fromhex(document["data"]), dtype=np.dtype(document["dtype"]))
	try:
		return array.reshape(document["shape"]).copy()
	except ValueError as error:
		raise ValueError("array document payload does not match its shape") from error


_SHAPE_TYPES = {
	target.__name__: target for target in (
		ProjOrder, SinoOrder, ReverseSinoOrder, ImageStack, ReconOrder,
		ReverseReconOrder, ImgSlice, Proj, YSlice, XSlice, Y, X, Theta, Z,
		GenericOrder,
	)
}


def _is_shape(target: Any) -> bool:
	return isinstance(target, type) and issubclass(target, DataShape)


def _shape_to_document(value: DataShape) -> dict[str, Any]:
	return {
		"type": type(value).__name__,
		"names": list(value.names),
		"values": list(value.as_tuple()),
	}


def _shape_from_document(document: Any, target: type[DataShape]) -> DataShape:
	type_name = document["type"]
	names = document["names"]
	if target is DataShape:
		resolved = _SHAPE_TYPES.get(type_name)
		if resolved is None:
			resolved = DataShape.gen(names)
	elif target.__name__ == type_name:
		resolved = target
	else:
		raise ValueError(f"expected {target.__name__}, received {type_name}")
	if resolved.__name__ != type_name or resolved.names != tuple(names):
		raise ValueError(f"invalid {type_name} shape document")
	return resolved(*document["values"])


CONVERTER.register_unstructure_hook(np.ndarray, _array_to_document)
CONVERTER.register_structure_hook(np.ndarray, _array_from_document)
CONVERTER.register_unstructure_hook(np.dtype, lambda value: value.str)
CONVERTER.register_structure_hook(np.dtype, lambda value, _target: np.dtype(value))
CONVERTER.register_unstructure_hook(slice, lambda value: [value.start, value.stop, value.step])
CONVERTER.register_structure_hook(slice, lambda value, _target: slice(*value))
CONVERTER.register_unstructure_hook(CTFov, lambda value: value._asdict())
CONVERTER.register_structure_hook(CTFov, lambda value, _target: CTFov(**value))
CONVERTER.register_unstructure_hook_func(_is_shape, _shape_to_document)
CONVERTER.register_structure_hook_func(_is_shape, _shape_from_document)
include_subclasses(
	DataRef, CONVERTER, subclasses=(TomoWindow, TomoDataSet),
	union_strategy=partial(configure_tagged_union, tag_name="type"),
)
