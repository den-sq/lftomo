"""Small declarative helpers for reversible, type-owned documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class DocumentCodec:
	encode: Callable[[Any], Any]
	decode: Callable[[Any], Any]


@dataclass(frozen=True)
class DocumentField:
	name: str
	codec: DocumentCodec
	storage: str


def encode_document(value: Any, fields: Sequence[DocumentField]) -> dict[str, Any]:
	"""Encode fields declared once by their owning type."""

	return {
		"type": type(value).__name__,
		"fields": {
			field.name: field.codec.encode(getattr(value, field.storage)) for field in fields
		},
	}


def decode_document(cls, document: dict[str, Any], fields: Sequence[DocumentField]):
	"""Restore an instance from the same field declaration used to encode it."""

	if not isinstance(document, dict) or set(document) != {"type", "fields"}:
		raise ValueError(f"expected {cls.__name__} document")
	values = document["fields"]
	names = tuple(field.name for field in fields)
	if document["type"] != cls.__name__ or not isinstance(values, dict) \
			or set(values) != set(names) or len(names) != len(set(names)):
		raise ValueError(f"fields do not match {cls.__name__}")
	result = cls.__new__(cls)
	for field in fields:
		object.__setattr__(result, field.storage, field.codec.decode(values[field.name]))
	return result


def array_to_document(value: np.ndarray) -> dict[str, Any]:
	array = np.ascontiguousarray(value)
	return {
		"dtype": array.dtype.str,
		"shape": [int(size) for size in array.shape],
		"data": array.tobytes().hex(),
	}


def array_from_document(document: dict[str, Any]) -> np.ndarray:
	if not isinstance(document, dict) or set(document) != {"dtype", "shape", "data"}:
		raise ValueError("array document is incomplete")
	if not isinstance(document["dtype"], str) or not isinstance(document["data"], str):
		raise TypeError("array document dtype and data must be strings")
	shape = document["shape"]
	if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
		raise TypeError("array document shape must contain non-negative integers")
	array = np.frombuffer(bytes.fromhex(document["data"]), dtype=np.dtype(document["dtype"]))
	try:
		return array.reshape(shape).copy()
	except ValueError as error:
		raise ValueError("array document payload does not match its shape") from error
