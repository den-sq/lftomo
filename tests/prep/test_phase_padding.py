"""Canonical canvas ownership and unchanged numerical phase boundaries."""

from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import json

import numpy as np
import pytest

from lftomo.prep import materials, phase
from lftomo.util.shapes import ProjOrder, SinoOrder
from lftomo.util.window import DataRef
from tests.prep.test_current_api import _shared, _window


def _forbid_padding_copy(*args, **kwargs):
	raise AssertionError("phase padding must use its canonical owned canvas")


@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("kind", ["spatial", "scalar"])
@pytest.mark.parametrize("batch", [False, True])
def test_affine_uses_prepared_window_and_owned_canvases(monkeypatch, pad, kind, batch):
	with np.load(Path(__file__).parents[1] / "fixtures/phase_affine_reference.npz") as reference:
		image = reference["transmission"]
		envelope = reference["envelope"] if kind == "spatial" else 3e-4
		expected = reference[f"{kind}_{int(pad)}"]
		if batch:
			image = np.stack((image, image))
			expected = np.stack((expected, expected))
		window = phase.phase_fft_window(image.shape[-2:], pad)
		shape = image.shape[:-2] + window.with_pad().shape.as_tuple()
		padding = np.empty(shape, dtype=np.float32)
		thickness = np.empty_like(padding)
		calls = []
		fill = DataRef.pad_array

		def observe(self, array, *, out=None):
			calls.append((self, out))
			return fill(self, array, out=out)

		monkeypatch.setattr(DataRef, "pad_array", observe)
		monkeypatch.setattr(phase, "phase_fft_window", _forbid_padding_copy)
		monkeypatch.setattr(np, "pad", _forbid_padding_copy)
		actual = phase.retrieve_single(
			image, 1e-6, 0.01, 1e-9, pad=not pad, material_alpha=2e-8, envelope=envelope,
			window=window, padding_work=padding, envelope_work=thickness,
		)
		np.testing.assert_array_equal(actual, expected)
		assert actual.dtype == np.float32
		assert len(calls) == 2
		assert calls[0][0] is calls[1][0] is window
		assert calls[0][1] is padding and calls[1][1] is thickness
		np.testing.assert_array_equal(phase.phase_unpad(padding, window), image)


@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("plane", [False, True])
def test_material_predictor_uses_owned_padding_and_preserves_rank(monkeypatch, pad, plane):
	path = Path(__file__).parents[1] / "fixtures/phase_materials_reference.npz"
	with np.load(path) as reference:
		arrays = dict(reference)
	metadata = json.loads(path.with_suffix(".json").read_text())
	psi, intensity = arrays["psi"], arrays["intensity"]
	expected = arrays["error_padded" if pad else "error_unpad"]
	if plane:
		psi, intensity, expected = psi[0], intensity[0], expected[:1]
	window = phase.phase_fft_window(psi.shape[-2:], pad)
	shape = psi.shape[:-2] + window.with_pad().shape.as_tuple()
	psi_work = np.empty(shape, dtype=np.float64)
	intensity_work = np.empty(shape, dtype=intensity.dtype)
	monkeypatch.setattr(phase, "phase_fft_window", _forbid_padding_copy)
	monkeypatch.setattr(np, "pad", _forbid_padding_copy)
	result = materials.predict_filter_error(
		psi, intensity, metadata["energy_kev"], metadata["pixel_m"], metadata["distance_m"],
		metadata["slope"], pad=not pad, workers=1,
		window=window, psi_work=psi_work, intensity_work=intensity_work,
	)
	np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-16)
	assert result.shape == expected.shape
	np.testing.assert_array_equal(phase.phase_unpad(psi_work, window), psi)
	np.testing.assert_array_equal(phase.phase_unpad(intensity_work, window), intensity)


@pytest.mark.parametrize("layout", [ProjOrder, SinoOrder])
@pytest.mark.parametrize("fail", [False, True])
def test_legacy_parent_owns_projection_canvases_and_retires_them(monkeypatch, layout, fail):
	shape = layout(Theta=2, Y=4, X=4)
	work = _shared(shape)
	work.array()[...] = 0.6
	canvases, windows = [], []
	actual_window = phase.legacy_phase_window

	def observe_window(*args, **kwargs):
		window = actual_window(*args, **kwargs)
		windows.append(window)
		return window

	class InlinePool:
		def __init__(self, processes):
			assert processes == 2

		def __enter__(self):
			return self

		def __exit__(self, *args):
			pass

		def starmap(self, kernel, arguments):
			for args in arguments:
				window, canvas = args[-2:]
				assert window is windows[0]
				assert canvas.shape == window.with_pad().shape
				assert not canvas.owns_unlink
				canvases.append(canvas.name)
				if fail:
					raise RuntimeError("worker failed")
				kernel(*args)

	monkeypatch.setattr(phase, "legacy_phase_window", observe_window)
	monkeypatch.setattr(phase, "Pool", InlinePool)
	monkeypatch.setattr(np, "pad", _forbid_padding_copy)
	try:
		if fail:
			with pytest.raises(RuntimeError, match="worker failed"):
				phase.phase_retrieval(_window(shape), work, 1e-3, 1e-3, 20, 2,
					sino_order=layout is SinoOrder, tp=True, height_pad=True, proc_count=2)
		else:
			phase.phase_retrieval(_window(shape), work, 1e-3, 1e-3, 20, 2,
				sino_order=layout is SinoOrder, tp=True, height_pad=True, proc_count=2)
			assert len(canvases) == 2
		assert len(windows) == len(set(canvases)) == 1
		with pytest.raises(FileNotFoundError):
			SharedMemory(name=canvases[0])
	finally:
		work.shutdown()
