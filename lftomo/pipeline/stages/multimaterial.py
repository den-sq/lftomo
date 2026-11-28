"""Damped, per-window multimaterial phase correction."""

from __future__ import annotations

from functools import cached_property

import numpy as np
from pydantic import Field, model_validator
from scipy import ndimage

from lftomo.pipeline.core import (
	BoundedWorkRequest, BufferSpec, PadRequirement, PipelineState, Stage, StageContract, StageParams,
)
from lftomo.pipeline.domains import DataDomain, PRESERVE
from lftomo.pipeline.stages.phase import slope_filter_shape
from lftomo.prep.materials import fbp, fit_material_line, forward_project, predict_filter_error, segment_materials
from lftomo.prep.phase import retrieve_single, wavelength_m
from lftomo.util.shapes import ProjOrder, ReconOrder, SinoOrder


class MultimaterialStage(Stage):
	"""Retrieve attenuation and damp its segmentation-predicted filter error.

	The working FBP is the reference classifier's reconstruction, not the
	configured downstream reconstruction. All halo rows participate in every
	iteration. FFT extent and the projection-wide air median affect the result.
	Only the Stage's produced extent is published back to the held input buffer.
	"""

	name = "phase.multimaterial"
	contract = StageContract(
		accepts=(ProjOrder, SinoOrder), prefers=ProjOrder, produces=PRESERVE,
		accepts_domain=(DataDomain.TRANSMISSION,), produces_domain=DataDomain.ATTENUATION,
		mutates_input=True, cacheable_prefix=True, window_axes=("Y", "Theta"),
		extent_sensitive_axes=("Y", "X"), requires_row_identity=True,
		requires_full_axes=("Theta",),
		required_metadata=("energy_keV", "pixel_size_um", "ssd_cm"),
		produces_artifacts=("phase_filter", "correction_norms", "label_change"),
		managed_artifacts=("phase_filter",),
	)

	class Params(StageParams):
		mu_table: tuple[float, ...] = Field(min_length=1)
		delta_table: tuple[float, ...] = Field(min_length=1)
		weights: tuple[float, ...] | None = None
		n_iter: int = Field(default=1, ge=0)
		omega: float = Field(default=0.7, ge=0, le=1)
		air_ratio: float | None = Field(default=0.05, ge=0, le=0.5)
		sino_sigma: float = Field(default=1.0, ge=0)
		classify_sigma: float = Field(default=1.5, ge=0)
		label_smooth: int = Field(default=3, ge=0)
		pad: bool = True

		@model_validator(mode="after")
		def material_columns(self):
			mu, delta = np.asarray(self.mu_table), np.asarray(self.delta_table)
			if len(mu) != len(delta):
				raise ValueError("mu_table and delta_table must have the same material order and length")
			if not np.isfinite(mu).all() or np.any(mu <= 0) or len(np.unique(mu)) != len(mu):
				raise ValueError("mu_table must contain distinct finite positive coefficients")
			if not np.isfinite(delta).all():
				raise ValueError("delta_table must be finite")
			if self.weights is not None:
				weights = np.asarray(self.weights)
				if weights.shape != mu.shape or not np.isfinite(weights).all() \
						or np.any(weights < 0) or not np.any(weights > 0):
					raise ValueError("weights must match the materials and be finite, nonnegative, and nonzero")
			return self

		@cached_property
		def line_fit(self):
			"""The reference loop's sole operative, through-origin material fit."""
			return fit_material_line(self.mu_table, self.delta_table, self.weights, through_origin=True)

	@classmethod
	def contract_for(cls, params):
		from dataclasses import replace

		return replace(cls.contract, required_pad=PadRequirement(
			phase_fft_y=True, phase_fft_passes=params.n_iter + 1,
		))

	def bounded_work_requests(self, input_spec, window=None):
		"""Declare the composite's mixed-layout scratch through the Stage hook."""
		# The iterative FFT and projection-wide air estimate consume this same
		# context on every pass; only the final publication narrows to the core.
		shape = input_spec.shape if window is None else window.with_pad().frange.length.to(input_spec.layout)
		specs = {
			"baseline": (shape, np.dtype(np.float32)),
			"current": (shape, np.dtype(np.float32)),
			"correction": (shape, np.dtype(np.float64)),
			"psi": (shape, np.dtype(np.float64)),
			"synthesis": (shape, np.dtype(np.float64)),
		}
		if self.params.n_iter > 1:
			specs["previous_labels"] = (ReconOrder.of(shape), np.dtype(np.int32))
		return tuple(
			BoundedWorkRequest(
				artifact_key=f"_multimaterial_{key}", buffer_spec=BufferSpec(work_shape, dtype),
				role=f"multimaterial_{key}", fill_value=0 if key == "correction" else None,
				required_for_fallback=True,
			)
			for key, (work_shape, dtype) in specs.items()
		)

	def run(self, state: PipelineState) -> PipelineState:
		from lftomo.pipeline.execution import run_multimaterial_stage

		return run_multimaterial_stage(self, state)

	def phase_filter_shape(self, shape):
		return slope_filter_shape(shape, self.params.pad)

	def retrieve(self, image, baseline, scan, denominator):
		baseline[...] = retrieve_single(
			image, scan.pixel_size_um * 1e-6, scan.ssd_cm / 100,
			self.params.line_fit["slope"], pad=self.params.pad, workers=1, denominator=denominator,
		)
		if self.params.air_ratio:
			margin = max(1, int(self.params.air_ratio * image.shape[-1]))
			baseline -= np.median(np.concatenate((baseline[:, :margin], baseline[:, -margin:]), axis=-1))

	def classify_row(self, sinogram, psi, synthesis, previous_labels, angles_deg, scan, iteration):
		"""Library return arrays are per-slice temporaries; reusable work is managed."""
		pixel = scan.pixel_size_um * 1e-6
		mu = fbp(sinogram, angles_deg, pixel)
		labels = segment_materials(mu, self.params.mu_table, sigma=self.params.classify_sigma)
		if self.params.label_smooth:
			labels = ndimage.median_filter(labels, size=self.params.label_smooth)
		if iteration == self.params.n_iter:
			return 0
		psi.fill(0)
		synthesis.fill(0)
		lam = wavelength_m(scan.energy_keV)
		for j, (coefficient, residual) in enumerate(zip(self.params.mu_table, self.params.line_fit["eps"], strict=True)):
			thickness = forward_project((labels == j + 1).astype(float), angles_deg, pixel)
			if self.params.sino_sigma:
				thickness = ndimage.gaussian_filter1d(thickness, self.params.sino_sigma, axis=-1)
			psi += -(2 * np.pi / lam) * residual * thickness
			synthesis += coefficient * thickness
		changed = 0
		if previous_labels is not None:
			if iteration:
				changed = int(np.count_nonzero(labels != previous_labels))
			previous_labels[...] = labels
		return changed

	def correct(self, psi, synthesis, correction, scan, denominator):
		# FFT/gradient return arrays are temporary outputs of SciPy/NumPy; no
		# nested executor or reusable, unmanaged work plane is created here.
		np.negative(synthesis, out=synthesis)
		np.exp(synthesis, out=synthesis)
		delta = predict_filter_error(
			psi, synthesis, scan.energy_keV, scan.pixel_size_um * 1e-6,
			scan.ssd_cm / 100, self.params.line_fit["slope"], pad=self.params.pad, workers=1,
			denominator=denominator,
		)[0]
		correction *= 1 - self.params.omega
		correction += self.params.omega * delta
