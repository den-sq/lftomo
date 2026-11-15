"""Dispatch classifications shared by the Stage registry and executor."""

from dataclasses import dataclass
from enum import Enum, auto


class DispatchKind(Enum):
	"""Executor selected for one registered Stage entry."""

	PROCESS = auto()
	THREAD = auto()
	SERIAL = auto()
	ENGINE = auto()
	EXEMPT = auto()


@dataclass(frozen=True)
class DispatchSpec:
	"""Registry-owned dispatch classification and its callable boundary."""

	kind: DispatchKind
	kernel: str | None = None
	reason: str | None = None
	projection_units: bool = False

	def __post_init__(self):
		if self.kind is DispatchKind.EXEMPT and not self.reason:
			raise ValueError("executor exemptions require a documented reason")


def _dispatch(
	kind: DispatchKind,
	kernel: str,
	*,
	projection_units: bool = False,
	reason: str | None = None,
) -> DispatchSpec:
	return DispatchSpec(
		kind=kind, kernel=kernel, projection_units=projection_units, reason=reason,
	)


_PROJECTION_PROCESS = {
	"filter.bilateral",
	"geometry.affine_skew",
	"geometry.movement",
	"movement_correct.linear_shift",
	"normalize.phase_contrast",
	"normalize.beam_basis_fit",
	"normalize.beam_tracking",
	"normalize.beam_keyframes",
	"normalize.taper_blended",
	"normalize.theta_switch",
	"normalize.weighted",
	"phase.bac",
	"phase.bac_correction",
	"phase.beta_scale",
	"phase.bronnikov_aided",
	"phase.negative_log",
	"phase.paganin",
	"phase.paganin_filter",
	"phase.paganin_phase_absorption",
}
_BOUNDED_PROCESS = {
	"stripe.fw", "stripe.vo_dead", "stripe.vo_family", "stripe.vo_large", "stripe.vo_sort",
}
_OUT_OF_PLACE_PROCESS = {
	"geometry.affine_skew_movement", "geometry.helical_pitch",
	"helical_correct.index_remap", "layout.convert",
}
_COMPOSITE_PROCESS = {"phase.multimaterial"}
_THREAD_IO = {
	"read.hdf5_dxchange", "read.tiff_stack", "write.hdf5_volume", "write.neuroglancer",
}
_PROCESS_WRITERS = {"write.tiff_slices"}
_ENGINE = {
	"center.entropy_local", "center.pc", "center.vo",
	"recon.bp_cuda", "recon.cgls_cuda", "recon.em_cuda", "recon.fbp_cuda",
	"recon.fp_cuda", "recon.gridrec", "recon.sart_cuda", "recon.sirt_cuda",
	"stripe.all_of_the_stripe", "super_resolve.drht", "super_resolve.ldm",
}
_SERIAL = {"center.manual", "evaluate.metrics", "skew_correct.cv2_affine"}


def _build_stage_dispatch() -> dict[str, DispatchSpec]:
	"""Build the single name-keyed dispatch table without importing Stage classes."""

	groups = (
		_PROJECTION_PROCESS, _BOUNDED_PROCESS, _OUT_OF_PLACE_PROCESS,
		_THREAD_IO, _PROCESS_WRITERS, _ENGINE, _SERIAL, _COMPOSITE_PROCESS,
	)
	seen: set[str] = set()
	for group in groups:
		overlap = seen & group
		if overlap:
			raise ValueError(f"Stages have multiple dispatch classifications: {sorted(overlap)}")
		seen |= group
	result = {}
	for key in _PROJECTION_PROCESS:
		result[key] = _dispatch(
			DispatchKind.PROCESS, "run_projection_kernel", projection_units=True,
		)
	for key in _BOUNDED_PROCESS:
		result[key] = _dispatch(DispatchKind.PROCESS, "run_sinogram_chain_kernel")
	for key in _OUT_OF_PLACE_PROCESS:
		result[key] = _dispatch(
			DispatchKind.PROCESS,
			"manager_owned_parallel_fill",
			reason="parent allocates output; unit windows own worker fill geometry",
		)
	for key in _PROCESS_WRITERS:
		result[key] = _dispatch(DispatchKind.PROCESS, "write_tiff_slice_kernel")
	for key in _THREAD_IO:
		result[key] = _dispatch(DispatchKind.THREAD, "stage_io_chunks")
	for key in _ENGINE:
		result[key] = _dispatch(DispatchKind.ENGINE, "internal_core_budget")
	for key in _SERIAL:
		result[key] = _dispatch(DispatchKind.SERIAL, "stage.run")
	for key in _COMPOSITE_PROCESS:
		result[key] = _dispatch(
			DispatchKind.PROCESS, "run_multimaterial_stage",
			reason="parent coordinates iteration barriers; the run executor owns row and projection kernels",
		)
	return result


STAGE_DISPATCH = _build_stage_dispatch()


def dispatch_for_stage(stage) -> DispatchSpec:
	"""Resolve dispatch from an attached spec or the canonical Stage name."""

	attached = getattr(stage, "dispatch", None)
	if isinstance(attached, DispatchSpec):
		return attached
	return STAGE_DISPATCH.get(
		getattr(stage, "name", None),
		DispatchSpec(
			DispatchKind.EXEMPT,
			reason="unregistered test or extension Stage uses serial hermetic execution",
		),
	)
