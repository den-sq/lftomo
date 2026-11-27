from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile as tf
import yaml

from lftomo.config import Acquisition, Detector, Scan, StageConfig, TIFFStackProjections
from lftomo.pipeline import (
	PadRequirement,
	Stage,
	StageContract,
	WindowConfig,
	WindowPlanError,
	resolve_streaming_plan,
	resolve_streaming_segments,
)
from lftomo.pipeline.stages import (
	BetaScaleStage,
	BilateralFilterStage,
	BronnikovAidedCorrectionFilterStage,
	BronnikovAidedCorrectionStage,
	HelicalIndexRemapStage,
	MovementCorrectStage,
	MultimaterialStage,
	NegativeLogStage,
	PaganinFilterStage,
	PaganinPhaseAbsorptionStage,
	PaganinStage,
	WeightedNormalizeStage,
	create_stage,
)
from lftomo.prep.phase import _phase_pad, _wavelength
from lftomo.runner import run_reconstruction
from lftomo.util.converter import CONVERTER
from lftomo.util.shapes import ProjOrder, Theta, X, Y
from lftomo.util.window import DataRef, TomoDataSet, TomoWindow
from tests.fixtures.tier1 import projection_stack


def _scan(rows=12, columns=8, theta=4):
	return Scan(
		energy_keV=20, pixel_size_um=1, ssd_cm=10,
		n_projections=theta, degrees_per_fov=180,
		detector=Detector(pixel_size_um=1, rows=rows, columns=columns),
		acquisition=Acquisition(n_projections=theta, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)


class _FullYStage(Stage):
	name = "test.full_y"
	contract = StageContract(requires_full_axes=("Y",))

	def run(self, state):
		return state


class _AsymmetricStage(Stage):
	name = "test.asymmetric"
	contract = StageContract(
		required_pad=PadRequirement(fixed=Y(1), fixed_after=Y(3)),
	)

	def run(self, state):
		return state


def test_stage_and_operator_padding_compose_into_the_canonical_dataset():
	stages = [
		BilateralFilterStage(diameter=5),
		BilateralFilterStage(diameter=5),
		MovementCorrectStage(y=3),
	]
	dataset = resolve_streaming_plan(
		stages, _scan(), WindowConfig(rows=3, extra_pad=Y(5)),
	)
	assert dataset.step == Y(3)
	assert dataset.pad_before == ProjOrder(Theta=0, Y=7, X=4)
	assert dataset.pad_after == ProjOrder(Theta=0, Y=7, X=4)

	user_dataset = resolve_streaming_plan(
		stages, _scan(), WindowConfig(rows=3, extra_pad=Y(10)),
	)
	assert user_dataset.pad_before.Y == 10
	assert user_dataset.pad_after.Y == 10


def test_asymmetric_stage_and_operator_requirements_compose_by_side():
	dataset = resolve_streaming_plan(
		[_AsymmetricStage()], _scan(),
		WindowConfig(rows=4, extra_pad=Y(2), extra_pad_after=Y(1)),
	)

	assert dataset.pad_before == ProjOrder(Theta=0, Y=2, X=0)
	assert dataset.pad_after == ProjOrder(Theta=0, Y=3, X=0)
	windows = tuple(dataset)
	assert windows[0].pad_before == ProjOrder(Theta=0, Y=0, X=0)
	assert windows[0].pad_after == ProjOrder(Theta=0, Y=3, X=0)
	assert windows[1].pad_before == ProjOrder(Theta=0, Y=2, X=0)
	assert windows[1].pad_after == ProjOrder(Theta=0, Y=3, X=0)
	record = CONVERTER.unstructure(dataset)
	assert CONVERTER.unstructure(CONVERTER.structure(record, DataRef)) == record


def test_phase_pad_uses_the_canonical_phase_rule_and_warns_about_extent(caplog):
	scan = _scan(rows=32)
	dataset = resolve_streaming_plan(
		[PaganinFilterStage(), NegativeLogStage()], scan, WindowConfig(rows=8),
	)
	assert dataset.pad_before.Y == _phase_pad(8, 1e-4, 10, 20)
	assert dataset.pad_after.Y == _phase_pad(8, 1e-4, 10, 20)
	assert "phase.paganin_filter" in caplog.text
	assert "whole-volume equivalence" in caplog.text


@pytest.mark.parametrize("rows", [154, 155, 256, 471, 472, 2160])
def test_phase_pad_covers_the_fresnel_width_for_every_window_height(rows):
	pixel_um, ssd_cm, energy = 0.637, 8.1795, 20
	scan = Scan(
		energy_keV=energy, pixel_size_um=pixel_um, ssd_cm=ssd_cm,
		n_projections=4, degrees_per_fov=180,
		detector=Detector(pixel_size_um=pixel_um, rows=2160, columns=8),
		acquisition=Acquisition(n_projections=4, rotation_end_deg=180),
		projections=TIFFStackProjections(format="tiff_stack", path="projections"),
	)
	fresnel = np.ceil(np.pi * _wavelength(energy) * ssd_cm / (pixel_um * 1e-4) ** 2)

	dataset = resolve_streaming_plan(
		[PaganinFilterStage(), NegativeLogStage()], scan, WindowConfig(rows=rows),
	)

	assert dataset.pad_before.Y >= fresnel // 2
	assert dataset.pad_after.Y == dataset.pad_before.Y


@pytest.mark.parametrize("stage", [
	PaganinStage(),
	PaganinPhaseAbsorptionStage(),
	BronnikovAidedCorrectionFilterStage(alpha=2.0),
	BronnikovAidedCorrectionStage(alpha=2.0),
	PaganinStage(enabled=False),
	BronnikovAidedCorrectionStage(alpha=2.0, enabled=False),
])
def test_phase_extent_warning_is_preserved(stage, caplog):
	resolve_streaming_plan([stage], _scan(), WindowConfig(rows=4))
	assert "Extent-sensitive" in caplog.text
	assert stage.name in caplog.text


def test_extent_warning_uses_partitioned_axis_and_instance_contract(caplog):
	stage = PaganinFilterStage()
	stage.name = "filter.extent_dependent"
	resolve_streaming_plan([stage], _scan(), WindowConfig())
	assert "Extent-sensitive" not in caplog.text
	resolve_streaming_plan(
		[stage, stage, NegativeLogStage(), BetaScaleStage()], _scan(), WindowConfig(rows=4),
	)
	assert caplog.text.count("filter.extent_dependent") == 1
	assert "phase.negative_log" not in caplog.text
	assert "phase.beta_scale" not in caplog.text

	caplog.clear()
	stage.name = "phase.paganin_filter"
	stage.contract = replace(stage.contract, extent_sensitive_axes=("X",))
	resolve_streaming_plan([stage], _scan(), WindowConfig(rows=4))
	assert "Extent-sensitive" not in caplog.text

	stage.contract = replace(stage.contract, extent_sensitive_axes=())
	resolve_streaming_plan([stage], _scan(), WindowConfig(rows=4))
	assert "Extent-sensitive" not in caplog.text


def test_full_axis_contract_rejects_silent_window_degradation():
	with pytest.raises(WindowPlanError, match="test.full_y.*full Y axis"):
		resolve_streaming_plan([_FullYStage()], _scan(), WindowConfig(rows=4))
	with pytest.raises(WindowPlanError, match="helical_correct.index_remap"):
		resolve_streaming_plan([
			HelicalIndexRemapStage(pitch_px_per_projection=0.1),
		], _scan(), WindowConfig(rows=4))


def test_canonical_dataset_reiterates_without_a_window_cache():
	dataset = resolve_streaming_plan(
		[BilateralFilterStage(diameter=5)], _scan(rows=10), WindowConfig(rows=3),
	)
	first_pass = tuple(dataset)
	second_pass = tuple(dataset)
	assert len(dataset.frange) == 4
	assert dataset.interval("Y") == Y.interval(0, 10)
	assert all(isinstance(window, TomoWindow) for window in first_pass)
	assert [window.index for window in first_pass] == [
		window.index for window in second_pass
	]


def test_streaming_windows_read_physical_neighbors_outside_selection(tmp_path):
	projection_dir = tmp_path / "projections"
	projection_dir.mkdir()
	for index in range(4):
		tf.imwrite(projection_dir / f"projection_{index}.tif", np.zeros((12, 8), dtype=np.uint16))
	scan = _scan(rows=12)
	scan.projections.path = projection_dir
	read = StageConfig(
		kind="read", algorithm="tiff_stack",
		params={"y_start": 2, "y_stop": 10},
	)
	stages = [create_stage(read), BilateralFilterStage(diameter=5)]
	dataset = resolve_streaming_plan(stages, scan, WindowConfig(rows=4))
	windows = tuple(dataset)

	assert dataset.physical_shape == ProjOrder(Theta=4, Y=12, X=8)
	assert dataset.interval("Y") == Y.interval(2, 10)
	assert dataset.step == Y(4)
	assert [window.index[1] for window in windows] == [
		slice(2, 6, 1), slice(6, 10, 1),
	]
	assert [window.with_pad(["Y"]).index[1] for window in windows] == [
		slice(0, 8, 1), slice(4, 12, 1),
	]


class _DirectionalStage(Stage):
	name = "test.directional"

	def run(self, state):
		return state


def _selected_source(tmp_path, *, rows=12, theta=10, **selection):
	for index in range(theta):
		tf.imwrite(tmp_path / f"projection_{index:03d}.tif", np.zeros((rows, 8), dtype=np.uint16))
	scan = _scan(rows=rows, theta=theta)
	scan.projections.path = tmp_path
	read = create_stage(StageConfig(kind="read", algorithm="tiff_stack", params=selection))
	return scan, read


@pytest.mark.parametrize("stage", [
	PaganinStage(slope=500),
	PaganinFilterStage(slope=500),
	BronnikovAidedCorrectionStage(alpha=2),
	BronnikovAidedCorrectionFilterStage(alpha=2),
])
@pytest.mark.parametrize("config", [WindowConfig(), WindowConfig(rows=3, theta=2)])
def test_directional_phase_context_clips_at_detector_edges(stage, config, caplog):
	segments = resolve_streaming_segments([stage], _scan(), config)
	assert len(segments) == 1
	assert segments[0].axis == ("Y" if config.rows is None else "Theta")
	assert all(window.shape.Y == 12 for window in segments[0].dataset)
	assert all(window.pad_before == window.pad_after == ProjOrder(0, 0, 0)
		for window in segments[0].dataset)
	assert "Extent-sensitive" not in caplog.text


@pytest.mark.parametrize("first", [
	HelicalIndexRemapStage(pitch_px_per_projection=0.1),
	create_stage(StageConfig(kind="geometry", algorithm="affine_skew_movement", params={
		"transition_to": "sinogram", "pad_x": 2,
	})),
])
def test_whole_plan_preserves_geometry_lifecycle_without_direction_boundaries(first):
	segments = resolve_streaming_segments([first, PaganinStage(slope=500)], _scan())
	assert [(segment.start, segment.stop) for segment in segments] == [(0, 2)]
	assert len(segments[0].dataset.frange) == 1
	assert segments[0].dataset.full_window().shape == ProjOrder(4, 12, 8)
	assert segments[0].dataset.full_window().pad_before.Y == 0
	assert segments[0].dataset.full_window().pad_after.Y == 0


def test_directional_segments_preserve_authored_order_and_stage_defaults(tmp_path):
	scan, read = _selected_source(tmp_path)
	y_stage = _DirectionalStage()
	y_stage.contract = StageContract(window_axes=("Y",))
	segments = resolve_streaming_segments(
		[read, WeightedNormalizeStage(), PaganinStage(slope=500), y_stage],
		scan, WindowConfig(rows=4),
	)
	assert [(segment.start, segment.stop, segment.axis) for segment in segments] == [
		(0, 2, "Y"), (2, 3, "Theta"), (3, 4, "Y"),
	]
	assert segments[1].dataset.step == Theta(4)
	assert segments[0].dataset.step == segments[2].dataset.step == Y(4)


@pytest.mark.parametrize("config, expected", [
	(WindowConfig(rows=4), [4, 4, 2]),
	(WindowConfig(rows=4, theta=3), [3, 3, 3, 1]),
	(WindowConfig(rows=20), [10]),
	(WindowConfig(columns=3), [6, 4]),
])
def test_theta_default_matches_row_batch_and_explicit_theta_overrides(config, expected):
	dataset = resolve_streaming_segments(
		[BronnikovAidedCorrectionStage(alpha=2)], _scan(rows=6, theta=10), config,
	)[0].dataset
	assert [window.frange.length.Theta for window in dataset] == expected


def test_theta_windows_preserve_selected_projection_stride(tmp_path):
	scan, read = _selected_source(tmp_path, theta_start=1, theta_stop=10, theta_stride=3)
	dataset = resolve_streaming_segments(
		[read], scan, WindowConfig(axis="Theta", theta=2),
	)[0].dataset
	assert dataset.step == Theta(6)
	assert [window.axis_slice("Theta") for window in dataset] == [
		slice(1, 7, 3), slice(7, 10, 3),
	]
	assert [window.frange.length.Theta for window in dataset] == [2, 1]
	assert dataset.sample_step.Theta == 3


@pytest.mark.parametrize("axis", ["Y", "Theta"])
@pytest.mark.parametrize("stride", [1, 3])
def test_multimaterial_holds_all_selected_angles_for_its_internal_reconstruction(
	tmp_path, axis, stride,
):
	scan, read = _selected_source(
		tmp_path, theta_start=1, theta_stop=10, theta_stride=stride,
	)
	stage = MultimaterialStage(mu_table=(1.0, 2.0), delta_table=(1e-10, 3e-10))
	segments = resolve_streaming_segments(
		[read, WeightedNormalizeStage(), stage], scan,
		WindowConfig(axis=axis, rows=3, theta=2),
	)
	segment = segments[-1]
	assert segment.start <= 2 < segment.stop
	assert segment.axis == "Y"
	assert all(window.axis_slice("Theta") == slice(1, 10, stride) for window in segment.dataset)
	assert all(window.frange.length.Theta == len(range(1, 10, stride)) for window in segment.dataset)
	with pytest.raises(WindowPlanError, match="requires the full Theta axis"):
		resolve_streaming_plan([stage], scan, WindowConfig(axis="Theta", theta=2))


def test_downstream_padding_expands_prior_materialized_coverage(tmp_path):
	scan, read = _selected_source(tmp_path, y_start=3, y_stop=9)
	segments = resolve_streaming_segments(
		[read, PaganinStage(slope=500), _AsymmetricStage()], scan,
		WindowConfig(rows=2, theta=3),
	)
	assert [segment.axis for segment in segments] == ["Y", "Theta", "Y"]
	assert segments[-1].dataset.interval("Y") == Y.interval(3, 9)
	assert segments[-1].dataset.pad_before.Y == 1
	assert segments[-1].dataset.pad_after.Y == 3
	assert segments[1].dataset.interval("Y") == Y.interval(2, 12)
	# The phase sweep also needs real detector context for this slope; its
	# input sweep must publish that context beyond the downstream filter halo.
	assert segments[0].dataset.interval("Y") == Y.interval(0, 12)
	assert segments[1].dataset.full_window().pad_before.Y == 2
	assert segments[1].dataset.full_window().pad_after.Y == 0


def test_explicit_direction_and_x_windows_use_canonical_geometry():
	stage = _DirectionalStage()
	stage.contract = StageContract(window_axes=("X", "Y"), required_window_axis="X")
	dataset = resolve_streaming_segments([stage], _scan(), WindowConfig(columns=3))[0].dataset
	assert dataset.step == X(3)
	assert [window.interval("X") for window in dataset] == [
		X.interval(0, 3), X.interval(3, 6), X.interval(6, 8),
	]
	pointwise = resolve_streaming_segments(
		[NegativeLogStage(), BetaScaleStage()], _scan(), WindowConfig(axis="X", columns=3),
	)
	assert len(pointwise) == 1
	assert pointwise[0].axis == "X"


@pytest.mark.parametrize("values", [
	{"required_window_axis": "Z", "window_axes": ("Z",)},
	{"preferred_window_axis": "Theta", "window_axes": ("Y",)},
	{"required_window_axis": "Y", "window_axes": ("Y",), "requires_full_axes": ("Y",)},
	{"required_window_axis": "Y", "preferred_window_axis": "X", "window_axes": ("Y", "X")},
])
def test_direction_contract_rejects_inconsistent_declarations(values):
	with pytest.raises(ValueError):
		StageContract(**values)


def test_single_direction_plan_honors_required_axis_and_capabilities():
	stage = _DirectionalStage()
	stage.contract = StageContract(window_axes=("Theta",), required_window_axis="Theta")
	with pytest.raises(WindowPlanError, match="requires window axis Theta"):
		resolve_streaming_plan([stage], _scan(), WindowConfig(rows=3))
	with pytest.raises(WindowPlanError, match="does not support X windows"):
		resolve_streaming_plan([WeightedNormalizeStage()], _scan(), WindowConfig(axis="X", columns=3))


def test_direction_planner_rejects_unsupported_artifact_boundary():
	producer, consumer = _DirectionalStage(), _DirectionalStage()
	producer.contract = StageContract(window_axes=("Y",), produces_artifacts=("per_window",))
	consumer.contract = StageContract(
		window_axes=("Theta",), required_window_axis="Theta", consumes_artifacts=("per_window",),
	)
	with pytest.raises(WindowPlanError, match="per_window"):
		resolve_streaming_segments([producer, consumer], _scan(), WindowConfig(rows=3))


def test_direction_planner_rejects_changed_geometry_before_boundary():
	class ExpandedStage(_DirectionalStage):
		def output_window(self, source):
			return source.with_output_padding(X(1))

	with pytest.raises(WindowPlanError, match="changes output geometry"):
		resolve_streaming_segments(
			[ExpandedStage(), PaganinStage(slope=500)], _scan(), WindowConfig(rows=3),
		)


def test_disabled_slope_phase_keeps_default_direction():
	segment, = resolve_streaming_segments(
		[PaganinStage(slope=500, enabled=False)], _scan(), WindowConfig(rows=3),
	)
	assert segment.axis == "Y"


def _write_yaml(path: Path, value):
	path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_windowed_runner_matches_whole_non_fft_and_replays_center_prefix(
	tmp_path, monkeypatch,
):
	import lftomo.pipeline.stages.read as read_module
	import lftomo.runner as runner_module

	bound_datasets = []
	read_stage_ids = []
	original_bind = read_module._bind_source_state
	original_read = read_module.HDF5DxchangeReadStage.run
	original_planned_pipeline = runner_module._planned_pipeline
	planned_scans = []

	def record_bind(state, working, read_window, stage):
		result = original_bind(state, working, read_window, stage)
		bound_datasets.append((state.dataset, result.dataset))
		return result

	def record_read(stage, state):
		read_stage_ids.append(id(stage))
		return original_read(stage, state)

	def record_planned_pipeline(stages, scan, *args, **kwargs):
		planned_scans.append(scan)
		return original_planned_pipeline(stages, scan, *args, **kwargs)

	monkeypatch.setattr(read_module, "_bind_source_state", record_bind)
	monkeypatch.setattr(read_module.HDF5DxchangeReadStage, "run", record_read)
	monkeypatch.setattr(runner_module, "_planned_pipeline", record_planned_pipeline)
	data_path = tmp_path / "scan.h5"
	rows, theta, columns = 6, 3, 6
	projections = 20 + 20 * projection_stack(size=rows, n_projections=theta)
	with h5py.File(data_path, "w") as handle:
		handle.create_dataset("exchange/data", data=projections)
		handle.create_dataset("exchange/data_white", data=np.full((2, rows, columns), 100))
		handle.create_dataset("exchange/data_dark", data=np.zeros((2, rows, columns)))
	scan_path = tmp_path / "scan.yaml"
	_write_yaml(scan_path, {
		"energy_keV": 20, "pixel_size_um": 1, "ssd_cm": 10,
		"n_projections": theta, "degrees_per_fov": 180,
		"detector": {"pixel_size_um": 1, "rows": rows, "columns": columns},
		"acquisition": {"n_projections": theta, "rotation_end_deg": 180},
		"projections": {
			"format": "hdf5_dxchange", "path": str(data_path),
			"data_path": "exchange/data", "white_path": "exchange/data_white",
			"dark_path": "exchange/data_dark",
		},
	})
	pipeline_path = tmp_path / "pipeline.yaml"
	_write_yaml(pipeline_path, {"pipeline": [
		{"kind": "read", "algorithm": "hdf5_dxchange", "params": {
			"order": "projection", "calibration_policy": "required",
		}},
		{"kind": "normalize", "algorithm": "weighted"},
		{"kind": "filter", "algorithm": "bilateral", "params": {"diameter": 3}},
		{"kind": "phase", "algorithm": "negative_log"},
		{"kind": "center", "algorithm": "pc"},
		{"kind": "recon", "algorithm": "gridrec", "params": {"circle_mask": False}},
		{"kind": "write", "algorithm": "tiff_slices", "params": {
			"naming_template": "slice_{slice:03d}.tif",
		}},
	]})

	import tomopy

	monkeypatch.setattr(tomopy, "find_center_pc", lambda *args, **kwargs: 2.0)
	reconstruction_inputs = []

	def reconstruct(data, *args, **kwargs):
		reconstruction_inputs.append(tuple(data.shape))
		kwargs["init_recon"][...] = np.repeat(
			np.asarray(data).mean(axis=0)[:, None, :], data.shape[2], axis=1,
		)
		return kwargs["init_recon"]

	monkeypatch.setattr(tomopy, "recon", reconstruct)
	whole_output = tmp_path / "whole"
	window_output = tmp_path / "windowed"
	whole = run_reconstruction(pipeline_path, scan_path, whole_output)[0]
	planned_scans.clear()
	bound_datasets.clear()
	read_stage_ids.clear()
	windowed = run_reconstruction(
		pipeline_path, scan_path, window_output,
		window_config=WindowConfig(rows=2),
	)[0]
	assert len(bound_datasets) == 6
	assert len(set(read_stage_ids)) == 1
	assert len({id(before) for before, _after in bound_datasets}) == 1
	assert all(before is after for before, after in bound_datasets)
	assert len({id(scan) for scan in planned_scans}) == 1
	assert reconstruction_inputs == [(theta, rows, columns)] + [
		(theta, 2, columns), (theta, 2, columns), (theta, 2, columns),
	]

	for row in range(rows):
		np.testing.assert_allclose(
			tf.imread(window_output / f"slice_{row:03d}.tif"),
			tf.imread(whole_output / f"slice_{row:03d}.tif"),
			rtol=1e-6, atol=1e-6,
		)
	assert len(windowed.artifacts["window_results"]) == 3
	assert all(
		item["center_artifacts"]["center"] == 2.0
		for item in windowed.artifacts["window_results"]
	)
	restored_dataset = CONVERTER.structure(windowed.artifacts["streaming_dataset"], DataRef)
	assert isinstance(restored_dataset, TomoDataSet)
	assert restored_dataset.interval("Y") == Y.interval(0, rows)
	assert restored_dataset.step == Y(2)
	assert windowed.artifacts["measured_shared_memory_bytes"] > 0
	manifest = yaml.safe_load((window_output / "recon_run.yaml").read_text())
	assert CONVERTER.structure(manifest["streaming_dataset"], DataRef).step == Y(2)
	assert "streaming_dataset" in windowed.resolved_recon["metadata"]
	assert "source_window" not in windowed.resolved_recon["metadata"]
