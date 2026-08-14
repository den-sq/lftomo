from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lftomo.experiment import (
	AttemptProgress,
	CancellationToken,
	ProgressEventWriter,
	read_progress_events,
)
from lftomo.experiment.events import ExecutionContext, StageProgressMember
from lftomo.pipeline import (
	DispatchKind,
	ExecutionKind,
	ExecutionSpan,
	ExecutionSpanEvent,
	ExecutionSpanPhase,
	LogicalStageMember,
	ManagedAllocationChange,
	MemberTiming,
	ObservationAttribution,
	ResourceObservation,
	WorkerTimingAggregate,
)


def _context(tmp_path):
	persisted = []
	context = ExecutionContext(
		writer=ProgressEventWriter(tmp_path, "experiment", durable=True),
		progress_sink=persisted.append,
		cancellation=CancellationToken(),
		progress=AttemptProgress(
			experiment_id="experiment",
			fidelity_id="screen",
			candidate_key="candidate",
			trial_id="trial",
			attempt_id="attempt",
			state="started",
			updated_at=datetime.now(timezone.utc),
			window_id="window-one",
			window_position=1,
			window_total=2,
		),
	)
	return context, persisted


def _configs():
	return (
		SimpleNamespace(id="stripe-one", kind="stripe", algorithm="vo_dead"),
		SimpleNamespace(id="stripe-two", kind="stripe", algorithm="vo_sort"),
	)


def _span(*, fused=True):
	members = (
		LogicalStageMember(
			0, "stripe.vo_dead", "lftomo.VoDead", DispatchKind.PROCESS,
		),
	)
	if fused:
		members += (LogicalStageMember(
			1, "stripe.vo_sort", "lftomo.VoSort", DispatchKind.PROCESS,
		),)
	return ExecutionSpan(
		"run", "run:0", 0, 1,
		ExecutionKind.BOUNDED_SINOGRAM if fused else ExecutionKind.SEQUENTIAL,
		0, len(members), members,
	)


def test_fused_span_projects_wall_worker_and_resource_semantics_once(tmp_path):
	context, persisted = _context(tmp_path)
	span = _span()
	context.execution_span(
		ExecutionSpanEvent(ExecutionSpanPhase.STARTED, span), _configs(),
	)
	context.execution_span(ExecutionSpanEvent(
		ExecutionSpanPhase.COMPLETED,
		span,
		coordinator_wall_seconds=0.4,
		member_timings=tuple(MemberTiming(
			member=member,
			worker=WorkerTimingAggregate(3, 0.1, 0.2, 0.3, 0.6),
		) for member in span.members),
		managed_allocation_changes=(ManagedAllocationChange(
			"allocate", "vo_work", "shared-one", 4096,
		),),
		resource_observations=(ResourceObservation(
			"process_private_peak", 2048, "bytes", "execution_span_process_tree",
			ObservationAttribution.OBSERVED, (10, 11),
		),),
	), _configs())

	events = read_progress_events(tmp_path / "progress.jsonl")
	assert [event.kind for event in events] == ["segment.started", "segment.completed"]
	terminal = events[-1]
	assert terminal.schema_version == 2
	assert terminal.context_update == "replace"
	assert terminal.segment_coordinator_wall_seconds == 0.4
	assert all(member.coordinator_wall_seconds is None for member in terminal.segment_members)
	assert [item.invocation_count for item in terminal.segment_worker_timings] == [3, 3]
	assert all("worker" in item.basis for item in terminal.segment_worker_timings)
	assert len(terminal.segment_allocation_changes) == 1
	assert terminal.segment_allocation_changes[0].scope == "execution_span"
	assert terminal.segment_resource_observations[0].process_ids == (10, 11)
	assert terminal.stage_occurrence_id is None
	assert context.progress.segment_id is None
	assert context.progress.segment_members == ()
	assert persisted[-1].segment_id is None


def test_standalone_span_retains_single_member_wall_and_legacy_stage_identity(tmp_path):
	context, _persisted = _context(tmp_path)
	span = _span(fused=False)
	context.execution_span(
		ExecutionSpanEvent(ExecutionSpanPhase.STARTED, span), _configs(),
	)
	context.execution_span(ExecutionSpanEvent(
		ExecutionSpanPhase.COMPLETED,
		span,
		coordinator_wall_seconds=0.25,
		member_timings=(MemberTiming(span.members[0], coordinator_wall_seconds=0.25),),
	), _configs())

	terminal = read_progress_events(tmp_path / "progress.jsonl")[-1]
	assert terminal.stage_occurrence_id == "stripe-one"
	assert terminal.stage_position == 1
	assert terminal.stage_total == 2
	assert terminal.stage_elapsed_seconds == 0.25
	assert terminal.segment_members[0].coordinator_wall_seconds == 0.25


def test_window_started_replaces_stale_segment_and_cached_context(tmp_path):
	context, persisted = _context(tmp_path)
	span = _span()
	context.execution_span(
		ExecutionSpanEvent(ExecutionSpanPhase.STARTED, span), _configs(),
	)
	context.cache_event("hit", cached_members=(StageProgressMember(
		authored_index=0,
		position=1,
		total=2,
		occurrence_id="stripe-one",
		algorithm="vo_dead",
		stage_name="stripe.vo_dead",
		stage_type="lftomo.VoDead",
		dispatch_kind="process",
		cache_state="cached",
	),))
	context.window_started("window-two", position=2, total=2)

	boundary = read_progress_events(tmp_path / "progress.jsonl")[-1]
	assert boundary.kind == "window.started"
	assert boundary.window_id == "window-two"
	assert boundary.segment_id is None
	assert boundary.segment_members == ()
	assert boundary.stage_occurrence_id is None
	assert boundary.cached_stage_count == 0
	assert boundary.cached_stage_members == ()
	assert persisted[-1].segment_id is None


@pytest.mark.parametrize("phase", (
	ExecutionSpanPhase.FAILED,
	ExecutionSpanPhase.INTERRUPTED,
))
def test_failed_and_interrupted_spans_preserve_current_context(tmp_path, phase):
	context, persisted = _context(tmp_path)
	span = _span()
	context.execution_span(
		ExecutionSpanEvent(ExecutionSpanPhase.STARTED, span), _configs(),
	)
	context.execution_span(ExecutionSpanEvent(
		phase,
		span,
		coordinator_wall_seconds=0.1,
		error_type="builtins.InterruptedError" if phase is ExecutionSpanPhase.INTERRUPTED else "builtins.ValueError",
		error_message="stopped",
	), _configs())

	terminal = read_progress_events(tmp_path / "progress.jsonl")[-1]
	assert terminal.kind == f"segment.{phase.value}"
	assert terminal.segment_id == "run:0"
	assert terminal.failure_message == "stopped"
	assert context.progress.failure.segment_id == "run:0"
	assert persisted[-1].segment_id == "run:0"
