"""Reconstruction evaluation, aggregation, and ranking."""

from .aggregation import aggregate_observations
from .experiment import validate_experiment_metric_references
from .metrics import METRIC_REGISTRY, PROTOCOL_VERSION, default_metric_registry
from .models import (
	AggregatedMetric,
	AggregationConfig,
	EvaluationArtifact,
	EvaluationWindow,
	ImageROI,
	MetricDefinition,
	MetricObservation,
	MetricRequest,
	PreviewContract,
	ReferenceMetricRequest,
	RingMetricRequest,
	RoiNoiseCnrMetricRequest,
	ValidityMetricRequest,
)
from .ranking import (
	AggregateScoreConfig,
	AggregateScoreTerm,
	CandidateMetrics,
	HardConstraint,
	ParetoObjective,
	ParetoOutcome,
	aggregate_score,
	pareto_rank,
)
from .registry import KernelValue, MetricRegistration, MetricRegistry
from .stage import EvaluationStage

__all__ = [
	"AggregatedMetric",
	"AggregationConfig",
	"EvaluationArtifact",
	"EvaluationStage",
	"EvaluationWindow",
	"ImageROI",
	"MetricDefinition",
	"MetricObservation",
	"MetricRequest",
	"KernelValue",
	"MetricRegistration",
	"MetricRegistry",
	"PreviewContract",
	"ReferenceMetricRequest",
	"RingMetricRequest",
	"RoiNoiseCnrMetricRequest",
	"ValidityMetricRequest",
	"METRIC_REGISTRY",
	"PROTOCOL_VERSION",
	"default_metric_registry",
	"aggregate_observations",
	"validate_experiment_metric_references",
	"AggregateScoreConfig",
	"AggregateScoreTerm",
	"CandidateMetrics",
	"HardConstraint",
	"ParetoObjective",
	"ParetoOutcome",
	"aggregate_score",
	"pareto_rank",
]
