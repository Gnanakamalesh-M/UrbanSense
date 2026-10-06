"""The static vs scheduled-retrain vs adaptive experiment (SPEC sections 38, 39).

Phase 7 reported that the adaptive system beat a frozen model by 17%. That was
measured honestly but against the wrong baseline: "better than a model that
never retrains" is a weak claim, because the cheap alternative nobody had tried
is retraining on a calendar. This package puts that baseline in the room.

Three arms replay the one stream and are scored on the one set of rows:

``static``
    The frozen pre-drift champion.

``scheduled``
    Retrained every N days on the expanding window, deployed unconditionally.
    Same training function, same windows, same gap and same leakage probe as the
    adaptive challenger -- so any leakage advantage would apply to both arms
    equally, which is what makes the comparison fair rather than merely close.

``adaptive``
    Drift detection to candidate update to promotion rules.

Everything is **pre-registered** in ``configs/experiments/`` before a run and
copied verbatim into the report, and results are reported as mean and spread
across several generated seeds. One dataset is an anecdote: the planted drift is
the same in every stream, but the noise, the rainfall and the missing-data draws
are not, and a difference between arms smaller than that variation is not a
finding.

**Nothing here reads ``ground_truth.json``.** Error is measured against observed
values; where the drift was planted is not an input to any arm or any metric.
"""

from urbansense.experiments.config import (
    ADAPTIVE,
    ARM_ORDER,
    SCHEDULED,
    STATIC,
    ExperimentConfig,
    ExperimentConfigError,
    load_experiment_config,
)
from urbansense.experiments.metrics import (
    Aggregate,
    ArmMetrics,
    ClassificationScore,
    congestion_score,
    score_arm,
    within_one_sd,
)
from urbansense.experiments.report import (
    STANDING_LIMITATIONS,
    ArmSummary,
    ExperimentReport,
    Verdict,
    build_report,
    decide,
    summarize_arm,
    write_report,
)
from urbansense.experiments.runner import ExperimentRunner, SeedRun

__all__ = [
    "ADAPTIVE",
    "ARM_ORDER",
    "SCHEDULED",
    "STANDING_LIMITATIONS",
    "STATIC",
    "Aggregate",
    "ArmMetrics",
    "ArmSummary",
    "ClassificationScore",
    "ExperimentConfig",
    "ExperimentConfigError",
    "ExperimentReport",
    "ExperimentRunner",
    "SeedRun",
    "Verdict",
    "build_report",
    "congestion_score",
    "decide",
    "load_experiment_config",
    "score_arm",
    "summarize_arm",
    "within_one_sd",
    "write_report",
]
