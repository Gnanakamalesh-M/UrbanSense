"""Adaptive learning and model versioning (SPEC sections 30-32).

    New Data -> Validation -> Drift Detection -> Candidate Update
    -> Train Candidate -> Evaluate -> Compare with Champion

A challenger is promoted only when it passes predefined criteria; otherwise the
champion is kept and the candidate is recorded as REJECTED. Every model version
stores its training period, features, algorithm, hyperparameters, metrics,
dataset version, timestamp and status, plus the drift episode and candidate that
prompted it.

Three invariants hold across the package. **Nothing retrains without a recorded
decision** -- ``candidate_from_event`` is the only path from drift to a fitted
model. **No challenger can see its own evaluation horizon**, enforced by a
filter on both sample and target time, a gap, and the Phase 4 leakage probe run
on each challenger's own feature set. And **exactly one champion serves at a
time**, checked after every registry mutation rather than only in the tests.
"""

from urbansense.adaptation.candidates import (
    DEFAULT_CANDIDATE_DIR,
    CandidateStatus,
    CandidateStore,
    CandidateUpdate,
    ChallengerEvaluation,
    RuleOutcome,
    candidate_from_event,
)
from urbansense.adaptation.challenger import (
    MIN_TRAINING_ROWS,
    ChallengerError,
    ChallengerFit,
    available_at,
    evaluation_window,
    train_challenger,
    window_rows,
)
from urbansense.adaptation.config import (
    DEFAULT_ADAPTATION_PATH,
    AdaptationConfig,
    AdaptationConfigError,
    CandidateRules,
    MonitoringRules,
    PromotionRules,
    TrainingWindow,
    load_adaptation_config,
)
from urbansense.adaptation.lineage import (
    LineageEntry,
    LineageError,
    assert_single_champion,
    lineage,
    promote,
    register_challenger,
    reject,
    render_lineage,
    replay_run_root,
)
from urbansense.adaptation.promotion import (
    decide,
    evaluate_challenger,
    promotion_summary,
    version_name,
)
from urbansense.adaptation.replay import (
    AdaptiveReplay,
    ArmResult,
    ReplayResult,
    RetrainCost,
    scenario_bounds,
)

__all__ = [
    "DEFAULT_ADAPTATION_PATH",
    "DEFAULT_CANDIDATE_DIR",
    "MIN_TRAINING_ROWS",
    "AdaptationConfig",
    "AdaptationConfigError",
    "AdaptiveReplay",
    "ArmResult",
    "CandidateRules",
    "CandidateStatus",
    "CandidateStore",
    "CandidateUpdate",
    "ChallengerError",
    "ChallengerEvaluation",
    "ChallengerFit",
    "LineageEntry",
    "LineageError",
    "MonitoringRules",
    "PromotionRules",
    "ReplayResult",
    "RetrainCost",
    "RuleOutcome",
    "TrainingWindow",
    "assert_single_champion",
    "available_at",
    "candidate_from_event",
    "decide",
    "evaluate_challenger",
    "evaluation_window",
    "lineage",
    "load_adaptation_config",
    "promote",
    "promotion_summary",
    "register_challenger",
    "reject",
    "render_lineage",
    "replay_run_root",
    "scenario_bounds",
    "train_challenger",
    "version_name",
    "window_rows",
]
