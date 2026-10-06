"""Evaluation, error intelligence and experiments (SPEC sections 28, 33, 38).

Stores how a model performed, sliced so that failures are visible rather than
averaged away: a model can be fine overall and unreliable on Friday evenings in
the one zone that congests, which is when a traffic forecast actually matters.

**Temporal splits only.** :mod:`urbansense.evaluation.splits` has no seed
parameter anywhere. A random split of a time series is not a weaker evaluation
but a meaningless one -- with ``lag_1h`` as a feature, shuffling puts hour 14 in
train and hour 15 in test, so the model predicts what it has already seen and
scores beautifully. Making that path unavailable is cheaper than remembering not
to take it.

**Honest reporting.** The renderer prints every model against every slice, in a
fixed order, with no filtering and no "best model" highlighting. A report that
could show only the favourable configuration would eventually be asked to.
"""

from urbansense.evaluation.error_analysis import (
    Breakdown,
    ErrorReport,
    analyse_errors,
    by_friday_evening,
    by_rainfall,
    by_severity,
    by_time_of_day,
    by_weekday,
    by_zone,
)
from urbansense.evaluation.metrics import (
    EvaluationResult,
    SliceFn,
    SliceMetrics,
    coverage,
    evaluate,
    evaluate_slice,
    friday_evening_slices,
    local_window_slice,
    mae,
    rmse,
    zone_slice,
)
from urbansense.evaluation.outcomes import (
    DEFAULT_OUTCOME_DIR,
    Outcome,
    OutcomeStore,
    build_outcomes,
    rain_band,
    time_of_day,
)
from urbansense.evaluation.report import render_comparison, render_table
from urbansense.evaluation.splits import (
    SplitBounds,
    TemporalSplit,
    default_gap_hours,
    temporal_split,
)

__all__ = [
    "DEFAULT_OUTCOME_DIR",
    "Breakdown",
    "ErrorReport",
    "EvaluationResult",
    "Outcome",
    "OutcomeStore",
    "SliceFn",
    "SliceMetrics",
    "SplitBounds",
    "TemporalSplit",
    "analyse_errors",
    "build_outcomes",
    "by_friday_evening",
    "by_rainfall",
    "by_severity",
    "by_time_of_day",
    "by_weekday",
    "by_zone",
    "coverage",
    "default_gap_hours",
    "evaluate",
    "evaluate_slice",
    "friday_evening_slices",
    "local_window_slice",
    "mae",
    "rain_band",
    "render_comparison",
    "render_table",
    "rmse",
    "temporal_split",
    "time_of_day",
    "zone_slice",
]
