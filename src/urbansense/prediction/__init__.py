"""Prediction engine (SPEC sections 25, 26, 31, 32).

Phase 4 predicts traffic volume at a configurable horizon. SPEC section 25's
fuller output -- probability, severity, expected window, confidence -- arrives in
a later phase; claiming it now with a point forecast behind it would oversell.

Baselines come first and are not a formality. A gradient model that cannot beat
"the traffic next hour will be what it is now" has learned nothing worth
deploying, and the comparison is printed next to it either way.

Every forecaster reads the same leakage-checked feature table, including the
baselines. A baseline reaching into the data by its own path could be
accidentally leaky, and would then set an impossible bar that the honest model
appears to fail.
"""

from urbansense.prediction.baselines import (
    Forecaster,
    NaiveLastValue,
    SameHourLastWeek,
)
from urbansense.prediction.calibration import (
    CalibrationReport,
    ProbabilityCalibrator,
    ReliabilityBin,
    brier_score,
    reliability_table,
)
from urbansense.prediction.congestion import (
    CongestionThresholds,
    congested_rate,
    fit_thresholds,
)
from urbansense.prediction.gradient import (
    DEFAULT_HYPERPARAMETERS,
    LOWER_QUANTILE,
    UPPER_QUANTILE,
    GradientForecaster,
    PredictionInterval,
)
from urbansense.prediction.output import (
    CongestionWindow,
    Prediction,
    build_predictions,
    congestion_windows,
    interval_confidence,
)
from urbansense.prediction.registry import (
    ARTIFACT_SUFFIX,
    REGISTRY_SUFFIX,
    ModelRegistry,
    ModelVersion,
)
from urbansense.prediction.sources import SourceBreakdown, SourceLedger, SourceShare

__all__ = [
    "ARTIFACT_SUFFIX",
    "DEFAULT_HYPERPARAMETERS",
    "LOWER_QUANTILE",
    "REGISTRY_SUFFIX",
    "UPPER_QUANTILE",
    "CalibrationReport",
    "CongestionThresholds",
    "CongestionWindow",
    "Forecaster",
    "GradientForecaster",
    "ModelRegistry",
    "ModelVersion",
    "NaiveLastValue",
    "Prediction",
    "PredictionInterval",
    "ProbabilityCalibrator",
    "ReliabilityBin",
    "SameHourLastWeek",
    "SourceBreakdown",
    "SourceLedger",
    "SourceShare",
    "brier_score",
    "build_predictions",
    "congested_rate",
    "congestion_windows",
    "fit_thresholds",
    "interval_confidence",
    "reliability_table",
]
