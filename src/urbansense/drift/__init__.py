"""Drift detection (SPEC section 29).

Monitors feature drift, prediction drift and performance drift over a stream in
time order, and raises a candidate-update signal rather than retraining
automatically.

Two things shape the package. **Gates are calibrated on the reference period and
then frozen**, so a detection can never be the consequence of a gate that moved.
And a detector that cannot reach a verdict says so -- ``INSUFFICIENT_REFERENCE``
is an outcome, not silence, because reporting a thin reference as "no drift"
turns absence of evidence into evidence of absence.

Measured on the demo history with the planted 2026-03-01 level shift: zero false
alarms across the stable stretch, and detection four days after the change. The
error-based detector is the one that found it first; distributional feature drift
is the least reliable of the three here, and the monitor's docstring records why.
"""

from urbansense.drift.detectors import (
    DEFAULT_KS_EFFECT,
    DEFAULT_MIN_REFERENCE_WINDOWS,
    DEFAULT_PSI_BINS,
    DEFAULT_SIGMA_MULTIPLIER,
    MIN_SAMPLE,
    PSI_MAJOR,
    PSI_MINOR,
    DriftKind,
    DriftSignal,
    DriftStatus,
    KSResult,
    feature_drift,
    ks_test,
    performance_drift,
    prediction_drift,
    psi,
    psi_severity,
    sigma_gate,
)
from urbansense.drift.events import (
    DEFAULT_DRIFT_DIR,
    DriftEpisode,
    DriftEvent,
    DriftStore,
    EpisodeTracker,
    detection_delay,
    false_alarm_count,
)
from urbansense.drift.monitor import (
    DEFAULT_STEP,
    DEFAULT_WATCHED_FEATURES,
    DEFAULT_WINDOW,
    MIN_WINDOW_ROWS,
    UNMONITORABLE_METRICS,
    DriftGates,
    DriftMonitor,
    MonitorConfig,
    summarize,
)

__all__ = [
    "DEFAULT_DRIFT_DIR",
    "DEFAULT_KS_EFFECT",
    "DEFAULT_MIN_REFERENCE_WINDOWS",
    "DEFAULT_PSI_BINS",
    "DEFAULT_SIGMA_MULTIPLIER",
    "DEFAULT_STEP",
    "DEFAULT_WATCHED_FEATURES",
    "DEFAULT_WINDOW",
    "MIN_SAMPLE",
    "MIN_WINDOW_ROWS",
    "PSI_MAJOR",
    "PSI_MINOR",
    "UNMONITORABLE_METRICS",
    "DriftEpisode",
    "DriftEvent",
    "DriftGates",
    "DriftKind",
    "DriftMonitor",
    "DriftSignal",
    "DriftStatus",
    "DriftStore",
    "EpisodeTracker",
    "KSResult",
    "MonitorConfig",
    "detection_delay",
    "false_alarm_count",
    "feature_drift",
    "ks_test",
    "performance_drift",
    "prediction_drift",
    "psi",
    "psi_severity",
    "sigma_gate",
    "summarize",
]
