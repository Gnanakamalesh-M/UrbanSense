"""Anomaly detection (SPEC section 24).

    Do not automatically classify an anomaly as an error.

That is the rule the whole package is built around. Readings are flagged relative
to their context and given **ranked candidate explanations** with evidence, and
``Anomaly.is_error`` is ``None`` on every record with no code path that sets it.

The reason is in the spec's own list of possibilities: weather, events,
accidents, road closures, sensor failure, holidays. A value of 8,700 where 5,000
is normal is consistent with all of them, and a system that picked one would be
wrong in the expensive direction -- discarding a festival or an accident as a
glitch, destroying exactly the record a reviewer needed.

Detection reuses Phase 2's robust median/MAD core rather than reimplementing it,
so there is one definition of "unusual" in the codebase. Its conditioning on
zone, weekday and hour is also what keeps a recurring peak from reading as a
surprise: within its own group, the Friday peak is the norm.
"""

from urbansense.anomaly.detector import (
    NEIGHBOUR_WINDOW,
    Anomaly,
    AnomalyResult,
    detect_anomalies,
    pre_quarantine_population,
)
from urbansense.anomaly.explanations import Explanation, ExplanationKind, explain
from urbansense.anomaly.store import (
    DEFAULT_ANOMALY_DIR,
    anomaly_payload,
    write_anomaly_report,
)

__all__ = [
    "DEFAULT_ANOMALY_DIR",
    "NEIGHBOUR_WINDOW",
    "Anomaly",
    "AnomalyResult",
    "Explanation",
    "ExplanationKind",
    "anomaly_payload",
    "detect_anomalies",
    "explain",
    "pre_quarantine_population",
    "write_anomaly_report",
]
