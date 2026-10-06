"""Canonical data schemas (SPEC sections 6, 7, 11, 12, 15, 18, 34).

This package is the contract every other module depends on. It is deliberately
the first thing built: the invariants the project promises -- provenance always
present, missing never zero, time always unambiguous, nothing silently
destroyed -- are enforced here in code rather than restated as prose in each
pipeline.
"""

from urbansense.schemas.enums import (
    AggregationLevel,
    ExtractionKind,
    ImputationMethod,
    ProblemType,
    QuarantineReason,
    Severity,
    SourceType,
    ValidationStatus,
)
from urbansense.schemas.observation import Observation, new_observation_id
from urbansense.schemas.provenance import (
    AnyProvenance,
    Confidence,
    DocumentProvenance,
    Provenance,
)
from urbansense.schemas.quarantine import QuarantineRecord
from urbansense.schemas.temporal import (
    FUTURE_TOLERANCE,
    UTCDateTime,
    is_future,
    require_utc,
)

__all__ = [
    "FUTURE_TOLERANCE",
    "AggregationLevel",
    "AnyProvenance",
    "Confidence",
    "DocumentProvenance",
    "ExtractionKind",
    "ImputationMethod",
    "Observation",
    "ProblemType",
    "Provenance",
    "QuarantineReason",
    "QuarantineRecord",
    "Severity",
    "SourceType",
    "UTCDateTime",
    "ValidationStatus",
    "is_future",
    "new_observation_id",
    "require_utc",
]
