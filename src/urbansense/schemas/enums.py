"""Controlled vocabularies for the canonical schema (SPEC sections 6, 7, 18, 34).

These are closed enumerations on purpose: an unknown source type or validation
status must fail loudly at the boundary rather than flow into the unified data
layer as a free-text string nobody checks.
"""

from __future__ import annotations

from enum import StrEnum


class SourceType(StrEnum):
    """Which independent stream an observation arrived on (SPEC section 4).

    The three streams stay separately traceable before being combined, so this
    field is never rewritten -- a real-time observation that later becomes
    finalized history keeps ``REAL_TIME`` and records the transition in its
    provenance lineage instead.
    """

    HISTORICAL = "historical"
    REAL_TIME = "real_time"
    DOCUMENT = "document"


class ProblemType(StrEnum):
    """Urban problem domain (SPEC section 3).

    Only ``TRAFFIC`` is implemented in v1. The others are declared so that
    adding them later is a configuration change, not a schema migration.
    """

    TRAFFIC = "traffic"
    FLOOD = "flood"
    WASTE = "waste"


class ValidationStatus(StrEnum):
    """Where a record stands in the integrity workflow (SPEC sections 8-18).

    ``QUARANTINED`` is a first-class state, not a deletion: invalid, suspicious
    and low-confidence records are preserved with their raw payload so the
    origin of a questionable value is never lost.
    """

    PENDING = "pending"
    VALID = "valid"
    SUSPECT = "suspect"
    CONFLICT = "conflict"
    DUPLICATE = "duplicate"
    QUARANTINED = "quarantined"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class AggregationLevel(StrEnum):
    """Temporal granularity the value represents (SPEC sections 10, 11).

    Two values at the same timestamp and location are not comparable unless
    they share an aggregation level and measurement duration.
    """

    INSTANT = "instant"
    MINUTE = "minute"
    QUARTER_HOUR = "quarter_hour"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ExtractionKind(StrEnum):
    """Whether a document value was read or reasoned (SPEC section 34).

    ``EXTRACTED_FACT`` means the value is literally present in the source text.
    ``INFERRED`` means a model derived it. Only extracted facts may become
    factual dataset records without review.
    """

    EXTRACTED_FACT = "extracted_fact"
    INFERRED = "inferred"


class ImputationMethod(StrEnum):
    """How a missing value was filled, when it was (SPEC section 15).

    Missing data is never silently turned into zero. If a value is imputed, the
    record says so and says how.
    """

    INTERPOLATION = "interpolation"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    MODEL_BASED = "model_based"
    SEASONAL_NAIVE = "seasonal_naive"


class QuarantineReason(StrEnum):
    """Why a record was quarantined (SPEC section 18)."""

    INVALID_VALUE = "invalid_value"
    INVALID_TIMESTAMP = "invalid_timestamp"
    UNKNOWN_UNIT = "unknown_unit"
    UNKNOWN_LOCATION = "unknown_location"
    #: A location string that did not match the registry exactly. Distinct from
    #: UNKNOWN_LOCATION: the name may well be a known zone under a spelling
    #: nobody has curated yet, so it needs a human decision rather than being
    #: declared nonexistent (SPEC section 13).
    LOCATION_REVIEW_REQUIRED = "location_review_required"
    SCHEMA_ERROR = "schema_error"
    CONFLICT = "conflict"
    SUSPECTED_DUPLICATE = "suspected_duplicate"
    LOW_CONFIDENCE_EXTRACTION = "low_confidence_extraction"
    SUSPICIOUS_MEASUREMENT = "suspicious_measurement"
    SENSOR_FAILURE = "sensor_failure"
    FUTURE_EVENT_TIME = "future_event_time"


class SourceStatus(StrEnum):
    """Health of one reporting source (SPEC section 17).

    Health is tracked per source because a dead sensor and a quiet road look
    identical in the data: both stop producing high readings. Only the gap and
    repetition structure tells them apart, and missing sensor data is never
    interpreted as zero.
    """

    OK = "ok"
    WARNING = "warning"
    FAILED = "failed"


class Recency(StrEnum):
    """How old an observation was when it reached the system (SPEC section 12).

    Separate from :class:`SourceType`, which records *which stream* a value
    arrived on. The two are independent axes: a reading can arrive on the live
    feed and still be months old, and conflating them would make a newly
    uploaded 2022 report indistinguishable from this hour's measurement.
    """

    #: Arrived promptly; safe to treat as describing current conditions.
    CURRENT = "current"
    #: Arrived late but recently enough to still inform the present.
    RECENT = "recent"
    #: Far older than its receipt. Historical evidence, never "current".
    HISTORICAL = "historical"


class ReconciliationPolicy(StrEnum):
    """How a conflict is meant to be settled (SPEC section 10).

    Only manual review exists in v1, and that is a deliberate limitation rather
    than an unfinished one. "Real-time is fresher" and "historical is
    authoritative" are both *policies*, not facts, and a policy that silently
    discards a value destroys the evidence that it was the wrong policy. The
    field exists so that when automatic policies arrive, every reconciled record
    says which rule produced it.
    """

    #: Held for a human decision. No value is chosen automatically.
    MANUAL_REVIEW = "manual_review"


class ReconciliationState(StrEnum):
    """Whether a detected conflict has been settled yet."""

    #: Detected and recorded; no value chosen. All candidates are retained.
    UNRESOLVED = "unresolved"
    #: A reviewer has dispositioned it. The candidates are still retained.
    RESOLVED = "resolved"


class ModelStatus(StrEnum):
    """Role of a model version (SPEC sections 31, 32).

    A rejected candidate keeps its entry rather than being deleted: knowing
    which updates failed evaluation, and by how much, is part of the
    experimental record. Phase 6's champion/challenger comparison reads this
    field; nothing in Phase 4 promotes or demotes anything.
    """

    #: Currently serving predictions.
    CHAMPION = "champion"
    #: A candidate under evaluation against the champion.
    CHALLENGER = "challenger"
    #: Evaluated and not promoted. Retained with its metrics.
    REJECTED = "rejected"
    #: Superseded, kept for the record.
    ARCHIVED = "archived"


class Severity(StrEnum):
    """How bad a predicted congestion episode is (SPEC section 25).

    Ordered, and the order is load-bearing: severity is assigned from the ratio
    of predicted volume to the zone's congestion threshold, so a higher
    predicted volume can never map to a lower band. :attr:`rank` makes that
    comparable without relying on the string.
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"

    @property
    def rank(self) -> int:
        """Position in the ordering, lowest first."""
        return _SEVERITY_ORDER.index(self)


#: Severity order, defined once so comparisons and the config loader agree.
_SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.LOW,
    Severity.MODERATE,
    Severity.HIGH,
    Severity.SEVERE,
)
