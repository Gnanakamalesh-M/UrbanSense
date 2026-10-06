"""The canonical observation record (SPEC section 6).

Every module in UrbanSense reads and writes this one type. Historical,
real-time and document data converge on it *after* source-specific validation,
which is why the invariants the project depends on are enforced here rather
than repeated in each pipeline:

* timestamps are timezone-aware and stored in UTC (SPEC section 12);
* ``event_time`` and ``received_time`` are distinct, and receipt cannot precede
  the event except in quarantine (SPEC sections 4B, 14);
* missing is missing -- never zero -- and an imputed value must say how it was
  imputed (SPEC section 15);
* provenance is mandatory, and document values must carry document provenance
  (SPEC sections 7, 34);
* unit normalization keeps the original value and unit alongside the normalized
  ones (SPEC section 11);
* an ``event_time`` in the future is a clock error or a leakage attempt and is
  only admissible as a quarantined record (SPEC section 33).

Records are frozen: a validated observation is never mutated in place. Changes
produce a new record whose provenance points back at the old one, so the chain
of custody survives.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated, Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from urbansense.schemas.enums import (
    AggregationLevel,
    ExtractionKind,
    ImputationMethod,
    ProblemType,
    QuarantineReason,
    SourceType,
    ValidationStatus,
)
from urbansense.schemas.provenance import AnyProvenance, Confidence, DocumentProvenance
from urbansense.schemas.temporal import FUTURE_TOLERANCE, UTCDateTime, is_future

#: Statuses under which a record may violate the "sane record" rules. Quarantine
#: exists precisely to hold malformed evidence without destroying it, so these
#: records are exempt from the consistency checks that would reject them.
_QUARANTINE_STATUSES = frozenset(
    {ValidationStatus.QUARANTINED, ValidationStatus.REJECTED, ValidationStatus.SUSPECT}
)


def new_observation_id() -> str:
    """Generate an opaque observation identifier."""
    return f"OBS-{uuid4().hex[:12].upper()}"


class Observation(BaseModel):
    """A single validated measurement of one metric, at one place and time.

    Attributes:
        observation_id: Stable identifier for this record.
        source_id: Identifier of the sensor, feed, dataset or document.
        source_type: Which independent stream it arrived on.
        event_time: When the measured phenomenon occurred (UTC).
        received_time: When this system received the information (UTC).
        source_time: Timestamp as asserted by the source, if it differs from
            ``event_time`` after normalization. Kept so a source's own clock can
            be audited rather than quietly corrected away.
        measurement_duration: Interval the value covers, e.g. 15 minutes for a
            15-minute vehicle count. ``None`` for instantaneous readings.
        reporting_period_start: Start of the period a document reports on. A
            report uploaded today about 2022 stays 2022 data (SPEC section 12).
        reporting_period_end: End of that period.
        location_id: Canonical location key from the location registry.
        latitude: Optional WGS84 latitude.
        longitude: Optional WGS84 longitude.
        problem_type: Urban problem domain. Traffic only in v1.
        metric: Canonical metric name, e.g. ``"traffic_volume"``.
        value: The normalized measurement, or ``None`` when genuinely missing.
        unit: Canonical unit of ``value``.
        original_value: Value as supplied, before unit conversion.
        original_unit: Unit as supplied, before conversion.
        aggregation_level: Temporal granularity ``value`` represents.
        confidence: Trust in this record, in [0, 1].
        validation_status: Where the record stands in the integrity workflow.
        provenance: Mandatory origin record.
        is_imputed: Whether ``value`` was filled rather than measured.
        imputation_method: How it was filled; required when ``is_imputed``.
        duplicate_of: Observation this one duplicates. Duplicates are linked,
            not deleted (SPEC section 9).
        conflicts_with: Observations this one disagrees with. Conflicts are
            recorded, never silently resolved (SPEC section 10).
        quarantine_reason: Why the record was quarantined, when it was.
        attributes: Extension point for problem-specific detail, keeping the
            core schema closed while remaining extensible (SPEC section 6).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)

    # --- identity ---------------------------------------------------------
    observation_id: str = Field(default_factory=new_observation_id, min_length=1)
    source_id: str = Field(min_length=1)
    source_type: SourceType

    # --- time (SPEC sections 4B, 12) --------------------------------------
    event_time: UTCDateTime
    received_time: UTCDateTime
    source_time: UTCDateTime | None = None
    measurement_duration: timedelta | None = None
    reporting_period_start: UTCDateTime | None = None
    reporting_period_end: UTCDateTime | None = None

    # --- place (SPEC section 13) ------------------------------------------
    location_id: str = Field(min_length=1)
    latitude: Annotated[float, Field(ge=-90.0, le=90.0)] | None = None
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)] | None = None

    # --- measurement (SPEC sections 6, 11) --------------------------------
    problem_type: ProblemType
    metric: str = Field(min_length=1)
    value: float | None = None
    unit: str = Field(min_length=1)
    original_value: float | None = None
    original_unit: str | None = None
    aggregation_level: AggregationLevel

    # --- trust (SPEC sections 7, 15, 18) ----------------------------------
    confidence: Confidence = 1.0
    validation_status: ValidationStatus = ValidationStatus.PENDING
    provenance: AnyProvenance
    is_imputed: bool = False
    imputation_method: ImputationMethod | None = None

    # --- integrity links (SPEC sections 9, 10) ----------------------------
    duplicate_of: str | None = None
    conflicts_with: tuple[str, ...] = ()
    quarantine_reason: QuarantineReason | None = None

    # --- extension --------------------------------------------------------
    attributes: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    @property
    def is_quarantined(self) -> bool:
        """Whether this record is held for review rather than trusted."""
        return self.validation_status in _QUARANTINE_STATUSES

    @property
    def is_missing(self) -> bool:
        """Whether the measurement is absent. Absent is not zero."""
        return self.value is None

    @model_validator(mode="after")
    def _received_not_before_event(self) -> Self:
        """Receipt cannot precede the event outside quarantine (SPEC section 14).

        A record claiming it was received before it happened is a clock bug. It
        is preserved as quarantined evidence rather than rejected outright.
        """
        if self.received_time < self.event_time and not self.is_quarantined:
            raise ValueError(
                f"received_time ({self.received_time.isoformat()}) precedes event_time "
                f"({self.event_time.isoformat()}). This is a clock error; set "
                "validation_status=QUARANTINED to preserve the record for review."
            )
        return self

    @model_validator(mode="after")
    def _event_time_not_in_future(self) -> Self:
        """Reject future event times outside quarantine (SPEC section 33).

        A measurement of something that has not happened yet is either a clock
        error or future data leaking into a training set. Either way it must not
        pass silently.
        """
        if not self.is_quarantined and is_future(self.event_time, tolerance=FUTURE_TOLERANCE):
            raise ValueError(
                f"event_time ({self.event_time.isoformat()}) is more than "
                f"{FUTURE_TOLERANCE} in the future. This is a clock error or future "
                "data leakage; set validation_status=QUARANTINED to preserve it."
            )
        return self

    @model_validator(mode="after")
    def _reporting_period_ordered(self) -> Self:
        """A reporting period must not end before it starts."""
        start, end = self.reporting_period_start, self.reporting_period_end
        if start is not None and end is not None and end < start:
            raise ValueError(
                f"reporting_period_end ({end.isoformat()}) precedes "
                f"reporting_period_start ({start.isoformat()})."
            )
        return self

    @model_validator(mode="after")
    def _missing_is_not_zero(self) -> Self:
        """Imputation must be explicit and complete (SPEC section 15).

        Missing data stays missing by default. If a value *was* filled, the
        record must carry both the filled value and the method used, so no
        downstream consumer can mistake an estimate for a measurement.
        """
        if self.is_imputed:
            if self.value is None:
                raise ValueError(
                    "is_imputed=True but value is None: an imputed record must carry "
                    "the imputed value. Leave is_imputed=False to represent missing data."
                )
            if self.imputation_method is None:
                raise ValueError(
                    "is_imputed=True requires imputation_method, so that an estimated "
                    "value is never mistaken for a measured one."
                )
        elif self.imputation_method is not None:
            raise ValueError(
                f"imputation_method={self.imputation_method!r} was given but "
                "is_imputed=False. Set is_imputed=True or drop the method."
            )
        return self

    @model_validator(mode="after")
    def _unit_normalization_keeps_original(self) -> Self:
        """Normalization never discards what arrived (SPEC section 11).

        Keeping the original value and unit is what lets a conversion be
        re-checked later, so neither half may be dropped on its own.
        """
        if self.original_value is not None and self.original_unit is None:
            raise ValueError(
                "original_value was kept without original_unit. A value without its "
                "unit cannot be compared or re-converted; record both or neither."
            )
        if self.original_unit is not None and self.original_value is None and not self.is_missing:
            raise ValueError(
                "original_unit was kept without original_value. Record both or neither."
            )
        return self

    @model_validator(mode="after")
    def _document_values_carry_document_provenance(self) -> Self:
        """Document data must be auditable back to its source (SPEC sections 7, 34).

        A document-derived number without the document, page and original text
        is unverifiable, so the type system refuses it. Conversely, document
        provenance on a sensor reading is a mislabelled source.
        """
        is_document_provenance = isinstance(self.provenance, DocumentProvenance)
        if self.source_type is SourceType.DOCUMENT and not is_document_provenance:
            raise ValueError(
                "source_type=DOCUMENT requires DocumentProvenance (document name, hash, "
                "original_text, extraction confidence and timestamp) so the value can be "
                "traced back to the source text."
            )
        if self.source_type is not SourceType.DOCUMENT and is_document_provenance:
            raise ValueError(
                f"DocumentProvenance was given with source_type={self.source_type.value!r}. "
                "Document-derived values must declare source_type=DOCUMENT."
            )
        return self

    @model_validator(mode="after")
    def _inferred_values_are_not_facts(self) -> Self:
        """Inferred values may not be born valid (SPEC section 34).

        An extractor that reasoned its way to a number has not read it from the
        source. Such a record must be reviewed before it is treated as fact, so
        it cannot start life as VALID.
        """
        provenance = self.provenance
        if (
            isinstance(provenance, DocumentProvenance)
            and provenance.extraction_kind is ExtractionKind.INFERRED
            and self.validation_status is ValidationStatus.VALID
        ):
            raise ValueError(
                "An INFERRED document value cannot have validation_status=VALID. "
                "Only extracted facts become factual records automatically; inferred "
                "values require review (use NEEDS_REVIEW)."
            )
        return self

    @model_validator(mode="after")
    def _quarantine_states_are_explained(self) -> Self:
        """A quarantined record must say why it is there (SPEC section 18)."""
        if self.validation_status is ValidationStatus.QUARANTINED and (
            self.quarantine_reason is None
        ):
            raise ValueError(
                "validation_status=QUARANTINED requires a quarantine_reason, so that "
                "questionable data is never held without an explanation."
            )
        return self

    @model_validator(mode="after")
    def _integrity_links_are_consistent(self) -> Self:
        """Duplicate and conflict links must not be self-referential."""
        if self.duplicate_of == self.observation_id:
            raise ValueError("duplicate_of cannot reference the observation itself.")
        if self.observation_id in self.conflicts_with:
            raise ValueError("conflicts_with cannot reference the observation itself.")
        return self
