"""Tests for the canonical observation schema.

Each test pins one invariant the project promises, so that a future change that
weakens a guarantee fails here rather than corrupting data silently.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from urbansense.schemas import (
    AggregationLevel,
    DocumentProvenance,
    ExtractionKind,
    ImputationMethod,
    Observation,
    ProblemType,
    Provenance,
    QuarantineReason,
    QuarantineRecord,
    SourceType,
    ValidationStatus,
    new_observation_id,
    require_utc,
)

from .conftest import EVENT_TIME, RECEIVED_TIME

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_valid_observation_is_accepted(observation):
    assert observation.metric == "traffic_volume"
    assert observation.value == 8200.0
    assert observation.problem_type is ProblemType.TRAFFIC
    assert observation.observation_id.startswith("OBS-")


def test_generated_ids_are_unique():
    assert new_observation_id() != new_observation_id()


def test_records_are_immutable(observation):
    """A validated observation is never mutated in place."""
    with pytest.raises(ValidationError):
        observation.value = 1.0


def test_unknown_field_is_rejected(make_observation):
    """A typo must fail loudly instead of being absorbed and ignored."""
    with pytest.raises(ValidationError, match=r"extra_forbidden|not permitted"):
        make_observation(trafic_volume=8200)


def test_json_round_trip_preserves_provenance_and_instants(observation):
    restored = Observation.model_validate_json(observation.model_dump_json())
    assert restored == observation
    assert restored.event_time == observation.event_time
    assert restored.provenance.origin == "sensor-A12"


def test_json_round_trip_keeps_document_provenance_class(make_observation, document_provenance):
    """The discriminated union must not collapse to the looser provenance type."""
    obs = make_observation(
        source_type=SourceType.DOCUMENT,
        provenance=document_provenance,
    )
    restored = Observation.model_validate_json(obs.model_dump_json())
    assert isinstance(restored.provenance, DocumentProvenance)
    assert restored.provenance.page == 7


# ---------------------------------------------------------------------------
# Invariant 1 — time is unambiguous and stored in UTC (SPEC section 12)
# ---------------------------------------------------------------------------


def test_naive_event_time_is_rejected(make_observation):
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_observation(event_time=datetime(2026, 9, 25, 18, 0))


def test_naive_received_time_is_rejected(make_observation):
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_observation(received_time=datetime(2026, 9, 25, 18, 2))


def test_aware_non_utc_time_is_normalized_to_utc(make_observation):
    ist = timezone(timedelta(hours=5, minutes=30))
    obs = make_observation(
        event_time=datetime(2026, 9, 25, 23, 30, tzinfo=ist),
        received_time=datetime(2026, 9, 25, 23, 35, tzinfo=ist),
    )
    assert obs.event_time.tzinfo == UTC
    assert obs.event_time == datetime(2026, 9, 25, 18, 0, tzinfo=UTC)


def test_require_utc_rejects_naive_directly():
    with pytest.raises(ValueError, match="timezone-aware"):
        require_utc(datetime(2026, 1, 1, 0, 0))


def test_event_and_received_time_stay_distinct(observation):
    """'When it happened' is never conflated with 'when we were told'."""
    assert observation.received_time > observation.event_time


def test_received_before_event_is_rejected(make_observation):
    with pytest.raises(ValidationError, match="precedes event_time"):
        make_observation(received_time=EVENT_TIME - timedelta(minutes=1))


def test_received_before_event_is_preserved_in_quarantine(make_observation):
    """A clock error is held as evidence, not thrown away (SPEC section 18)."""
    obs = make_observation(
        received_time=EVENT_TIME - timedelta(minutes=1),
        validation_status=ValidationStatus.QUARANTINED,
        quarantine_reason=QuarantineReason.INVALID_TIMESTAMP,
    )
    assert obs.is_quarantined


def test_old_report_received_today_stays_historical(make_observation, document_provenance):
    """A 2022 report uploaded in 2026 is 2022 data (SPEC section 12)."""
    obs = make_observation(
        source_type=SourceType.DOCUMENT,
        provenance=document_provenance,
        event_time=datetime(2022, 6, 1, 18, 0, tzinfo=UTC),
        received_time=RECEIVED_TIME,
        reporting_period_start=datetime(2022, 6, 1, 0, 0, tzinfo=UTC),
        reporting_period_end=datetime(2022, 6, 30, 23, 59, tzinfo=UTC),
    )
    assert obs.event_time.year == 2022
    assert obs.received_time.year == 2026


def test_inverted_reporting_period_is_rejected(make_observation):
    with pytest.raises(ValidationError, match="reporting_period_end"):
        make_observation(
            reporting_period_start=datetime(2022, 6, 30, tzinfo=UTC),
            reporting_period_end=datetime(2022, 6, 1, tzinfo=UTC),
        )


# ---------------------------------------------------------------------------
# Invariant: no future event times (SPEC section 33)
# ---------------------------------------------------------------------------


def test_future_event_time_is_rejected(make_observation):
    future = datetime.now(UTC) + timedelta(days=1)
    with pytest.raises(ValidationError, match="future"):
        make_observation(event_time=future, received_time=future + timedelta(minutes=1))


def test_future_event_time_is_allowed_only_in_quarantine(make_observation):
    future = datetime.now(UTC) + timedelta(days=1)
    obs = make_observation(
        event_time=future,
        received_time=future + timedelta(minutes=1),
        validation_status=ValidationStatus.QUARANTINED,
        quarantine_reason=QuarantineReason.FUTURE_EVENT_TIME,
    )
    assert obs.is_quarantined


def test_small_clock_skew_is_tolerated(make_observation):
    """A sensor a minute ahead of us is skew, not leakage."""
    skewed = datetime.now(UTC) + timedelta(minutes=1)
    obs = make_observation(event_time=skewed, received_time=skewed)
    assert obs.event_time == skewed


# ---------------------------------------------------------------------------
# Invariant 2 — missing is never zero (SPEC section 15)
# ---------------------------------------------------------------------------


def test_missing_value_is_representable_and_is_not_zero(make_observation):
    obs = make_observation(value=None)
    assert obs.is_missing
    assert obs.value is None
    assert obs.value != 0


def test_missing_value_may_be_valid(make_observation):
    """Genuinely absent data is a valid record, not an error."""
    obs = make_observation(value=None, validation_status=ValidationStatus.VALID)
    assert obs.is_missing and not obs.is_quarantined


def test_imputed_record_requires_a_value(make_observation):
    with pytest.raises(ValidationError, match="imputed record must carry"):
        make_observation(
            value=None, is_imputed=True, imputation_method=ImputationMethod.FORWARD_FILL
        )


def test_imputed_record_requires_a_method(make_observation):
    with pytest.raises(ValidationError, match="requires imputation_method"):
        make_observation(value=5000.0, is_imputed=True)


def test_imputation_method_without_flag_is_rejected(make_observation):
    """An estimate must never be able to masquerade as a measurement."""
    with pytest.raises(ValidationError, match="is_imputed=False"):
        make_observation(imputation_method=ImputationMethod.INTERPOLATION)


def test_properly_declared_imputation_is_accepted(make_observation):
    obs = make_observation(
        value=5000.0,
        is_imputed=True,
        imputation_method=ImputationMethod.MODEL_BASED,
        confidence=0.5,
    )
    assert obs.is_imputed
    assert obs.imputation_method is ImputationMethod.MODEL_BASED


# ---------------------------------------------------------------------------
# Invariant 3 — provenance is mandatory (SPEC sections 7, 34)
# ---------------------------------------------------------------------------


def test_provenance_is_required(provenance):
    with pytest.raises(ValidationError, match="provenance"):
        Observation(
            source_id="sensor-A12",
            source_type=SourceType.REAL_TIME,
            event_time=EVENT_TIME,
            received_time=RECEIVED_TIME,
            location_id="ZONE-B",
            problem_type=ProblemType.TRAFFIC,
            metric="traffic_volume",
            value=1.0,
            unit="vehicles/hour",
            aggregation_level=AggregationLevel.HOURLY,
        )


def test_document_source_requires_document_provenance(make_observation):
    with pytest.raises(ValidationError, match="requires DocumentProvenance"):
        make_observation(source_type=SourceType.DOCUMENT)


def test_document_provenance_on_non_document_source_is_rejected(
    make_observation, document_provenance
):
    with pytest.raises(ValidationError, match="must declare source_type=DOCUMENT"):
        make_observation(
            source_type=SourceType.REAL_TIME,
            provenance=document_provenance,
        )


def test_document_observation_is_traceable_to_its_source(make_observation, document_provenance):
    obs = make_observation(source_type=SourceType.DOCUMENT, provenance=document_provenance)
    assert obs.provenance.document_name == "traffic_report_q3.pdf"
    assert obs.provenance.page == 7
    assert "8,200 vehicles per hour" in obs.provenance.original_text
    assert obs.provenance.extraction_confidence == pytest.approx(0.94)


def test_document_provenance_requires_original_text():
    with pytest.raises(ValidationError):
        DocumentProvenance(
            document_name="x.pdf",
            document_hash="sha256:abc",
            original_text="",
            extraction_kind=ExtractionKind.EXTRACTED_FACT,
            extraction_confidence=0.9,
            extracted_at=RECEIVED_TIME,
            extractor_version="docint-0.1.0",
            ingested_at=RECEIVED_TIME,
            pipeline_version="0.1.0",
        )


def test_extraction_cannot_postdate_ingestion():
    with pytest.raises(ValidationError, match="cannot be ingested before"):
        DocumentProvenance(
            document_name="x.pdf",
            document_hash="sha256:abc",
            original_text="Traffic rose.",
            extraction_kind=ExtractionKind.EXTRACTED_FACT,
            extraction_confidence=0.9,
            extracted_at=RECEIVED_TIME + timedelta(hours=1),
            extractor_version="docint-0.1.0",
            ingested_at=RECEIVED_TIME,
            pipeline_version="0.1.0",
        )


def test_lineage_is_recorded_rather_than_overwritten(observation):
    """Real-time data finalized as history points back; it does not overwrite."""
    finalized = Provenance(
        origin="historical_warehouse",
        ingested_at=RECEIVED_TIME + timedelta(days=1),
        pipeline_version="0.1.0",
        upstream_observation_ids=(observation.observation_id,),
    )
    assert finalized.upstream_observation_ids == (observation.observation_id,)


# ---------------------------------------------------------------------------
# Hallucination guard (SPEC section 34)
# ---------------------------------------------------------------------------


def test_inferred_document_value_cannot_be_born_valid(make_observation, document_provenance):
    inferred = document_provenance.model_copy(update={"extraction_kind": ExtractionKind.INFERRED})
    with pytest.raises(ValidationError, match="INFERRED document value"):
        make_observation(
            source_type=SourceType.DOCUMENT,
            provenance=inferred,
            validation_status=ValidationStatus.VALID,
        )


def test_inferred_document_value_may_await_review(make_observation, document_provenance):
    inferred = document_provenance.model_copy(update={"extraction_kind": ExtractionKind.INFERRED})
    obs = make_observation(
        source_type=SourceType.DOCUMENT,
        provenance=inferred,
        validation_status=ValidationStatus.NEEDS_REVIEW,
    )
    assert obs.validation_status is ValidationStatus.NEEDS_REVIEW


def test_extracted_fact_may_be_valid(make_observation, document_provenance):
    obs = make_observation(
        source_type=SourceType.DOCUMENT,
        provenance=document_provenance,
        validation_status=ValidationStatus.VALID,
    )
    assert obs.validation_status is ValidationStatus.VALID


# ---------------------------------------------------------------------------
# Invariant 4 — unit normalization keeps the original (SPEC section 11)
# ---------------------------------------------------------------------------


def test_normalization_retains_original_value_and_unit(make_observation):
    obs = make_observation(
        value=8200.0,
        unit="vehicles/hour",
        original_value=2050.0,
        original_unit="vehicles/15min",
    )
    assert obs.original_value == 2050.0
    assert obs.original_unit == "vehicles/15min"
    assert obs.unit == "vehicles/hour"


def test_original_value_without_unit_is_rejected(make_observation):
    with pytest.raises(ValidationError, match="without original_unit"):
        make_observation(original_value=2050.0)


def test_original_unit_without_value_is_rejected(make_observation):
    with pytest.raises(ValidationError, match="without original_value"):
        make_observation(original_unit="vehicles/15min")


def test_original_unit_alone_is_allowed_when_value_is_missing(make_observation):
    """A missing reading can still record the unit it would have been in."""
    obs = make_observation(value=None, original_unit="vehicles/15min")
    assert obs.is_missing and obs.original_unit == "vehicles/15min"


def test_unit_is_required(make_observation):
    with pytest.raises(ValidationError):
        make_observation(unit="")


# ---------------------------------------------------------------------------
# Confidence, coordinates, integrity links
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_confidence_must_be_a_probability(make_observation, bad):
    with pytest.raises(ValidationError):
        make_observation(confidence=bad)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(91.0, 80.0), (-91.0, 80.0), (13.0, 181.0), (13.0, -181.0)],
)
def test_impossible_coordinates_are_rejected(make_observation, lat, lon):
    with pytest.raises(ValidationError):
        make_observation(latitude=lat, longitude=lon)


def test_duplicates_are_linked_not_deleted(make_observation, observation):
    """A duplicate keeps its own record and points at the original."""
    duplicate = make_observation(
        source_id="sensor-A12-mirror",
        validation_status=ValidationStatus.DUPLICATE,
        duplicate_of=observation.observation_id,
    )
    assert duplicate.duplicate_of == observation.observation_id
    assert duplicate.value == observation.value


def test_conflicts_are_recorded_not_resolved(make_observation, observation):
    """Disagreeing sources are both kept and marked CONFLICT (SPEC section 10)."""
    conflicting = make_observation(
        source_id="historical_warehouse",
        source_type=SourceType.HISTORICAL,
        value=7900.0,
        validation_status=ValidationStatus.CONFLICT,
        conflicts_with=(observation.observation_id,),
    )
    assert conflicting.value == 7900.0
    assert observation.observation_id in conflicting.conflicts_with


def test_self_referential_duplicate_link_is_rejected(make_observation):
    with pytest.raises(ValidationError, match="itself"):
        make_observation(observation_id="OBS-1", duplicate_of="OBS-1")


def test_self_referential_conflict_link_is_rejected(make_observation):
    with pytest.raises(ValidationError, match="itself"):
        make_observation(observation_id="OBS-1", conflicts_with=("OBS-1",))


def test_quarantined_record_must_state_a_reason(make_observation):
    with pytest.raises(ValidationError, match="requires a quarantine_reason"):
        make_observation(validation_status=ValidationStatus.QUARANTINED)


def test_attributes_allow_extension_without_schema_change(make_observation):
    obs = make_observation(attributes={"lane_count": 4, "detector_type": "loop"})
    assert obs.attributes["lane_count"] == 4


# ---------------------------------------------------------------------------
# Quarantine records (SPEC sections 14, 18)
# ---------------------------------------------------------------------------


def test_quarantine_preserves_an_unparseable_payload():
    """A record that fails schema validation is exactly what a reviewer needs."""
    record = QuarantineRecord(
        quarantine_id="QTN-0001",
        reason=QuarantineReason.INVALID_VALUE,
        detail="traffic_volume was -42, which is impossible",
        quarantined_at=RECEIVED_TIME,
        source_id="sensor-A12",
        raw_payload={"zone": "B", "vehicles": -42, "ts": "2026-09-25T18:00:00+05:30"},
        detector="range_check",
    )
    assert record.raw_payload["vehicles"] == -42
    assert record.observation is None
    assert not record.resolved


def test_quarantine_can_carry_the_parsed_observation(observation):
    record = QuarantineRecord(
        quarantine_id="QTN-0002",
        reason=QuarantineReason.SUSPICIOUS_MEASUREMENT,
        detail="value is 4x the plausible maximum for this location",
        quarantined_at=RECEIVED_TIME,
        raw_payload={"vehicles": 8200},
        observation=observation,
        detector="outlier_check",
    )
    assert record.observation == observation


def test_quarantine_requires_an_explanation():
    with pytest.raises(ValidationError):
        QuarantineRecord(
            quarantine_id="QTN-0003",
            reason=QuarantineReason.SCHEMA_ERROR,
            detail="",
            quarantined_at=RECEIVED_TIME,
            raw_payload={},
            detector="schema_check",
        )
