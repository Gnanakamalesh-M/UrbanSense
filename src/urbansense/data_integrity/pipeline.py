"""The validation and normalization pipeline (SPEC sections 2, 8, 11-18).

Implements the NORMALIZE -> VALIDATE -> QUARANTINE stretch of the lifecycle for
traffic, taking an :class:`~urbansense.ingestion.result.IngestionResult` and
producing a :class:`~urbansense.data_integrity.result.ValidatedResult`.

**Stage order is fixed and load-bearing:**

1. resolve the **location** -- an unresolvable zone makes everything downstream
   meaningless, so it is settled first;
2. resolve the **metric** and convert the **unit** -- range bounds are declared
   in canonical units, so checking before converting would compare
   ``2050 vehicles/15min`` against a per-hour ceiling and pass a value four
   times the limit;
3. classify **recency** -- cheap, and needed on every surviving record;
4. **range-check** against the metric's declared bounds;
5. detect **frozen sensors** -- needs the whole series, so it runs after the
   per-row passes;
6. detect **outliers** -- needs the surviving population, so it runs last.

Steps 1-4 can quarantine a row. Steps 5-6 are the two that need context rather
than a single row: a frozen run is invisible one reading at a time, and an
outlier is only unusual relative to its peers.

**No detector reads ``ground_truth.json``.** Nothing in this package imports or
references it. The planted answers exist to *score* these detectors from the
tests; a detector that consulted them would make every later measurement a
tautology.

**Duplicates and conflicts are untouched.** They normalize like any other row
and pass through with ``duplicate_of`` and ``conflicts_with`` unset. Detecting
them is a later phase, and guessing here would pre-empt it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from urbansense.config.models import Settings
from urbansense.data_integrity.outliers import detect_outliers
from urbansense.data_integrity.ranges import check_range
from urbansense.data_integrity.result import ValidatedResult, ValidationStats
from urbansense.data_integrity.sensors import (
    assess_source_health,
    detect_frozen_runs,
)
from urbansense.ingestion.result import IngestionResult
from urbansense.preprocessing.locations import LocationRegistry
from urbansense.preprocessing.recency import arrival_lag, classify_recency
from urbansense.preprocessing.units import UnitError, normalize_value
from urbansense.schemas.enums import ProblemType, QuarantineReason, ValidationStatus
from urbansense.schemas.observation import Observation
from urbansense.schemas.quarantine import QuarantineRecord

#: Recorded on every quarantine record this module produces.
DETECTOR_PREFIX = "data_integrity"


def _raw_payload(observation: Observation) -> dict[str, str]:
    """Recover the row exactly as the source sent it.

    Ingestion stashes it under ``attributes["source_row"]`` so a quarantine
    record is always complete without re-reading the file. When it is absent
    (an observation built by hand, say), fall back to a reconstruction and say
    so, rather than writing an empty payload and losing the trail.
    """
    stored = observation.attributes.get("source_row")
    if isinstance(stored, dict):
        return {str(key): str(value) for key, value in stored.items()}

    return {
        "record_id": str(observation.attributes.get("record_id", "")),
        "source_id": observation.source_id,
        "location_id": observation.location_id,
        "metric": str(observation.attributes.get("source_metric", observation.metric)),
        "value": "" if observation.value is None else repr(observation.value),
        "unit": observation.unit,
        "_note": "reconstructed; original source row was not retained",
    }


def _quarantine(
    observation: Observation,
    *,
    reason: QuarantineReason,
    detail: str,
    detector: str,
    now: datetime,
) -> QuarantineRecord:
    """Build a quarantine record for an observation, payload intact."""
    record_id = observation.attributes.get("record_id") or observation.observation_id
    return QuarantineRecord(
        quarantine_id=f"QTN-{reason.value}-{record_id}",
        reason=reason,
        detail=detail,
        quarantined_at=now,
        source_id=observation.source_id,
        raw_payload=_raw_payload(observation),
        # The parsed observation too: the row parsed fine and was rejected on a
        # content judgement, so a reviewer benefits from seeing both.
        observation=observation,
        detector=f"{DETECTOR_PREFIX}.{detector}",
    )


def validate_and_normalize(
    ingestion_result: IngestionResult,
    *,
    settings: Settings,
    registry: LocationRegistry,
    problem_type: ProblemType = ProblemType.TRAFFIC,
    now: datetime | None = None,
    expected_interval: timedelta | None = None,
    min_frozen_run: int | None = None,
    outlier_threshold: float | None = None,
    outlier_min_group_size: int | None = None,
    current_within: timedelta | None = None,
    recent_within: timedelta | None = None,
    warning_gap: timedelta | None = None,
    failed_gap: timedelta | None = None,
) -> ValidatedResult:
    """Normalize and validate an ingestion result.

    Args:
        ingestion_result: Output of a Phase 1 loader. Its own rejections are
            carried forward, so the row-count invariant holds against the file.
        settings: Problem configuration supplying metric aliases, canonical
            units and plausibility bounds.
        registry: Canonical location registry.
        problem_type: Problem domain being validated.
        now: Reference time for quarantine stamps and the trailing-gap
            calculation. Defaults to the ingestion receipt time when available,
            so a historical file validates reproducibly.
        expected_interval: Expected spacing between readings, for gap analysis.
        min_frozen_run: Identical consecutive readings before a run is frozen.
        outlier_threshold: MAD-sigma threshold for flagging.
        outlier_min_group_size: Minimum comparison-group size.
        current_within: Lag at or under which data counts as current.
        recent_within: Lag at or under which data counts as recent.
        warning_gap: Gap beyond which a source is flagged.
        failed_gap: Gap beyond which a source is considered failed.

    Returns:
        A :class:`ValidatedResult` whose usable and quarantined records together
        account for every input row.
    """
    validation = settings.validation
    reference = now if now is not None else _default_reference(ingestion_result)

    frozen_run_length = min_frozen_run if min_frozen_run is not None else validation.min_frozen_run
    threshold = (
        outlier_threshold if outlier_threshold is not None else validation.outlier_mad_threshold
    )
    group_size = (
        outlier_min_group_size
        if outlier_min_group_size is not None
        else validation.outlier_min_group_size
    )
    current_band = (
        current_within
        if current_within is not None
        else timedelta(hours=validation.current_within_hours)
    )
    recent_band = (
        recent_within
        if recent_within is not None
        else timedelta(hours=validation.recent_within_hours)
    )
    warn_gap = (
        warning_gap
        if warning_gap is not None
        else timedelta(hours=validation.source_warning_gap_hours)
    )
    fail_gap = (
        failed_gap
        if failed_gap is not None
        else timedelta(hours=validation.source_failed_gap_hours)
    )

    # Ingestion's rejections come along unchanged. Dropping them here would make
    # the per-file totals stop reconciling and hide rows that were already lost.
    quarantined: list[QuarantineRecord] = list(ingestion_result.rejected)
    carried_forward = len(quarantined)

    normalized: list[Observation] = []
    unit_conversions = 0

    for observation in ingestion_result.observations:
        # --- 1. location (SPEC section 13) ------------------------------
        match = registry.resolve(observation.location_id)
        if not match.resolved:
            quarantined.append(
                _quarantine(
                    observation,
                    reason=QuarantineReason.LOCATION_REVIEW_REQUIRED,
                    detail=match.detail,
                    detector="locations",
                    now=reference,
                )
            )
            continue
        assert match.zone is not None  # resolved implies a zone

        # --- 2. metric and unit (SPEC sections 5, 11) -------------------
        source_metric = str(observation.attributes.get("source_metric", observation.metric))
        metric_config = settings.resolve_metric(problem_type, source_metric)

        if metric_config is None:
            # Not a metric this problem defines. The rainfall covariate lands
            # here: it belongs to the flood domain, which v1 does not enable.
            # Discarding it would throw away a perfectly good predictor, so it
            # passes through un-range-checked but still normalized for time and
            # place -- there are no configured bounds to check it against.
            normalized.append(
                _finalize(
                    observation,
                    location_id=match.zone.location_id,
                    latitude=match.zone.latitude,
                    longitude=match.zone.longitude,
                    value=observation.value,
                    unit=observation.unit,
                    original_value=None,
                    original_unit=None,
                    reference=reference,
                    current_within=current_band,
                    recent_within=recent_band,
                    metric_known=False,
                )
            )
            continue

        try:
            converted = normalize_value(
                observation.value, observation.unit, metric_config.canonical_unit
            )
        except UnitError as exc:
            quarantined.append(
                _quarantine(
                    observation,
                    reason=QuarantineReason.UNKNOWN_UNIT,
                    detail=(
                        f"{exc} -- not coerced to {metric_config.canonical_unit!r}; "
                        "comparing values without understanding their units is how "
                        "silent corruption gets in"
                    ),
                    detector="units",
                    now=reference,
                )
            )
            continue

        if (
            observation.unit.strip().lower()
            not in {unit.strip().lower() for unit in metric_config.accepted_units}
            and metric_config.accepted_units
        ):
            quarantined.append(
                _quarantine(
                    observation,
                    reason=QuarantineReason.UNKNOWN_UNIT,
                    detail=(
                        f"unit {observation.unit!r} is not accepted for metric "
                        f"{metric_config.name!r}; accepted units are "
                        f"{sorted(metric_config.accepted_units)}"
                    ),
                    detector="units",
                    now=reference,
                )
            )
            continue

        if converted.converted:
            unit_conversions += 1

        # --- 4. range check, in canonical units (SPEC section 14) -------
        violation = check_range(converted.value, metric_config, unit=converted.unit)
        if violation is not None:
            quarantined.append(
                _quarantine(
                    observation,
                    reason=violation.reason,
                    detail=violation.detail,
                    detector="ranges",
                    now=reference,
                )
            )
            continue

        normalized.append(
            _finalize(
                observation,
                location_id=match.zone.location_id,
                latitude=match.zone.latitude,
                longitude=match.zone.longitude,
                value=converted.value,
                unit=converted.unit,
                original_value=converted.original_value if converted.converted else None,
                original_unit=converted.original_unit if converted.converted else None,
                reference=reference,
                current_within=current_band,
                recent_within=recent_band,
                metric_known=True,
                canonical_metric=metric_config.name,
            )
        )

    # --- 5. frozen sensors (SPEC section 17) ----------------------------
    frozen_runs = detect_frozen_runs(
        normalized,
        min_run_length=frozen_run_length,
        ignored_values=validation.frozen_ignored_values,
    )
    frozen_ids = {observation_id for run in frozen_runs for observation_id in run.observation_ids}
    frozen_detail = {
        observation_id: run for run in frozen_runs for observation_id in run.observation_ids
    }

    surviving: list[Observation] = []
    for observation in normalized:
        if observation.observation_id not in frozen_ids:
            surviving.append(observation)
            continue
        run = frozen_detail[observation.observation_id]
        quarantined.append(
            _quarantine(
                observation,
                reason=QuarantineReason.SENSOR_FAILURE,
                detail=(
                    f"{run.metric} at {run.location_id} reported {run.value:g} "
                    f"identically for {run.length} consecutive readings "
                    f"({run.start.isoformat()} to {run.end.isoformat()}, "
                    f"{run.duration}). A stuck sensor's output is not a "
                    "measurement, and it looks plausible, which is what makes it "
                    "more dangerous than a gap."
                ),
                detector="sensors",
                now=reference,
            )
        )

    # --- 6. outliers: flag and keep (SPEC section 16) -------------------
    flags = detect_outliers(surviving, threshold=threshold, min_group_size=group_size)

    final: list[Observation] = []
    for observation in surviving:
        flag = flags.get(observation.observation_id)
        if flag is None:
            final.append(observation)
            continue
        attributes = dict(observation.attributes)
        attributes["outlier_mad_sigma"] = round(flag.deviation, 3)
        attributes["outlier_group_median"] = flag.group_median
        final.append(
            observation.model_copy(
                update={
                    # SUSPECT, not quarantined: unusual is not impossible, and
                    # a festival or an accident is real data (SPEC section 16).
                    "validation_status": ValidationStatus.SUSPECT,
                    "attributes": attributes,
                }
            )
        )

    health = assess_source_health(
        # Includes the frozen readings on purpose: they are the evidence that
        # the sensor is unhealthy, so excluding them would hide the failure.
        normalized,
        frozen_runs=frozen_runs,
        now=None,
        expected_interval=expected_interval,
        warning_gap=warn_gap,
        failed_gap=fail_gap,
        warning_missing_rate=validation.source_warning_missing_rate,
    )

    suspicious_count = sum(1 for obs in final if obs.validation_status is ValidationStatus.SUSPECT)

    return ValidatedResult(
        valid=tuple(final),
        quarantined=tuple(quarantined),
        source_health=health,
        frozen_runs=frozen_runs,
        outlier_flags=flags,
        stats=ValidationStats(
            rows_in=ingestion_result.stats.rows_read,
            valid=len(final),
            suspicious=suspicious_count,
            quarantined=len(quarantined),
            missing_values=sum(1 for obs in final if obs.value is None),
            carried_forward=carried_forward,
            normalized_units=unit_conversions,
            source_path=ingestion_result.stats.source_path,
        ),
    )


def _default_reference(ingestion_result: IngestionResult) -> datetime:
    """Pick a reference time, preferring the data's own receipt time.

    Using the ingestion receipt rather than "now" keeps a run over a historical
    file reproducible: the same file validated twice produces the same recency
    classification and the same quarantine timestamps.
    """
    for observation in ingestion_result.observations:
        return observation.received_time
    for record in ingestion_result.rejected:
        return record.quarantined_at
    from datetime import UTC

    return datetime.now(UTC)


def _finalize(
    observation: Observation,
    *,
    location_id: str,
    latitude: float | None,
    longitude: float | None,
    value: float | None,
    unit: str,
    original_value: float | None,
    original_unit: str | None,
    reference: datetime,
    current_within: timedelta,
    recent_within: timedelta,
    metric_known: bool,
    canonical_metric: str | None = None,
) -> Observation:
    """Produce the normalized observation.

    A new record rather than a mutation: ``Observation`` is frozen, so a
    normalized value is a *new* record and the original is never altered in
    place.
    """
    recency = classify_recency(
        observation.event_time,
        observation.received_time,
        current_within=current_within,
        recent_within=recent_within,
    )
    lag = arrival_lag(observation.event_time, observation.received_time)

    attributes = dict(observation.attributes)
    attributes["recency"] = recency.value
    attributes["arrival_lag_seconds"] = int(lag.total_seconds())
    attributes["metric_in_problem_config"] = metric_known
    if original_unit is not None:
        attributes["unit_normalized_from"] = original_unit

    update: dict[str, object] = {
        "location_id": location_id,
        "value": value,
        "unit": unit,
        # Both representations retained, so the conversion stays auditable
        # (SPEC section 11).
        "original_value": original_value,
        "original_unit": original_unit,
        # Normalization vouches for shape, not for content; the record is
        # valid unless a detector says otherwise.
        "validation_status": ValidationStatus.VALID,
        "attributes": attributes,
    }
    if latitude is not None:
        update["latitude"] = latitude
    if longitude is not None:
        update["longitude"] = longitude
    if canonical_metric is not None:
        update["metric"] = canonical_metric

    return observation.model_copy(update=update)


__all__ = ["DETECTOR_PREFIX", "validate_and_normalize"]
