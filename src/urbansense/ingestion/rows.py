"""Parsing raw feed rows into typed fields (SPEC sections 4, 12, 15).

This is the boundary where untrusted text becomes typed data, and it is the only
place in ingestion allowed to fail. Two outcomes, never a third:

* a :class:`ParsedRow` -- enough structure to build an ``Observation``;
* a :class:`RowParseError` -- carrying the original row untouched, so the
  rejected record can be quarantined with its payload intact.

A row is **never silently dropped**. Ingestion accounts for every input line.

Note what is *not* done here: no range checking, no outlier judgement, no
plausibility test. A negative vehicle count parses perfectly well and becomes a
``PENDING`` observation. Catching it is Phase 2's job, and doing it here would
pre-empt the thing Phase 2 exists to prove.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from urbansense.schemas.enums import QuarantineReason

#: Columns a traffic feed row must contain to be interpretable at all.
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"record_id", "source_id", "interval_end_local", "location_id", "metric", "value", "unit"}
)


def snap_to_interval(moment: datetime, interval: timedelta) -> datetime:
    """Round ``moment`` to the nearest ``interval`` boundary.

    Sensors have imperfect clocks: a reading for the 18:00-19:00 hour may be
    stamped 18:59:08 or 19:00:37. Those are the same bucket, and treating them
    as different instants would make every join across sources fail and would
    scatter one hour's readings across two.

    So a feed's clock is trusted only to within half an interval, which is a
    normalization step, not a correction: the raw instant is kept on the
    observation as ``source_time``, so the skew stays visible and a frozen or
    badly-drifting sensor clock remains detectable (SPEC section 17).

    Args:
        moment: The timestamp as asserted by the source.
        interval: The measurement interval to snap to.

    Returns:
        ``moment`` rounded to the nearest interval boundary, in the same
        timezone. Returned unchanged when ``interval`` is zero or negative.
    """
    if interval <= timedelta(0):
        return moment

    step = interval.total_seconds()
    # Snap in the timestamp's own offset frame, not against the UTC epoch.
    # A feed on a half-hour offset such as IST has its hour boundaries at
    # :30 past the UTC hour, so rounding to UTC-epoch multiples would place
    # every reading half an interval away from the boundary it belongs to.
    local = moment.replace(tzinfo=None)
    origin = datetime(1970, 1, 1)
    offset_seconds = (local - origin).total_seconds()
    snapped = origin + timedelta(seconds=round(offset_seconds / step) * step)
    return snapped.replace(tzinfo=moment.tzinfo)


@dataclass(frozen=True, slots=True)
class ParsedRow:
    """One feed row with its fields typed.

    Attributes:
        record_id: The source's reading id.
        source_id: Sensor or feed that produced it.
        location_id: Zone the reading is for.
        metric: Metric name as the source calls it.
        unit: Unit as the source declares it.
        value: The measurement, or ``None`` when the cell was empty. ``None``
            means *missing* and is never converted to zero (SPEC section 15).
        source_time: The instant the source asserted, normalized to UTC. For
            this feed that is the end of the measurement interval, with the
            sensor's own clock skew baked in.
        event_time: The instant the measurement interval began, derived from
            ``source_time`` and the interval length. Distinct from
            ``source_time``, which is the point of keeping both.
        interval_end_local: The raw timestamp string exactly as the source wrote
            it, preserved so the normalization can be re-checked later.
        measurement_duration: Interval the value covers.
    """

    record_id: str
    source_id: str
    location_id: str
    metric: str
    unit: str
    value: float | None
    source_time: datetime
    event_time: datetime
    interval_end_local: str
    measurement_duration: timedelta


@dataclass(frozen=True, slots=True)
class RowParseError:
    """A row that could not be parsed, with everything needed to quarantine it.

    Attributes:
        reason: Which quarantine category this falls into.
        detail: Human-readable explanation for a reviewer.
        payload: The original row as a mapping, untouched. Normalizing it on the
            way in would destroy the evidence a reviewer needs.
        record_id: The source's reading id, when the row had a usable one.
    """

    reason: QuarantineReason
    detail: str
    payload: dict[str, str]
    record_id: str | None = None


def parse_row(
    payload: dict[str, str],
    *,
    interval: timedelta,
    accepted_units: frozenset[str] | None = None,
) -> ParsedRow | RowParseError:
    """Parse one feed row.

    Args:
        payload: The raw row, as read from CSV.
        interval: Length of the measurement interval, used to derive
            ``event_time`` from the source's interval-end timestamp.
        accepted_units: Units this metric may legitimately arrive in. A unit
            outside the set is rejected rather than coerced -- comparing values
            without understanding their units is how silent corruption happens
            (SPEC section 11). ``None`` skips the check.

    Returns:
        A :class:`ParsedRow`, or a :class:`RowParseError` describing why not.
    """
    missing_columns = REQUIRED_COLUMNS - payload.keys()
    if missing_columns:
        return RowParseError(
            reason=QuarantineReason.SCHEMA_ERROR,
            detail=f"row is missing required columns: {sorted(missing_columns)}",
            payload=payload,
            record_id=payload.get("record_id"),
        )

    record_id = payload["record_id"].strip()
    if not record_id:
        return RowParseError(
            reason=QuarantineReason.SCHEMA_ERROR,
            detail="record_id is empty",
            payload=payload,
        )

    raw_stamp = payload["interval_end_local"].strip()
    try:
        stamped = datetime.fromisoformat(raw_stamp)
    except ValueError:
        return RowParseError(
            reason=QuarantineReason.INVALID_TIMESTAMP,
            detail=f"interval_end_local {raw_stamp!r} is not an ISO-8601 timestamp",
            payload=payload,
            record_id=record_id,
        )

    if stamped.tzinfo is None:
        # A naive timestamp cannot be placed on the UTC timeline without
        # guessing a zone, and guessing is how an 18:00 peak silently becomes a
        # 12:30 one (SPEC section 12). The feed is specified to carry an offset.
        return RowParseError(
            reason=QuarantineReason.INVALID_TIMESTAMP,
            detail=(
                f"interval_end_local {raw_stamp!r} has no UTC offset; an instant "
                "without an offset is ambiguous and will not be guessed at"
            ),
            payload=payload,
            record_id=record_id,
        )

    source_time = stamped.astimezone(UTC)
    # Snap while the timestamp still carries the source's own offset. Snapping
    # after the UTC conversion would align to UTC hour boundaries, which for a
    # half-hour offset such as IST sit half an interval away from the source's
    # own hour boundaries.
    event_time = (snap_to_interval(stamped, interval) - interval).astimezone(UTC)

    location_id = payload["location_id"].strip()
    metric = payload["metric"].strip()
    unit = payload["unit"].strip()
    for name, field in (("location_id", location_id), ("metric", metric), ("unit", unit)):
        if not field:
            return RowParseError(
                reason=QuarantineReason.SCHEMA_ERROR,
                detail=f"{name} is empty",
                payload=payload,
                record_id=record_id,
            )

    if accepted_units is not None and unit not in accepted_units:
        return RowParseError(
            reason=QuarantineReason.UNKNOWN_UNIT,
            detail=(
                f"unit {unit!r} is not accepted for metric {metric!r}; "
                f"accepted units are {sorted(accepted_units)}"
            ),
            payload=payload,
            record_id=record_id,
        )

    raw_value = payload["value"].strip()
    value: float | None
    if raw_value == "":
        # Missing stays missing. A sensor that reported nothing did not report
        # zero vehicles (SPEC section 15).
        value = None
    else:
        try:
            value = float(raw_value)
        except ValueError:
            return RowParseError(
                reason=QuarantineReason.INVALID_VALUE,
                detail=f"value {raw_value!r} is not a number",
                payload=payload,
                record_id=record_id,
            )

    return ParsedRow(
        record_id=record_id,
        source_id=payload["source_id"].strip(),
        location_id=location_id,
        metric=metric,
        unit=unit,
        value=value,
        # Kept raw, skew and all: this is what the source actually asserted, and
        # discarding it would hide a drifting sensor clock.
        source_time=source_time,
        # The feed stamps the end of the interval, so the event began one
        # interval earlier. Snapping absorbs the sensor's clock skew, so
        # readings for the same hour share an event_time and can be joined.
        event_time=event_time,
        interval_end_local=raw_stamp,
        measurement_duration=interval,
    )
