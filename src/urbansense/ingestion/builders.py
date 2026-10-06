"""Building canonical records from parsed rows (SPEC sections 6, 7, 8).

Shared by the historical loader and the real-time simulator so both produce
*identical* observation shapes. If they diverged, a value's meaning would depend
on which pipeline happened to ingest it, which defeats having a canonical schema
at all. The only intended difference between the two is ``source_type`` and when
``received_time`` is stamped.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

from urbansense.ingestion.rows import ParsedRow, RowParseError
from urbansense.schemas.enums import (
    AggregationLevel,
    ProblemType,
    SourceType,
    ValidationStatus,
)
from urbansense.schemas.observation import Observation
from urbansense.schemas.provenance import Provenance
from urbansense.schemas.quarantine import QuarantineRecord

#: Recorded on every observation's provenance so a value can be traced back to
#: the code that produced it (SPEC section 7).
PIPELINE_VERSION = "ingestion-1.0.0"

#: Maps an interval length to the aggregation level it represents, so two values
#: are never compared across incompatible granularities (SPEC section 10).
_AGGREGATION_BY_MINUTES: Mapping[int, AggregationLevel] = {
    1: AggregationLevel.MINUTE,
    15: AggregationLevel.QUARTER_HOUR,
    60: AggregationLevel.HOURLY,
    1440: AggregationLevel.DAILY,
}


def aggregation_for(interval: timedelta) -> AggregationLevel:
    """Return the aggregation level a measurement interval represents."""
    minutes = int(interval.total_seconds() // 60)
    return _AGGREGATION_BY_MINUTES.get(minutes, AggregationLevel.INSTANT)


class ZoneRegistry:
    """Coordinates for known zones, loaded from the dataset's ``zones.csv``.

    The feed sends a zone id, not coordinates on every reading, so this is where
    latitude and longitude come from. A zone absent from the registry is left
    **without** coordinates rather than guessed at: SPEC section 13 is explicit
    that similar names are not assumed identical, and an unknown zone is a
    location-review case for Phase 2, not something to invent here.
    """

    def __init__(self, coordinates: Mapping[str, tuple[float, float]]) -> None:
        self._coordinates = dict(coordinates)

    @classmethod
    def empty(cls) -> ZoneRegistry:
        """A registry with no entries; observations get no coordinates."""
        return cls({})

    @classmethod
    def from_csv(cls, path: Path) -> ZoneRegistry:
        """Load a registry from a ``zones.csv`` written by the generator.

        Malformed entries are skipped rather than fatal: a broken registry row
        should cost coordinates on one zone, not abort the whole ingestion.
        """
        coordinates: dict[str, tuple[float, float]] = {}
        import csv

        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                location_id = (row.get("location_id") or "").strip()
                try:
                    latitude = float(row["latitude"])
                    longitude = float(row["longitude"])
                except (KeyError, TypeError, ValueError):
                    continue
                if location_id:
                    coordinates[location_id] = (latitude, longitude)
        return cls(coordinates)

    def get(self, location_id: str) -> tuple[float | None, float | None]:
        """Return ``(latitude, longitude)`` for a zone, or ``(None, None)``."""
        found = self._coordinates.get(location_id)
        return found if found is not None else (None, None)

    def __len__(self) -> int:
        """Number of known zones."""
        return len(self._coordinates)


def to_observation(
    row: ParsedRow,
    *,
    received_time: datetime,
    source_type: SourceType,
    problem_type: ProblemType,
    registry: ZoneRegistry,
    canonical_metric: str | None = None,
    source_row: Mapping[str, str] | None = None,
) -> Observation:
    """Build a canonical observation from a parsed row.

    Args:
        row: The parsed feed row.
        received_time: When this system received the data.
        source_type: Which independent stream the row arrived on.
        problem_type: Problem domain.
        registry: Zone coordinates.
        canonical_metric: The configured canonical name, when config recognized
            the source's spelling. ``None`` keeps the source's own term --
            renaming a metric we do not recognize would be inventing
            information (SPEC section 5).
        source_row: The row exactly as read. Retained in ``attributes`` so a
            later stage that quarantines this record can write the *original*
            payload without re-reading the file (SPEC section 18). Omit it only
            where the raw row genuinely is not available.

    Returns:
        An ``Observation`` with ``validation_status=PENDING``.
    """
    latitude, longitude = registry.get(row.location_id)

    attributes: dict[str, object] = {
        "record_id": row.record_id,
        "source_metric": row.metric,
    }
    if source_row is not None:
        # The row as received. Carried on the record so a downstream quarantine
        # always has the original payload to hand (SPEC section 18).
        attributes["source_row"] = dict(source_row)

    return Observation(
        observation_id=f"OBS-{row.record_id}",
        source_id=row.source_id,
        source_type=source_type,
        event_time=row.event_time,
        received_time=received_time,
        source_time=row.source_time,
        measurement_duration=row.measurement_duration,
        location_id=row.location_id,
        latitude=latitude,
        longitude=longitude,
        problem_type=problem_type,
        metric=canonical_metric if canonical_metric is not None else row.metric,
        value=row.value,
        unit=row.unit,
        aggregation_level=aggregation_for(row.measurement_duration),
        # PENDING, not VALID: ingestion transcribes, it does not vouch. Deciding
        # what is valid is the integrity engine's job (SPEC section 8).
        validation_status=ValidationStatus.PENDING,
        provenance=Provenance(
            origin=row.source_id,
            ingested_at=received_time,
            pipeline_version=PIPELINE_VERSION,
            # The source's own wall-clock string, verbatim, so the UTC
            # normalization can be audited afterwards (SPEC section 12).
            notes=f"interval_end_local={row.interval_end_local}",
        ),
        attributes=attributes,
    )


def to_quarantine_record(
    error: RowParseError,
    *,
    source_path: Path,
    received_time: datetime,
    index: int,
) -> QuarantineRecord:
    """Wrap a parse failure as a quarantine record, payload intact.

    Args:
        error: The parse failure.
        source_path: File the row came from, used to build a stable id.
        received_time: When the row was read.
        index: 1-based row number, used when the row had no usable id.

    Returns:
        A ``QuarantineRecord`` carrying the original payload unaltered.
    """
    identifier = error.record_id if error.record_id else f"row{index}"
    return QuarantineRecord(
        quarantine_id=f"QTN-{source_path.stem}-{identifier}",
        reason=error.reason,
        detail=error.detail,
        quarantined_at=received_time,
        source_id=error.payload.get("source_id") or None,
        # Exactly as received. A record that failed to parse is the one a
        # reviewer most needs to see unaltered (SPEC section 18).
        raw_payload=dict(error.payload),
        detector="ingestion.parse_row",
    )
