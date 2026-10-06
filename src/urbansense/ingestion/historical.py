"""Historical CSV ingestion (SPEC sections 4A, 7, 12, 15).

Reads a traffic feed export and produces canonical observations. Three
properties are load-bearing:

**Raw files are never modified.** This module opens its input read-only and
writes nothing anywhere. Rejected rows are returned to the caller, not written,
so nothing here can touch `data/raw/`.

**All three timestamps are preserved distinctly.** ``source_time`` is what the
source asserted, ``event_time`` is when the measurement actually began, and
``received_time`` is when we read it. Collapsing them would make late-arriving
data and newly uploaded old reports indistinguishable from fresh readings.

**Nothing is validated beyond representability.** Negative volumes and absurd
speeds pass through as ``PENDING``. Ingestion's job is faithful transcription;
judging the values is Phase 2's.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

from urbansense.config.models import Settings
from urbansense.ingestion.builders import (
    PIPELINE_VERSION,
    ZoneRegistry,
    aggregation_for,
    to_observation,
    to_quarantine_record,
)
from urbansense.ingestion.result import IngestionResult, IngestionStats
from urbansense.ingestion.rows import RowParseError, parse_row
from urbansense.schemas.enums import ProblemType, SourceType
from urbansense.schemas.observation import Observation
from urbansense.schemas.quarantine import QuarantineRecord

__all__ = [
    "PIPELINE_VERSION",
    "ZoneRegistry",
    "aggregation_for",
    "load_historical",
]


def load_historical(
    path: Path,
    *,
    settings: Settings | None = None,
    problem_type: ProblemType = ProblemType.TRAFFIC,
    interval: timedelta = timedelta(hours=1),
    registry: ZoneRegistry | None = None,
    received_time: datetime | None = None,
    source_type: SourceType = SourceType.HISTORICAL,
) -> IngestionResult:
    """Ingest a historical traffic CSV into canonical observations.

    Args:
        path: CSV to read. Opened read-only; never modified.
        settings: Problem configuration, used to resolve metric aliases and
            accepted units. ``None`` skips both checks.
        problem_type: Problem domain the file belongs to.
        interval: Measurement interval, used to derive ``event_time`` from the
            source's interval-end timestamps.
        registry: Zone coordinates. Defaults to empty, leaving observations
            without coordinates rather than inventing them.
        received_time: When the data was received. Defaults to now. Injectable
            so tests are deterministic.
        source_type: Which stream this file represents.

    Returns:
        An :class:`IngestionResult` whose observations and rejections together
        account for every data row in the file.
    """
    zones = registry if registry is not None else ZoneRegistry.empty()
    received = received_time if received_time is not None else datetime.now(UTC)

    observations: list[Observation] = []
    rejected: list[QuarantineRecord] = []
    rows_read = 0
    missing_values = 0

    # Read-only, and the only file access this module performs.
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, payload in enumerate(csv.DictReader(handle), start=1):
            rows_read += 1
            cleaned = {key: ("" if value is None else value) for key, value in payload.items()}

            metric_name = (cleaned.get("metric") or "").strip()
            configured = (
                settings.resolve_metric(problem_type, metric_name)
                if settings is not None and metric_name
                else None
            )
            units = (
                frozenset(configured.accepted_units)
                if (configured is not None and configured.accepted_units)
                else None
            )

            parsed = parse_row(cleaned, interval=interval, accepted_units=units)
            if isinstance(parsed, RowParseError):
                rejected.append(
                    to_quarantine_record(
                        parsed, source_path=path, received_time=received, index=index
                    )
                )
                continue

            if parsed.value is None:
                missing_values += 1

            observations.append(
                to_observation(
                    parsed,
                    received_time=received,
                    source_type=source_type,
                    problem_type=problem_type,
                    registry=zones,
                    canonical_metric=configured.name if configured is not None else None,
                    source_row=cleaned,
                )
            )

    return IngestionResult(
        observations=tuple(observations),
        rejected=tuple(rejected),
        stats=IngestionStats(
            rows_read=rows_read,
            observations=len(observations),
            rejected=len(rejected),
            missing_values=missing_values,
            source_path=path,
        ),
    )
