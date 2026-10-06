"""Builders for synthetic observations used by the detector tests.

Detector tests need small, hand-shaped series -- a run of identical values, a
gap in the middle, one extreme point -- which the generated dataset cannot
provide on demand. These build them directly from the schema so the tests stay
independent of the generator.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from urbansense.schemas.enums import (
    AggregationLevel,
    ProblemType,
    SourceType,
    ValidationStatus,
)
from urbansense.schemas.observation import Observation
from urbansense.schemas.provenance import Provenance

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for the annotation
    from urbansense.data_integrity.result import ValidatedResult

#: Far enough in the past that even a long weekly series stays behind "now".
#: The schema rejects a future event_time, so a test series starting too late
#: would fail construction rather than testing the detector.
DEFAULT_START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)


def make_observation(
    value: float | None,
    *,
    event_time: datetime,
    source_id: str = "sensor-TEST",
    location_id: str = "ZONE-T",
    metric: str = "traffic_volume",
    unit: str = "vehicles/hour",
    record_id: str | None = None,
) -> Observation:
    """Build one observation with sensible defaults."""
    identifier = record_id if record_id is not None else f"REC-{int(event_time.timestamp())}"
    return Observation(
        observation_id=f"OBS-{identifier}",
        source_id=source_id,
        source_type=SourceType.HISTORICAL,
        event_time=event_time,
        received_time=event_time + timedelta(hours=1),
        source_time=event_time + timedelta(hours=1),
        measurement_duration=timedelta(hours=1),
        location_id=location_id,
        problem_type=ProblemType.TRAFFIC,
        metric=metric,
        value=value,
        unit=unit,
        aggregation_level=AggregationLevel.HOURLY,
        validation_status=ValidationStatus.PENDING,
        provenance=Provenance(
            origin=source_id,
            ingested_at=event_time + timedelta(hours=1),
            pipeline_version="test",
        ),
        attributes={"record_id": identifier, "source_metric": metric},
    )


def make_series(
    values: list[float | None],
    *,
    start: datetime = DEFAULT_START,
    hours_apart: int = 1,
    source_id: str = "sensor-TEST",
    location_id: str = "ZONE-T",
    metric: str = "traffic_volume",
) -> list[Observation]:
    """Build a series of observations from a list of values.

    ``hours_apart`` of ``24 * 7`` puts every point in the same weekday-and-hour
    group, which is what the outlier detector conditions on -- so a test can
    build one comparison group deliberately.
    """
    return [
        make_observation(
            value,
            event_time=start + timedelta(hours=hours_apart * index),
            source_id=source_id,
            location_id=location_id,
            metric=metric,
            record_id=f"REC-{index:06d}",
        )
        for index, value in enumerate(values)
    ]


def make_validated(
    observations: list[Observation],
    *,
    rows_in: int | None = None,
) -> ValidatedResult:
    """Wrap hand-built observations as a ValidatedResult for reconcile().

    Lets a reconciliation test construct one exact group -- a cross-source
    agreement, a three-way dispute -- which the generated dataset cannot
    provide on demand.
    """
    from urbansense.data_integrity.result import ValidatedResult, ValidationStats

    return ValidatedResult(
        valid=tuple(observations),
        stats=ValidationStats(
            rows_in=rows_in if rows_in is not None else len(observations),
            valid=len(observations),
        ),
    )
