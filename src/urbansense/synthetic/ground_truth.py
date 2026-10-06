"""The ground-truth record written alongside generated data (SPEC section 38).

This file is the whole reason the generator exists. Later phases claim to
discover patterns, detect drift and catch bad records; those claims are only
testable if something independent states what was actually planted.

``ground_truth.json`` is therefore written next to the data and holds:

* the exact parameters of every injected effect, so a discovered pattern's
  strength can be compared against the real one;
* the exact ``record_id`` of every planted duplicate, conflict and invalid row,
  so Phase 2 can be scored on precision as well as recall -- "you quarantined
  these 23 records" is a stronger claim than "you quarantined 23 records";
* the windows to *exclude* when measuring a pattern, since a gap or a frozen
  sensor inside the measurement window would otherwise distort the baseline.

Crucially, none of this appears in the CSV. Labels in the data would be a
leakage channel: a phase that could read ``is_friday_evening`` off its input
would be credited with a discovery it never made.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from urbansense.synthetic.config import GeneratorConfig


class TimeWindow(BaseModel):
    """A local-time window, inclusive of start and exclusive of end."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    location_id: str
    start: datetime
    end: datetime
    reason: str = ""


class FridayEveningTruth(BaseModel):
    """What was planted for the Friday-evening congestion pattern."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    location_id: str
    weekday: int
    weekday_name: str
    start_hour: int
    end_hour: int
    volume_multiplier: float
    speed_multiplier: float
    affected_record_ids: tuple[str, ...] = ()
    #: Zones deliberately left untouched, so a discovery phase must localize the
    #: pattern rather than asserting it city-wide.
    control_location_ids: tuple[str, ...] = ()


class RainEffectTruth(BaseModel):
    """What was planted for the rainfall-to-traffic relationship."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    volume_coefficient: float
    speed_coefficient: float
    cap_mm: float
    rainfall_metric: str
    rainfall_location_id: str
    wet_hour_count: int
    dry_hour_count: int


class DriftTruth(BaseModel):
    """What was planted for the mid-series baseline shift."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_date: date
    location_ids: tuple[str, ...]
    volume_multiplier: float
    reason: str
    control_location_ids: tuple[str, ...] = ()


class SensorFailureTruth(BaseModel):
    """What was planted for the frozen-value sensor failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    location_id: str
    start: datetime
    end: datetime
    frozen_value: float
    affected_record_ids: tuple[str, ...] = ()


class MissingDataTruth(BaseModel):
    """Which observations are missing, and why.

    ``record_ids`` are rows that exist in the CSV with an **empty** value cell.
    Missing is represented explicitly; it is never written as zero.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    windows: tuple[TimeWindow, ...] = ()
    scattered_fraction: float = 0.0
    scattered_location_ids: tuple[str, ...] = ()
    window_record_ids: tuple[str, ...] = ()
    scattered_record_ids: tuple[str, ...] = ()

    @property
    def all_record_ids(self) -> tuple[str, ...]:
        """Every record id whose value is missing."""
        return self.window_record_ids + self.scattered_record_ids


class DuplicateTruth(BaseModel):
    """One planted duplicate: ``record_id`` repeats ``duplicate_of``'s content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    duplicate_of: str


class ConflictTruth(BaseModel):
    """One planted conflict: two sources disagreeing on the same observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    conflicts_with: str
    location_id: str
    metric: str
    interval_end_local: datetime
    original_value: float
    conflicting_value: float


class InvalidTruth(BaseModel):
    """One planted invalid row, and why it is invalid.

    ``kind`` is one of ``negative_value``, ``impossible_speed``,
    ``unparseable_timestamp`` or ``unknown_unit``. The first two are
    representable as observations and must reach Phase 2 as ``PENDING``; the
    last two cannot become observations at all and must be rejected by
    ingestion without crashing it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    kind: str
    detail: str
    ingestible: bool


class PlantedRecords(BaseModel):
    """Every deliberately corrupted row, addressed by record id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    duplicates: tuple[DuplicateTruth, ...] = ()
    conflicts: tuple[ConflictTruth, ...] = ()
    invalid: tuple[InvalidTruth, ...] = ()

    @property
    def unparseable_record_ids(self) -> tuple[str, ...]:
        """Rows ingestion must reject rather than turn into observations."""
        return tuple(item.record_id for item in self.invalid if not item.ingestible)


class DatasetTruth(BaseModel):
    """Shape of one generated file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    row_count: int
    start: datetime
    end: datetime
    columns: tuple[str, ...]


class GroundTruth(BaseModel):
    """The complete record of what was planted in a generated dataset.

    Attributes:
        generator_version: Version of the generator that produced the data, so a
            stale ground-truth file can be detected.
        seed: Seed used. Regenerating with it reproduces the data exactly.
        config: The full generator configuration, echoed verbatim.
        timezone_offset_hours: Offset of the source's local wall clock, needed
            to interpret every local timestamp in this file.
        datasets: The files written.
        patterns: Injected pattern parameters, keyed by pattern name.
        anomalies: Injected anomaly parameters, keyed by anomaly name.
        missing_data: What is missing and why.
        planted_records: Duplicates, conflicts and invalid rows by record id.
        clock_skew_seconds: Per-sensor clock offset, which is why
            ``source_time`` differs from ``event_time``.
        exclude_windows: Windows to exclude when measuring pattern strength,
            because a gap or frozen sensor inside the window would distort it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    generator_version: str
    seed: int
    config: GeneratorConfig
    timezone_offset_hours: float
    datasets: dict[str, DatasetTruth] = Field(default_factory=dict)
    patterns: dict[str, FridayEveningTruth | RainEffectTruth] = Field(default_factory=dict)
    anomalies: dict[str, DriftTruth | SensorFailureTruth] = Field(default_factory=dict)
    missing_data: MissingDataTruth = MissingDataTruth()
    planted_records: PlantedRecords = PlantedRecords()
    clock_skew_seconds: dict[str, int] = Field(default_factory=dict)
    exclude_windows: tuple[TimeWindow, ...] = ()

    # -- typed accessors -------------------------------------------------
    #
    # ``patterns`` and ``anomalies`` are dicts so new effects can be added
    # without a schema change, but that makes their values unions. These
    # accessors give the known entries a precise type, so a consumer reaching
    # for ``friday_evening.volume_multiplier`` is checked rather than hoping.

    #: Key under which the Friday-evening pattern is stored.
    FRIDAY_EVENING_KEY: ClassVar[str] = "friday_evening_congestion"
    #: Key under which the rainfall relationship is stored.
    RAIN_EFFECT_KEY: ClassVar[str] = "rain_traffic_effect"
    #: Key under which the baseline drift is stored.
    DRIFT_KEY: ClassVar[str] = "baseline_drift"
    #: Key under which the sensor failure is stored.
    SENSOR_FAILURE_KEY: ClassVar[str] = "sensor_failure"

    @property
    def friday_evening(self) -> FridayEveningTruth:
        """The planted Friday-evening congestion pattern."""
        found = self.patterns[self.FRIDAY_EVENING_KEY]
        assert isinstance(found, FridayEveningTruth)
        return found

    @property
    def rain_effect(self) -> RainEffectTruth:
        """The planted rainfall-to-traffic relationship."""
        found = self.patterns[self.RAIN_EFFECT_KEY]
        assert isinstance(found, RainEffectTruth)
        return found

    @property
    def drift(self) -> DriftTruth:
        """The planted baseline drift."""
        found = self.anomalies[self.DRIFT_KEY]
        assert isinstance(found, DriftTruth)
        return found

    @property
    def sensor_failure(self) -> SensorFailureTruth:
        """The planted frozen-sensor failure."""
        found = self.anomalies[self.SENSOR_FAILURE_KEY]
        assert isinstance(found, SensorFailureTruth)
        return found

    def to_json(self) -> str:
        """Serialize deterministically, so regeneration is byte-stable."""
        payload: Any = self.model_dump(mode="json")
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def write(self, path: Path) -> None:
        """Write ``ground_truth.json`` to ``path``."""
        path.write_text(self.to_json(), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path) -> GroundTruth:
        """Read a ground-truth file written by :meth:`write`."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
