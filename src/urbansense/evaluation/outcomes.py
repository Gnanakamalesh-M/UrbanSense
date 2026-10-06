"""Append-only prediction outcome tracking (SPEC section 28).

    Store every prediction outcome:
    prediction, actual, error, location, time, conditions, model_version

Append-only, as JSONL. A prediction's record is what it said *at the time*, and
rewriting it later to match what happened would destroy the only evidence that
the model was ever wrong. Each run appends; nothing is edited, and
:meth:`OutcomeStore.append` opens the file in ``"a"`` mode so the code could not
truncate it even by accident.

**Conditions are recorded at target time, and that is not leakage.** The
rainfall stored against a prediction is the rainfall during the hour being
predicted, which the model never saw -- it is an attribute of the *outcome*, for
answering "where does the model fail". It is written after the fact and is never
read back as an input: the feature path is probed by
:func:`~urbansense.features.leakage.detect_leakage`, and this store is not on
it. Recording the conditions the model *did* see instead would make the error
analysis unable to answer the only question it exists for -- whether heavy rain
breaks the model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from urbansense.features.builder import FeatureTable
from urbansense.prediction.congestion import CongestionThresholds
from urbansense.schemas.enums import Severity

#: Default directory for outcome logs.
DEFAULT_OUTCOME_DIR = Path("reports") / "outcomes"

#: Rainfall band edges in mm. "Light" and "heavy" are the bands SPEC section 28
#: asks the error report to separate; the 10mm cut matches the saturation point
#: Phase 5's conditional pattern fitted on.
LIGHT_RAIN_MM = 0.1
HEAVY_RAIN_MM = 10.0


def rain_band(rainfall_mm: float | None) -> str:
    """Bucket a rainfall reading.

    An unknown reading is ``"unknown"``, never ``"dry"``. Missing stays missing:
    treating an absent gauge reading as zero rain would move those rows into the
    dry bucket and quietly flatter the dry-weather numbers.
    """
    if rainfall_mm is None or np.isnan(rainfall_mm):
        return "unknown"
    if rainfall_mm < LIGHT_RAIN_MM:
        return "dry"
    return "light" if rainfall_mm <= HEAVY_RAIN_MM else "heavy"


def _optional_bool(value: object) -> bool | None:
    """Read a stored flag that may legitimately be absent."""
    return None if value is None else bool(value)


def _optional_float(value: object) -> float | None:
    """Read a stored number that may legitimately be absent.

    ``None`` survives as ``None`` rather than becoming 0.0: an unrecorded
    rainfall is not a dry hour.
    """
    return None if value is None else float(value)  # type: ignore[arg-type]


def time_of_day(hour: int) -> str:
    """Bucket a local hour into the bands the error report breaks out."""
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


@dataclass(frozen=True)
class Outcome:
    """One prediction and what actually happened.

    Attributes:
        prediction_id: Stable id, derived from model, zone and target time so
            the same prediction keeps its id across runs.
        model_version: Which model said it.
        location_id: Zone.
        target_time: The hour predicted, in UTC.
        local_hour: Target hour in local time, for the time-of-day breakdown.
        weekday: Local weekday of the target, Monday 0.
        predicted: What the model said.
        actual: What was observed.
        probability: Calibrated congestion probability, when one was produced.
        severity: Severity band the prediction carried.
        congested_predicted: Whether the prediction exceeded the threshold.
        congested_actual: Whether the observation did.
        rainfall_mm: Rainfall during the predicted hour, or ``None`` if unknown.
        record_id: Source record behind the actual, so an outcome traces back to
            a feed line.
        recorded_at: When this row was written.
    """

    prediction_id: str
    model_version: str
    location_id: str
    target_time: datetime
    local_hour: int
    weekday: int
    predicted: float
    actual: float
    probability: float | None = None
    severity: Severity | None = None
    congested_predicted: bool | None = None
    congested_actual: bool | None = None
    rainfall_mm: float | None = None
    record_id: str = ""
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def error(self) -> float:
        """Predicted minus actual. Signed, so bias stays visible."""
        return self.predicted - self.actual

    @property
    def abs_error(self) -> float:
        """Absolute error."""
        return abs(self.error)

    @property
    def rain_band(self) -> str:
        """Rainfall bucket for the predicted hour."""
        return rain_band(self.rainfall_mm)

    @property
    def time_of_day(self) -> str:
        """Time-of-day bucket for the predicted hour."""
        return time_of_day(self.local_hour)

    @property
    def is_friday_evening(self) -> bool:
        """Whether the target sits in the planted Friday 18:00-20:00 window."""
        return self.weekday == 4 and 18 <= self.local_hour < 20

    def as_dict(self) -> dict[str, object]:
        """Serializable form. Every field SPEC section 28 names is present."""
        return {
            "prediction_id": self.prediction_id,
            "model_version": self.model_version,
            "location_id": self.location_id,
            "target_time": self.target_time.isoformat(),
            "predicted": round(self.predicted, 4),
            "actual": round(self.actual, 4),
            "error": round(self.error, 4),
            "abs_error": round(self.abs_error, 4),
            "probability": (round(self.probability, 6) if self.probability is not None else None),
            "severity": self.severity.value if self.severity else None,
            "congested_predicted": self.congested_predicted,
            "congested_actual": self.congested_actual,
            "conditions": {
                "rainfall_mm": (
                    round(self.rainfall_mm, 3) if self.rainfall_mm is not None else None
                ),
                "rain_band": self.rain_band,
                "local_hour": self.local_hour,
                "time_of_day": self.time_of_day,
                "weekday": self.weekday,
                "is_friday_evening": self.is_friday_evening,
            },
            "record_id": self.record_id,
            "recorded_at": self.recorded_at.isoformat(),
            "conditions_note": (
                "conditions describe the predicted hour, not what the model saw; "
                "they are written after the fact and are never read back as inputs"
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Outcome:
        """Rebuild from a stored row."""
        conditions = payload.get("conditions")
        conditions = conditions if isinstance(conditions, dict) else {}
        severity = payload.get("severity")
        rainfall = conditions.get("rainfall_mm")
        probability = payload.get("probability")
        return cls(
            prediction_id=str(payload["prediction_id"]),
            model_version=str(payload["model_version"]),
            location_id=str(payload["location_id"]),
            target_time=datetime.fromisoformat(str(payload["target_time"])),
            local_hour=int(conditions.get("local_hour", 0)),
            weekday=int(conditions.get("weekday", 0)),
            predicted=float(payload["predicted"]),  # type: ignore[arg-type]
            actual=float(payload["actual"]),  # type: ignore[arg-type]
            probability=_optional_float(probability),
            severity=Severity(str(severity)) if severity else None,
            congested_predicted=_optional_bool(payload.get("congested_predicted")),
            congested_actual=_optional_bool(payload.get("congested_actual")),
            rainfall_mm=_optional_float(rainfall),
            record_id=str(payload.get("record_id", "")),
            recorded_at=datetime.fromisoformat(str(payload["recorded_at"])),
        )


class OutcomeStore:
    """Append-only JSONL log of prediction outcomes.

    Attributes:
        path: The log file. Created on demand.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Initialize.

        Args:
            path: The JSONL file to append to. Defaults to
                ``reports/outcomes/outcomes.jsonl``.
        """
        self.path = path if path is not None else DEFAULT_OUTCOME_DIR / "outcomes.jsonl"

    def append(self, outcomes: Iterable[Outcome]) -> int:
        """Append outcomes to the log.

        Opened in append mode, so existing rows cannot be truncated even by a
        caller who meant to replace them. Returns the number of rows written.
        """
        rows = list(outcomes)
        if not rows:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for outcome in rows:
                handle.write(json.dumps(outcome.as_dict(), sort_keys=True) + "\n")
        return len(rows)

    def read(self) -> tuple[Outcome, ...]:
        """Read every outcome back, in the order written."""
        if not self.path.is_file():
            return ()
        rows: list[Outcome] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(Outcome.from_dict(json.loads(line)))
        return tuple(rows)

    def __len__(self) -> int:
        """Rows currently in the log."""
        if not self.path.is_file():
            return 0
        return sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_outcomes(
    table: FeatureTable,
    predicted: NDArray[np.float64],
    *,
    model_version: str,
    thresholds: CongestionThresholds | None = None,
    probabilities: Sequence[float] | None = None,
    rainfall_by_time: dict[datetime, float] | None = None,
    timezone_offset_hours: float = 0.0,
    recorded_at: datetime | None = None,
) -> tuple[Outcome, ...]:
    """Pair every prediction with its actual and the conditions that held.

    Args:
        table: The evaluated rows.
        predicted: One prediction per row.
        model_version: Which model produced them.
        thresholds: For the congested flags, when available.
        probabilities: Calibrated congestion probabilities, when available.
        rainfall_by_time: City rainfall keyed by event time. A target hour with
            no entry records ``None``, never zero.
        timezone_offset_hours: Local offset, for the hour and weekday fields.
        recorded_at: Timestamp for the rows. Defaults to now.

    Raises:
        ValueError: if the prediction count does not match the table, which
            would silently pair every prediction with the wrong actual.
    """
    if predicted.shape[0] != len(table):
        raise ValueError(
            f"{predicted.shape[0]} predictions for {len(table)} rows; refusing to "
            "record a misaligned pair"
        )

    stamp = recorded_at if recorded_at is not None else datetime.now(UTC)
    offset = timedelta(hours=timezone_offset_hours)
    rows: list[Outcome] = []
    for index in range(len(table)):
        target = table.target_times[index]
        local = target + offset
        zone = table.locations[index]
        value = float(predicted[index])
        actual = float(table.y[index])
        rainfall = rainfall_by_time.get(target) if rainfall_by_time else None
        rows.append(
            Outcome(
                prediction_id=f"{model_version}:{zone}:{target.isoformat()}",
                model_version=model_version,
                location_id=zone,
                target_time=target,
                local_hour=local.hour,
                weekday=local.weekday(),
                predicted=value,
                actual=actual,
                probability=(float(probabilities[index]) if probabilities is not None else None),
                severity=thresholds.severity(zone, value) if thresholds else None,
                congested_predicted=(thresholds.is_congested(zone, value) if thresholds else None),
                congested_actual=(thresholds.is_congested(zone, actual) if thresholds else None),
                rainfall_mm=rainfall,
                record_id=table.record_ids[index],
                recorded_at=stamp,
            )
        )
    return tuple(rows)
