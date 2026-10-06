"""The full prediction output (SPEC section 25).

    For each urban problem, predict:
    probability / severity / expected time / location / confidence

    Traffic Risk: 87%  Zone: B  Expected: 18:00-20:00  Severity: High
    Confidence: 89%

This module assembles the pieces the rest of the package produces -- a volume
from the forecaster, a threshold from :mod:`congestion`, a calibrated
probability from :mod:`calibration`, a source breakdown from :mod:`sources` and
factors from :mod:`urbansense.explainability` -- into one record per zone-hour,
plus the contiguous windows those hours form.

**Two deliberate departures from the example above.**

*Confidence is a spread, not a second probability.* Section 25's example prints
"Confidence: 89%" beside a risk of 87%, which invites reading both as
likelihoods. Only one of them is calibrated. So :attr:`Prediction.confidence`
is derived from the width of the prediction interval relative to the predicted
value and is labelled as an interval-width measure everywhere it is rendered.
Two percentages side by side, one meaning something precise and one meaning
something vague, is worse than one percentage and a width.

*A congestion window is not a Friday detector.* On the demo data 14 of 14
ZONE-B Friday evening hours clear the threshold -- and so do 65 of 82
*other*-weekday evening hours, because the threshold is anchored to a training
period when the city was quieter. Windows are therefore common, and what
distinguishes the planted Friday peak is its **severity**, not the fact that a
window exists. Rendering says so rather than letting a reader infer that a
window means something unusual.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from urbansense.explainability.attribution import Explanation, FactorAttributor
from urbansense.features.builder import FeatureTable
from urbansense.prediction.calibration import ProbabilityCalibrator
from urbansense.prediction.congestion import CongestionThresholds
from urbansense.prediction.sources import SourceBreakdown, SourceLedger
from urbansense.schemas.enums import Severity

#: Interval width, relative to the prediction, at or above which confidence is
#: reported as zero. A band as wide as the prediction itself says nothing.
CONFIDENCE_FLOOR_RATIO = 1.0


def interval_confidence(predicted: float, lower: float, upper: float) -> float:
    """How tight the prediction interval is, as a 0-1 score.

    **Not a probability.** It is ``1 - width/prediction``, clipped: a band half
    as wide as the prediction scores 0.5, one as wide scores 0. The calibrated
    congestion probability is the number to read as a likelihood; this one says
    how precise the volume estimate is.
    """
    if np.isnan(predicted) or np.isnan(lower) or np.isnan(upper) or predicted <= 0:
        return float("nan")
    width = max(upper - lower, 0.0)
    return float(np.clip(1.0 - (width / predicted) / CONFIDENCE_FLOOR_RATIO, 0.0, 1.0))


@dataclass(frozen=True)
class Prediction:
    """One zone-hour prediction, with everything needed to judge it.

    Attributes:
        location_id: Zone.
        target_time: The hour predicted, in UTC.
        local_time: The same hour in the source's local clock.
        predicted_volume: The forecast.
        threshold: The zone's congestion threshold, from training data.
        congestion_probability: Calibrated P(actual > threshold).
        severity: Band from the predicted-to-threshold ratio.
        interval_lower: Lower bound of the prediction interval.
        interval_upper: Upper bound.
        model_version: Which model said it.
        probability_is_calibrated: Whether a fitted calibrator was applied. A
            raw score and a calibrated one must never look alike.
        sources: Where the inputs came from, when computed.
        explanation: Contributing factors, when computed.
    """

    location_id: str
    target_time: datetime
    local_time: datetime
    predicted_volume: float
    threshold: float
    congestion_probability: float
    severity: Severity
    interval_lower: float
    interval_upper: float
    model_version: str
    probability_is_calibrated: bool
    sources: SourceBreakdown | None = None
    explanation: Explanation | None = None

    @property
    def is_congested(self) -> bool:
        """Whether the predicted volume clears the zone's threshold."""
        return bool(self.predicted_volume > self.threshold)

    @property
    def ratio(self) -> float:
        """Predicted volume as a multiple of the threshold."""
        return self.predicted_volume / self.threshold if self.threshold else float("nan")

    @property
    def confidence(self) -> float:
        """Interval tightness. See :func:`interval_confidence` -- not a chance."""
        return interval_confidence(self.predicted_volume, self.interval_lower, self.interval_upper)

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "location_id": self.location_id,
            "target_time": self.target_time.isoformat(),
            "local_time": self.local_time.isoformat(),
            "predicted_volume": round(self.predicted_volume, 2),
            "threshold": round(self.threshold, 2),
            "ratio_to_threshold": round(self.ratio, 4),
            "congestion_probability": round(self.congestion_probability, 4),
            "probability_is_calibrated": self.probability_is_calibrated,
            "severity": self.severity.value,
            "is_congested": self.is_congested,
            "interval": {
                "lower": round(self.interval_lower, 2),
                "upper": round(self.interval_upper, 2),
                "confidence": round(self.confidence, 4),
                "confidence_note": (
                    "interval tightness, not a probability; the calibrated "
                    "congestion probability is the figure to read as a likelihood"
                ),
            },
            "model_version": self.model_version,
            "sources": self.sources.as_dict() if self.sources else None,
            "explanation": self.explanation.as_dict() if self.explanation else None,
        }

    def describe(self) -> str:
        """Human-readable summary, in the shape SPEC section 25 shows."""
        label = "calibrated" if self.probability_is_calibrated else "UNCALIBRATED"
        lines = [
            f"{self.location_id}  {self.local_time:%Y-%m-%d %H:%M} local",
            f"  congestion probability  {self.congestion_probability:>6.1%}  ({label})",
            f"  severity                {self.severity.value:>6}  "
            f"(x{self.ratio:.2f} the zone threshold of {self.threshold:,.0f})",
            f"  predicted volume        {self.predicted_volume:>6,.0f} "
            f"[{self.interval_lower:,.0f} to {self.interval_upper:,.0f}]",
            f"  interval tightness      {self.confidence:>6.1%}  "
            "(how precise the volume is, not a chance)",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class CongestionWindow:
    """A run of consecutive hours predicted above the threshold.

    Attributes:
        location_id: Zone.
        start_local: First congested hour, local.
        end_local: Hour after the last congested one, local, so the window
            reads as a half-open 18:00-20:00 span.
        peak_severity: Worst severity in the run.
        peak_volume: Highest predicted volume in the run.
        peak_probability: Highest calibrated probability in the run.
        hours: Number of hours in the run.
    """

    location_id: str
    start_local: datetime
    end_local: datetime
    peak_severity: Severity
    peak_volume: float
    peak_probability: float
    hours: int

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "location_id": self.location_id,
            "start_local": self.start_local.isoformat(),
            "end_local": self.end_local.isoformat(),
            "window": self.label,
            "hours": self.hours,
            "peak_severity": self.peak_severity.value,
            "peak_volume": round(self.peak_volume, 2),
            "peak_probability": round(self.peak_probability, 4),
        }

    @property
    def label(self) -> str:
        """The window as ``18:00-20:00``."""
        return f"{self.start_local:%H:%M}-{self.end_local:%H:%M}"

    def describe(self) -> str:
        """One line."""
        return (
            f"{self.location_id}  {self.start_local:%Y-%m-%d}  {self.label}  "
            f"peak {self.peak_severity.value} at {self.peak_volume:,.0f} "
            f"(probability {self.peak_probability:.0%})"
        )


def congestion_windows(predictions: Sequence[Prediction]) -> tuple[CongestionWindow, ...]:
    """Merge consecutive congested hours into windows.

    A gap in the predictions breaks a window rather than being bridged: two
    congested hours either side of an unpredicted hour are not evidence of a
    three-hour episode, and merging them would invent the middle one.
    """
    by_zone: dict[str, list[Prediction]] = {}
    for item in predictions:
        by_zone.setdefault(item.location_id, []).append(item)

    windows: list[CongestionWindow] = []
    for zone, rows in sorted(by_zone.items()):
        rows.sort(key=lambda item: item.target_time)
        run: list[Prediction] = []

        def flush(batch: list[Prediction], zone: str = zone) -> None:
            if not batch:
                return
            windows.append(
                CongestionWindow(
                    location_id=zone,
                    start_local=batch[0].local_time,
                    end_local=batch[-1].local_time + timedelta(hours=1),
                    peak_severity=max((item.severity for item in batch), key=lambda s: s.rank),
                    peak_volume=max(item.predicted_volume for item in batch),
                    peak_probability=max(item.congestion_probability for item in batch),
                    hours=len(batch),
                )
            )

        for item in rows:
            if not item.is_congested:
                flush(run)
                run = []
                continue
            if run and item.target_time - run[-1].target_time != timedelta(hours=1):
                # Not contiguous: an unpredicted hour sits between them.
                flush(run)
                run = []
            run.append(item)
        flush(run)

    return tuple(windows)


def build_predictions(
    table: FeatureTable,
    predicted: NDArray[np.float64],
    *,
    thresholds: CongestionThresholds,
    calibrator: ProbabilityCalibrator,
    model_version: str,
    interval_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    timezone_offset_hours: float = 0.0,
    ledger: SourceLedger | None = None,
    attributor: FactorAttributor | None = None,
    explain_rows: bool = False,
) -> tuple[Prediction, ...]:
    """Assemble full prediction records for every row of a table.

    Args:
        table: Rows to predict for.
        predicted: One predicted volume per row.
        thresholds: Per-zone thresholds, fitted on training data.
        calibrator: Probability calibrator. Used fitted or not -- when unfitted,
            every record is marked uncalibrated rather than quietly presenting
            a raw score as a calibrated one.
        model_version: Which model produced these.
        interval_bounds: Lower and upper bounds per row, when available.
        timezone_offset_hours: Local offset for the rendered hour.
        ledger: Source index, for the contribution breakdown.
        attributor: Explainer, for the contributing factors.
        explain_rows: Whether to attach a per-row explanation. Off by default:
            explaining thousands of rows is slow and almost never wanted, while
            a day's worth is instant.

    Raises:
        ValueError: if the prediction count does not match the table.
    """
    if predicted.shape[0] != len(table):
        raise ValueError(
            f"{predicted.shape[0]} predictions for {len(table)} rows; refusing to "
            "build misaligned prediction records"
        )

    zone_thresholds = np.array(
        [thresholds.by_zone.get(zone, float("nan")) for zone in table.locations],
        dtype=np.float64,
    )
    raw = calibrator.raw(predicted, zone_thresholds)
    probabilities = calibrator.calibrate(raw)
    offset = timedelta(hours=timezone_offset_hours)

    records: list[Prediction] = []
    for index in range(len(table)):
        zone = table.locations[index]
        value = float(predicted[index])
        lower, upper = (
            (float(interval_bounds[0][index]), float(interval_bounds[1][index]))
            if interval_bounds is not None
            else (float("nan"), float("nan"))
        )
        records.append(
            Prediction(
                location_id=zone,
                target_time=table.target_times[index],
                local_time=table.target_times[index] + offset,
                predicted_volume=value,
                threshold=float(zone_thresholds[index]),
                congestion_probability=float(probabilities[index]),
                severity=thresholds.severity(zone, value),
                interval_lower=lower,
                interval_upper=upper,
                model_version=model_version,
                probability_is_calibrated=calibrator.is_fitted,
                sources=(
                    ledger.attribute(
                        table.feature_set,
                        location_id=zone,
                        sample_time=table.sample_times[index],
                    )
                    if ledger is not None
                    else None
                ),
                explanation=(
                    attributor.explain_row(table.x[index])
                    if attributor is not None and explain_rows
                    else None
                ),
            )
        )
    return tuple(records)
