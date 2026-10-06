"""Error metrics, sliced (SPEC sections 28, 38).

    Overall MAE: 8.4 / Heavy rain: 16.7 / Normal weather: 6.1

A single aggregate hides exactly the failures worth fixing. A model can be fine
on average and unreliable on Friday evenings in the one zone that congests --
which is when a traffic forecast actually matters. So every metric here is
computed per slice as well as overall, and the slices are named rather than
implied.

MAE and RMSE are both reported because they disagree usefully: RMSE punishes
large errors more, so RMSE rising faster than MAE between two models means the
second is failing badly on a few samples rather than slightly on many.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from urbansense.features.builder import FeatureTable


def mae(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    """Mean absolute error.

    ``NaN`` predictions are counted as errors rather than skipped: a model that
    declines to answer has not got it right, and dropping those rows would
    flatter a baseline that is often unable to predict at all.
    """
    if actual.size == 0:
        return float("nan")
    errors = np.abs(predicted - actual)
    errors = np.where(np.isnan(errors), np.abs(actual), errors)
    return float(np.mean(errors))


def rmse(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    """Root mean squared error, with the same treatment of unanswered rows."""
    if actual.size == 0:
        return float("nan")
    errors = predicted - actual
    errors = np.where(np.isnan(errors), actual, errors)
    return float(np.sqrt(np.mean(errors**2)))


def coverage(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    """Fraction of rows the model actually produced a number for.

    Reported alongside the errors because a baseline with excellent MAE over the
    quarter of rows it could answer is not better than a model that answers all
    of them.
    """
    if predicted.size == 0:
        return float("nan")
    return float(np.mean(~np.isnan(predicted)))


@dataclass(frozen=True)
class SliceMetrics:
    """Errors over one named subset of rows.

    Attributes:
        name: What the slice is.
        rows: Rows in the slice.
        mae: Mean absolute error.
        rmse: Root mean squared error.
        coverage: Fraction of rows the model answered.
        mean_actual: Mean observed value, so an error can be read relative to
            the level -- an MAE of 400 means different things on a six-lane
            arterial and a residential collector.
    """

    name: str
    rows: int
    mae: float
    rmse: float
    coverage: float
    mean_actual: float

    @property
    def relative_mae(self) -> float:
        """MAE as a fraction of the mean observed value."""
        if self.mean_actual == 0 or np.isnan(self.mean_actual):
            return float("nan")
        return self.mae / self.mean_actual

    def as_dict(self) -> dict[str, object]:
        """Serializable form, for the model registry."""
        return {
            "name": self.name,
            "rows": self.rows,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "relative_mae": round(self.relative_mae, 4),
            "coverage": round(self.coverage, 4),
            "mean_actual": round(self.mean_actual, 4),
        }


def evaluate_slice(
    name: str,
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
) -> SliceMetrics:
    """Compute metrics for one slice."""
    return SliceMetrics(
        name=name,
        rows=int(actual.size),
        mae=mae(actual, predicted),
        rmse=rmse(actual, predicted),
        coverage=coverage(actual, predicted),
        mean_actual=float(np.mean(actual)) if actual.size else float("nan"),
    )


#: A named row filter over a feature table.
SliceFn = Callable[[FeatureTable], NDArray[np.bool_]]


def zone_slice(location_id: str) -> SliceFn:
    """Rows for one zone."""

    def chooser(table: FeatureTable) -> NDArray[np.bool_]:
        return np.array([location == location_id for location in table.locations], dtype=bool)

    return chooser


def local_window_slice(
    location_id: str,
    *,
    weekday: int,
    start_hour: int,
    end_hour: int,
    timezone_offset_hours: float,
) -> SliceFn:
    """Rows whose **target** falls in a local weekday-and-hour window.

    Keyed on the target time rather than the sample time, because the question
    "how well does it predict the Friday evening peak?" is about when the peak
    happens, not when the prediction was made. At a 24h horizon those differ by a
    day, and slicing on the sample time would measure Thursday evenings.
    """

    def chooser(table: FeatureTable) -> NDArray[np.bool_]:
        flags: list[bool] = []
        for location, target_time in zip(table.locations, table.target_times, strict=True):
            local = target_time + timedelta(hours=timezone_offset_hours)
            flags.append(
                location == location_id
                and local.weekday() == weekday
                and start_hour <= local.hour < end_hour
            )
        return np.array(flags, dtype=bool)

    return chooser


@dataclass(frozen=True)
class EvaluationResult:
    """Every slice's metrics for one model on one dataset.

    Attributes:
        model_name: Which model produced the predictions.
        dataset_name: Which dataset it was evaluated on.
        overall: Metrics across all rows.
        slices: Metrics per named slice, in declaration order.
    """

    model_name: str
    dataset_name: str
    overall: SliceMetrics
    slices: tuple[SliceMetrics, ...] = ()

    def slice_named(self, name: str) -> SliceMetrics | None:
        """One slice's metrics by name."""
        return next((item for item in self.slices if item.name == name), None)

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "overall": self.overall.as_dict(),
            "slices": [item.as_dict() for item in self.slices],
        }


def evaluate(
    model_name: str,
    dataset_name: str,
    table: FeatureTable,
    predicted: NDArray[np.float64],
    *,
    slices: Sequence[tuple[str, SliceFn]] = (),
) -> EvaluationResult:
    """Evaluate predictions overall and per slice.

    Raises:
        ValueError: if the prediction count does not match the table. A silent
            misalignment would compare each prediction against the wrong row and
            still produce a plausible number.
    """
    if predicted.shape[0] != len(table):
        raise ValueError(
            f"{predicted.shape[0]} predictions for {len(table)} rows; refusing to "
            "evaluate a misaligned pair"
        )

    computed: list[SliceMetrics] = []
    for name, chooser in slices:
        mask = chooser(table)
        computed.append(evaluate_slice(name, table.y[mask], predicted[mask]))

    return EvaluationResult(
        model_name=model_name,
        dataset_name=dataset_name,
        overall=evaluate_slice("overall", table.y, predicted),
        slices=tuple(computed),
    )


def friday_evening_slices(
    table: FeatureTable, *, timezone_offset_hours: float
) -> list[tuple[str, SliceFn]]:
    """Standard slices: every zone, plus Friday 18:00-20:00 in each.

    The Friday evening window is the planted spatio-temporal pattern, so these
    slices are what show whether the model learned it or averaged it away.
    """
    zones = sorted(set(table.locations))
    named: list[tuple[str, SliceFn]] = [(f"zone {zone}", zone_slice(zone)) for zone in zones]
    named.extend(
        (
            f"{zone} Fri 18-20",
            local_window_slice(
                zone,
                weekday=4,
                start_hour=18,
                end_hour=20,
                timezone_offset_hours=timezone_offset_hours,
            ),
        )
        for zone in zones
    )
    return named


def date_of(moment: datetime) -> str:
    """Render an instant as a date, for report headings."""
    return moment.strftime("%Y-%m-%d")
