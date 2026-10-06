"""Where the model fails (SPEC section 28).

    Analyze where the model fails.
    Overall MAE: 8.4 / Heavy rain: 16.7 / Normal weather: 6.1 / Evening: 11.8

An aggregate MAE is the least useful number a model reports: it is an average
over conditions that behave nothing like each other, and a model can look
healthy overall while being useless exactly when someone needs it. So the
breakdown is the product here and the headline is the footnote.

Slices are cut from the :class:`~urbansense.evaluation.outcomes.Outcome` log
rather than recomputed from a feature table, because the log is what records the
conditions that actually held during the predicted hour.

Error maths is **not** reimplemented: every cell goes through
:func:`~urbansense.evaluation.metrics.evaluate_slice`, so a slice and the
overall figure cannot drift apart through two implementations of a mean.

Two reporting rules, both aimed at the same failure mode -- a report nobody
distrusts:

- **Small slices are labelled.** An MAE over nine rows is a rumour, and it is
  printed with its row count so it reads as one.
- **The weakest slices are ranked and printed**, not buried. A report that
  showed only the favourable cuts would be the cherry-picking section 38 warns
  about.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from urbansense.evaluation.metrics import SliceMetrics, evaluate_slice
from urbansense.evaluation.outcomes import Outcome

#: Below this many rows a slice is flagged as too thin to read much into.
THIN_SLICE_ROWS = 30

#: Weekday names for the weekday breakdown, Monday first.
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _metrics(name: str, rows: Sequence[Outcome]) -> SliceMetrics:
    """Error metrics for a group of outcomes.

    Delegates to the Phase 4 slice function so there is exactly one definition
    of MAE in the codebase.
    """
    actual = np.array([row.actual for row in rows], dtype=np.float64)
    predicted = np.array([row.predicted for row in rows], dtype=np.float64)
    return evaluate_slice(name, actual, predicted)


@dataclass(frozen=True)
class Breakdown:
    """Error metrics across one way of cutting the outcomes.

    Attributes:
        dimension: What the rows were grouped by.
        slices: One entry per group, ordered by the grouping's natural order.
    """

    dimension: str
    slices: tuple[SliceMetrics, ...]

    def named(self, name: str) -> SliceMetrics | None:
        """One slice by name."""
        return next((item for item in self.slices if item.name == name), None)

    @property
    def names(self) -> tuple[str, ...]:
        """Slice names in this breakdown."""
        return tuple(item.name for item in self.slices)

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "dimension": self.dimension,
            "slices": [item.as_dict() for item in self.slices],
        }

    def render(self) -> str:
        """A table, with row counts so thin slices read as thin."""
        lines = [
            f"{self.dimension}",
            f"  {'slice':<22}{'rows':>7}{'MAE':>12}{'RMSE':>12}{'rel MAE':>10}  ",
        ]
        for item in self.slices:
            flag = "  (thin)" if item.rows < THIN_SLICE_ROWS else ""
            lines.append(
                f"  {item.name:<22}{item.rows:>7}{item.mae:>12,.1f}"
                f"{item.rmse:>12,.1f}{item.relative_mae:>9.1%}{flag}"
            )
        return "\n".join(lines)


def _group(
    outcomes: Sequence[Outcome],
    dimension: str,
    key: Callable[[Outcome], str],
    *,
    order: Sequence[str] | None = None,
) -> Breakdown:
    """Group outcomes by a key and compute metrics per group.

    Empty groups are omitted rather than shown as zero rows: "we never predicted
    in heavy rain" and "we predicted perfectly in heavy rain" are opposite
    findings, and a zero row reads as the second.
    """
    buckets: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(key(outcome), []).append(outcome)

    names = [name for name in (order or sorted(buckets))] if order else sorted(buckets)
    return Breakdown(
        dimension=dimension,
        slices=tuple(_metrics(name, buckets[name]) for name in names if name in buckets),
    )


def by_rainfall(outcomes: Sequence[Outcome]) -> Breakdown:
    """MAE by rainfall band.

    ``unknown`` is a band of its own. Folding hours with no gauge reading into
    ``dry`` would be treating missing as zero.
    """
    return _group(
        outcomes,
        "by rainfall",
        lambda item: item.rain_band,
        order=("dry", "light", "heavy", "unknown"),
    )


def by_time_of_day(outcomes: Sequence[Outcome]) -> Breakdown:
    """MAE by time of day, in the source's local clock."""
    return _group(
        outcomes,
        "by time of day",
        lambda item: item.time_of_day,
        order=("night", "morning", "afternoon", "evening"),
    )


def by_zone(outcomes: Sequence[Outcome]) -> Breakdown:
    """MAE per zone."""
    return _group(outcomes, "by zone", lambda item: item.location_id)


def by_weekday(outcomes: Sequence[Outcome]) -> Breakdown:
    """MAE per local weekday."""
    return _group(
        outcomes,
        "by weekday",
        lambda item: WEEKDAY_NAMES[item.weekday],
        order=WEEKDAY_NAMES,
    )


def by_friday_evening(outcomes: Sequence[Outcome]) -> Breakdown:
    """The planted Friday 18:00-20:00 window against everything else."""
    return _group(
        outcomes,
        "Friday evening window",
        lambda item: "Friday 18-20" if item.is_friday_evening else "all other hours",
        order=("Friday 18-20", "all other hours"),
    )


def by_severity(outcomes: Sequence[Outcome]) -> Breakdown:
    """MAE by the severity the prediction claimed.

    Worth cutting because a model that is accurate when it says "low" and wild
    when it says "severe" is more dangerous than its average suggests.
    """
    return _group(
        outcomes,
        "by predicted severity",
        lambda item: item.severity.value if item.severity else "unscored",
    )


@dataclass(frozen=True)
class ErrorReport:
    """The full breakdown for one dataset.

    Attributes:
        dataset: Which split or period this covers.
        model_version: Which model produced the predictions.
        overall: Metrics across every row.
        breakdowns: Each way of cutting the rows.
        rows: Outcomes analysed.
    """

    dataset: str
    model_version: str
    overall: SliceMetrics
    breakdowns: tuple[Breakdown, ...]
    rows: int

    def breakdown(self, dimension: str) -> Breakdown | None:
        """One breakdown by name."""
        return next((item for item in self.breakdowns if item.dimension == dimension), None)

    @property
    def all_slices(self) -> tuple[SliceMetrics, ...]:
        """Every slice from every breakdown."""
        return tuple(item for breakdown in self.breakdowns for item in breakdown.slices)

    def weakest(
        self, limit: int = 5, *, min_rows: int = THIN_SLICE_ROWS
    ) -> tuple[SliceMetrics, ...]:
        """Slices with the worst relative error.

        Ranked by relative MAE rather than MAE, because a 400-vehicle error
        means something different on a six-lane arterial and a quiet collector.
        Slices thinner than ``min_rows`` are excluded from the ranking -- not
        hidden, they still appear in their own table, but a nine-row slice
        topping the list would be noise presented as a finding.
        """
        eligible = [
            item
            for item in self.all_slices
            if item.rows >= min_rows and not np.isnan(item.relative_mae)
        ]
        eligible.sort(key=lambda item: -item.relative_mae)
        return tuple(eligible[:limit])

    def as_dict(self) -> dict[str, object]:
        """Serializable form, for the registry."""
        return {
            "dataset": self.dataset,
            "model_version": self.model_version,
            "rows": self.rows,
            "overall": self.overall.as_dict(),
            "breakdowns": [item.as_dict() for item in self.breakdowns],
            "weakest_slices": [item.as_dict() for item in self.weakest()],
        }

    def render(self) -> str:
        """The whole report, breakdowns first and the headline last."""
        lines = [
            f"error analysis: {self.dataset} ({self.rows:,} outcomes, model {self.model_version})",
            "",
        ]
        for breakdown in self.breakdowns:
            lines.append(breakdown.render())
            lines.append("")

        lines.append("weakest slices by relative error")
        weakest = self.weakest()
        if not weakest:
            lines.append("  none with enough rows to rank")
        for item in weakest:
            lines.append(
                f"  {item.name:<22}{item.rows:>7}{item.mae:>12,.1f}{item.relative_mae:>9.1%}"
            )
        lines.append(
            f"  (slices under {THIN_SLICE_ROWS} rows are excluded from this ranking; "
            "they still appear in their own tables)"
        )
        lines.append("")
        lines.append(
            f"overall MAE {self.overall.mae:,.1f} "
            f"({self.overall.relative_mae:.1%} of the mean observed "
            f"{self.overall.mean_actual:,.0f}) -- an average over conditions that "
            "behave nothing like each other, which is why the tables above come first"
        )
        return "\n".join(lines)


def analyse_errors(
    outcomes: Sequence[Outcome],
    *,
    dataset: str,
    model_version: str = "",
) -> ErrorReport:
    """Break prediction errors down every way SPEC section 28 asks for.

    Raises:
        ValueError: if there are no outcomes. An empty report would print a row
            of NaNs that reads like a healthy result.
    """
    if not outcomes:
        raise ValueError("cannot analyse errors with no outcomes")

    version = model_version or outcomes[0].model_version
    return ErrorReport(
        dataset=dataset,
        model_version=version,
        overall=_metrics("overall", outcomes),
        breakdowns=(
            by_rainfall(outcomes),
            by_time_of_day(outcomes),
            by_zone(outcomes),
            by_weekday(outcomes),
            by_friday_evening(outcomes),
            by_severity(outcomes),
        ),
        rows=len(outcomes),
    )
