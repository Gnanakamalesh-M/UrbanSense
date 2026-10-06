"""Temporal splits (SPEC section 33).

    Temporal splits only; no future leakage. Train on the past, test on the
    future. No random shuffling of time series.

There is deliberately **no seed parameter** anywhere in this module. A random
split of a time series is not a weaker evaluation, it is a meaningless one: with
`lag_1h` as a feature, a shuffled split puts hour 14 in train and hour 15 in
test, so the model is asked to predict a value it has effectively already seen.
The resulting score is excellent and tells you nothing. Making the shuffled path
unavailable is cheaper than remembering not to use it.

**The gap.** Each later segment starts a configurable number of hours after the
previous one ends, and those hours belong to no segment. Without it, a test
sample at the boundary reads lags stretching back into the training period, and
its target may be a value the model saw as a label. The default gap is the
furthest a feature reaches or the horizon, whichever is larger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from urbansense.features.builder import FeatureTable


@dataclass(frozen=True)
class SplitBounds:
    """The time range one segment covers.

    Attributes:
        name: Segment name.
        start: First sample time in the segment.
        end: Last sample time in the segment.
        rows: Number of samples.
    """

    name: str
    start: datetime | None
    end: datetime | None
    rows: int

    def as_dict(self) -> dict[str, object]:
        """Serializable form, for the model registry."""
        return {
            "name": self.name,
            "start": None if self.start is None else self.start.isoformat(),
            "end": None if self.end is None else self.end.isoformat(),
            "rows": self.rows,
        }

    def describe(self) -> str:
        """Human-readable range."""
        if self.start is None or self.end is None:
            return f"{self.name}: empty"
        return (
            f"{self.name}: {self.start:%Y-%m-%d %H:%M} .. {self.end:%Y-%m-%d %H:%M} "
            f"({self.rows:,} rows)"
        )


@dataclass(frozen=True)
class TemporalSplit:
    """Train, validation and test tables in time order.

    Attributes:
        train: Earliest segment.
        validation: Middle segment, for tuning and the residual bands.
        test: Latest segment, touched once.
        gap_hours: Hours excluded between consecutive segments.
        excluded_rows: Samples that fell inside a gap and belong to no segment.
    """

    train: FeatureTable
    validation: FeatureTable
    test: FeatureTable
    gap_hours: int
    excluded_rows: int

    @property
    def bounds(self) -> tuple[SplitBounds, SplitBounds, SplitBounds]:
        """Ranges of the three segments."""
        return (
            _bounds("train", self.train),
            _bounds("validation", self.validation),
            _bounds("test", self.test),
        )

    @property
    def is_strictly_ordered(self) -> bool:
        """Whether every train sample precedes every validation sample, and so on.

        The property the whole module exists to guarantee. Asserted by the tests
        rather than assumed from the construction.
        """
        train, validation, test = self.bounds
        if train.end is None or validation.start is None:
            return False
        if validation.end is None or test.start is None:
            return False
        return train.end < validation.start and validation.end < test.start

    def gap_before_validation(self) -> timedelta | None:
        """Actual gap between the train and validation segments."""
        train, validation, _ = self.bounds
        if train.end is None or validation.start is None:
            return None
        return validation.start - train.end

    def gap_before_test(self) -> timedelta | None:
        """Actual gap between the validation and test segments."""
        _, validation, test = self.bounds
        if validation.end is None or test.start is None:
            return None
        return test.start - validation.end

    def describe(self) -> str:
        """Human-readable summary of the split."""
        lines = [f"temporal split (gap {self.gap_hours}h, never shuffled):"]
        lines.extend(f"  {bound.describe()}" for bound in self.bounds)
        before_validation = self.gap_before_validation()
        before_test = self.gap_before_test()
        if before_validation is not None:
            lines.append(f"  gap train -> validation: {before_validation}")
        if before_test is not None:
            lines.append(f"  gap validation -> test:  {before_test}")
        lines.append(f"  excluded by gaps: {self.excluded_rows:,} rows")
        return "\n".join(lines)


def _bounds(name: str, table: FeatureTable) -> SplitBounds:
    """Summarize one table's time range."""
    if not table.sample_times:
        return SplitBounds(name=name, start=None, end=None, rows=0)
    return SplitBounds(
        name=name,
        start=min(table.sample_times),
        end=max(table.sample_times),
        rows=len(table),
    )


def default_gap_hours(table: FeatureTable) -> int:
    """The gap a feature table needs: its furthest reach or its horizon.

    A test sample reaching back ``max_reach_hours`` must not touch the training
    period, and its target sits ``horizon_hours`` ahead. Whichever is larger is
    the span over which the two segments could otherwise overlap.
    """
    feature_set = table.feature_set
    return max(feature_set.max_reach_hours, feature_set.horizon_hours)


def temporal_split(
    table: FeatureTable,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    gap_hours: int | None = None,
) -> TemporalSplit:
    """Split a feature table by time, never at random.

    Boundaries are chosen on the ordered distinct sample times, so every zone is
    split at the same instants. Splitting each zone independently would put the
    same hour in train for one zone and test for another, and a cross-zone
    feature would then straddle the boundary.

    Args:
        table: The feature table to split.
        train_fraction: Share of the time span used for training.
        validation_fraction: Share used for validation. The remainder, minus the
            gaps, is the test segment.
        gap_hours: Hours to exclude between segments. Defaults to
            :func:`default_gap_hours`.

    Returns:
        A :class:`TemporalSplit` in time order.

    Raises:
        ValueError: if the fractions are not a sensible division, or if a
            segment would be empty. An empty test set would make every metric
            vacuous while still producing a number.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be between 0 and 1, got {validation_fraction}")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            f"train_fraction ({train_fraction}) + validation_fraction "
            f"({validation_fraction}) must leave room for a test segment"
        )
    if len(table) == 0:
        raise ValueError("cannot split an empty feature table")

    gap = gap_hours if gap_hours is not None else default_gap_hours(table)
    if gap < 0:
        raise ValueError(f"gap_hours must not be negative, got {gap}")

    ordered = sorted(set(table.sample_times))
    count = len(ordered)
    train_end = ordered[max(0, int(count * train_fraction) - 1)]
    validation_end = ordered[max(0, int(count * (train_fraction + validation_fraction)) - 1)]

    offset = timedelta(hours=gap)
    validation_start = train_end + offset
    test_start = validation_end + offset

    times = np.array(table.sample_times, dtype=object)
    train_mask = np.array([moment <= train_end for moment in times], dtype=bool)
    validation_mask = np.array(
        [validation_start <= moment <= validation_end for moment in times], dtype=bool
    )
    test_mask = np.array([moment >= test_start for moment in times], dtype=bool)

    split = TemporalSplit(
        train=table.select(train_mask),
        validation=table.select(validation_mask),
        test=table.select(test_mask),
        gap_hours=gap,
        excluded_rows=int(len(table) - train_mask.sum() - validation_mask.sum() - test_mask.sum()),
    )

    for name, segment in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if len(segment) == 0:
            raise ValueError(
                f"the {name} segment is empty after applying a {gap}h gap to "
                f"{count} distinct sample times. Widen the data or narrow the gap "
                "-- an empty segment still produces a metric, and that metric "
                "would be meaningless."
            )

    return split
