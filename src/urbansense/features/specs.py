"""Feature declarations with explicit backward offsets (SPEC section 33).

Every feature states, as data, how far back in time it reads. That is the first
of two independent leakage defences: a spec whose offset is negative cannot be
constructed, so a forward-looking feature cannot be declared through the normal
path at all.

It is deliberately only the *first* defence. A declaration describes intent, and
a hand-rolled builder could ignore it, so
:mod:`urbansense.features.leakage` verifies the actual behaviour by perturbing
the future and checking that no feature value moves. Declarations catch the
careless mistake; the probe catches the determined one.

``offset_hours`` is how far before the sample time ``t`` the feature reads, and
``window_hours`` is how far back the window extends from there. Both are
non-negative, so the furthest any feature can reach is
``t - offset_hours - window_hours`` and the nearest is ``t``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum


class FeatureKind(StrEnum):
    """What a feature is derived from."""

    #: Computed from the sample timestamp alone. Reads no observations.
    CALENDAR = "calendar"
    #: A single past value of the target series.
    LAG = "lag"
    #: An aggregate over a past window of the target series.
    ROLLING = "rolling"
    #: A value from another metric, such as the rainfall covariate.
    COVARIATE = "covariate"
    #: A property of the location itself.
    SPATIAL = "spatial"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One declared feature.

    Attributes:
        name: Column name in the feature table.
        kind: What the feature is derived from.
        offset_hours: Hours before ``t`` that the feature starts reading. Zero
            means it may read the observation at ``t`` itself, which is legal --
            ``t`` is the prediction *time*, not the target time.
        window_hours: Hours the reading window extends back from the offset.
            Zero for a point lookup.
        metric: Metric read, for lag, rolling and covariate features.
        location: Fixed location to read from, for city-wide covariates. ``None``
            means the sample's own zone.
        description: Why the feature exists, for the registry record.

    Raises:
        ValueError: if either offset is negative. A negative offset is a
            forward-looking feature, which is the thing SPEC section 33 forbids,
            so it fails at construction rather than at review.
    """

    name: str
    kind: FeatureKind
    offset_hours: int = 0
    window_hours: int = 0
    metric: str | None = None
    location: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Reject a feature that would read the future."""
        if self.offset_hours < 0:
            raise ValueError(
                f"feature {self.name!r} declares offset_hours={self.offset_hours}, "
                "which reads the future. A feature at time t may only use data at "
                "or before t (SPEC section 33)."
            )
        if self.window_hours < 0:
            raise ValueError(
                f"feature {self.name!r} declares window_hours={self.window_hours}; "
                "a window extends backwards, so it cannot be negative."
            )
        if not self.name:
            raise ValueError("a feature must have a name")

    @property
    def reach_hours(self) -> int:
        """Furthest number of hours before ``t`` this feature reads."""
        return self.offset_hours + self.window_hours

    def earliest(self, sample_time: datetime) -> datetime:
        """Earliest instant this feature may read for a given sample time."""
        return sample_time - timedelta(hours=self.reach_hours)

    def latest(self, sample_time: datetime) -> datetime:
        """Latest instant this feature may read. Never after ``sample_time``."""
        return sample_time - timedelta(hours=self.offset_hours)


#: Calendar feature builders, keyed by name. Each takes the sample time and
#: returns a float. They read no observations at all, which is why their offset
#: is zero and the leakage probe can never flag them.
CalendarFn = Callable[[datetime], float]


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """The ordered set of features a model is trained on.

    Order is fixed and explicit, because the design matrix is positional: a
    column reordered between training and prediction would silently feed the
    model the wrong variable.

    Attributes:
        specs: The features, in column order.
        target_metric: Metric being predicted.
        horizon_hours: How far ahead the target sits from the sample time.
    """

    specs: tuple[FeatureSpec, ...]
    target_metric: str
    horizon_hours: int

    def __post_init__(self) -> None:
        """Reject duplicate column names and a non-positive horizon."""
        names = [spec.name for spec in self.specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate feature names: {duplicates}")
        if not self.specs:
            raise ValueError("a feature set needs at least one feature")
        if self.horizon_hours <= 0:
            raise ValueError(
                f"horizon_hours must be positive, got {self.horizon_hours}; "
                "predicting the present or the past is not forecasting"
            )

    @property
    def names(self) -> tuple[str, ...]:
        """Column names, in order."""
        return tuple(spec.name for spec in self.specs)

    @property
    def max_reach_hours(self) -> int:
        """Furthest any feature reaches back. Sets the warm-up period."""
        return max(spec.reach_hours for spec in self.specs)

    def describe(self) -> list[dict[str, object]]:
        """Serializable description, for the model registry."""
        return [
            {
                "name": spec.name,
                "kind": spec.kind.value,
                "offset_hours": spec.offset_hours,
                "window_hours": spec.window_hours,
                "metric": spec.metric,
                "location": spec.location,
                "description": spec.description,
            }
            for spec in self.specs
        ]
