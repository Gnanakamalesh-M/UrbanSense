"""Calendar features derived from the sample timestamp (SPEC section 21).

These encode the temporal patterns section 21 asks about -- hourly, daily,
weekly, monthly -- and they read **no observations at all**. That makes them the
only features that are leakage-free by construction rather than by checking:
there is no data for them to read the future from.

Hour and month are encoded as sine/cosine pairs rather than integers. Hour 23
and hour 0 are adjacent in reality but 23 apart as integers, and a tree asked to
split on that boundary has to spend two splits to say "late night". The cyclical
pair makes the adjacency geometric. Weekday is left as an integer because the
week's shape here is not smooth -- Saturday and Sunday differ sharply from
Friday, so there is no cycle worth preserving.

Timestamps are UTC throughout the system. Calendar features are computed in the
source's **local** time, because "Friday evening" and "the 18:00 peak" are
local-clock facts: a model given UTC hours would be looking for a pattern that
sits at 12:30 in its own frame and smears across two hour buckets.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from urbansense.features.specs import FeatureKind, FeatureSpec

#: Saturday and Sunday, as Python weekday numbers (Monday is 0).
WEEKEND_DAYS = frozenset({5, 6})


def to_local(moment: datetime, offset_hours: float) -> datetime:
    """Shift a UTC instant into the source's local wall clock.

    Returned naive, because what matters downstream is the local hour and
    weekday rather than the instant, and carrying a tzinfo here would invite
    a second conversion.
    """
    return (moment + timedelta(hours=offset_hours)).replace(tzinfo=None)


def hour_sin(local: datetime) -> float:
    """Sine component of the hour-of-day cycle."""
    return math.sin(2.0 * math.pi * local.hour / 24.0)


def hour_cos(local: datetime) -> float:
    """Cosine component of the hour-of-day cycle."""
    return math.cos(2.0 * math.pi * local.hour / 24.0)


def month_sin(local: datetime) -> float:
    """Sine component of the month-of-year cycle."""
    return math.sin(2.0 * math.pi * (local.month - 1) / 12.0)


def month_cos(local: datetime) -> float:
    """Cosine component of the month-of-year cycle."""
    return math.cos(2.0 * math.pi * (local.month - 1) / 12.0)


def hour_of_day(local: datetime) -> float:
    """Local hour, kept alongside the cyclical pair.

    Trees can split a raw ordinal cleanly, and the planted pattern is a sharp
    18:00-20:00 window rather than a smooth curve, so the plain hour is the more
    useful encoding for exactly the effect being measured. The cyclical pair
    handles the midnight wrap that this one cannot.
    """
    return float(local.hour)


def weekday(local: datetime) -> float:
    """Local day of week, Monday is 0."""
    return float(local.weekday())


def is_weekend(local: datetime) -> float:
    """Whether the local day is Saturday or Sunday."""
    return 1.0 if local.weekday() in WEEKEND_DAYS else 0.0


def month_of_year(local: datetime) -> float:
    """Local month, 1 to 12."""
    return float(local.month)


#: Calendar features, by name. Each reads only the sample timestamp.
CALENDAR_FEATURES: dict[str, tuple[str, object]] = {
    "hour": ("local hour of day, 0-23", hour_of_day),
    "hour_sin": ("sine of the hour cycle, so 23:00 and 00:00 are adjacent", hour_sin),
    "hour_cos": ("cosine of the hour cycle", hour_cos),
    "weekday": ("local day of week, Monday is 0", weekday),
    "is_weekend": ("1 on Saturday or Sunday", is_weekend),
    "month": ("local month, 1-12", month_of_year),
    "month_sin": ("sine of the month cycle, for seasonality", month_sin),
    "month_cos": ("cosine of the month cycle", month_cos),
}


def calendar_specs(names: tuple[str, ...]) -> tuple[FeatureSpec, ...]:
    """Build specs for the named calendar features.

    Raises:
        KeyError: if a name is not a known calendar feature. Silently dropping an
            unknown name would train a model on fewer features than the config
            asked for, and nothing would say so.
    """
    specs: list[FeatureSpec] = []
    for name in names:
        if name not in CALENDAR_FEATURES:
            raise KeyError(
                f"unknown calendar feature {name!r}; known features are {sorted(CALENDAR_FEATURES)}"
            )
        description, _ = CALENDAR_FEATURES[name]
        specs.append(
            FeatureSpec(
                name=name,
                kind=FeatureKind.CALENDAR,
                # Zero offset: derived from t itself, reading no observations.
                offset_hours=0,
                window_hours=0,
                description=description,
            )
        )
    return tuple(specs)


def compute_calendar(name: str, local: datetime) -> float:
    """Compute one calendar feature for a local timestamp."""
    _, function = CALENDAR_FEATURES[name]
    return float(function(local))  # type: ignore[operator]
