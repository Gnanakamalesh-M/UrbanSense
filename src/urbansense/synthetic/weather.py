"""Deterministic synthetic rainfall (SPEC sections 21, 37).

Rain is generated as its own series and emitted as its own observations, with
its own source. It is a **covariate, not a label**: nothing in the traffic rows
says "it rained here". A later phase that wants the conditional pattern
``rain + Friday evening -> congestion`` has to join the two series itself,
which is the discovery we actually want to test.

The series is wet-spell based rather than independent per hour, because real
rain arrives in spells and an hour-independent series would make the rainfall
effect trivially separable from everything else.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta

#: Relative monthly rainfall weight, index 0 = January. Shaped like a monsoon
#: climate (heavy October-December), so wet and dry seasons both appear.
# fmt: off
_MONTHLY_WEIGHT: tuple[float, ...] = (
    0.25, 0.15, 0.20, 0.35, 0.55, 0.70,  # Jan-Jun
    0.75, 0.80, 0.95, 1.60, 1.80, 0.90,  # Jul-Dec
)
# fmt: on


def _spell_probability(moment: date) -> float:
    """Chance that a new wet spell starts on a given day."""
    return 0.06 * _MONTHLY_WEIGHT[moment.month - 1]


def generate_rainfall(
    *,
    start: datetime,
    end: datetime,
    interval: timedelta,
    rng: random.Random,
) -> dict[datetime, float]:
    """Generate an hourly rainfall series in mm.

    Args:
        start: First local timestamp, inclusive.
        end: Last local timestamp, exclusive.
        interval: Spacing between observations.
        rng: Seeded generator; all randomness comes from here so the series is
            reproducible from the dataset seed alone.

    Returns:
        Mapping of local timestamp to millimetres of rain in that interval.
        Dry intervals are present with a value of ``0.0`` -- a dry hour is a
        measurement of no rain, which is not the same as a missing reading.
    """
    series: dict[datetime, float] = {}
    moment = start
    #: Intervals remaining in the current wet spell.
    spell_remaining = 0
    spell_intensity = 0.0

    while moment < end:
        if spell_remaining == 0 and rng.random() < _spell_probability(moment.date()):
            # Spells last a few hours; intensity is skewed so that most spells
            # are light drizzle and heavy downpours are rare but present.
            spell_remaining = rng.randint(2, 14)
            spell_intensity = rng.lognormvariate(0.2, 0.9)

        if spell_remaining > 0:
            # Within a spell, intensity rises and falls rather than being flat.
            shape = math.sin(math.pi * (1.0 - spell_remaining / 15.0))
            rain = max(0.0, spell_intensity * (0.4 + shape) * rng.uniform(0.6, 1.4))
            spell_remaining -= 1
        else:
            rain = 0.0

        series[moment] = round(rain, 2)
        moment += interval

    return series


def wet_dry_counts(series: dict[datetime, float], *, threshold: float = 0.2) -> tuple[int, int]:
    """Count wet and dry intervals in a rainfall series.

    Args:
        series: Output of :func:`generate_rainfall`.
        threshold: Millimetres above which an interval counts as wet. A trace
            of rain does not change driving behaviour.

    Returns:
        ``(wet_count, dry_count)``.
    """
    wet = sum(1 for value in series.values() if value > threshold)
    return wet, len(series) - wet
