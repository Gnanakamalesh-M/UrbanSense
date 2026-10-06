"""Deterministic signal functions for the synthetic generator (SPEC section 37).

These build the traffic series as a product of interpretable factors:

    volume = base x diurnal x weekly x seasonal x friday_boost x rain x drift x noise

Keeping each factor a separate pure function is what makes the dataset
*verifiable*: a test can assert the Friday factor is present at roughly its
configured strength, because that factor is the only thing that touches those
hours. A single opaque expression would make the planted truth unmeasurable.

Everything is stdlib ``math`` plus an injected ``random.Random``, so output
depends only on the seed -- no numpy, and no reliance on global random state.
"""

from __future__ import annotations

import math
from datetime import date, datetime

from urbansense.synthetic.config import (
    DriftConfig,
    FridayEveningConfig,
    GeneratorConfig,
    RainEffectConfig,
    ZoneConfig,
)

#: Relative traffic level for each hour of the day, index 0..23.
#: Two commute peaks around 09:00 and 18:00, a deep overnight trough. Expressed
#: as an explicit table rather than a sum of sinusoids so the shape is readable
#: and a test can point at a specific hour.
# fmt: off
_DIURNAL_SHAPE: tuple[float, ...] = (
    0.18, 0.12, 0.09, 0.08, 0.11, 0.22,  # 00-05
    0.45, 0.72, 0.95, 1.00, 0.86, 0.78,  # 06-11
    0.80, 0.82, 0.80, 0.84, 0.92, 1.00,  # 12-17
    0.98, 0.88, 0.70, 0.52, 0.36, 0.25,  # 18-23
)
# fmt: on

#: Weekday multipliers, Monday=0 .. Sunday=6. Weekends are quieter, Friday
#: slightly busier than a mid-week day even before the evening pattern.
_WEEKLY_SHAPE: tuple[float, ...] = (0.98, 1.00, 1.01, 1.03, 1.06, 0.78, 0.66)


def diurnal_factor(hour: int) -> float:
    """Relative traffic level for a local hour of day."""
    return _DIURNAL_SHAPE[hour % 24]


def weekly_factor(weekday: int) -> float:
    """Relative traffic level for a day of week (Monday=0)."""
    return _WEEKLY_SHAPE[weekday % 7]


def seasonal_factor(day_of_year: int) -> float:
    """Mild annual seasonality, peaking in the second half of the year.

    Amplitude is deliberately small (+/-6%): large enough to be real, small
    enough that it cannot be mistaken for the planted drift.
    """
    return 1.0 + 0.06 * math.sin(2.0 * math.pi * (day_of_year - 80) / 365.25)


def friday_evening_factor(moment: datetime, location_id: str, config: FridayEveningConfig) -> float:
    """Volume multiplier for the Friday-evening congestion window.

    Returns 1.0 everywhere outside the configured zone and hours, so this
    factor is the sole cause of the pattern and nothing else leaks into it.
    """
    if not config.enabled or location_id != config.location_id:
        return 1.0
    if moment.weekday() != config.weekday:
        return 1.0
    if not (config.start_hour <= moment.hour < config.end_hour):
        return 1.0
    return config.volume_multiplier


def friday_evening_speed_factor(
    moment: datetime, location_id: str, config: FridayEveningConfig
) -> float:
    """Speed multiplier inside the congestion window.

    Congestion means more vehicles *and* slower ones; a dataset where volume
    rose while speed held constant would be physically incoherent.
    """
    if friday_evening_factor(moment, location_id, config) == 1.0:
        return 1.0
    return config.speed_multiplier


def rain_volume_factor(rain_mm: float, config: RainEffectConfig) -> float:
    """Volume multiplier for a given hour's rainfall.

    The effect saturates at ``cap_mm``: the first few millimetres change
    behaviour, a cloudburst does not scale without limit.
    """
    if not config.enabled or rain_mm <= 0.0:
        return 1.0
    return 1.0 + config.volume_coefficient * min(rain_mm, config.cap_mm)


def rain_speed_factor(rain_mm: float, config: RainEffectConfig) -> float:
    """Speed multiplier for a given hour's rainfall."""
    if not config.enabled or rain_mm <= 0.0:
        return 1.0
    return max(0.1, 1.0 - config.speed_coefficient * min(rain_mm, config.cap_mm))


def drift_factor(moment_date: date, location_id: str, config: DriftConfig) -> float:
    """Volume multiplier for the permanent post-drift baseline shift.

    Applies from ``start_date`` onward and never reverts -- this models the city
    changing, not a transient anomaly.
    """
    if not config.enabled or location_id not in config.location_ids:
        return 1.0
    if moment_date < config.start_date:
        return 1.0
    return config.volume_multiplier


def compose_volume(
    *,
    moment: datetime,
    zone: ZoneConfig,
    rain_mm: float,
    config: GeneratorConfig,
    noise: float,
) -> float:
    """Compute one traffic-volume observation from its factors.

    Args:
        moment: Local timestamp of the interval start.
        zone: Zone being generated.
        rain_mm: Rainfall for this hour, in mm.
        config: Full generator configuration.
        noise: Multiplicative noise factor, supplied by the caller so all
            randomness stays in one seeded place.

    Returns:
        Vehicles per hour, never negative.
    """
    value = (
        zone.base_volume
        * diurnal_factor(moment.hour)
        * weekly_factor(moment.weekday())
        * seasonal_factor(moment.timetuple().tm_yday)
        * friday_evening_factor(moment, zone.location_id, config.friday_evening)
        * rain_volume_factor(rain_mm, config.rain_effect)
        * drift_factor(moment.date(), zone.location_id, config.drift)
        * noise
    )
    return max(0.0, value)


def compose_speed(
    *,
    moment: datetime,
    zone: ZoneConfig,
    rain_mm: float,
    config: GeneratorConfig,
    noise: float,
    volume: float,
) -> float:
    """Compute one average-speed observation.

    Speed falls as volume approaches the zone's capacity, and falls further in
    rain and inside the congestion window. Clamped to a plausible road range so
    the generator never emits an impossible speed by accident -- the impossible
    speeds in the dataset are planted deliberately and recorded in ground truth.
    """
    load = volume / max(zone.base_volume, 1.0)
    congestion_penalty = 1.0 / (1.0 + 0.35 * max(0.0, load - 0.6))
    value = (
        zone.base_speed
        * congestion_penalty
        * friday_evening_speed_factor(moment, zone.location_id, config.friday_evening)
        * rain_speed_factor(rain_mm, config.rain_effect)
        * noise
    )
    return min(zone.base_speed * 1.25, max(3.0, value))
