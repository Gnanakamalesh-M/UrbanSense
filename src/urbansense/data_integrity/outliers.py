"""Outlier detection (SPEC section 16).

Detection only. The spec is explicit that an outlier is not automatically an
error:

    Normal traffic = 5000, Observed = 20000

could be sensor failure, an accident, a festival, a road diversion, or a
genuine unusual event. Which one it is cannot be decided by looking at the
number, so this module **flags and keeps**. Nothing is deleted, nothing is
clipped, nothing is "corrected". Flagged observations stay in the usable stream
with ``validation_status=SUSPECT`` so a consumer can include or exclude them
deliberately.

Two design choices carry most of the weight:

**Median and MAD, not mean and standard deviation.** The mean is dragged by the
very outliers being looked for, so a few extreme values raise the threshold
until they look acceptable, while quietly making their neighbours look extreme.
The median absolute deviation is unmoved by a minority of extreme points.

**Conditioned on zone, metric, weekday and hour.** Traffic at 03:00 and traffic
at 18:00 are different populations, and so are Friday and Sunday. Comparing a
value against the wrong population is what makes a naive detector report the
*signal* as error: unconditioned, this would flag every Friday-evening peak and
every post-drift reading -- the two strongest real patterns in the data -- as
anomalies. Within its own weekday-and-hour group, a recurring peak is the norm,
which is exactly right: a pattern is not an anomaly.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from urbansense.schemas.observation import Observation

#: Deviations from the group median, in MAD units, before a value is flagged.
#: Deliberately conservative: a false "suspicious" label on real data costs
#: trust in the flag, and the flag is only useful if it is rare.
DEFAULT_THRESHOLD = 6.0

#: Minimum samples in a group before it can judge anything. With fewer, the
#: median and MAD are noise and would flag ordinary variation.
DEFAULT_MIN_GROUP_SIZE = 12

#: Scale factor making MAD comparable to a standard deviation for normally
#: distributed data, so the threshold reads like a sigma multiple.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True, slots=True)
class OutlierFlag:
    """One observation flagged as unusual but retained.

    Attributes:
        observation_id: The flagged observation.
        value: Its value.
        group_median: Median of its comparison group.
        group_mad: Scaled median absolute deviation of the group.
        deviation: How many MAD-sigmas the value sits from the median.
        group_size: Samples in the comparison group.
        detail: Human-readable explanation for the review note.
    """

    observation_id: str
    value: float
    group_median: float
    group_mad: float
    deviation: float
    group_size: int
    detail: str


def _group_key(observation: Observation) -> tuple[str, str, int, int]:
    """Comparison group: zone, metric, weekday and hour of day."""
    moment = observation.event_time
    return (
        observation.location_id,
        observation.metric,
        moment.weekday(),
        moment.hour,
    )


def detect_outliers(
    observations: Iterable[Observation],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> dict[str, OutlierFlag]:
    """Flag unusual values without altering or removing any of them.

    Args:
        observations: Observations to scan. Missing values are skipped -- an
            absent reading is not an unusual one.
        threshold: Deviations from the group median, in MAD units, before a
            value is flagged.
        min_group_size: Minimum group size before the group may judge.

    Returns:
        Flags keyed by ``observation_id``, for the observations judged unusual.
        An observation absent from the mapping was not flagged.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    groups: dict[tuple[str, str, int, int], list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.value is None:
            continue
        groups[_group_key(observation)].append(observation)

    flags: dict[str, OutlierFlag] = {}
    for key, group in groups.items():
        if len(group) < min_group_size:
            # Too few samples to distinguish an outlier from ordinary spread.
            # Saying nothing is the honest answer; guessing would manufacture
            # suspicion out of a small sample.
            continue

        values = [obs.value for obs in group if obs.value is not None]
        median = statistics.median(values)
        mad = statistics.median([abs(value - median) for value in values]) * _MAD_TO_SIGMA

        if mad == 0.0:
            # A group with no spread at all. Every value is identical, which is
            # a frozen-sensor signature rather than an outlier question -- the
            # frozen-run detector owns that case, so this one stays quiet
            # instead of flagging the whole group against itself.
            continue

        location_id, metric, weekday, hour = key
        for observation in group:
            assert observation.value is not None  # filtered above
            deviation = abs(observation.value - median) / mad
            if deviation <= threshold:
                continue
            flags[observation.observation_id] = OutlierFlag(
                observation_id=observation.observation_id,
                value=observation.value,
                group_median=median,
                group_mad=mad,
                deviation=deviation,
                group_size=len(group),
                detail=(
                    f"{metric} at {location_id} is {observation.value:g}, "
                    f"{deviation:.1f} MAD-sigma from the median {median:g} for "
                    f"weekday {weekday} hour {hour:02d} (n={len(group)}). "
                    "Unusual, not impossible: could be an incident, an event or a "
                    "sensor problem, so it is flagged for investigation and kept."
                ),
            )

    return flags
