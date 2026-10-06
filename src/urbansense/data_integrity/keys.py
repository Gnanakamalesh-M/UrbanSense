"""What "the same observation" means (SPEC sections 9, 10).

Before two records can be called duplicates or conflicts, they have to be
describing the same thing. SPEC section 10 gives the checklist directly:

    Same timestamp? Same location? Same metric? Same unit?
    Same measurement duration? Same spatial coverage? Same aggregation?

If any of those differ, the records are **different observations** and are kept
separately -- not reconciled, not compared, not merged. That is why the key is
a single explicit tuple rather than an ad-hoc comparison scattered through the
detectors: a field accidentally omitted from the key would silently merge
records that measure different things, and nothing downstream could detect it
afterwards because the evidence would be gone.

``value`` and ``source_id`` are deliberately **not** part of this key. They are
what distinguishes the three outcomes *within* a group: agreeing values from one
source are a repeat, agreeing values from two sources are corroboration, and
disagreeing values are a conflict. Putting them in the key would make a conflict
unrecognizable, because the two disagreeing records would land in different
groups and never be compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from urbansense.schemas.enums import AggregationLevel
from urbansense.schemas.observation import Observation


@dataclass(frozen=True, slots=True)
class ObservationKey:
    """Identifies the phenomenon an observation measures.

    Two observations sharing this key are claims about the same measurement;
    two that differ in any component are claims about different measurements.

    Attributes:
        location_id: Canonical zone, after location normalization. Comparing
            un-normalized names would miss duplicates reported under different
            spellings of the same zone.
        event_time: When the measured interval began, in UTC and snapped to the
            interval boundary by ingestion, so sensor clock skew cannot split
            one hour's readings across two keys.
        metric: Canonical metric name.
        unit: Canonical unit. Part of the key because 2050 vehicles/15min and
            8200 vehicles/hour are the same road but not the same *claim*;
            normalization is what brings them into one key, and until it has run
            they must not be compared (SPEC section 11).
        measurement_duration: Interval the value covers. A 15-minute count and
            an hourly count at the same instant measure different things.
        aggregation_level: Temporal granularity the value represents.
    """

    location_id: str
    event_time: datetime
    metric: str
    unit: str
    measurement_duration: timedelta | None
    aggregation_level: AggregationLevel

    @classmethod
    def of(cls, observation: Observation) -> ObservationKey:
        """Build the key for an observation."""
        return cls(
            location_id=observation.location_id,
            event_time=observation.event_time,
            metric=observation.metric,
            unit=observation.unit,
            measurement_duration=observation.measurement_duration,
            aggregation_level=observation.aggregation_level,
        )

    def describe(self) -> str:
        """Human-readable rendering, for review records."""
        duration = (
            "instant" if self.measurement_duration is None else str(self.measurement_duration)
        )
        return (
            f"{self.metric} at {self.location_id} for the {duration} beginning "
            f"{self.event_time.isoformat()} ({self.aggregation_level.value}, {self.unit})"
        )

    def as_dict(self) -> dict[str, str]:
        """Serializable form, for writing into a review record."""
        return {
            "location_id": self.location_id,
            "event_time": self.event_time.isoformat(),
            "metric": self.metric,
            "unit": self.unit,
            "measurement_duration": (
                "" if self.measurement_duration is None else str(self.measurement_duration)
            ),
            "aggregation_level": self.aggregation_level.value,
        }


def group_by_key(
    observations: list[Observation],
) -> dict[ObservationKey, list[Observation]]:
    """Group observations by the phenomenon they measure.

    Groups are ordered deterministically within themselves -- by
    ``(received_time, event_time, observation_id)`` -- so representative
    selection and the resulting links are reproducible across runs. Ordering by
    insertion alone would make the output depend on the input file's row order.
    """
    groups: dict[ObservationKey, list[Observation]] = {}
    for observation in observations:
        groups.setdefault(ObservationKey.of(observation), []).append(observation)

    for members in groups.values():
        members.sort(key=lambda obs: (obs.received_time, obs.event_time, obs.observation_id))
    return groups
