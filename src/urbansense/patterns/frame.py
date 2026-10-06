"""A read-only view of the unified layer for discovery (SPEC sections 19, 21).

Discovery reads :attr:`ReconciledResult.unified` and nothing else. The frame is
built from that one collection, so the duplicate-linked, conflicted and
quarantined records are not reachable from anything in this package -- the same
structural guarantee the Phase 4 feature builder relies on, rather than a comment
asking people to remember.

Everything here is in the source's **local** time. "Friday 18:00" is a
local-clock fact: in UTC the planted window sits at 12:30 and straddles two hour
buckets, so a scan over UTC hours would be hunting a smeared pattern and would
find a weaker version of it in two adjacent cells.

Unknown values are dropped rather than filled. A cell's median is then the median
of what was actually observed, and a cell with too little data is reported as
untested rather than quietly given a number (SPEC section 15).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from urbansense.data_integrity.unified import ReconciledResult

#: Rainfall above this counts as a wet hour. A trace of rain does not change
#: driving behaviour, and treating 0.1 mm as "wet" would dilute the contrast.
WET_THRESHOLD_MM = 0.5

#: Location the city-wide rain gauge reports for.
CITY_LOCATION = "CITY"

#: Metric carrying the rainfall covariate.
RAINFALL_METRIC = "rainfall"


@dataclass(frozen=True, slots=True)
class Reading:
    """One observation, in local time, ready to be grouped.

    Attributes:
        location_id: Zone.
        metric: Metric.
        local_time: Event time in the source's local wall clock, naive.
        value: The measurement. Never ``None`` -- unknown readings are excluded
            when the frame is built.
        rain_mm: Concurrent city rainfall, or ``None`` when unknown. Kept
            alongside rather than joined later, so every analysis conditions on
            the same rainfall series.
    """

    location_id: str
    metric: str
    local_time: datetime
    value: float
    rain_mm: float | None

    @property
    def weekday(self) -> int:
        """Local day of week, Monday 0."""
        return self.local_time.weekday()

    @property
    def hour(self) -> int:
        """Local hour of day."""
        return self.local_time.hour

    @property
    def month_key(self) -> str:
        """Calendar month label, used as the evolution window."""
        return f"{self.local_time.year:04d}-{self.local_time.month:02d}"

    @property
    def is_wet(self) -> bool:
        """Whether rain was falling. ``False`` when rainfall is unknown.

        Unknown weather is not evidence of rain, so an unknown hour is treated as
        not-wet for the *condition* while still being excluded from rain
        regressions, where an unknown amount cannot be a predictor.
        """
        return self.rain_mm is not None and self.rain_mm > WET_THRESHOLD_MM


@dataclass(frozen=True)
class ObservationFrame:
    """Unified observations grouped for discovery.

    Attributes:
        readings: Every known reading, in local time.
        timezone_offset_hours: Offset applied to reach local time, recorded so a
            reader knows which clock the hours refer to.
        source_rows: Unified rows the frame was built from, including the ones
            dropped for an unknown value.
        dropped_unknown: Rows excluded because their value was unknown.
    """

    readings: tuple[Reading, ...]
    timezone_offset_hours: float
    source_rows: int
    dropped_unknown: int
    _by_metric: dict[str, tuple[Reading, ...]] = field(default_factory=dict, repr=False)
    _zones: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_result(
        cls, result: ReconciledResult, *, timezone_offset_hours: float = 0.0
    ) -> ObservationFrame:
        """Build a frame from a reconciliation result.

        Only ``result.unified`` is read. Rainfall is joined onto every reading by
        local timestamp, so a conditional analysis never has to re-derive the
        join and two analyses cannot disagree about which hours were wet.
        """
        offset = timedelta(hours=timezone_offset_hours)

        rain_by_time: dict[datetime, float] = {}
        for observation in result.unified:
            if (
                observation.metric == RAINFALL_METRIC
                and observation.location_id == CITY_LOCATION
                and observation.value is not None
            ):
                rain_by_time[observation.event_time + offset] = observation.value

        readings: list[Reading] = []
        dropped = 0
        for observation in result.unified:
            if observation.value is None:
                dropped += 1
                continue
            local_time = observation.event_time + offset
            readings.append(
                Reading(
                    location_id=observation.location_id,
                    metric=observation.metric,
                    local_time=local_time.replace(tzinfo=None),
                    value=observation.value,
                    rain_mm=rain_by_time.get(local_time),
                )
            )

        readings.sort(key=lambda item: (item.local_time, item.location_id, item.metric))
        return cls(
            readings=tuple(readings),
            timezone_offset_hours=timezone_offset_hours,
            source_rows=len(result.unified),
            dropped_unknown=dropped,
        )

    def for_metric(self, metric: str) -> tuple[Reading, ...]:
        """Readings for one metric, cached."""
        cached = self._by_metric.get(metric)
        if cached is None:
            cached = tuple(item for item in self.readings if item.metric == metric)
            self._by_metric[metric] = cached
        return cached

    def zones(self, metric: str) -> tuple[str, ...]:
        """Zones reporting a metric, excluding the city-wide gauge.

        ``CITY`` is a reference point for the rain series rather than a road, so
        including it in a spatial scan would compare a rain gauge against traffic
        detectors.
        """
        cached = self._zones.get(metric)
        if cached is None:
            cached = tuple(
                sorted(
                    {
                        item.location_id
                        for item in self.for_metric(metric)
                        if item.location_id != CITY_LOCATION
                    }
                )
            )
            self._zones[metric] = cached
        return cached

    def cells(self, metric: str) -> dict[tuple[str, int, int], list[Reading]]:
        """Readings grouped by ``(zone, weekday, hour)``.

        This is the unit the spatio-temporal scan tests, and conditioning on hour
        and zone together is what removes the two obvious confounds: 18:00 is a
        peak everywhere, and ZONE-A carries four times ZONE-C.
        """
        grouped: dict[tuple[str, int, int], list[Reading]] = defaultdict(list)
        for item in self.for_metric(metric):
            if item.location_id == CITY_LOCATION:
                continue
            grouped[(item.location_id, item.weekday, item.hour)].append(item)
        return dict(grouped)

    def windows(self) -> tuple[str, ...]:
        """Evolution windows present, in order."""
        return tuple(sorted({item.month_key for item in self.readings}))

    @property
    def first_local_time(self) -> datetime | None:
        """Earliest local timestamp in the frame."""
        return self.readings[0].local_time if self.readings else None

    @property
    def last_local_time(self) -> datetime | None:
        """Latest local timestamp in the frame."""
        return self.readings[-1].local_time if self.readings else None

    def __len__(self) -> int:
        """Number of known readings."""
        return len(self.readings)
