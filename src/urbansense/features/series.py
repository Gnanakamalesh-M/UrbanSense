"""Timestamp-keyed lookup over the unified layer (SPEC sections 15, 33).

Lag and rolling features are looked up **by timestamp**, never by row position.
That is a correctness requirement, not a style preference, and this project's own
data demonstrates why: quarantining the frozen ZONE-D window left a **2 day 13
hour hole** in that series, and ZONE-A has four two-hour holes. A positional
shift such as ``series[i - 1]`` across that hole returns the value from two and a
half days earlier and labels it ``lag_1h``. The model would be trained on a
feature that is silently, confidently wrong — and nothing downstream could
detect it, because the number looks entirely plausible.

Keying on the instant makes an absent hour return ``None`` instead. ``None``
becomes ``NaN`` in the design matrix, which the gradient model handles natively,
so missing stays missing and nothing is imputed (SPEC section 15).

Two kinds of absence are deliberately *not* distinguished here: an hour with no
record at all, and an hour whose record carries ``value=None``. Both mean "we do
not know what the traffic was", which is the only thing a feature can act on.
The distinction is preserved upstream, on the observation itself.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from urbansense.schemas.observation import Observation

#: One hour, the grid step for every series in this project.
_HOUR = timedelta(hours=1)


@dataclass(frozen=True, slots=True)
class _Grid:
    """Prefix sums over one series' dense hourly grid.

    Attributes:
        origin: Instant of index 0.
        length: Number of hourly slots.
        sums: Cumulative sum of known values, length ``length + 1``.
        counts: Cumulative count of known values, length ``length + 1``.
    """

    origin: datetime
    length: int
    sums: NDArray[np.float64]
    counts: NDArray[np.int64]


@dataclass(frozen=True)
class SeriesIndex:
    """Values from the unified layer, addressable by instant.

    Attributes:
        values: ``(location_id, metric, event_time) -> value``. A present key
            with a ``None`` value means the reading was missing; an absent key
            means no record existed for that hour. Both read as unknown.
        received: ``(location_id, metric, event_time) -> received_time``, so the
            receipt-time clause can be probed the same way the event-time clause
            is. Kept parallel rather than merged into ``values`` so the common
            lookup stays a plain dict access.
        locations: Locations present in the index, sorted.
        metrics: Metrics present in the index, sorted.
        earliest: First instant in the index.
        latest: Last instant in the index.
    """

    values: dict[tuple[str, str, datetime], float | None]
    received: dict[tuple[str, str, datetime], datetime]
    locations: tuple[str, ...]
    metrics: tuple[str, ...]
    earliest: datetime | None
    latest: datetime | None
    #: Lazily built prefix sums per series, for rolling means. A cache only --
    #: :meth:`window_mean_exact` stays the definition, and a test asserts the two
    #: agree on the real dataset so the fast path cannot drift from it.
    _grids: dict[tuple[str, str], _Grid] = field(default_factory=dict, repr=False)

    @classmethod
    def from_observations(cls, observations: Iterable[Observation]) -> SeriesIndex:
        """Build an index from unified observations.

        Args:
            observations: The unified layer. Reconciliation has already
                collapsed duplicates to one logical observation per key, so a
                repeated key here would mean reconciliation failed -- and the
                last writer would silently win. Asserted rather than assumed.

        Raises:
            ValueError: if two observations share a key, which should be
                impossible after reconciliation.
        """
        values: dict[tuple[str, str, datetime], float | None] = {}
        received: dict[tuple[str, str, datetime], datetime] = {}
        locations: set[str] = set()
        metrics: set[str] = set()
        earliest: datetime | None = None
        latest: datetime | None = None

        for observation in observations:
            key = (observation.location_id, observation.metric, observation.event_time)
            if key in values:
                raise ValueError(
                    f"two observations share the key {key}; the unified layer should "
                    "hold one logical observation per key, so reconciliation has "
                    "either not run or has a bug. Refusing to let one silently "
                    "overwrite the other."
                )
            values[key] = observation.value
            received[key] = observation.received_time
            locations.add(observation.location_id)
            metrics.add(observation.metric)
            if earliest is None or observation.event_time < earliest:
                earliest = observation.event_time
            if latest is None or observation.event_time > latest:
                latest = observation.event_time

        return cls(
            values=values,
            received=received,
            locations=tuple(sorted(locations)),
            metrics=tuple(sorted(metrics)),
            earliest=earliest,
            latest=latest,
        )

    def at(self, location_id: str, metric: str, moment: datetime) -> float | None:
        """Value at exactly this instant, or ``None`` when unknown.

        No nearest-neighbour fallback and no forward fill. Returning the closest
        available reading would be imputation wearing a lookup's clothing, and
        could reach across a gap into data the caller did not ask for.
        """
        return self.values.get((location_id, metric, moment))

    def window(
        self,
        location_id: str,
        metric: str,
        *,
        end: datetime,
        hours: int,
    ) -> list[float]:
        """Known values in ``(end - hours, end]``, oldest first.

        The window is inclusive of ``end`` and exclusive of its start, so two
        adjacent windows never share an hour. Unknown hours are simply absent
        from the result rather than filled, so a mean over a sparse window is a
        mean of what was actually observed.

        Raises:
            ValueError: if ``hours`` is not positive. A zero or negative window
                would silently return nothing or reach forwards.
        """
        if hours <= 0:
            raise ValueError(f"window hours must be positive, got {hours}")

        found: list[float] = []
        for step in range(hours - 1, -1, -1):
            value = self.at(location_id, metric, end - timedelta(hours=step))
            if value is not None:
                found.append(value)
        return found

    def window_mean_exact(
        self,
        location_id: str,
        metric: str,
        *,
        end: datetime,
        hours: int,
        min_observations: int = 1,
    ) -> float | None:
        """Mean of known values in the window, by direct lookup.

        The reference implementation: one dict lookup per hour, obviously correct
        and obviously backward-looking. :meth:`window_mean` computes the same
        thing with prefix sums, and a test pins the two together.
        """
        found = self.window(location_id, metric, end=end, hours=hours)
        if len(found) < min_observations:
            return None
        return statistics.fmean(found)

    def window_mean(
        self,
        location_id: str,
        metric: str,
        *,
        end: datetime,
        hours: int,
        min_observations: int = 1,
    ) -> float | None:
        """Mean of known values in ``(end - hours, end]``, or ``None`` if sparse.

        ``min_observations`` guards against a "mean" computed from one reading
        standing in for a week. Below the threshold the answer is unknown, which
        is honest, rather than a number carrying false weight.

        Uses cached prefix sums, falling back to :meth:`window_mean_exact` when
        the series does not sit on a whole-hour grid. A 168-hour window costs 168
        dict lookups per row otherwise, which is nine seconds per feature table
        on this dataset.
        """
        if hours <= 0:
            raise ValueError(f"window hours must be positive, got {hours}")

        grid = self._grid(location_id, metric)
        if grid is None:
            return self.window_mean_exact(
                location_id,
                metric,
                end=end,
                hours=hours,
                min_observations=min_observations,
            )

        offset = (end - grid.origin) / _HOUR
        if offset != int(offset):  # pragma: no cover - _grid() rejects such series
            return self.window_mean_exact(
                location_id,
                metric,
                end=end,
                hours=hours,
                min_observations=min_observations,
            )

        last = min(int(offset), grid.length - 1)
        if last < 0:
            return None
        first = max(0, int(offset) - hours + 1)
        if last < first:
            return None

        count = int(grid.counts[last + 1] - grid.counts[first])
        if count < min_observations:
            return None
        return float(grid.sums[last + 1] - grid.sums[first]) / count

    def _grid(self, location_id: str, metric: str) -> _Grid | None:
        """Prefix sums for one series, built on first use.

        Returns ``None`` when the series does not sit on a whole-hour grid, so
        the caller falls back to the exact lookup rather than indexing wrongly.
        """
        key = (location_id, metric)
        cached = self._grids.get(key)
        if cached is not None:
            return cached

        times = self.times_for(location_id, metric)
        if not times:
            return None

        origin = times[0]
        span = (times[-1] - origin) / _HOUR
        if span != int(span):
            return None
        length = int(span) + 1

        dense = np.full(length, np.nan, dtype=np.float64)
        for moment in times:
            step = (moment - origin) / _HOUR
            if step != int(step):
                return None
            value = self.values[(location_id, metric, moment)]
            if value is not None:
                dense[int(step)] = value

        known = ~np.isnan(dense)
        grid = _Grid(
            origin=origin,
            length=length,
            sums=np.concatenate(([0.0], np.cumsum(np.where(known, dense, 0.0)))),
            counts=np.concatenate(([0], np.cumsum(known.astype(np.int64)))),
        )
        self._grids[key] = grid
        return grid

    def times_for(self, location_id: str, metric: str) -> tuple[datetime, ...]:
        """Every instant with a record for this series, in order."""
        return tuple(
            sorted(
                moment for (loc, met, moment) in self.values if loc == location_id and met == metric
            )
        )

    def with_future_perturbed(
        self, pivot: datetime, *, factor: float = 10.0, shift: float = 1000.0
    ) -> SeriesIndex:
        """Return a copy with every value after ``pivot`` changed.

        This is the engine of the leakage probe in
        :mod:`urbansense.features.leakage`. Features rebuilt against this copy
        must produce identical values to the original; any that differ read the
        future.

        Both a multiply and an add are applied so the perturbation cannot be a
        no-op for a value that happens to be zero -- a leaky feature reading a
        single dry rainfall hour would otherwise go undetected.
        """
        return self._perturbed(lambda key: key[2] > pivot, factor=factor, shift=shift)

    def with_late_arrivals_perturbed(
        self, pivot: datetime, *, factor: float = 10.0, shift: float = 1000.0
    ) -> SeriesIndex:
        """Return a copy with every value that *arrived* after ``pivot`` changed.

        The receipt-time counterpart of :meth:`with_future_perturbed`. A feature
        rebuilt against this copy must be unchanged, or it used a record before
        that record had arrived -- which is leakage even though the record's
        ``event_time`` is safely in the past.

        Only meaningful when records carry distinct receipt times. After a batch
        load they all share one instant, and the caller reports the clause as not
        applicable rather than pretending to have checked it.
        """
        return self._perturbed(
            lambda key: self.received.get(key, pivot) > pivot, factor=factor, shift=shift
        )

    def _perturbed(
        self,
        should_change: Callable[[tuple[str, str, datetime]], bool],
        *,
        factor: float,
        shift: float,
    ) -> SeriesIndex:
        """Copy the index, perturbing the values the predicate selects.

        Both a multiply and an add are applied so the perturbation cannot be a
        no-op for a value that happens to be zero -- a leaky feature reading a
        single dry rainfall hour would otherwise go undetected.
        """
        perturbed: dict[tuple[str, str, datetime], float | None] = {}
        for key, value in self.values.items():
            if value is not None and should_change(key):
                perturbed[key] = value * factor + shift
            else:
                perturbed[key] = value

        # A fresh index, so the perturbed copy builds its own prefix sums
        # rather than reusing the original's and hiding the perturbation.
        return SeriesIndex(
            values=perturbed,
            received=self.received,
            locations=self.locations,
            metrics=self.metrics,
            earliest=self.earliest,
            latest=self.latest,
        )

    @property
    def distinct_received_times(self) -> int:
        """How many distinct receipt instants the data carries.

        One means a single batch load. More than one does **not** imply the
        receipts are informative: concatenating two files gives two load
        instants seconds apart, both long after every event. Use
        :meth:`receipts_straddle` to decide whether the receipt clause can
        actually be probed.
        """
        return len(set(self.received.values()))

    def arrived_by(self, moment: datetime) -> int:
        """How many records had been received at or before ``moment``."""
        return sum(1 for received in self.received.values() if received <= moment)

    def receipts_straddle(self, moment: datetime) -> bool:
        """Whether receipts fall on both sides of ``moment``.

        The condition under which "was this record available yet?" is a
        question with two possible answers. If every record arrived after
        ``moment``, perturbing the late arrivals perturbs the whole dataset, and
        a clean feature moves exactly as much as a leaky one -- so the probe
        cannot distinguish them and must say so rather than report a violation.
        """
        arrived = self.arrived_by(moment)
        return 0 < arrived < len(self.received)

    def __len__(self) -> int:
        """Number of indexed records."""
        return len(self.values)
