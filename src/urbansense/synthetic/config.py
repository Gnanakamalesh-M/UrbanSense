"""Configuration for the synthetic data generator (SPEC section 37).

The generator exists so later phases can be *measured*. A phase that claims to
discover a Friday congestion pattern, detect a drift event or quarantine an
invalid record is only testable against data where we planted the answer — so
every injected effect is declared here, echoed verbatim into
``ground_truth.json``, and nothing is hidden in the generator's code.

Date ranges are explicit and absolute rather than relative to "now". The
committed sample data must be byte-stable across regenerations, and a relative
range would also eventually produce a future ``event_time``, which the schema
rightly rejects.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from urbansense.schemas.enums import ProblemType

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Positive = Annotated[float, Field(gt=0.0)]


class ZoneConfig(BaseModel):
    """One spatial zone and its baseline traffic behaviour.

    Attributes:
        location_id: Canonical zone identifier, e.g. ``"ZONE-B"``.
        description: What kind of road this stands for.
        latitude: WGS84 latitude of the zone's reference point.
        longitude: WGS84 longitude.
        base_volume: Mean vehicles/hour before any diurnal or weekly shaping.
        base_speed: Mean km/hour under free-flow conditions.
        noise_fraction: Gaussian noise as a fraction of the shaped value.
        sensor_id: Identifier of the sensor reporting this zone.
        clock_skew_seconds: This sensor's clock offset. Non-zero on purpose, so
            ``source_time`` and ``event_time`` differ for a real reason rather
            than being trivially equal (SPEC section 4B).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    location_id: str = Field(min_length=1)
    description: str = ""
    latitude: Annotated[float, Field(ge=-90.0, le=90.0)]
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)]
    base_volume: Positive
    base_speed: Positive = 45.0
    noise_fraction: Fraction = 0.08
    sensor_id: str = Field(min_length=1)
    clock_skew_seconds: int = 0


class FridayEveningConfig(BaseModel):
    """Recurring Friday-evening congestion in one zone (SPEC section 21).

    The headline spatio-temporal pattern: ``Zone B + Friday + 18:00-20:00``.

    Attributes:
        enabled: Whether to inject it.
        location_id: Zone the pattern applies to. Other zones are untouched, so
            a discovery phase has to find *where* as well as *when*.
        weekday: Day of week, Monday=0 .. Sunday=6. Friday is 4.
        start_hour: First affected local hour, inclusive.
        end_hour: First unaffected local hour, exclusive.
        volume_multiplier: Factor applied to volume inside the window.
        speed_multiplier: Factor applied to speed inside the window.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    location_id: str = "ZONE-B"
    weekday: Annotated[int, Field(ge=0, le=6)] = 4
    start_hour: Annotated[int, Field(ge=0, le=23)] = 18
    end_hour: Annotated[int, Field(ge=1, le=24)] = 20
    volume_multiplier: Positive = 2.2
    speed_multiplier: Positive = 0.55

    @model_validator(mode="after")
    def _window_is_ordered(self) -> Self:
        """The window must span at least one hour."""
        if self.end_hour <= self.start_hour:
            raise ValueError(
                f"end_hour ({self.end_hour}) must be after start_hour ({self.start_hour})."
            )
        return self


class RainEffectConfig(BaseModel):
    """Conditional rainfall effect on traffic (SPEC section 21).

    Rain is a *covariate*, not a label: the rainfall series is emitted as its own
    observations, so a later phase has to connect rain to traffic itself.

    Attributes:
        enabled: Whether to inject it.
        volume_coefficient: Volume gain per mm of rain, before the cap.
        speed_coefficient: Speed loss per mm of rain, before the cap.
        cap_mm: Rain above this adds no further effect; heavy rain saturates
            rather than scaling without limit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    volume_coefficient: float = 0.03
    speed_coefficient: float = 0.02
    cap_mm: Positive = 20.0


class DriftConfig(BaseModel):
    """A permanent baseline shift partway through the series (SPEC section 29).

    Models something physical changing in the city -- a new flyover opening --
    rather than a data error. Drift detection in Phase 6 is scored against this.

    Attributes:
        enabled: Whether to inject it.
        start_date: Local date from which the shift applies, inclusive.
        location_ids: Zones affected. Unaffected zones act as a control.
        volume_multiplier: Post-drift factor on volume.
        reason: Human-readable explanation, carried into ground truth.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    start_date: date = date(2026, 5, 1)
    location_ids: tuple[str, ...] = ("ZONE-A", "ZONE-B")
    volume_multiplier: Positive = 1.18
    reason: str = "new flyover opened, shifting baseline demand"


class SensorFailureConfig(BaseModel):
    """A frozen-value sensor failure window (SPEC section 17).

    The sensor keeps reporting, but reports the same number forever. This is
    deliberately worse than missing data: a naive pipeline accepts it silently.

    Attributes:
        enabled: Whether to inject it.
        location_id: Zone whose sensor fails. Kept away from the pattern zone so
            the failure cannot distort pattern-strength assertions.
        start: Local start of the window, inclusive.
        end: Local end of the window, exclusive.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    location_id: str = "ZONE-D"
    start: datetime = datetime(2026, 3, 10, 2, 0)
    end: datetime = datetime(2026, 3, 12, 14, 0)

    @model_validator(mode="after")
    def _window_is_ordered(self) -> Self:
        """The failure window must not end before it starts."""
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be after start ({self.start}).")
        return self


class MissingWindowConfig(BaseModel):
    """One contiguous window of missing observations (SPEC section 15).

    Written as an **empty** value cell, never as zero. A sensor that reported
    nothing did not report zero vehicles.

    Attributes:
        location_id: Affected zone.
        start: Local start, inclusive.
        end: Local end, exclusive.
        reason: Why it is missing, for the ground-truth record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    location_id: str
    start: datetime
    end: datetime
    reason: str = "feed outage"

    @model_validator(mode="after")
    def _window_is_ordered(self) -> Self:
        """The gap must not end before it starts."""
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be after start ({self.start}).")
        return self


class MissingDataConfig(BaseModel):
    """All missing-data injection.

    Attributes:
        enabled: Whether to inject missing data at all.
        windows: Contiguous outage windows.
        scattered_fraction: Fraction of remaining rows dropped at random, which
            is the ordinary background rate of a real feed.
        scattered_location_ids: Zones subject to scattered loss.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    windows: tuple[MissingWindowConfig, ...] = ()
    scattered_fraction: Fraction = 0.02
    scattered_location_ids: tuple[str, ...] = ("ZONE-C",)


class IntegrityInjectionConfig(BaseModel):
    """Planted duplicates, conflicts and invalid values for Phase 2 to catch.

    These rows are emitted into the raw CSV exactly as a misbehaving source
    would produce them. Phase 1 does **not** detect or fix any of it; the
    detectors are Phase 2's job and this is what they will be scored against.

    Attributes:
        duplicate_count: Rows re-emitted verbatim under a new ``record_id``.
        conflict_count: Timestamps where a second source reports a materially
            different value for the same zone and metric.
        conflict_divergence: Relative difference applied to the conflicting copy.
        conflict_source_id: Source id used for the disagreeing copy.
        negative_value_count: Impossible negative volumes.
        impossible_speed_count: Speeds far beyond any plausible road.
        impossible_speed_value: The value to use for those rows.
        unparseable_timestamp_count: Rows whose timestamp is not a timestamp, so
            ingestion must have a rejection path rather than crashing.
        unknown_unit_count: Rows declaring a unit absent from the metric config.
        unknown_unit: The bogus unit string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    duplicate_count: Annotated[int, Field(ge=0)] = 12
    conflict_count: Annotated[int, Field(ge=0)] = 8
    conflict_divergence: float = 0.22
    conflict_source_id: str = "city-warehouse"
    negative_value_count: Annotated[int, Field(ge=0)] = 4
    impossible_speed_count: Annotated[int, Field(ge=0)] = 2
    impossible_speed_value: float = 480.0
    unparseable_timestamp_count: Annotated[int, Field(ge=0)] = 2
    unknown_unit_count: Annotated[int, Field(ge=0)] = 1
    unknown_unit: str = "vehicles/fortnight"


class GeneratorConfig(BaseModel):
    """Everything needed to generate one reproducible dataset.

    Attributes:
        seed: Random seed. The same seed must always produce identical output.
        problem_type: Problem domain. Traffic only in v1.
        timezone_offset_hours: Offset of the source's local wall clock. Sources
            stamp readings in local time; ingestion normalizes to UTC.
        history_start: First local date of the historical period, inclusive.
        history_end: Last local date of the historical period, inclusive.
        holdout_end: Last local date of the held-out period, inclusive. The
            held-out period begins the day after ``history_end`` and feeds the
            real-time simulator, so no model ever trains on it.
        interval_minutes: Spacing between observations.
        zones: Zones to generate.
        volume_metric: Canonical metric name for traffic volume.
        speed_metric: Canonical metric name for average speed.
        rainfall_metric: Canonical metric name for the rainfall covariate.
        rainfall_unit: Unit for rainfall. Rain belongs to the flood domain, so
            it is emitted as a separate source and is not validated against the
            traffic problem config.
        rainfall_location_id: Zone the rain gauge reports for (city-wide).
        friday_evening: Friday-evening congestion pattern.
        rain_effect: Rainfall-to-traffic relationship.
        drift: Mid-series baseline shift.
        sensor_failure: Frozen-value failure window.
        missing_data: Missing-data injection.
        integrity: Planted duplicates, conflicts and invalid values.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 20260101
    problem_type: ProblemType = ProblemType.TRAFFIC
    timezone_offset_hours: float = 5.5

    history_start: date = date(2025, 10, 1)
    history_end: date = date(2026, 8, 31)
    holdout_end: date = date(2026, 9, 30)
    interval_minutes: Annotated[int, Field(gt=0)] = 60

    zones: tuple[ZoneConfig, ...]
    volume_metric: str = "traffic_volume"
    speed_metric: str = "average_speed"
    rainfall_metric: str = "rainfall"
    rainfall_unit: str = "mm"
    rainfall_location_id: str = "CITY"

    friday_evening: FridayEveningConfig = FridayEveningConfig()
    rain_effect: RainEffectConfig = RainEffectConfig()
    drift: DriftConfig = DriftConfig()
    sensor_failure: SensorFailureConfig = SensorFailureConfig()
    missing_data: MissingDataConfig = MissingDataConfig()
    integrity: IntegrityInjectionConfig = IntegrityInjectionConfig()

    @property
    def holdout_start(self) -> date:
        """First local date of the held-out period."""
        return self.history_end + timedelta(days=1)

    @property
    def interval(self) -> timedelta:
        """Spacing between consecutive observations."""
        return timedelta(minutes=self.interval_minutes)

    def zone(self, location_id: str) -> ZoneConfig | None:
        """Return the zone configuration for ``location_id``, or ``None``."""
        return next((z for z in self.zones if z.location_id == location_id), None)

    @model_validator(mode="after")
    def _dates_are_ordered(self) -> Self:
        """History must precede the holdout, and neither may be empty."""
        if self.history_end < self.history_start:
            raise ValueError(
                f"history_end ({self.history_end}) precedes history_start ({self.history_start})."
            )
        if self.holdout_end <= self.history_end:
            raise ValueError(
                f"holdout_end ({self.holdout_end}) must be after history_end "
                f"({self.history_end}); the holdout period would otherwise be empty."
            )
        return self

    @model_validator(mode="after")
    def _zones_are_unique_and_present(self) -> Self:
        """At least one zone, with distinct ids."""
        if not self.zones:
            raise ValueError("at least one zone must be configured.")
        ids = [zone.location_id for zone in self.zones]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate zone location_ids: {sorted(duplicates)}.")
        return self

    @model_validator(mode="after")
    def _injected_effects_reference_real_zones(self) -> Self:
        """Every injected effect must name a zone that exists.

        A typo here would silently produce a dataset with no pattern in it,
        which is the one failure mode that would quietly invalidate every later
        phase's test.
        """
        known = {zone.location_id for zone in self.zones}

        referenced: list[tuple[str, str]] = []
        if self.friday_evening.enabled:
            referenced.append(("friday_evening", self.friday_evening.location_id))
        if self.sensor_failure.enabled:
            referenced.append(("sensor_failure", self.sensor_failure.location_id))
        if self.drift.enabled:
            referenced.extend(("drift", loc) for loc in self.drift.location_ids)
        if self.missing_data.enabled:
            referenced.extend(
                ("missing_data.windows", w.location_id) for w in self.missing_data.windows
            )
            referenced.extend(
                ("missing_data.scattered", loc) for loc in self.missing_data.scattered_location_ids
            )

        unknown = sorted({f"{field}={loc!r}" for field, loc in referenced if loc not in known})
        if unknown:
            raise ValueError(
                f"injected effects reference zones that do not exist: {', '.join(unknown)}; "
                f"configured zones are {sorted(known)}."
            )
        return self

    @model_validator(mode="after")
    def _windows_fall_inside_the_series(self) -> Self:
        """Effect windows must overlap the generated period to have any effect."""
        series_start = datetime.combine(self.history_start, time.min)
        series_end = datetime.combine(self.holdout_end, time.max)

        windows: list[tuple[str, datetime, datetime]] = []
        if self.sensor_failure.enabled:
            windows.append(("sensor_failure", self.sensor_failure.start, self.sensor_failure.end))
        if self.missing_data.enabled:
            windows.extend(
                ("missing_data window", w.start, w.end) for w in self.missing_data.windows
            )

        for name, start, end in windows:
            if end <= series_start or start >= series_end:
                raise ValueError(
                    f"{name} ({start} to {end}) falls outside the generated period "
                    f"({series_start} to {series_end}) and would have no effect."
                )

        if self.drift.enabled and not (
            self.history_start < self.drift.start_date <= self.holdout_end
        ):
            raise ValueError(
                f"drift.start_date ({self.drift.start_date}) must fall inside the generated "
                f"period ({self.history_start} to {self.holdout_end}), with data on both sides."
            )
        return self
