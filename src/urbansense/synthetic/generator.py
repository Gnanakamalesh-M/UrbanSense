"""Synthetic dataset generation (SPEC section 37).

Produces a traffic dataset with **known ground truth** so later phases can be
scored rather than merely demonstrated. The output is:

* ``traffic_history.csv`` -- the training period;
* ``traffic_holdout.csv`` -- a later, held-out period the real-time simulator
  replays, so no model can have trained on it;
* ``ground_truth.json`` -- every planted effect, with parameters and record ids.

Determinism is a hard requirement. All randomness is drawn from one seeded
``random.Random``, consumed in a fixed order, so the same seed always yields
byte-identical files. That is what lets the committed sample data be verified by
regenerating it.

Generation order matters and is fixed: the clean series first, then the planted
corruptions layered on top. Corrupting afterwards means the clean signal is
never perturbed by the thing meant to be found, so pattern-strength assertions
stay meaningful.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import date, datetime, time, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from urbansense.synthetic import patterns, weather
from urbansense.synthetic.config import GeneratorConfig
from urbansense.synthetic.ground_truth import (
    ConflictTruth,
    DatasetTruth,
    DriftTruth,
    DuplicateTruth,
    FridayEveningTruth,
    GroundTruth,
    InvalidTruth,
    MissingDataTruth,
    PlantedRecords,
    RainEffectTruth,
    SensorFailureTruth,
    TimeWindow,
)
from urbansense.synthetic.writer import (
    CSV_COLUMNS,
    ZONE_COLUMNS,
    RawRow,
    WriteRefused,
    ZoneRow,
    write_rows,
)

#: Bumped when the generated output changes shape, so a ground-truth file from
#: an older generator can be spotted rather than silently trusted.
GENERATOR_VERSION = "1.0.0"

HISTORY_FILENAME = "traffic_history.csv"
HOLDOUT_FILENAME = "traffic_holdout.csv"
ZONES_FILENAME = "zones.csv"
GROUND_TRUTH_FILENAME = "ground_truth.json"

_WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


class GeneratedDataset(BaseModel):
    """What a generation run produced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    history_path: Path
    holdout_path: Path
    zones_path: Path
    ground_truth_path: Path
    history_rows: int
    holdout_rows: int
    ground_truth: GroundTruth


def _local_offset(config: GeneratorConfig) -> timedelta:
    """The source's local-time offset from UTC."""
    return timedelta(hours=config.timezone_offset_hours)


def _format_local(moment: datetime, config: GeneratorConfig) -> str:
    """Render a local timestamp the way a source would stamp it.

    Includes the offset, so the string is unambiguous and ingestion never has to
    guess a timezone (SPEC section 12).
    """
    offset = _local_offset(config)
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"{moment.strftime('%Y-%m-%d %H:%M:%S')}{sign}{hours:02d}:{minutes:02d}"


def _timestamps(start: date, end: date, interval: timedelta) -> Iterator[datetime]:
    """Yield local timestamps from ``start`` 00:00 through ``end`` 23:59."""
    moment = datetime.combine(start, time.min)
    limit = datetime.combine(end, time.min) + timedelta(days=1)
    while moment < limit:
        yield moment
        moment += interval


def _in_window(moment: datetime, start: datetime, end: datetime) -> bool:
    """Whether ``moment`` falls in ``[start, end)``."""
    return start <= moment < end


class _Builder:
    """Accumulates rows and ground truth for one generation run.

    Holds the single seeded RNG. Instance state rather than module state so two
    runs in one process cannot interfere -- a property the determinism tests
    depend on.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self._counter = 0

        self.rows: list[RawRow] = []
        self.holdout_rows: list[RawRow] = []

        self.friday_record_ids: list[str] = []
        self.failure_record_ids: list[str] = []
        self.window_missing_ids: list[str] = []
        self.scattered_missing_ids: list[str] = []
        self.frozen_value: float = 0.0

        #: Candidate rows for planted corruption: only clean, present-valued
        #: volume rows from the history period, so a planted duplicate is never
        #: itself a missing or already-corrupted record.
        self.corruption_candidates: list[RawRow] = []

    # -- identity ------------------------------------------------------

    def next_record_id(self) -> str:
        """Allocate the next deterministic record id.

        The seed is not embedded: it is recorded once in ``ground_truth.json``,
        and repeating it on 70,000 rows only inflates the file.
        """
        self._counter += 1
        return f"REC-{self._counter:06d}"

    # -- row construction ----------------------------------------------

    def make_row(
        self,
        *,
        moment: datetime,
        source_id: str,
        location_id: str,
        metric: str,
        value: float | None,
        unit: str,
        decimals: int = 1,
        clock_skew_seconds: int = 0,
        record_id: str | None = None,
    ) -> RawRow:
        """Build one raw row.

        ``value=None`` renders as an empty cell. It must never render as ``"0"``:
        a sensor that reported nothing did not report zero vehicles
        (SPEC section 15).

        ``decimals=0`` is used for vehicle counts, which are integers in reality
        -- a loop detector counts whole vehicles, not 1197.59 of them.

        ``moment`` is the interval **start**; the row is stamped with the
        interval end plus this sensor's clock skew, which is what the source
        actually asserts. Ingestion recovers event_time from the convention and
        cannot recover the skew, exactly as in reality.
        """
        stamp = moment + self.config.interval + timedelta(seconds=clock_skew_seconds)
        return RawRow(
            record_id=record_id if record_id is not None else self.next_record_id(),
            source_id=source_id,
            interval_end_local=_format_local(stamp, self.config),
            location_id=location_id,
            metric=metric,
            value="" if value is None else f"{value:.{decimals}f}",
            unit=unit,
        )

    # -- the clean series ----------------------------------------------

    def build_series(
        self, *, start: date, end: date, rainfall: dict[datetime, float], holdout: bool
    ) -> None:
        """Generate the clean observations for one period.

        Missing windows and the sensor failure are applied here because they
        change what the source *emits*. The integrity corruptions (duplicates,
        conflicts, invalid values) are layered on afterwards, in
        :meth:`plant_integrity_issues`.
        """
        config = self.config
        sink = self.holdout_rows if holdout else self.rows
        failure = config.sensor_failure
        missing = config.missing_data

        for moment in _timestamps(start, end, config.interval):
            rain_mm = rainfall.get(moment, 0.0)

            # One rain observation per interval for the whole city, from its own
            # source. It is a covariate: nothing ties it to a traffic row except
            # the timestamp, so the relationship has to be discovered.
            sink.append(
                self.make_row(
                    moment=moment,
                    source_id="rain-gauge-01",
                    location_id=config.rainfall_location_id,
                    metric=config.rainfall_metric,
                    value=rain_mm,
                    unit=config.rainfall_unit,
                )
            )

            for zone in config.zones:
                # Draw noise unconditionally, before any branch that might skip
                # this row, so the RNG consumption order is identical whatever
                # the injected effects are. Otherwise toggling one effect would
                # reshuffle the entire dataset.
                volume_noise = self.rng.gauss(1.0, zone.noise_fraction)
                speed_noise = self.rng.gauss(1.0, zone.noise_fraction * 0.5)
                scattered_roll = self.rng.random()

                volume = patterns.compose_volume(
                    moment=moment,
                    zone=zone,
                    rain_mm=rain_mm,
                    config=config,
                    noise=max(0.35, volume_noise),
                )
                speed = patterns.compose_speed(
                    moment=moment,
                    zone=zone,
                    rain_mm=rain_mm,
                    config=config,
                    noise=max(0.5, speed_noise),
                    volume=volume,
                )

                in_failure = (
                    failure.enabled
                    and zone.location_id == failure.location_id
                    and _in_window(moment, failure.start, failure.end)
                )
                if in_failure:
                    # A frozen sensor keeps reporting the same number. This is
                    # worse than missing data: the value looks plausible, so a
                    # pipeline that does not check for repetition accepts it.
                    if self.frozen_value == 0.0:
                        self.frozen_value = float(round(volume))
                    volume = self.frozen_value
                    speed = round(speed, 1)

                missing_window = self._matching_missing_window(moment, zone.location_id)
                is_window_missing = missing_window is not None
                is_scattered_missing = (
                    missing.enabled
                    and not is_window_missing
                    and zone.location_id in missing.scattered_location_ids
                    and scattered_roll < missing.scattered_fraction
                )

                volume_value: float | None = (
                    None if (is_window_missing or is_scattered_missing) else round(volume)
                )

                volume_row = self.make_row(
                    moment=moment,
                    source_id=zone.sensor_id,
                    location_id=zone.location_id,
                    metric=config.volume_metric,
                    value=volume_value,
                    unit="vehicles/hour",
                    decimals=0,
                    clock_skew_seconds=zone.clock_skew_seconds,
                )
                sink.append(volume_row)

                speed_row = self.make_row(
                    moment=moment,
                    source_id=zone.sensor_id,
                    location_id=zone.location_id,
                    metric=config.speed_metric,
                    value=None if is_window_missing else round(speed, 1),
                    unit="km/hour",
                    clock_skew_seconds=zone.clock_skew_seconds,
                )
                sink.append(speed_row)

                if holdout:
                    continue

                if is_window_missing:
                    self.window_missing_ids.extend([volume_row.record_id, speed_row.record_id])
                elif is_scattered_missing:
                    self.scattered_missing_ids.append(volume_row.record_id)

                if in_failure:
                    self.failure_record_ids.append(volume_row.record_id)

                if (
                    patterns.friday_evening_factor(moment, zone.location_id, config.friday_evening)
                    != 1.0
                ):
                    self.friday_record_ids.append(volume_row.record_id)

                if volume_value is not None and not in_failure:
                    self.corruption_candidates.append(volume_row)

    def _matching_missing_window(self, moment: datetime, location_id: str) -> TimeWindow | None:
        """Return the configured missing window covering ``moment``, if any."""
        if not self.config.missing_data.enabled:
            return None
        for window in self.config.missing_data.windows:
            if window.location_id == location_id and _in_window(moment, window.start, window.end):
                return TimeWindow(
                    location_id=window.location_id,
                    start=window.start,
                    end=window.end,
                    reason=window.reason,
                )
        return None

    # -- planted corruption --------------------------------------------

    def plant_integrity_issues(self) -> PlantedRecords:
        """Append duplicates, conflicts and invalid rows to the history.

        These are emitted exactly as a misbehaving source would produce them.
        Phase 1 does not detect or repair any of it -- the detectors are Phase 2,
        and this is what they will be scored against.
        """
        settings = self.config.integrity
        pool = self.corruption_candidates
        if not pool:
            return PlantedRecords()

        chosen = self.rng.sample(
            pool,
            k=min(
                len(pool),
                settings.duplicate_count
                + settings.conflict_count
                + settings.negative_value_count
                + settings.impossible_speed_count
                + settings.unparseable_timestamp_count
                + settings.unknown_unit_count,
            ),
        )
        cursor = 0

        def take(count: int) -> list[RawRow]:
            nonlocal cursor
            batch = chosen[cursor : cursor + count]
            cursor += count
            return batch

        duplicates: list[DuplicateTruth] = []
        for source in take(settings.duplicate_count):
            # Identical content, new record id: the same observation reported
            # twice. Detection must link them, not delete one (SPEC section 9).
            copy = source.model_copy(update={"record_id": self.next_record_id()})
            self.rows.append(copy)
            duplicates.append(
                DuplicateTruth(record_id=copy.record_id, duplicate_of=source.record_id)
            )

        conflicts: list[ConflictTruth] = []
        for source in take(settings.conflict_count):
            # A second source reporting a materially different value for the
            # same zone, metric and instant. Both must be kept and the
            # disagreement recorded, never silently resolved (SPEC section 10).
            original = float(source.value)
            diverged = float(round(original * (1.0 + settings.conflict_divergence)))
            copy = source.model_copy(
                update={
                    "record_id": self.next_record_id(),
                    "source_id": settings.conflict_source_id,
                    "value": f"{diverged:.0f}",
                }
            )
            self.rows.append(copy)
            conflicts.append(
                ConflictTruth(
                    record_id=copy.record_id,
                    conflicts_with=source.record_id,
                    location_id=source.location_id,
                    metric=source.metric,
                    interval_end_local=_parse_local_naive(source.interval_end_local),
                    original_value=original,
                    conflicting_value=diverged,
                )
            )

        invalid: list[InvalidTruth] = []

        for source in take(settings.negative_value_count):
            value = -abs(float(round(float(source.value) * 0.5)))
            copy = source.model_copy(
                update={"record_id": self.next_record_id(), "value": f"{value:.0f}"}
            )
            self.rows.append(copy)
            invalid.append(
                InvalidTruth(
                    record_id=copy.record_id,
                    kind="negative_value",
                    detail=f"negative traffic volume {value}",
                    # Representable as an Observation, so ingestion passes it
                    # through as PENDING; it is Phase 2 that must quarantine it.
                    ingestible=True,
                )
            )

        for source in take(settings.impossible_speed_count):
            copy = source.model_copy(
                update={
                    "record_id": self.next_record_id(),
                    "metric": self.config.speed_metric,
                    "value": f"{settings.impossible_speed_value:.1f}",
                    "unit": "km/hour",
                }
            )
            self.rows.append(copy)
            invalid.append(
                InvalidTruth(
                    record_id=copy.record_id,
                    kind="impossible_speed",
                    detail=f"average speed {settings.impossible_speed_value} km/hour",
                    ingestible=True,
                )
            )

        for source in take(settings.unparseable_timestamp_count):
            copy = source.model_copy(
                update={
                    "record_id": self.next_record_id(),
                    "interval_end_local": "not-a-timestamp",
                }
            )
            self.rows.append(copy)
            invalid.append(
                InvalidTruth(
                    record_id=copy.record_id,
                    kind="unparseable_timestamp",
                    detail="interval_end_local is not a timestamp",
                    # Cannot become an Observation at all, so ingestion must
                    # reject it into quarantine without crashing.
                    ingestible=False,
                )
            )

        for source in take(settings.unknown_unit_count):
            copy = source.model_copy(
                update={
                    "record_id": self.next_record_id(),
                    "unit": settings.unknown_unit,
                }
            )
            self.rows.append(copy)
            invalid.append(
                InvalidTruth(
                    record_id=copy.record_id,
                    kind="unknown_unit",
                    detail=f"unit {settings.unknown_unit!r} is not an accepted unit",
                    ingestible=False,
                )
            )

        return PlantedRecords(
            duplicates=tuple(duplicates),
            conflicts=tuple(conflicts),
            invalid=tuple(invalid),
        )


def _parse_local_naive(text: str) -> datetime:
    """Parse a local timestamp string back to a naive datetime."""
    return datetime.fromisoformat(text).replace(tzinfo=None)


def generate(config: GeneratorConfig) -> tuple[list[RawRow], list[RawRow], GroundTruth]:
    """Generate the dataset in memory.

    Returns:
        ``(history_rows, holdout_rows, ground_truth)``.
    """
    builder = _Builder(config)

    # One rainfall series spanning both periods, drawn first so its RNG
    # consumption is independent of how many zones there are.
    rainfall = weather.generate_rainfall(
        start=datetime.combine(config.history_start, time.min),
        end=datetime.combine(config.holdout_end, time.min) + timedelta(days=1),
        interval=config.interval,
        rng=builder.rng,
    )

    builder.build_series(
        start=config.history_start,
        end=config.history_end,
        rainfall=rainfall,
        holdout=False,
    )
    builder.build_series(
        start=config.holdout_start,
        end=config.holdout_end,
        rainfall=rainfall,
        holdout=True,
    )
    planted = builder.plant_integrity_issues()

    history_rainfall = {
        moment: value for moment, value in rainfall.items() if moment.date() <= config.history_end
    }
    wet, dry = weather.wet_dry_counts(history_rainfall)

    control_zones = tuple(
        zone.location_id
        for zone in config.zones
        if zone.location_id != config.friday_evening.location_id
    )
    drift_controls = tuple(
        zone.location_id
        for zone in config.zones
        if zone.location_id not in config.drift.location_ids
    )

    missing_windows = tuple(
        TimeWindow(
            location_id=window.location_id,
            start=window.start,
            end=window.end,
            reason=window.reason,
        )
        for window in config.missing_data.windows
    )
    failure_window = TimeWindow(
        location_id=config.sensor_failure.location_id,
        start=config.sensor_failure.start,
        end=config.sensor_failure.end,
        reason="frozen sensor values",
    )

    ground_truth = GroundTruth(
        generator_version=GENERATOR_VERSION,
        seed=config.seed,
        config=config,
        timezone_offset_hours=config.timezone_offset_hours,
        patterns={
            GroundTruth.FRIDAY_EVENING_KEY: FridayEveningTruth(
                location_id=config.friday_evening.location_id,
                weekday=config.friday_evening.weekday,
                weekday_name=_WEEKDAY_NAMES[config.friday_evening.weekday],
                start_hour=config.friday_evening.start_hour,
                end_hour=config.friday_evening.end_hour,
                volume_multiplier=config.friday_evening.volume_multiplier,
                speed_multiplier=config.friday_evening.speed_multiplier,
                affected_record_ids=tuple(builder.friday_record_ids),
                control_location_ids=control_zones,
            ),
            GroundTruth.RAIN_EFFECT_KEY: RainEffectTruth(
                volume_coefficient=config.rain_effect.volume_coefficient,
                speed_coefficient=config.rain_effect.speed_coefficient,
                cap_mm=config.rain_effect.cap_mm,
                rainfall_metric=config.rainfall_metric,
                rainfall_location_id=config.rainfall_location_id,
                wet_hour_count=wet,
                dry_hour_count=dry,
            ),
        },
        anomalies={
            GroundTruth.DRIFT_KEY: DriftTruth(
                start_date=config.drift.start_date,
                location_ids=config.drift.location_ids,
                volume_multiplier=config.drift.volume_multiplier,
                reason=config.drift.reason,
                control_location_ids=drift_controls,
            ),
            GroundTruth.SENSOR_FAILURE_KEY: SensorFailureTruth(
                location_id=config.sensor_failure.location_id,
                start=config.sensor_failure.start,
                end=config.sensor_failure.end,
                frozen_value=builder.frozen_value,
                affected_record_ids=tuple(builder.failure_record_ids),
            ),
        },
        missing_data=MissingDataTruth(
            windows=missing_windows,
            scattered_fraction=config.missing_data.scattered_fraction,
            scattered_location_ids=config.missing_data.scattered_location_ids,
            window_record_ids=tuple(builder.window_missing_ids),
            scattered_record_ids=tuple(builder.scattered_missing_ids),
        ),
        planted_records=planted,
        clock_skew_seconds={zone.sensor_id: zone.clock_skew_seconds for zone in config.zones},
        # Measuring pattern strength inside a gap or a frozen-sensor window
        # would distort the baseline, so ground truth names the windows to skip.
        exclude_windows=(*missing_windows, failure_window),
    )

    return builder.rows, builder.holdout_rows, ground_truth


def generate_to_directory(
    config: GeneratorConfig, out_dir: Path, *, force: bool = False
) -> GeneratedDataset:
    """Generate the dataset and write it to ``out_dir``.

    Args:
        config: Generator configuration.
        out_dir: Destination directory, created if absent.
        force: Permit replacing existing files. Without it, an existing dataset
            is left untouched rather than silently overwritten.

    Returns:
        A :class:`GeneratedDataset` describing what was written.
    """
    history_rows, holdout_rows, ground_truth = generate(config)

    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / HISTORY_FILENAME
    holdout_path = out_dir / HOLDOUT_FILENAME
    zones_path = out_dir / ZONES_FILENAME
    truth_path = out_dir / GROUND_TRUTH_FILENAME

    if truth_path.exists() and not force:
        raise WriteRefused(
            f"{truth_path} already exists. Pass force=True (or --force) to replace it."
        )

    written_history = write_rows(history_path, history_rows, force=force)
    written_holdout = write_rows(holdout_path, holdout_rows, force=force)

    # The location registry: coordinates live here rather than on every reading
    # (SPEC section 13). The rain gauge reports for the city as a whole.
    zone_rows = [
        ZoneRow(
            location_id=zone.location_id,
            description=zone.description,
            latitude=f"{zone.latitude:.6f}",
            longitude=f"{zone.longitude:.6f}",
            sensor_id=zone.sensor_id,
        )
        for zone in config.zones
    ]
    zone_rows.append(
        ZoneRow(
            location_id=config.rainfall_location_id,
            description="city-wide rain gauge reference point",
            latitude="13.082700",
            longitude="80.270700",
            sensor_id="rain-gauge-01",
        )
    )
    write_rows(zones_path, zone_rows, force=force, columns=ZONE_COLUMNS)

    def span(rows: list[RawRow]) -> tuple[datetime, datetime]:
        stamps = [
            _parse_local_naive(row.interval_end_local)
            for row in rows
            if row.interval_end_local != "not-a-timestamp"
        ]
        return min(stamps), max(stamps)

    history_span = span(history_rows)
    holdout_span = span(holdout_rows)

    ground_truth = ground_truth.model_copy(
        update={
            "datasets": {
                "history": DatasetTruth(
                    path=HISTORY_FILENAME,
                    row_count=written_history,
                    start=history_span[0],
                    end=history_span[1],
                    columns=CSV_COLUMNS,
                ),
                "holdout": DatasetTruth(
                    path=HOLDOUT_FILENAME,
                    row_count=written_holdout,
                    start=holdout_span[0],
                    end=holdout_span[1],
                    columns=CSV_COLUMNS,
                ),
                "zones": DatasetTruth(
                    path=ZONES_FILENAME,
                    row_count=len(zone_rows),
                    start=history_span[0],
                    end=holdout_span[1],
                    columns=ZONE_COLUMNS,
                ),
            }
        }
    )
    ground_truth.write(truth_path)

    return GeneratedDataset(
        history_path=history_path,
        holdout_path=holdout_path,
        zones_path=zones_path,
        ground_truth_path=truth_path,
        history_rows=written_history,
        holdout_rows=written_holdout,
        ground_truth=ground_truth,
    )
