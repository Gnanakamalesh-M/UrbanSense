"""Tests for the synthetic generator.

The generator's purpose is to produce data whose answers are known, so these
tests check two separate things:

1. **Determinism** -- the same seed yields byte-identical output, which is what
   makes the committed sample data verifiable by regeneration.
2. **The planted effects are really in the data** -- not merely in the config.
   A generator that recorded a Friday pattern in ``ground_truth.json`` without
   putting one in the CSV would silently invalidate every later phase's tests,
   so each effect is measured back out of the generated rows.

Where an effect is measured, it is measured **against a control zone**. Friday
has a higher weekday factor than Tuesday on its own, so comparing Friday
evenings to other evenings overstates the planted pattern by that factor. Taking
the ratio of ratios (affected zone vs untouched zone) cancels the shared weekly
and seasonal shape and recovers the planted multiplier itself.
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from urbansense.ingestion.rows import snap_to_interval
from urbansense.synthetic import (
    CSV_COLUMNS,
    FORBIDDEN_COLUMN_PATTERNS,
    GroundTruth,
    WriteRefused,
    generate,
    generate_to_directory,
    load_generator_config,
    write_rows,
)
from urbansense.synthetic.generator import (
    GROUND_TRUTH_FILENAME,
    HISTORY_FILENAME,
    HOLDOUT_FILENAME,
    ZONES_FILENAME,
)
from urbansense.synthetic.loader import GeneratorConfigError
from urbansense.synthetic.writer import ZONE_COLUMNS, RawRow

# ---------------------------------------------------------------------------
# Fixtures: generate once per module, since the full dataset takes ~1.5s
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    return load_generator_config()


@pytest.fixture(scope="module")
def generated(config):
    """``(history_rows, holdout_rows, ground_truth)`` from the shipped config."""
    return generate(config)


@pytest.fixture(scope="module")
def history(generated):
    return generated[0]


@pytest.fixture(scope="module")
def truth(generated):
    return generated[2]


def _parse_stamp(row: RawRow) -> datetime:
    """The source's interval-end timestamp as an aware datetime."""
    return datetime.fromisoformat(row.interval_end_local)


def _event_start(row: RawRow, interval: timedelta) -> datetime:
    """The interval start in the source's **local** wall-clock time.

    The feed stamps interval *end*, so the Friday 18:00-20:00 window appears in
    the raw data as stamps of 19:00 and 20:00. Deriving the start is what a
    consumer of this feed has to do, and getting it wrong would shift every
    pattern by an hour.

    Returned naive, in local time, because that is the frame the planted
    patterns and ground-truth windows are expressed in -- "Friday 18:00" is a
    local-clock claim, not a UTC one.

    Uses the same :func:`snap_to_interval` the real pipeline uses, which absorbs
    the sensor clock skew baked into the raw stamps. Without it, a sensor 52
    seconds behind would place its 18:00 reading at 17:59:08 and the pattern
    would appear to span the wrong hour.
    """
    snapped = snap_to_interval(_parse_stamp(row), interval)
    return (snapped - interval).replace(tzinfo=None)


def _values(
    rows: list[RawRow], *, metric: str, location_id: str, interval: timedelta
) -> list[tuple[datetime, float]]:
    """Clean present values for one metric and zone, as ``(event_start, value)``.

    Skips missing cells, the unparseable-timestamp rows and the planted negative
    values, so a measurement of a pattern is not polluted by planted corruption.
    """
    out: list[tuple[datetime, float]] = []
    for row in rows:
        if row.metric != metric or row.location_id != location_id or not row.value:
            continue
        if row.interval_end_local == "not-a-timestamp":
            continue
        value = float(row.value)
        if value < 0:
            continue
        out.append((_event_start(row, interval), value))
    return out


def _mean_ratio(
    rows: list[RawRow],
    *,
    location_id: str,
    interval: timedelta,
    metric: str,
    weekday: int,
    start_hour: int,
    end_hour: int,
) -> float:
    """Mean inside the target window divided by mean in the same hours elsewhere.

    Comparing like hours to like hours removes the diurnal shape; what remains
    is the weekday effect plus any planted pattern.
    """
    series = _values(rows, metric=metric, location_id=location_id, interval=interval)
    inside = [v for t, v in series if t.weekday() == weekday and start_hour <= t.hour < end_hour]
    outside = [v for t, v in series if t.weekday() != weekday and start_hour <= t.hour < end_hour]
    assert inside and outside, f"no data to compare for {location_id}"
    return statistics.mean(inside) / statistics.mean(outside)


# ---------------------------------------------------------------------------
# Determinism (SPEC section 39)
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_rows(config):
    first_history, first_holdout, first_truth = generate(config)
    second_history, second_holdout, second_truth = generate(config)

    assert first_history == second_history
    assert first_holdout == second_holdout
    assert first_truth.to_json() == second_truth.to_json()


def test_different_seed_produces_different_data(config):
    other = config.model_copy(update={"seed": config.seed + 1})
    baseline, _, _ = generate(config)
    changed, _, _ = generate(other)

    assert len(baseline) == len(changed), "row count should not depend on the seed"
    assert baseline != changed, "a different seed must produce different values"


def test_same_seed_produces_byte_identical_files(config, tmp_path):
    """Byte-identical output is what makes the committed sample verifiable."""
    first = generate_to_directory(config, tmp_path / "a")
    second = generate_to_directory(config, tmp_path / "b")

    for name in (HISTORY_FILENAME, HOLDOUT_FILENAME, ZONES_FILENAME, GROUND_TRUTH_FILENAME):
        assert (first.history_path.parent / name).read_bytes() == (
            second.history_path.parent / name
        ).read_bytes(), f"{name} is not reproducible"


def test_generation_is_independent_of_call_order(config):
    """Two runs in one process must not share state through a module-level RNG."""
    other = config.model_copy(update={"seed": config.seed + 7})
    generate(other)
    after_other, _, _ = generate(config)
    direct, _, _ = generate(config)
    assert after_other == direct


# ---------------------------------------------------------------------------
# No label leakage: the CSV carries only what a source would send
# ---------------------------------------------------------------------------


def test_csv_header_is_exactly_the_source_columns(config, tmp_path):
    dataset = generate_to_directory(config, tmp_path)
    with dataset.history_path.open(encoding="utf-8", newline="") as handle:
        header = tuple(next(csv.reader(handle)))
    assert header == CSV_COLUMNS


def test_csv_has_no_label_or_ground_truth_columns(config, tmp_path):
    """A label column in the data would be a leakage channel.

    A later phase that could read ``is_friday_evening`` off its input would be
    credited with a discovery it never made, so the header is checked against
    the shapes such a column would take.
    """
    dataset = generate_to_directory(config, tmp_path)
    for path in (dataset.history_path, dataset.holdout_path):
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        for column in header:
            lowered = column.lower()
            for pattern in FORBIDDEN_COLUMN_PATTERNS:
                assert pattern not in lowered, (
                    f"{path.name} column {column!r} looks like a planted-truth label; "
                    "labels belong in ground_truth.json only"
                )


def test_zone_registry_holds_the_coordinates(config, tmp_path):
    """Coordinates live in the registry, not repeated on every reading."""
    dataset = generate_to_directory(config, tmp_path)
    with dataset.zones_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0].keys()) == ZONE_COLUMNS
    registered = {row["location_id"] for row in rows}
    for zone in config.zones:
        assert zone.location_id in registered


# ---------------------------------------------------------------------------
# Friday-evening congestion is really present (SPEC section 21)
# ---------------------------------------------------------------------------


def test_friday_evening_congestion_is_present_in_zone_b(config, history, truth):
    planted = truth.patterns["friday_evening_congestion"]
    ratio = _mean_ratio(
        history,
        location_id=planted.location_id,
        interval=config.interval,
        metric=config.volume_metric,
        weekday=planted.weekday,
        start_hour=planted.start_hour,
        end_hour=planted.end_hour,
    )
    assert ratio > 1.8, (
        f"Friday {planted.start_hour}-{planted.end_hour} in {planted.location_id} is only "
        f"{ratio:.2f}x other evenings; the planted pattern is not in the data"
    )


def test_friday_pattern_is_localized_to_one_zone(config, history, truth):
    """The control zones must not show the pattern.

    If every zone rose on Friday evenings, a discovery phase could claim a
    city-wide effect and be scored correct for the wrong reason.
    """
    planted = truth.patterns["friday_evening_congestion"]
    target = _mean_ratio(
        history,
        location_id=planted.location_id,
        interval=config.interval,
        metric=config.volume_metric,
        weekday=planted.weekday,
        start_hour=planted.start_hour,
        end_hour=planted.end_hour,
    )
    for control_id in planted.control_location_ids:
        control = _mean_ratio(
            history,
            location_id=control_id,
            interval=config.interval,
            metric=config.volume_metric,
            weekday=planted.weekday,
            start_hour=planted.start_hour,
            end_hour=planted.end_hour,
        )
        assert target > control * 1.5, (
            f"{planted.location_id} ({target:.2f}x) is not clearly above control "
            f"{control_id} ({control:.2f}x)"
        )


def test_friday_multiplier_recovers_the_planted_value(config, history, truth):
    """The ratio of ratios recovers the configured multiplier.

    Friday's own weekday factor lifts every zone, so dividing the affected
    zone's Friday lift by an untouched zone's cancels the shared weekly and
    seasonal shape, leaving the planted multiplier.
    """
    planted = truth.patterns["friday_evening_congestion"]
    kwargs = {
        "interval": config.interval,
        "metric": config.volume_metric,
        "weekday": planted.weekday,
        "start_hour": planted.start_hour,
        "end_hour": planted.end_hour,
    }
    target = _mean_ratio(history, location_id=planted.location_id, **kwargs)
    controls = [
        _mean_ratio(history, location_id=control_id, **kwargs)
        for control_id in planted.control_location_ids
    ]
    recovered = target / statistics.mean(controls)
    assert recovered == pytest.approx(planted.volume_multiplier, rel=0.15), (
        f"recovered multiplier {recovered:.2f} differs from planted {planted.volume_multiplier}"
    )


def test_congestion_also_slows_traffic(config, history, truth):
    """Volume up and speed unchanged would be physically incoherent."""
    planted = truth.patterns["friday_evening_congestion"]
    ratio = _mean_ratio(
        history,
        location_id=planted.location_id,
        interval=config.interval,
        metric=config.speed_metric,
        weekday=planted.weekday,
        start_hour=planted.start_hour,
        end_hour=planted.end_hour,
    )
    assert ratio < 0.9, f"speed ratio {ratio:.2f} shows no slowdown during congestion"


def test_friday_record_ids_are_recorded_and_real(config, history, truth):
    planted = truth.patterns["friday_evening_congestion"]
    assert planted.affected_record_ids, "no Friday records were recorded in ground truth"

    by_id = {row.record_id: row for row in history}
    for record_id in planted.affected_record_ids:
        row = by_id[record_id]
        start = _event_start(row, config.interval)
        assert row.location_id == planted.location_id
        assert start.weekday() == planted.weekday
        assert planted.start_hour <= start.hour < planted.end_hour


# ---------------------------------------------------------------------------
# Rain effect (SPEC section 21)
# ---------------------------------------------------------------------------


def test_rainfall_series_is_emitted_as_its_own_observations(config, history, truth):
    planted = truth.patterns["rain_traffic_effect"]
    rain_rows = [row for row in history if row.metric == planted.rainfall_metric]
    assert rain_rows, "no rainfall observations were generated"
    assert {row.location_id for row in rain_rows} == {planted.rainfall_location_id}
    assert planted.wet_hour_count > 0 and planted.dry_hour_count > 0


def test_wet_hours_carry_more_traffic_than_dry_hours(config, history, truth):
    planted = truth.patterns["rain_traffic_effect"]
    rain = {
        _event_start(row, config.interval): float(row.value)
        for row in history
        if row.metric == planted.rainfall_metric and row.value
    }
    series = _values(
        history,
        metric=config.volume_metric,
        location_id="ZONE-C",  # no drift, no Friday pattern: a clean read
        interval=config.interval,
    )
    wet = [v for t, v in series if rain.get(t, 0.0) > 2.0]
    dry = [v for t, v in series if rain.get(t, 0.0) == 0.0]
    assert wet and dry
    assert statistics.mean(wet) > statistics.mean(dry), "rain shows no effect on volume"


def test_rain_effect_magnitude_is_near_the_planted_coefficient(config, history, truth):
    """Mean uplift should track ``1 + coefficient * mean_rain`` within tolerance."""
    planted = truth.patterns["rain_traffic_effect"]
    rain = {
        _event_start(row, config.interval): float(row.value)
        for row in history
        if row.metric == planted.rainfall_metric and row.value
    }
    series = _values(
        history, metric=config.volume_metric, location_id="ZONE-C", interval=config.interval
    )
    # Compare like hours to like hours, so the diurnal shape cannot masquerade
    # as a rain effect (rain is seasonal, not uniform across the day).
    by_hour_wet: dict[int, list[float]] = {}
    by_hour_dry: dict[int, list[float]] = {}
    wet_depths: list[float] = []
    for moment, value in series:
        depth = rain.get(moment, 0.0)
        if depth > 2.0:
            by_hour_wet.setdefault(moment.hour, []).append(value)
            wet_depths.append(min(depth, planted.cap_mm))
        elif depth == 0.0:
            by_hour_dry.setdefault(moment.hour, []).append(value)

    shared = sorted(set(by_hour_wet) & set(by_hour_dry))
    ratios = [
        statistics.mean(by_hour_wet[hour]) / statistics.mean(by_hour_dry[hour]) for hour in shared
    ]
    observed = statistics.mean(ratios)
    expected = 1.0 + planted.volume_coefficient * statistics.mean(wet_depths)
    assert observed == pytest.approx(expected, rel=0.08), (
        f"observed rain uplift {observed:.3f} vs expected {expected:.3f}"
    )


# ---------------------------------------------------------------------------
# Drift (SPEC section 29)
# ---------------------------------------------------------------------------


def test_drift_shifts_the_baseline_in_affected_zones(config, history, truth):
    planted = truth.anomalies["baseline_drift"]
    for location_id in planted.location_ids:
        series = _values(
            history,
            metric=config.volume_metric,
            location_id=location_id,
            interval=config.interval,
        )
        before = [v for t, v in series if t.date() < planted.start_date]
        after = [v for t, v in series if t.date() >= planted.start_date]
        assert before and after
        assert statistics.mean(after) > statistics.mean(before), (
            f"{location_id} shows no baseline shift after {planted.start_date}"
        )


def test_drift_magnitude_recovers_against_control_zones(config, history, truth):
    """Seasonality lifts every zone, so the control cancels it out."""
    planted = truth.anomalies["baseline_drift"]

    def shift(location_id: str) -> float:
        series = _values(
            history,
            metric=config.volume_metric,
            location_id=location_id,
            interval=config.interval,
        )
        before = [v for t, v in series if t.date() < planted.start_date]
        after = [v for t, v in series if t.date() >= planted.start_date]
        return statistics.mean(after) / statistics.mean(before)

    affected = statistics.mean([shift(loc) for loc in planted.location_ids])
    controls = statistics.mean([shift(loc) for loc in planted.control_location_ids])
    recovered = affected / controls
    assert recovered == pytest.approx(planted.volume_multiplier, rel=0.12), (
        f"recovered drift {recovered:.3f} differs from planted {planted.volume_multiplier}"
    )


def test_control_zones_do_not_drift(config, history, truth):
    """An unaffected zone must stay flat, or drift detection has nothing to find."""
    planted = truth.anomalies["baseline_drift"]
    for location_id in planted.control_location_ids:
        series = _values(
            history,
            metric=config.volume_metric,
            location_id=location_id,
            interval=config.interval,
        )
        before = [v for t, v in series if t.date() < planted.start_date]
        after = [v for t, v in series if t.date() >= planted.start_date]
        ratio = statistics.mean(after) / statistics.mean(before)
        assert ratio < 1.15, f"control zone {location_id} drifted by {ratio:.2f}x"


# ---------------------------------------------------------------------------
# Sensor failure (SPEC section 17)
# ---------------------------------------------------------------------------


def test_sensor_failure_window_holds_one_frozen_value(config, history, truth):
    planted = truth.anomalies["sensor_failure"]
    rows = [row for row in history if row.record_id in set(planted.affected_record_ids)]
    assert rows, "no frozen-sensor records were recorded"
    assert {float(row.value) for row in rows} == {planted.frozen_value}, (
        "a frozen sensor must report exactly one repeated value"
    )


def test_frozen_records_fall_inside_the_recorded_window(config, history, truth):
    planted = truth.anomalies["sensor_failure"]
    by_id = {row.record_id: row for row in history}
    for record_id in planted.affected_record_ids:
        row = by_id[record_id]
        start = _event_start(row, config.interval)
        assert row.location_id == planted.location_id
        assert planted.start <= start < planted.end


def test_frozen_value_is_plausible_not_zero(truth):
    """A frozen sensor reports a plausible-looking number.

    That is exactly why it is more dangerous than an obvious gap.
    """
    planted = truth.anomalies["sensor_failure"]
    assert planted.frozen_value > 0


# ---------------------------------------------------------------------------
# Missing data stays missing (SPEC section 15)
# ---------------------------------------------------------------------------


def test_missing_values_are_empty_cells_never_zero(history, truth):
    missing_ids = set(truth.missing_data.all_record_ids)
    assert missing_ids, "no missing records were generated"

    for row in history:
        if row.record_id in missing_ids:
            assert row.value == "", (
                f"{row.record_id} is recorded as missing but holds {row.value!r}; "
                "missing must never be written as a number"
            )
            assert row.value != "0"


def test_missing_windows_cover_their_configured_span(config, history, truth):
    by_id = {row.record_id: row for row in history}
    for record_id in truth.missing_data.window_record_ids:
        row = by_id[record_id]
        start = _event_start(row, config.interval)
        assert any(
            window.location_id == row.location_id and window.start <= start < window.end
            for window in truth.missing_data.windows
        ), f"{record_id} is marked a window gap but falls outside every window"


def test_scattered_missing_rate_is_near_the_configured_fraction(config, history, truth):
    zone_id = truth.missing_data.scattered_location_ids[0]
    rows = [
        row for row in history if row.location_id == zone_id and row.metric == config.volume_metric
    ]
    observed = sum(1 for row in rows if row.value == "") / len(rows)
    assert observed == pytest.approx(truth.missing_data.scattered_fraction, abs=0.01)


def test_the_literal_zero_is_not_used_for_absence(config, history):
    """A dry hour of rain is a real zero; a missing traffic reading is not.

    Both exist in the data, and conflating them is the mistake this guards
    against.
    """
    rain_zeros = [
        row for row in history if row.metric == config.rainfall_metric and row.value == "0.0"
    ]
    assert rain_zeros, "expected measured zero rainfall to be present"


# ---------------------------------------------------------------------------
# Planted duplicates, conflicts and invalid rows (SPEC sections 9, 10, 14)
# ---------------------------------------------------------------------------


def test_duplicates_repeat_their_originals_content(history, truth):
    by_id = {row.record_id: row for row in history}
    planted = truth.planted_records.duplicates
    assert len(planted) == truth.config.integrity.duplicate_count

    for item in planted:
        copy, original = by_id[item.record_id], by_id[item.duplicate_of]
        assert copy.record_id != original.record_id
        assert copy.interval_end_local == original.interval_end_local
        assert copy.location_id == original.location_id
        assert copy.metric == original.metric
        assert copy.value == original.value
        assert copy.source_id == original.source_id


def test_conflicts_disagree_on_value_from_a_different_source(history, truth):
    by_id = {row.record_id: row for row in history}
    planted = truth.planted_records.conflicts
    assert len(planted) == truth.config.integrity.conflict_count

    for item in planted:
        copy, original = by_id[item.record_id], by_id[item.conflicts_with]
        assert copy.interval_end_local == original.interval_end_local
        assert copy.location_id == original.location_id
        assert copy.metric == original.metric
        assert copy.source_id != original.source_id
        assert float(copy.value) != float(original.value)
        assert float(copy.value) == pytest.approx(item.conflicting_value)


def test_invalid_rows_are_present_and_classified(history, truth):
    by_id = {row.record_id: row for row in history}
    kinds = {item.kind for item in truth.planted_records.invalid}
    assert kinds == {
        "negative_value",
        "impossible_speed",
        "unparseable_timestamp",
        "unknown_unit",
    }

    for item in truth.planted_records.invalid:
        row = by_id[item.record_id]
        if item.kind == "negative_value":
            assert float(row.value) < 0
        elif item.kind == "impossible_speed":
            assert float(row.value) == truth.config.integrity.impossible_speed_value
        elif item.kind == "unparseable_timestamp":
            with pytest.raises(ValueError):
                datetime.fromisoformat(row.interval_end_local)
        elif item.kind == "unknown_unit":
            assert row.unit == truth.config.integrity.unknown_unit


def test_unparseable_rows_are_flagged_as_not_ingestible(truth):
    """Ground truth separates rows ingestion must reject from rows it must keep.

    The keepable ones pass through to Phase 2 as PENDING.
    """
    not_ingestible = {item.kind for item in truth.planted_records.invalid if not item.ingestible}
    assert not_ingestible == {"unparseable_timestamp", "unknown_unit"}


def test_planted_record_ids_are_unique_and_not_reused(history, truth):
    ids = [row.record_id for row in history]
    assert len(ids) == len(set(ids)), "record ids must be unique"

    planted_ids = (
        {item.record_id for item in truth.planted_records.duplicates}
        | {item.record_id for item in truth.planted_records.conflicts}
        | {item.record_id for item in truth.planted_records.invalid}
    )
    missing_ids = set(truth.missing_data.all_record_ids)
    assert not (planted_ids & missing_ids), "a planted corruption must not also be a missing record"


# ---------------------------------------------------------------------------
# Holdout period (SPEC section 33)
# ---------------------------------------------------------------------------


def test_holdout_period_follows_the_history_and_never_overlaps(config, generated):
    history, holdout, _ = generated
    interval = config.interval

    history_end = max(
        _event_start(row, interval)
        for row in history
        if row.interval_end_local != "not-a-timestamp"
    )
    holdout_start = min(_event_start(row, interval) for row in holdout)
    assert holdout_start > history_end, (
        "the holdout period must start after the history ends, or a model could "
        "train on what it is later evaluated against"
    )


def test_holdout_contains_no_planted_corruption(generated, truth):
    """The holdout stays clean.

    It feeds the live simulator, so stream tests measure streaming rather than
    corruption handling.
    """
    _, holdout, _ = generated
    holdout_ids = {row.record_id for row in holdout}
    planted = (
        {item.record_id for item in truth.planted_records.duplicates}
        | {item.record_id for item in truth.planted_records.conflicts}
        | {item.record_id for item in truth.planted_records.invalid}
    )
    assert not (holdout_ids & planted)


# ---------------------------------------------------------------------------
# Config matches the problem definition (SPEC section 3)
# ---------------------------------------------------------------------------


def test_generated_metrics_match_the_traffic_problem_config(config, settings_fixture=None):
    """The generator must not invent metric names the project does not define.

    If these drifted apart, ingestion would silently fail to resolve the metric
    and every downstream join would be on the wrong key.
    """
    from urbansense.config import load_settings
    from urbansense.schemas.enums import ProblemType

    settings = load_settings()
    for metric_name in (config.volume_metric, config.speed_metric):
        resolved = settings.resolve_metric(ProblemType.TRAFFIC, metric_name)
        assert resolved is not None, f"{metric_name} is not defined in configs/problems"
        assert resolved.name == metric_name


def test_generated_units_are_accepted_by_the_problem_config(config, history):
    from urbansense.config import load_settings
    from urbansense.schemas.enums import ProblemType

    settings = load_settings()
    unknown_unit = config.integrity.unknown_unit

    for row in history:
        if row.metric not in (config.volume_metric, config.speed_metric):
            continue
        if row.unit == unknown_unit:  # planted on purpose
            continue
        resolved = settings.resolve_metric(ProblemType.TRAFFIC, row.metric)
        assert resolved is not None
        assert row.unit in resolved.accepted_units, (
            f"{row.unit!r} is not an accepted unit for {row.metric}"
        )


# ---------------------------------------------------------------------------
# Writer behaviour (SPEC section 4A)
# ---------------------------------------------------------------------------


def test_writer_refuses_to_overwrite_without_force(tmp_path):
    """Raw data is immutable; a rerun must not silently destroy a dataset."""
    path = tmp_path / "rows.csv"
    row = RawRow(
        record_id="REC-000001",
        source_id="sensor-A12",
        interval_end_local="2026-01-01 01:00:00+05:30",
        location_id="ZONE-A",
        metric="traffic_volume",
        value="100",
        unit="vehicles/hour",
    )
    assert write_rows(path, [row]) == 1

    with pytest.raises(WriteRefused, match="already exists"):
        write_rows(path, [row])

    assert write_rows(path, [row, row], force=True) == 2


def test_generate_to_directory_refuses_to_overwrite(config, tmp_path):
    generate_to_directory(config, tmp_path)
    with pytest.raises(WriteRefused):
        generate_to_directory(config, tmp_path)
    generate_to_directory(config, tmp_path, force=True)


def test_writer_leaves_no_temporary_file_behind(tmp_path):
    path = tmp_path / "rows.csv"
    write_rows(path, [])
    assert list(tmp_path.iterdir()) == [path]


# ---------------------------------------------------------------------------
# Ground-truth file round-trip
# ---------------------------------------------------------------------------


def test_ground_truth_file_round_trips(config, tmp_path):
    dataset = generate_to_directory(config, tmp_path)
    reloaded = GroundTruth.load(dataset.ground_truth_path)
    assert reloaded.seed == config.seed
    assert reloaded.generator_version == dataset.ground_truth.generator_version
    assert reloaded.to_json() == dataset.ground_truth.to_json()


def test_ground_truth_json_is_sorted_and_readable(config, tmp_path):
    dataset = generate_to_directory(config, tmp_path)
    payload = json.loads(dataset.ground_truth_path.read_text(encoding="utf-8"))
    assert list(payload) == sorted(payload)
    assert payload["seed"] == config.seed
    assert "friday_evening_congestion" in payload["patterns"]


def test_ground_truth_records_dataset_shape(config, tmp_path):
    dataset = generate_to_directory(config, tmp_path)
    datasets = dataset.ground_truth.datasets
    assert datasets["history"].row_count == dataset.history_rows
    assert datasets["holdout"].row_count == dataset.holdout_rows
    assert datasets["history"].columns == CSV_COLUMNS


def test_ground_truth_names_windows_to_exclude_from_measurements(truth):
    """Measuring a pattern inside a gap or a frozen window would distort it."""
    assert truth.exclude_windows
    locations = {window.location_id for window in truth.exclude_windows}
    assert truth.anomalies["sensor_failure"].location_id in locations


def test_clock_skew_is_recorded_per_sensor(config, truth):
    assert truth.clock_skew_seconds
    for zone in config.zones:
        assert truth.clock_skew_seconds[zone.sensor_id] == zone.clock_skew_seconds


# ---------------------------------------------------------------------------
# Config validation guards the dataset's purpose
# ---------------------------------------------------------------------------


def test_effect_referencing_an_unknown_zone_is_rejected(config):
    """An effect naming a nonexistent zone must be rejected.

    A typo here would produce a dataset with no pattern in it, quietly
    invalidating every later phase's tests.
    """
    broken = config.friday_evening.model_copy(update={"location_id": "ZONE-NOPE"})
    with pytest.raises(ValueError, match="reference zones that do not exist"):
        config.model_copy(update={"friday_evening": broken}).model_validate(
            config.model_copy(update={"friday_evening": broken}).model_dump()
        )


def test_drift_outside_the_series_is_rejected(config):
    from urbansense.synthetic.config import GeneratorConfig

    payload = config.model_dump()
    payload["drift"]["start_date"] = "2030-01-01"
    with pytest.raises(ValueError, match="must fall inside the generated period"):
        GeneratorConfig.model_validate(payload)


def test_holdout_must_follow_history(config):
    from urbansense.synthetic.config import GeneratorConfig

    payload = config.model_dump()
    payload["holdout_end"] = payload["history_end"]
    with pytest.raises(ValueError, match="must be after history_end"):
        GeneratorConfig.model_validate(payload)


def test_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(GeneratorConfigError, match="not found"):
        load_generator_config(tmp_path / "nope.yaml")


def test_malformed_config_file_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("zones: [\n", encoding="utf-8")
    with pytest.raises(GeneratorConfigError, match="not valid YAML"):
        load_generator_config(path)


def test_config_seed_can_be_overridden(tmp_path):
    overridden = load_generator_config(seed=99)
    assert overridden.seed == 99


def test_shipped_config_path_exists():
    from urbansense.synthetic import DEFAULT_CONFIG_PATH

    assert DEFAULT_CONFIG_PATH.is_file()
    assert Path(DEFAULT_CONFIG_PATH).name == "traffic_demo.yaml"
