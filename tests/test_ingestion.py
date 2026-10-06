"""Tests for historical and real-time ingestion.

These pin the properties the rest of the system relies on:

* **nothing is lost** -- every input row becomes an observation or a quarantine
  record, and the result says so;
* **the three timestamps stay distinct** -- event, source and receipt, each
  meaning something different;
* **missing stays missing** -- never silently zero;
* **raw files are never modified** -- verified by hashing before and after;
* **ingestion transcribes rather than judges** -- an impossible value passes
  through as ``PENDING`` for the integrity engine to catch later.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from urbansense.config import load_settings
from urbansense.ingestion import (
    PIPELINE_VERSION,
    REQUIRED_COLUMNS,
    StreamSimulator,
    ZoneRegistry,
    aggregation_for,
    load_historical,
    parse_row,
)
from urbansense.ingestion.rows import ParsedRow, RowParseError, snap_to_interval
from urbansense.schemas.enums import (
    AggregationLevel,
    ProblemType,
    QuarantineReason,
    SourceType,
    ValidationStatus,
)
from urbansense.synthetic import generate_to_directory, load_generator_config

HOUR = timedelta(hours=1)
FIXED_RECEIVED = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Fixtures: one generated dataset for the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generator_config():
    return load_generator_config()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, generator_config):
    """A generated dataset written to a temporary directory."""
    out = tmp_path_factory.mktemp("dataset")
    return generate_to_directory(generator_config, out)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def registry(dataset):
    return ZoneRegistry.from_csv(dataset.zones_path)


@pytest.fixture(scope="module")
def ingested(dataset, settings, registry):
    """The full historical ingestion, done once."""
    return load_historical(
        dataset.history_path,
        settings=settings,
        registry=registry,
        received_time=FIXED_RECEIVED,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(**overrides: str) -> dict[str, str]:
    """A minimal valid feed row, with overrides."""
    payload = {
        "record_id": "REC-000001",
        "source_id": "sensor-B07",
        "interval_end_local": "2026-02-06 19:00:00+05:30",
        "location_id": "ZONE-B",
        "metric": "traffic_volume",
        "value": "8200",
        "unit": "vehicles/hour",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------


def test_valid_row_parses(ingested=None):
    parsed = parse_row(_row(), interval=HOUR)
    assert isinstance(parsed, ParsedRow)
    assert parsed.value == 8200.0
    assert parsed.location_id == "ZONE-B"
    assert parsed.measurement_duration == HOUR


def test_required_columns_are_declared():
    assert "interval_end_local" in REQUIRED_COLUMNS
    assert "value" in REQUIRED_COLUMNS


def test_missing_column_is_rejected_with_payload():
    payload = _row()
    del payload["unit"]
    result = parse_row(payload, interval=HOUR)
    assert isinstance(result, RowParseError)
    assert result.reason is QuarantineReason.SCHEMA_ERROR
    assert result.payload == payload


def test_unparseable_timestamp_is_rejected():
    result = parse_row(_row(interval_end_local="not-a-timestamp"), interval=HOUR)
    assert isinstance(result, RowParseError)
    assert result.reason is QuarantineReason.INVALID_TIMESTAMP
    assert result.payload["interval_end_local"] == "not-a-timestamp"


def test_naive_timestamp_is_rejected_rather_than_guessed():
    """An instant without an offset is ambiguous and will not be guessed at."""
    result = parse_row(_row(interval_end_local="2026-02-06 19:00:00"), interval=HOUR)
    assert isinstance(result, RowParseError)
    assert result.reason is QuarantineReason.INVALID_TIMESTAMP
    assert "offset" in result.detail


def test_non_numeric_value_is_rejected():
    result = parse_row(_row(value="eight thousand"), interval=HOUR)
    assert isinstance(result, RowParseError)
    assert result.reason is QuarantineReason.INVALID_VALUE


def test_unknown_unit_is_rejected_not_coerced():
    result = parse_row(
        _row(unit="vehicles/fortnight"),
        interval=HOUR,
        accepted_units=frozenset({"vehicles/hour"}),
    )
    assert isinstance(result, RowParseError)
    assert result.reason is QuarantineReason.UNKNOWN_UNIT


def test_empty_required_field_is_rejected():
    for field in ("record_id", "location_id", "metric", "unit"):
        result = parse_row(_row(**{field: "  "}), interval=HOUR)
        assert isinstance(result, RowParseError), f"{field} empty should be rejected"


def test_negative_value_parses_fine():
    """Ingestion transcribes; judging the value is the integrity engine's job."""
    parsed = parse_row(_row(value="-4200"), interval=HOUR)
    assert isinstance(parsed, ParsedRow)
    assert parsed.value == -4200.0


# ---------------------------------------------------------------------------
# Clock-skew snapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stamp", "expected_hour", "expected_minute"),
    [
        ("2026-02-06 19:00:00+05:30", 19, 0),
        ("2026-02-06 19:00:37+05:30", 19, 0),  # sensor 37s fast
        ("2026-02-06 18:59:08+05:30", 19, 0),  # sensor 52s slow
        ("2026-02-06 19:01:28+05:30", 19, 0),  # sensor 88s fast
    ],
)
def test_snapping_absorbs_sensor_clock_skew(stamp, expected_hour, expected_minute):
    """Readings for one hour must share an event_time, or joins break."""
    snapped = snap_to_interval(datetime.fromisoformat(stamp), HOUR)
    assert (snapped.hour, snapped.minute, snapped.second) == (
        expected_hour,
        expected_minute,
        0,
    )


def test_snapping_respects_a_half_hour_utc_offset():
    """IST hour boundaries sit at :30 past the UTC hour.

    Snapping against the UTC epoch instead of the local offset would put every
    reading half an interval from where it belongs.
    """
    snapped = snap_to_interval(datetime.fromisoformat("2026-02-06 19:00:14+05:30"), HOUR)
    assert snapped.minute == 0
    assert snapped.utcoffset() == timedelta(hours=5, minutes=30)


def test_snapping_is_a_no_op_for_a_zero_interval():
    moment = datetime.fromisoformat("2026-02-06 19:00:37+05:30")
    assert snap_to_interval(moment, timedelta(0)) == moment


def test_raw_source_time_keeps_the_skew_for_later_detection(dataset, settings, registry):
    """Skew is normalized out of event_time but preserved in source_time.

    That is what keeps a drifting sensor clock detectable (SPEC section 17).
    """
    parsed = parse_row(_row(interval_end_local="2026-02-06 19:00:37+05:30"), interval=HOUR)
    assert isinstance(parsed, ParsedRow)
    assert parsed.source_time.second == 37
    assert parsed.event_time.second == 0


# ---------------------------------------------------------------------------
# The three timestamps (SPEC sections 4B, 12)
# ---------------------------------------------------------------------------


def test_all_three_timestamps_are_present_and_distinct(ingested):
    observation = ingested.observations[0]
    assert observation.event_time is not None
    assert observation.source_time is not None
    assert observation.received_time is not None
    assert observation.event_time != observation.received_time
    assert observation.source_time != observation.event_time


def test_source_time_leads_event_time_by_one_interval(ingested):
    """The feed stamps interval end; event_time is interval start."""
    checked = 0
    for observation in ingested.observations[:500]:
        gap = observation.source_time - observation.event_time
        # One interval, plus that sensor's clock skew, which stays in
        # source_time because nobody told the pipeline what the skew is.
        assert timedelta(minutes=58) <= gap <= timedelta(minutes=62)
        checked += 1
    assert checked > 0


def test_received_time_is_recorded_separately_from_the_event(ingested):
    for observation in ingested.observations[:200]:
        assert observation.received_time == FIXED_RECEIVED
        assert observation.received_time > observation.event_time


def test_timestamps_are_utc(ingested):
    for observation in ingested.observations[:200]:
        assert observation.event_time.tzinfo == UTC
        assert observation.source_time is not None
        assert observation.source_time.tzinfo == UTC


def test_local_timestamp_string_is_recoverable_from_provenance(ingested, dataset):
    """The source's own wall-clock string survives ingestion.

    Keeping it is what lets the UTC normalization be audited after the fact.
    """
    with dataset.history_path.open(encoding="utf-8", newline="") as handle:
        first_row = next(csv.DictReader(handle))

    observation = ingested.observations[0]
    assert observation.provenance.notes is not None
    assert first_row["interval_end_local"] in observation.provenance.notes


def test_local_time_converts_to_the_correct_utc_instant():
    parsed = parse_row(_row(interval_end_local="2026-02-06 19:00:00+05:30"), interval=HOUR)
    assert isinstance(parsed, ParsedRow)
    # 19:00 IST is 13:30 UTC; the event began an hour earlier, at 12:30 UTC.
    assert parsed.source_time == datetime(2026, 2, 6, 13, 30, tzinfo=UTC)
    assert parsed.event_time == datetime(2026, 2, 6, 12, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Provenance (SPEC section 7)
# ---------------------------------------------------------------------------


def test_every_observation_carries_provenance(ingested):
    for observation in ingested.observations:
        assert observation.provenance is not None
        assert observation.provenance.origin
        assert observation.provenance.pipeline_version == PIPELINE_VERSION
        assert observation.provenance.ingested_at == FIXED_RECEIVED


def test_provenance_origin_is_the_reporting_source(ingested):
    for observation in ingested.observations[:200]:
        assert observation.provenance.origin == observation.source_id


def test_source_record_id_is_retained(ingested):
    for observation in ingested.observations[:200]:
        assert observation.attributes["record_id"]
        assert observation.observation_id.endswith(observation.attributes["record_id"])


# ---------------------------------------------------------------------------
# Missing stays missing (SPEC section 15)
# ---------------------------------------------------------------------------


def test_missing_values_stay_none_and_never_become_zero(ingested, dataset):
    missing_ids = set(dataset.ground_truth.missing_data.all_record_ids)
    assert missing_ids

    checked = 0
    for observation in ingested.observations:
        if observation.attributes["record_id"] in missing_ids:
            assert observation.value is None, "a missing reading must not acquire a value"
            assert observation.value != 0
            assert observation.is_missing
            assert observation.is_imputed is False
            assert observation.imputation_method is None
            checked += 1

    assert checked == len(missing_ids), (
        f"expected {len(missing_ids)} missing observations, found {checked}"
    )


def test_missing_count_is_reported(ingested, dataset):
    expected = len(dataset.ground_truth.missing_data.all_record_ids)
    assert ingested.stats.missing_values == expected


def test_an_empty_cell_and_a_zero_cell_are_different(ingested):
    empty = parse_row(_row(value=""), interval=HOUR)
    zero = parse_row(_row(value="0"), interval=HOUR)
    assert isinstance(empty, ParsedRow) and isinstance(zero, ParsedRow)
    assert empty.value is None
    assert zero.value == 0.0
    assert empty.value != zero.value


def test_measured_zero_rainfall_is_kept_as_a_measurement(ingested, generator_config):
    """A dry hour is a measurement of no rain, not an absence of measurement."""
    zeros = [
        obs
        for obs in ingested.observations
        if obs.metric == generator_config.rainfall_metric and obs.value == 0.0
    ]
    assert zeros, "expected measured zeroes for dry hours"
    for observation in zeros[:20]:
        assert observation.is_missing is False


# ---------------------------------------------------------------------------
# Nothing is lost (SPEC section 18)
# ---------------------------------------------------------------------------


def test_every_row_is_accounted_for(ingested, dataset):
    assert ingested.accounts_for_every_row
    assert ingested.stats.rows_read == dataset.history_rows
    assert len(ingested.observations) + len(ingested.rejected) == dataset.history_rows


def test_unrepresentable_rows_are_rejected_not_dropped(ingested, dataset):
    expected = set(dataset.ground_truth.planted_records.unparseable_record_ids)
    assert expected, "the dataset should contain rows ingestion must reject"

    rejected_ids = {record.raw_payload["record_id"] for record in ingested.rejected}
    assert expected <= rejected_ids, (
        f"planted unparseable rows were not rejected: {sorted(expected - rejected_ids)}"
    )


def test_rejected_records_preserve_the_original_payload(ingested):
    assert ingested.rejected
    for record in ingested.rejected:
        assert record.raw_payload
        assert set(record.raw_payload) >= {"record_id", "metric", "value"}
        assert record.detector == "ingestion.parse_row"
        assert record.detail
        assert record.resolved is False


def test_rejected_reasons_match_the_planted_defects(ingested, dataset):
    by_record = {record.raw_payload["record_id"]: record for record in ingested.rejected}
    for item in dataset.ground_truth.planted_records.invalid:
        if item.ingestible:
            continue
        record = by_record[item.record_id]
        if item.kind == "unparseable_timestamp":
            assert record.reason is QuarantineReason.INVALID_TIMESTAMP
        elif item.kind == "unknown_unit":
            assert record.reason is QuarantineReason.UNKNOWN_UNIT


def test_rejection_count_is_reported(ingested):
    assert ingested.stats.rejected == len(ingested.rejected)
    assert ingested.stats.observations == len(ingested.observations)


# ---------------------------------------------------------------------------
# Ingestion transcribes rather than judges (SPEC section 8)
# ---------------------------------------------------------------------------


def test_observations_arrive_pending_not_valid(ingested):
    """Ingestion does not vouch for data; the integrity engine decides."""
    for observation in ingested.observations[:300]:
        assert observation.validation_status is ValidationStatus.PENDING


def test_impossible_values_pass_through_for_phase_two(ingested, dataset):
    """A negative volume is representable, so it becomes a PENDING observation.

    Catching it is Phase 2's job; rejecting it here would pre-empt the thing
    Phase 2 exists to prove.
    """
    planted = {
        item.record_id
        for item in dataset.ground_truth.planted_records.invalid
        if item.kind == "negative_value"
    }
    assert planted

    found = {
        obs.attributes["record_id"]
        for obs in ingested.observations
        if obs.attributes["record_id"] in planted
    }
    assert found == planted
    for observation in ingested.observations:
        if observation.attributes["record_id"] in planted:
            assert observation.value is not None and observation.value < 0
            assert observation.validation_status is ValidationStatus.PENDING


def test_planted_duplicates_both_survive_ingestion(ingested, dataset):
    """Duplicates are linked later, not dropped now."""
    by_record = {obs.attributes["record_id"]: obs for obs in ingested.observations}
    for item in dataset.ground_truth.planted_records.duplicates:
        assert item.record_id in by_record
        assert item.duplicate_of in by_record
        copy, original = by_record[item.record_id], by_record[item.duplicate_of]
        assert copy.value == original.value
        assert copy.event_time == original.event_time
        # Ingestion does not yet link them; that is Phase 2's job.
        assert copy.duplicate_of is None


def test_planted_conflicts_both_survive_ingestion(ingested, dataset):
    by_record = {obs.attributes["record_id"]: obs for obs in ingested.observations}
    for item in dataset.ground_truth.planted_records.conflicts:
        copy, original = by_record[item.record_id], by_record[item.conflicts_with]
        assert copy.event_time == original.event_time
        assert copy.location_id == original.location_id
        assert copy.value != original.value
        assert copy.conflicts_with == ()


# ---------------------------------------------------------------------------
# Raw files are never modified (SPEC section 4A)
# ---------------------------------------------------------------------------


def test_raw_file_is_unchanged_after_ingestion(dataset, settings, registry):
    path = dataset.history_path
    before_hash, before_mtime, before_size = (
        _sha256(path),
        path.stat().st_mtime_ns,
        path.stat().st_size,
    )

    load_historical(path, settings=settings, registry=registry)

    assert _sha256(path) == before_hash, "ingestion modified the raw file"
    assert path.stat().st_mtime_ns == before_mtime
    assert path.stat().st_size == before_size


def test_ingestion_creates_no_files(dataset, settings, registry, tmp_path, monkeypatch):
    """Ingestion writes nothing at all.

    Rejections are returned rather than written, so the IO decision stays with
    the caller and nothing here can touch data/raw.
    """
    monkeypatch.chdir(tmp_path)
    result = load_historical(dataset.history_path, settings=settings, registry=registry)
    assert result.rejected
    assert list(tmp_path.iterdir()) == []


def test_stream_does_not_modify_its_input(dataset):
    path = dataset.holdout_path
    before = _sha256(path)
    simulator = StreamSimulator(path, speed=0.0)
    list(simulator.observations(limit=50))
    assert _sha256(path) == before


# ---------------------------------------------------------------------------
# Metric and unit resolution (SPEC sections 5, 11)
# ---------------------------------------------------------------------------


def test_canonical_metric_names_are_used(ingested, generator_config):
    names = {obs.metric for obs in ingested.observations}
    assert generator_config.volume_metric in names
    assert generator_config.speed_metric in names


def test_source_metric_spelling_is_retained(ingested):
    for observation in ingested.observations[:100]:
        assert "source_metric" in observation.attributes


def test_alias_resolves_to_the_canonical_metric(tmp_path, settings):
    """A feed calling it "vehicle count" lands on traffic_volume."""
    path = tmp_path / "aliased.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(_row(metric="vehicle count"))

    result = load_historical(path, settings=settings, received_time=FIXED_RECEIVED)
    assert len(result.observations) == 1
    observation = result.observations[0]
    assert observation.metric == "traffic_volume"
    assert observation.attributes["source_metric"] == "vehicle count"


def test_unrecognized_metric_keeps_its_own_name(tmp_path, settings):
    """Renaming a metric we do not recognize would be inventing information."""
    path = tmp_path / "unknown.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(_row(metric="sunspot_index", unit="count"))

    result = load_historical(path, settings=settings, received_time=FIXED_RECEIVED)
    assert result.observations[0].metric == "sunspot_index"


def test_rainfall_is_ingested_even_though_flood_is_disabled(ingested, generator_config):
    """Rain is ingested even though the flood domain is disabled.

    Discarding it for that reason would throw away a perfectly good covariate.
    """
    rainfall = [
        obs for obs in ingested.observations if obs.metric == generator_config.rainfall_metric
    ]
    assert rainfall
    assert all(obs.unit == generator_config.rainfall_unit for obs in rainfall[:50])


def test_original_unit_is_preserved(ingested):
    for observation in ingested.observations[:100]:
        assert observation.unit


# ---------------------------------------------------------------------------
# Aggregation level (SPEC section 10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (timedelta(minutes=1), AggregationLevel.MINUTE),
        (timedelta(minutes=15), AggregationLevel.QUARTER_HOUR),
        (timedelta(hours=1), AggregationLevel.HOURLY),
        (timedelta(days=1), AggregationLevel.DAILY),
        (timedelta(seconds=30), AggregationLevel.INSTANT),
    ],
)
def test_aggregation_level_follows_the_interval(interval, expected):
    assert aggregation_for(interval) is expected


def test_hourly_data_is_marked_hourly(ingested):
    for observation in ingested.observations[:100]:
        assert observation.aggregation_level is AggregationLevel.HOURLY
        assert observation.measurement_duration == HOUR


# ---------------------------------------------------------------------------
# Zone registry (SPEC section 13)
# ---------------------------------------------------------------------------


def test_coordinates_come_from_the_registry(ingested, registry):
    assert len(registry) > 0
    located = [obs for obs in ingested.observations[:200] if obs.latitude is not None]
    assert located
    for observation in located:
        assert -90 <= observation.latitude <= 90
        assert -180 <= observation.longitude <= 180


def test_unknown_zone_gets_no_coordinates_rather_than_a_guess(tmp_path, settings):
    """SPEC section 13 is explicit that similar names are not assumed identical."""
    path = tmp_path / "unknown_zone.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(_row(location_id="ZONE-UNMAPPED"))

    result = load_historical(
        path, settings=settings, registry=ZoneRegistry.empty(), received_time=FIXED_RECEIVED
    )
    observation = result.observations[0]
    assert observation.latitude is None
    assert observation.longitude is None


def test_registry_skips_malformed_entries(tmp_path):
    path = tmp_path / "zones.csv"
    path.write_text(
        "location_id,description,latitude,longitude,sensor_id\n"
        "ZONE-A,ok,13.0,80.0,sensor-A12\n"
        "ZONE-B,bad,not-a-number,80.0,sensor-B07\n",
        encoding="utf-8",
    )
    registry = ZoneRegistry.from_csv(path)
    assert registry.get("ZONE-A") == (13.0, 80.0)
    assert registry.get("ZONE-B") == (None, None)


# ---------------------------------------------------------------------------
# Source type separation (SPEC section 4)
# ---------------------------------------------------------------------------


def test_historical_load_is_marked_historical(ingested):
    for observation in ingested.observations[:100]:
        assert observation.source_type is SourceType.HISTORICAL


def test_stream_observations_are_marked_real_time(dataset, settings, registry):
    simulator = StreamSimulator(
        dataset.holdout_path, settings=settings, registry=registry, speed=0.0
    )
    for observation in simulator.observations(limit=20):
        assert observation.source_type is SourceType.REAL_TIME


# ---------------------------------------------------------------------------
# Real-time simulator (SPEC sections 4B, 37)
# ---------------------------------------------------------------------------


def test_stream_emits_in_file_order_with_non_decreasing_event_times(dataset, settings):
    simulator = StreamSimulator(dataset.holdout_path, settings=settings, speed=0.0)
    observations = list(simulator.observations(limit=200))
    assert len(observations) == 200
    event_times = [obs.event_time for obs in observations]
    assert event_times == sorted(event_times)


def test_stream_stamps_received_time_as_it_emits(dataset, settings):
    """received_time must advance across the stream while event_time does not.

    That is what makes this a live source rather than a slow file read, and it
    is what lets late-arriving data be recognized as late.
    """
    ticks = iter([datetime(2026, 10, 1, 9, 0, second=i, tzinfo=UTC) for i in range(0, 60)])
    simulator = StreamSimulator(
        dataset.holdout_path, settings=settings, speed=0.0, clock=lambda: next(ticks)
    )
    observations = list(simulator.observations(limit=20))
    received = [obs.received_time for obs in observations]
    assert received == sorted(received)
    assert received[0] < received[-1], "received_time did not advance"


def test_fast_mode_never_sleeps(dataset, settings):
    """speed=0 is what tests and demos use; it must not wait at all."""
    calls: list[float] = []
    simulator = StreamSimulator(
        dataset.holdout_path,
        settings=settings,
        speed=0.0,
        sleep=calls.append,
    )
    list(simulator.observations(limit=50))
    assert calls == []
    assert simulator.delay_seconds == 0.0


def test_speed_divides_the_configured_interval(dataset, settings):
    calls: list[float] = []
    simulator = StreamSimulator(
        dataset.holdout_path,
        settings=settings,
        interval_seconds=2.0,
        speed=4.0,
        sleep=calls.append,
    )
    list(simulator.observations(limit=5))
    assert simulator.delay_seconds == pytest.approx(0.5)
    assert calls == [pytest.approx(0.5)] * 5


def test_real_time_speed_uses_the_full_interval(dataset, settings):
    calls: list[float] = []
    simulator = StreamSimulator(
        dataset.holdout_path,
        settings=settings,
        interval_seconds=1.5,
        speed=1.0,
        sleep=calls.append,
    )
    list(simulator.observations(limit=3))
    assert calls == [pytest.approx(1.5)] * 3


def test_limit_stops_the_stream_early(dataset, settings):
    simulator = StreamSimulator(dataset.holdout_path, settings=settings, speed=0.0)
    assert len(list(simulator.observations(limit=7))) == 7


def test_stream_without_limit_replays_the_whole_file(dataset, settings):
    simulator = StreamSimulator(dataset.holdout_path, settings=settings, speed=0.0)
    result = simulator.collect()
    assert result.stats.rows_read == dataset.holdout_rows
    assert result.accounts_for_every_row


def test_stream_can_be_restarted(dataset, settings):
    simulator = StreamSimulator(dataset.holdout_path, settings=settings, speed=0.0)
    first = [obs.observation_id for obs in simulator.observations(limit=5)]
    second = [obs.observation_id for obs in simulator.observations(limit=5)]
    assert first == second


def test_negative_speed_and_interval_are_rejected(dataset):
    with pytest.raises(ValueError, match="speed"):
        StreamSimulator(dataset.holdout_path, speed=-1.0)
    with pytest.raises(ValueError, match="interval_seconds"):
        StreamSimulator(dataset.holdout_path, interval_seconds=-1.0)


def test_stream_survives_a_malformed_row(tmp_path, settings):
    """A live feed that dies on one bad message is not a live feed."""
    path = tmp_path / "mixed.csv"
    fieldnames = sorted(REQUIRED_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(_row(record_id="REC-000001"))
        writer.writerow(_row(record_id="REC-000002", interval_end_local="broken"))
        writer.writerow(_row(record_id="REC-000003"))

    simulator = StreamSimulator(path, settings=settings, speed=0.0)
    result = simulator.collect()
    assert len(result.observations) == 2
    assert len(result.rejected) == 1
    assert result.accounts_for_every_row


def test_holdout_events_precede_their_receipt(dataset, settings):
    """The holdout period is in the past, so replaying it is always valid."""
    simulator = StreamSimulator(dataset.holdout_path, settings=settings, speed=0.0)
    for observation in simulator.observations(limit=100):
        assert observation.event_time < observation.received_time


# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------


def test_result_reports_its_source_path(ingested, dataset):
    assert ingested.stats.source_path == dataset.history_path


def test_result_length_is_the_observation_count(ingested):
    assert len(ingested) == len(ingested.observations)


def test_accounts_for_every_row_detects_loss():
    from urbansense.ingestion.result import IngestionResult, IngestionStats

    lying = IngestionResult(stats=IngestionStats(rows_read=5))
    assert lying.accounts_for_every_row is False


def test_problem_type_is_recorded(ingested):
    for observation in ingested.observations[:100]:
        assert observation.problem_type is ProblemType.TRAFFIC
