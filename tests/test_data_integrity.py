"""Tests for validation, detection and quarantine, scored against ground truth.

The detectors in :mod:`urbansense.data_integrity` never read
``ground_truth.json`` -- only these tests do, and one test enforces that by
grepping the source tree. Scoring a detector against answers it can see would
make every number here a tautology.

The headline test is :func:`test_detection_precision_and_recall`, which reports
both figures and names the offending record ids on failure. Recall alone is easy
to game by quarantining everything, so precision is asserted alongside it.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from urbansense.config import load_settings
from urbansense.data_integrity import (
    QuarantineWriter,
    assess_source_health,
    check_range,
    detect_frozen_runs,
    detect_outliers,
    validate_and_normalize,
)
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.preprocessing import load_location_registry
from urbansense.schemas.enums import (
    ProblemType,
    QuarantineReason,
    Recency,
    SourceStatus,
    ValidationStatus,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = REPO_ROOT / "data" / "sample" / "traffic_history.csv"
ZONES_CSV = REPO_ROOT / "data" / "sample" / "zones.csv"
GROUND_TRUTH = REPO_ROOT / "data" / "sample" / "ground_truth.json"
HOUR = timedelta(hours=1)

#: The four sensors and the rain gauge that report continuously. The planted
#: conflict source ``city-warehouse`` is excluded on purpose: it contributes
#: eight readings in total and then goes silent, so it is genuinely a stale
#: batch feed and health correctly reports it as such.
CONTINUOUS_SOURCES = ("sensor-A12", "sensor-B07", "sensor-C03", "sensor-D21", "rain-gauge-01")


# ---------------------------------------------------------------------------
# Fixtures: one full pipeline run for the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def truth() -> dict:
    """The planted answers. Only tests may read this."""
    return json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def location_registry():
    return load_location_registry()


@pytest.fixture(scope="module")
def ingested():
    """Ingestion without config, so Phase 2 owns all semantic validation.

    Passing ``settings=None`` keeps ingestion to representability only, which
    is why the bogus-unit row reaches the validator instead of being rejected
    at the boundary.
    """
    return load_historical(HISTORY_CSV, registry=ZoneRegistry.from_csv(ZONES_CSV))


@pytest.fixture(scope="module")
def validated(ingested, settings, location_registry):
    return validate_and_normalize(ingested, settings=settings, registry=location_registry)


@pytest.fixture(scope="module")
def quarantined_ids(validated) -> set[str]:
    """Record ids of everything quarantined."""
    return {
        record.raw_payload.get("record_id", "")
        for record in validated.quarantined
        if record.raw_payload.get("record_id")
    }


@pytest.fixture(scope="module")
def valid_by_record(validated) -> dict[str, object]:
    return {
        str(obs.attributes["record_id"]): obs
        for obs in validated.valid
        if "record_id" in obs.attributes
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The invariant: nothing is lost
# ---------------------------------------------------------------------------


def test_valid_plus_quarantined_equals_input(validated, ingested):
    """The two-bucket partition must be exact.

    If this fails, records have gone missing between stages, which the project
    forbids outright.
    """
    assert validated.accounts_for_every_row
    assert validated.stats.rows_in == ingested.stats.rows_read
    assert len(validated.valid) + len(validated.quarantined) == ingested.stats.rows_read


def test_input_count_matches_the_file(validated):
    with HISTORY_CSV.open(encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    assert validated.stats.rows_in == rows


def test_ingestion_rejections_are_carried_forward(validated, ingested):
    """Dropping them would make the per-file totals stop reconciling."""
    assert validated.stats.carried_forward == len(ingested.rejected)
    assert validated.stats.carried_forward > 0
    carried = {r.quarantine_id for r in ingested.rejected}
    present = {r.quarantine_id for r in validated.quarantined}
    assert carried <= present


def test_suspicious_is_a_view_over_valid_not_a_third_bucket(validated):
    """An outlier is retained data, so it must still be inside ``valid``."""
    valid_ids = {obs.observation_id for obs in validated.valid}
    for observation in validated.suspicious:
        assert observation.observation_id in valid_ids
    assert len(validated.clean) + len(validated.suspicious) == len(validated.valid)


# ---------------------------------------------------------------------------
# Scoring the detectors against the planted answers
# ---------------------------------------------------------------------------


def test_detection_precision_and_recall(validated, truth, quarantined_ids, capsys):
    """Score quarantine decisions against the planted defects.

    Recall on its own is trivially gamed by quarantining everything, so
    precision is asserted alongside it. Both are printed so the numbers are
    visible with ``-s``.
    """
    planted_invalid = {
        item["record_id"]: item["kind"] for item in truth["planted_records"]["invalid"]
    }
    frozen = set(truth["anomalies"]["sensor_failure"]["affected_record_ids"])
    expected = set(planted_invalid) | frozen

    caught = expected & quarantined_ids
    missed = expected - quarantined_ids
    false_positives = quarantined_ids - expected

    recall = len(caught) / len(expected)
    precision = len(caught) / len(quarantined_ids) if quarantined_ids else 0.0

    with capsys.disabled():
        print(
            f"\n  quarantine scoring over {len(expected)} planted defects "
            f"({len(planted_invalid)} invalid + {len(frozen)} frozen):"
            f"\n    recall    {recall:.3f}  ({len(caught)}/{len(expected)} caught)"
            f"\n    precision {precision:.3f}  ({len(caught)}/{len(quarantined_ids)} "
            "quarantined were planted)"
        )

    assert not missed, f"planted defects not quarantined: {sorted(missed)}"
    assert not false_positives, (
        f"{len(false_positives)} rows were quarantined but are not planted defects: "
        f"{sorted(false_positives)[:20]}"
    )
    assert recall == 1.0
    assert precision == 1.0


def test_each_planted_defect_gets_a_sensible_reason(validated, truth):
    """The reason code must describe the actual defect, not just "bad"."""
    by_record = {record.raw_payload.get("record_id"): record for record in validated.quarantined}
    expected_reason = {
        "negative_value": QuarantineReason.INVALID_VALUE,
        "impossible_speed": QuarantineReason.INVALID_VALUE,
        "unparseable_timestamp": QuarantineReason.INVALID_TIMESTAMP,
        "unknown_unit": QuarantineReason.UNKNOWN_UNIT,
    }
    for item in truth["planted_records"]["invalid"]:
        record = by_record[item["record_id"]]
        assert record.reason is expected_reason[item["kind"]], (
            f"{item['record_id']} ({item['kind']}) got reason {record.reason.value}"
        )
        assert record.detail, "a quarantined record must say why"


def test_frozen_rows_are_quarantined_as_sensor_failure(validated, truth):
    frozen = set(truth["anomalies"]["sensor_failure"]["affected_record_ids"])
    by_record = {record.raw_payload.get("record_id"): record for record in validated.quarantined}
    for record_id in frozen:
        assert by_record[record_id].reason is QuarantineReason.SENSOR_FAILURE


def test_quarantine_counts_by_reason(validated):
    counts = validated.by_reason
    assert counts[QuarantineReason.SENSOR_FAILURE] == 60
    assert counts[QuarantineReason.INVALID_VALUE] == 6  # 4 negative + 2 impossible speed
    assert counts[QuarantineReason.INVALID_TIMESTAMP] == 2
    assert counts[QuarantineReason.UNKNOWN_UNIT] == 1


def test_no_genuine_row_is_falsely_quarantined(validated, truth, quarantined_ids):
    """The planted Friday peak and the post-drift rows must survive.

    These are the two strongest real patterns in the data. A detector that
    quarantined them would be reporting the signal as error -- and SPEC section
    16 is explicit that an unusual value is not automatically wrong.
    """
    friday = set(truth["patterns"]["friday_evening_congestion"]["affected_record_ids"])
    assert friday, "expected planted Friday rows"
    assert not (friday & quarantined_ids), (
        f"{len(friday & quarantined_ids)} Friday-peak rows were quarantined"
    )


def test_post_drift_rows_survive(validated, truth):
    """An 18% baseline shift is the city changing, not a data error."""
    drift = truth["anomalies"]["baseline_drift"]
    start = datetime.fromisoformat(drift["start_date"]).date()
    post = [
        obs
        for obs in validated.valid
        if obs.location_id in drift["location_ids"]
        and obs.metric == "traffic_volume"
        and obs.event_time.date() >= start
    ]
    assert len(post) > 1000, "post-drift data should pass validation in bulk"


def test_missing_rows_are_not_quarantined(validated, truth, quarantined_ids):
    """Absence of a reading is not an invalid reading."""
    missing = set(truth["missing_data"]["window_record_ids"]) | set(
        truth["missing_data"]["scattered_record_ids"]
    )
    assert not (missing & quarantined_ids)


# ---------------------------------------------------------------------------
# Frozen sensor detection (SPEC section 17)
# ---------------------------------------------------------------------------


def test_frozen_window_matches_ground_truth(validated, truth):
    planted = truth["anomalies"]["sensor_failure"]
    assert len(validated.frozen_runs) == 1, (
        f"expected exactly one frozen run, got {len(validated.frozen_runs)}"
    )
    run = validated.frozen_runs[0]

    assert run.location_id == planted["location_id"]
    assert run.metric == "traffic_volume"
    assert run.value == pytest.approx(planted["frozen_value"])
    assert run.length == len(planted["affected_record_ids"])

    # Ground truth records the window in the source's local time; event times
    # are UTC, so compare after shifting by the configured offset.
    offset = timedelta(hours=truth["timezone_offset_hours"])
    expected_start = datetime.fromisoformat(planted["start"]).replace(tzinfo=UTC) - offset
    expected_end = datetime.fromisoformat(planted["end"]).replace(tzinfo=UTC) - offset

    assert run.start == expected_start
    # The window end is exclusive, so the last frozen reading begins one
    # interval before it.
    assert run.end == expected_end - HOUR


def test_frozen_run_ids_match_ground_truth_exactly(validated, truth):
    planted = set(truth["anomalies"]["sensor_failure"]["affected_record_ids"])
    detected = {
        observation_id.removeprefix("OBS-")
        for run in validated.frozen_runs
        for observation_id in run.observation_ids
    }
    assert detected == planted


def test_dry_rainfall_runs_are_not_called_frozen(validated):
    """A rain gauge reading 0 mm through a dry month is working correctly.

    Repetition only indicates a stuck device when the value represents
    activity; without this guard every dry spell reads as a broken gauge.
    """
    assert all(run.metric != "rainfall" for run in validated.frozen_runs)
    assert validated.source_health["rain-gauge-01"].status is SourceStatus.OK


def test_missing_values_break_a_run(settings, location_registry):
    """A gap is an absence of evidence, not a repeated reading."""
    from tests.helpers import make_series

    observations = make_series(
        [100.0, 100.0, 100.0, None, 100.0, 100.0, 100.0],
    )
    assert detect_frozen_runs(observations, min_run_length=6) == ()
    # Without the gap, the same values do form a run.
    unbroken = make_series([100.0] * 7)
    assert len(detect_frozen_runs(unbroken, min_run_length=6)) == 1


def test_frozen_detector_rejects_a_nonsensical_threshold():
    from tests.helpers import make_series

    with pytest.raises(ValueError, match="at least 2"):
        detect_frozen_runs(make_series([1.0, 1.0]), min_run_length=1)


# ---------------------------------------------------------------------------
# Source health (SPEC section 17)
# ---------------------------------------------------------------------------


def test_frozen_sensor_is_reported_failed(validated):
    health = validated.source_health["sensor-D21"]
    assert health.status is SourceStatus.FAILED
    assert health.frozen_run_count == 1
    assert health.frozen_observation_count == 60
    assert "frozen" in health.detail


def test_healthy_sensors_are_ok(validated):
    for source_id in CONTINUOUS_SOURCES:
        if source_id == "sensor-D21":
            continue
        health = validated.source_health[source_id]
        assert health.status is SourceStatus.OK, (
            f"{source_id} reported {health.status.value}: {health.detail}"
        )


def test_health_reports_last_observation_and_gaps(validated):
    health = validated.source_health["sensor-A12"]
    assert health.last_observation_time is not None
    assert health.observation_count > 10_000
    assert health.max_gap is not None


def test_health_counts_missing_without_zeroing_it(validated):
    """Missing readings are counted as missing, never as zero."""
    health = validated.source_health["sensor-C03"]
    assert health.missing_count > 0
    assert 0.0 < health.missing_rate < 1.0


def test_a_source_that_had_an_outage_and_recovered_warns():
    """A historical gap is degradation, not a current failure.

    The source is reporting again, so calling it FAILED would be wrong -- and
    would make the status useless for answering "can I trust this feed now?"
    """
    from tests.helpers import make_series

    start = datetime(2024, 1, 1, tzinfo=UTC)
    observations = [
        *make_series([100.0, 110.0, 120.0], start=start),
        *make_series([130.0], start=start + timedelta(days=3)),
    ]
    health = assess_source_health(observations, failed_gap=timedelta(hours=6))
    record = health["sensor-TEST"]
    assert record.status is SourceStatus.WARNING
    assert record.max_gap is not None and record.max_gap > timedelta(hours=6)


def test_a_source_that_went_silent_is_failed():
    """Nothing for days as of now, so the feed is down (SPEC section 17)."""
    from tests.helpers import make_series

    start = datetime(2024, 1, 1, tzinfo=UTC)
    observations = make_series([100.0, 110.0, 120.0], start=start)
    health = assess_source_health(
        observations,
        now=start + timedelta(days=5),
        failed_gap=timedelta(hours=6),
    )
    record = health["sensor-TEST"]
    assert record.status is SourceStatus.FAILED
    assert record.gap_duration is not None
    assert record.gap_duration > timedelta(days=4)
    assert record.last_observation_time == start + timedelta(hours=2)


def test_a_high_missing_rate_warns():
    from tests.helpers import make_series

    observations = make_series([100.0, None, None, 110.0, None, 120.0])
    health = assess_source_health(observations, failed_gap=timedelta(days=9999))
    record = health["sensor-TEST"]
    assert record.status is SourceStatus.WARNING
    assert record.missing_count == 3


def test_unhealthy_sources_view(validated):
    unhealthy = validated.unhealthy_sources
    assert "sensor-D21" in unhealthy
    assert "sensor-A12" not in unhealthy


# ---------------------------------------------------------------------------
# Outliers: detection only (SPEC section 16)
# ---------------------------------------------------------------------------


def test_outliers_are_flagged_and_retained(validated):
    """Flagged, never deleted: an outlier may be a real event."""
    suspicious = validated.suspicious
    assert suspicious, "expected some values to be flagged"
    valid_ids = {obs.observation_id for obs in validated.valid}
    for observation in suspicious:
        assert observation.validation_status is ValidationStatus.SUSPECT
        assert observation.observation_id in valid_ids
        assert "outlier_mad_sigma" in observation.attributes


def test_flagging_is_rare(validated):
    """A flag is only useful if it is rare."""
    share = len(validated.suspicious) / len(validated.valid)
    assert share < 0.02, f"{share:.2%} of rows were flagged suspicious"


def test_recurring_patterns_are_not_flagged_as_outliers(validated, truth):
    """A pattern is not an anomaly.

    The detector conditions on weekday and hour, so the Friday peak is the norm
    within its own group. An unconditioned detector would flag all ~96 planted
    Friday rows -- reporting the signal as error.
    """
    friday = set(truth["patterns"]["friday_evening_congestion"]["affected_record_ids"])
    flagged = {
        str(obs.attributes["record_id"])
        for obs in validated.suspicious
        if "record_id" in obs.attributes
    }
    overlap = friday & flagged
    assert len(overlap) <= len(friday) * 0.1, (
        f"{len(overlap)} of {len(friday)} Friday-peak rows were flagged as outliers"
    )


def test_an_extreme_value_is_flagged():
    from tests.helpers import make_series

    normal = [5000.0 + (index % 5) * 50 for index in range(40)]
    observations = make_series([*normal, 40000.0], hours_apart=24 * 7)
    flags = detect_outliers(observations, threshold=6.0, min_group_size=12)
    assert len(flags) == 1
    flag = next(iter(flags.values()))
    assert flag.value == 40000.0
    assert flag.deviation > 6.0


def test_small_groups_do_not_judge():
    """With few samples, median and MAD are noise."""
    from tests.helpers import make_series

    observations = make_series([5000.0, 5100.0, 40000.0], hours_apart=24 * 7)
    assert detect_outliers(observations, min_group_size=12) == {}


def test_a_zero_variance_group_is_left_to_the_frozen_detector():
    """Flagging a constant group against itself would be meaningless."""
    from tests.helpers import make_series

    observations = make_series([100.0] * 20, hours_apart=24 * 7)
    assert detect_outliers(observations, min_group_size=12) == {}


def test_outlier_detector_skips_missing_values():
    from tests.helpers import make_series

    observations = make_series([None] * 20, hours_apart=24 * 7)
    assert detect_outliers(observations, min_group_size=12) == {}


def test_outlier_threshold_must_be_positive():
    from tests.helpers import make_series

    with pytest.raises(ValueError, match="must be positive"):
        detect_outliers(make_series([1.0]), threshold=0.0)


# ---------------------------------------------------------------------------
# Range checks (SPEC section 14)
# ---------------------------------------------------------------------------


def test_negative_volume_is_rejected(settings):
    metric = settings.resolve_metric(ProblemType.TRAFFIC, "traffic_volume")
    assert metric is not None
    violation = check_range(-4200.0, metric, unit="vehicles/hour")
    assert violation is not None
    assert violation.reason is QuarantineReason.INVALID_VALUE
    assert "non-negative" in violation.detail


def test_impossible_speed_is_rejected(settings):
    metric = settings.resolve_metric(ProblemType.TRAFFIC, "average_speed")
    assert metric is not None
    violation = check_range(480.0, metric, unit="km/hour")
    assert violation is not None
    assert "plausible maximum" in violation.detail


def test_plausible_values_pass(settings):
    metric = settings.resolve_metric(ProblemType.TRAFFIC, "traffic_volume")
    assert metric is not None
    assert check_range(8200.0, metric, unit="vehicles/hour") is None


def test_missing_value_is_not_a_range_violation(settings):
    """Quarantining every gap would be the wrong reading of 'invalid'."""
    metric = settings.resolve_metric(ProblemType.TRAFFIC, "traffic_volume")
    assert metric is not None
    assert check_range(None, metric, unit="vehicles/hour") is None


def test_impossible_values_are_preserved_not_clipped(validated, truth):
    """A negative count is kept as evidence, not silenced with max(0, x)."""
    negatives = [
        item["record_id"]
        for item in truth["planted_records"]["invalid"]
        if item["kind"] == "negative_value"
    ]
    by_record = {record.raw_payload.get("record_id"): record for record in validated.quarantined}
    for record_id in negatives:
        record = by_record[record_id]
        assert float(record.raw_payload["value"]) < 0, "the raw payload keeps the negative"
        assert record.observation is not None
        assert record.observation.value is not None
        assert record.observation.value < 0


# ---------------------------------------------------------------------------
# Normalization applied through the pipeline
# ---------------------------------------------------------------------------


def test_gap_rows_stay_none_after_normalization(validated, truth, valid_by_record):
    """Missing stays missing: not zero, not imputed."""
    missing = set(truth["missing_data"]["window_record_ids"]) | set(
        truth["missing_data"]["scattered_record_ids"]
    )
    checked = 0
    for record_id in missing:
        observation = valid_by_record.get(record_id)
        if observation is None:
            continue
        assert observation.value is None, f"{record_id} acquired a value"
        assert observation.value != 0
        assert observation.is_imputed is False
        assert observation.imputation_method is None
        checked += 1
    assert checked == len(missing), f"only {checked} of {len(missing)} gap rows survived"
    assert validated.stats.missing_values == len(missing)


def test_locations_resolve_to_canonical_ids(validated, location_registry):
    for observation in validated.valid[:500]:
        assert location_registry.get(observation.location_id) is not None


def test_coordinates_come_from_the_registry(validated, location_registry):
    observation = next(obs for obs in validated.valid if obs.location_id == "ZONE-B")
    zone = location_registry.get("ZONE-B")
    assert zone is not None
    assert observation.latitude == pytest.approx(zone.latitude)


def test_recency_is_recorded_and_historical_for_old_data(validated):
    """The whole demo history predates its receipt by months."""
    assert validated.by_recency == {Recency.HISTORICAL: len(validated.valid)}
    for observation in validated.valid[:200]:
        assert observation.attributes["recency"] == Recency.HISTORICAL.value
        assert observation.attributes["arrival_lag_seconds"] > 0


def test_source_type_is_not_overwritten_by_recency(validated):
    """Arrival stream and age are independent axes (SPEC section 4).

    Overwriting the stream label to mean "old" would lose the ability to ask
    what the live feed actually reported.
    """
    from urbansense.schemas.enums import SourceType

    for observation in validated.valid[:200]:
        assert observation.source_type is SourceType.HISTORICAL
        assert observation.attributes["recency"] == Recency.HISTORICAL.value


def test_three_timestamps_survive_validation(validated):
    for observation in validated.valid[:200]:
        assert observation.source_time is not None
        assert observation.event_time.tzinfo == UTC
        assert observation.source_time.tzinfo == UTC
        assert observation.received_time.tzinfo == UTC
        assert observation.event_time != observation.source_time
        assert observation.received_time > observation.event_time


def test_unit_conversion_through_the_pipeline_keeps_the_original(
    tmp_path, settings, location_registry
):
    """A 15-minute feed becomes hourly, and both representations survive."""
    path = tmp_path / "quarter_hourly.csv"
    path.write_text(
        "record_id,source_id,interval_end_local,location_id,metric,value,unit\n"
        "REC-000001,sensor-B07,2026-02-06 19:00:00+05:30,ZONE-B,traffic_volume,"
        "2050,vehicles/15min\n",
        encoding="utf-8",
    )
    result = validate_and_normalize(
        load_historical(path), settings=settings, registry=location_registry
    )
    assert len(result.valid) == 1
    observation = result.valid[0]
    assert observation.value == pytest.approx(8200.0)
    assert observation.unit == "vehicles/hour"
    assert observation.original_value == pytest.approx(2050.0)
    assert observation.original_unit == "vehicles/15min"
    assert observation.attributes["unit_normalized_from"] == "vehicles/15min"
    assert result.stats.normalized_units == 1


def test_unknown_unit_is_quarantined_not_coerced(tmp_path, settings, location_registry):
    path = tmp_path / "bad_unit.csv"
    path.write_text(
        "record_id,source_id,interval_end_local,location_id,metric,value,unit\n"
        "REC-000001,sensor-B07,2026-02-06 19:00:00+05:30,ZONE-B,traffic_volume,"
        "8200,vehicles/fortnight\n",
        encoding="utf-8",
    )
    result = validate_and_normalize(
        load_historical(path), settings=settings, registry=location_registry
    )
    assert len(result.valid) == 0
    assert len(result.quarantined) == 1
    assert result.quarantined[0].reason is QuarantineReason.UNKNOWN_UNIT


def test_unknown_location_is_quarantined_for_review(tmp_path, settings, location_registry):
    path = tmp_path / "bad_zone.csv"
    path.write_text(
        "record_id,source_id,interval_end_local,location_id,metric,value,unit\n"
        "REC-000001,sensor-X,2026-02-06 19:00:00+05:30,Anna Nagar East,"
        "traffic_volume,8200,vehicles/hour\n",
        encoding="utf-8",
    )
    result = validate_and_normalize(
        load_historical(path), settings=settings, registry=location_registry
    )
    assert len(result.quarantined) == 1
    record = result.quarantined[0]
    assert record.reason is QuarantineReason.LOCATION_REVIEW_REQUIRED
    assert "similar is not identical" in record.detail
    assert "ZONE-B" in record.detail, "the reviewer should be offered the candidate"


def test_rainfall_covariate_survives_without_traffic_config(validated):
    """Rain belongs to the disabled flood domain but is a useful predictor."""
    rainfall = [obs for obs in validated.valid if obs.metric == "rainfall"]
    assert rainfall
    for observation in rainfall[:20]:
        assert observation.attributes["metric_in_problem_config"] is False
        assert observation.unit == "mm"


# ---------------------------------------------------------------------------
# Phase 3 boundary: duplicates and conflicts are untouched
# ---------------------------------------------------------------------------


def test_planted_duplicates_pass_through_unlinked(validated, truth, valid_by_record):
    """Linking them is Phase 3's job; guessing here would pre-empt it."""
    for item in truth["planted_records"]["duplicates"]:
        copy = valid_by_record[item["record_id"]]
        original = valid_by_record[item["duplicate_of"]]
        assert copy.duplicate_of is None
        assert original.duplicate_of is None
        assert copy.value == original.value
        assert copy.validation_status is not ValidationStatus.DUPLICATE


def test_planted_conflicts_pass_through_unflagged(validated, truth, valid_by_record):
    for item in truth["planted_records"]["conflicts"]:
        copy = valid_by_record[item["record_id"]]
        original = valid_by_record[item["conflicts_with"]]
        assert copy.conflicts_with == ()
        assert original.conflicts_with == ()
        assert copy.value != original.value
        assert copy.validation_status is not ValidationStatus.CONFLICT


# ---------------------------------------------------------------------------
# Quarantine output (SPEC section 18)
# ---------------------------------------------------------------------------


def test_quarantine_records_keep_the_original_payload(validated):
    for record in validated.quarantined:
        assert record.raw_payload, "a quarantine record must carry its payload"
        assert record.reason
        assert record.detail
        assert record.quarantined_at.tzinfo is not None
        assert record.detector
        assert record.resolved is False


def test_quarantine_payload_matches_the_file_exactly(validated):
    """The payload is the row as received, not a normalized rendering."""
    with HISTORY_CSV.open(encoding="utf-8", newline="") as handle:
        rows = {row["record_id"]: row for row in csv.DictReader(handle)}

    checked = 0
    for record in validated.quarantined:
        record_id = record.raw_payload.get("record_id")
        if record_id not in rows:
            continue
        assert record.raw_payload == rows[record_id], (
            f"{record_id}'s payload differs from the source row"
        )
        checked += 1
    assert checked == len(validated.quarantined)


def test_writer_persists_records_with_payloads(validated, tmp_path):
    writer = QuarantineWriter(tmp_path)
    target = writer.write_records(validated.quarantined, now=datetime(2026, 10, 2, tzinfo=UTC))
    assert target is not None and target.exists()

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert len(payload) == len(validated.quarantined)
    assert {"reason", "detail", "quarantined_at", "raw_payload"} <= payload[0].keys()
    assert payload[0]["raw_payload"]


def test_writer_writes_nothing_when_there_is_nothing_to_write(tmp_path):
    assert QuarantineWriter(tmp_path).write_records([]) is None
    assert not list(tmp_path.iterdir())


def test_review_notes_are_notes_not_removals(validated, tmp_path):
    writer = QuarantineWriter(tmp_path)
    target = writer.write_review_notes(validated.suspicious, validated.outlier_flags)
    assert target is not None
    notes = json.loads(target.read_text(encoding="utf-8"))
    assert len(notes) == len(validated.suspicious)
    for note in notes:
        assert note["retained"] is True
        assert note["note"]


def test_each_run_writes_a_new_file(validated, tmp_path):
    """Quarantine is append-only in spirit: nothing overwrites a prior run."""
    writer = QuarantineWriter(tmp_path)
    first = writer.write_records(
        validated.quarantined[:2], now=datetime(2026, 10, 2, 9, 0, tzinfo=UTC)
    )
    second = writer.write_records(
        validated.quarantined[:2], now=datetime(2026, 10, 2, 10, 0, tzinfo=UTC)
    )
    assert first != second
    assert first is not None and second is not None
    assert first.exists() and second.exists()


# ---------------------------------------------------------------------------
# Raw data is immutable
# ---------------------------------------------------------------------------


def test_raw_files_are_unchanged_by_validation(settings, location_registry):
    before_hash = _sha256(HISTORY_CSV)
    before_mtime = HISTORY_CSV.stat().st_mtime_ns

    ingestion = load_historical(HISTORY_CSV)
    validate_and_normalize(ingestion, settings=settings, registry=location_registry)

    assert _sha256(HISTORY_CSV) == before_hash, "validation modified the raw file"
    assert HISTORY_CSV.stat().st_mtime_ns == before_mtime


def test_validation_writes_nothing_by_itself(tmp_path, monkeypatch, settings, location_registry):
    """Persisting quarantine is the caller's decision, not the pipeline's."""
    monkeypatch.chdir(tmp_path)
    result = validate_and_normalize(
        load_historical(HISTORY_CSV), settings=settings, registry=location_registry
    )
    assert result.quarantined
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# No leakage: detectors must never read the answers
# ---------------------------------------------------------------------------


def test_no_source_file_reads_ground_truth():
    """A detector that consulted the answers would be scoring itself.

    Every later phase's measurement would then be a tautology, so this is
    enforced mechanically rather than by review.

    Checked against the parsed code rather than the raw text: a docstring that
    explains a module does *not* read ground truth is not a violation, and a
    text search would flag it. Identifiers, attribute access and non-docstring
    string literals are what actually matter.
    """
    import ast

    offenders: list[str] = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if "synthetic" in path.parts:
            continue  # the generator legitimately writes it
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        }
        for node in ast.walk(tree):
            hit = False
            if isinstance(node, ast.Name) and "ground_truth" in node.id:
                hit = True
            elif isinstance(node, ast.Attribute) and "ground_truth" in node.attr:
                hit = True
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "ground_truth" in node.value
                and node not in docstrings
            ):
                hit = True
            if hit:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not offenders, (
        f"these source locations reference ground truth in code: {offenders}. "
        "Only the generator may write it, and only tests may read it."
    )


def test_detectors_do_not_import_the_synthetic_package():
    """Validation must work on any feed, not only on generated data."""
    offenders: list[str] = []
    for package in ("data_integrity", "preprocessing", "ingestion"):
        for path in (REPO_ROOT / "src" / "urbansense" / package).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "urbansense.synthetic" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"these modules import the generator: {offenders}"


# ---------------------------------------------------------------------------
# Determinism and reporting
# ---------------------------------------------------------------------------


def test_validation_is_deterministic(ingested, settings, location_registry):
    first = validate_and_normalize(ingested, settings=settings, registry=location_registry)
    second = validate_and_normalize(ingested, settings=settings, registry=location_registry)
    assert first.stats == second.stats
    assert [r.quarantine_id for r in first.quarantined] == [
        r.quarantine_id for r in second.quarantined
    ]


def test_stats_are_reported(validated):
    stats = validated.stats
    assert stats.valid == len(validated.valid)
    assert stats.quarantined == len(validated.quarantined)
    assert stats.suspicious == len(validated.suspicious)
    assert stats.source_path is not None
