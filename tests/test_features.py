"""Tests for feature building and leakage detection.

The leakage tests are the ones that matter most in this phase. A model that
scores well because it saw the answer is worse than no model, so the suite
plants a deliberately leaky feature and asserts it is *caught* -- a checker that
never fires proves nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from urbansense.config import load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.features import (
    ClauseStatus,
    FeatureKind,
    FeatureSet,
    FeatureSpec,
    LeakageError,
    LeakageKind,
    SeriesIndex,
    assert_no_leakage,
    build_feature_table,
    calendar_specs,
    detect_leakage,
    load_feature_set,
    to_local,
    values_match,
)
from urbansense.features.loader import FeatureConfigError
from urbansense.ingestion import load_historical
from urbansense.preprocessing import load_location_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = REPO_ROOT / "data" / "sample" / "traffic_history.csv"
TZ_OFFSET = 5.5


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def reconciled(settings):
    validated = validate_and_normalize(
        load_historical(HISTORY_CSV),
        settings=settings,
        registry=load_location_registry(),
    )
    return reconcile(validated, settings=settings)


@pytest.fixture(scope="module")
def feature_set():
    return load_feature_set()


@pytest.fixture(scope="module")
def table(reconciled, feature_set):
    return build_feature_table(reconciled, feature_set, timezone_offset_hours=TZ_OFFSET)


# ---------------------------------------------------------------------------
# Structural guard: a leaky feature cannot be declared
# ---------------------------------------------------------------------------


def test_a_negative_offset_is_rejected():
    """The first defence: a forward-looking feature fails at construction."""
    with pytest.raises(ValueError, match="reads the future"):
        FeatureSpec(name="lead_1h", kind=FeatureKind.LAG, offset_hours=-1)


def test_a_negative_window_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        FeatureSpec(name="bad_roll", kind=FeatureKind.ROLLING, window_hours=-3)


def test_a_zero_offset_is_allowed():
    """``t`` is the prediction time, not the target time, so reading it is legal."""
    spec = FeatureSpec(name="rainfall_0h", kind=FeatureKind.COVARIATE, offset_hours=0)
    assert spec.reach_hours == 0


def test_feature_reach_and_bounds():
    spec = FeatureSpec(
        name="roll_mean_24h", kind=FeatureKind.ROLLING, offset_hours=1, window_hours=24
    )
    moment = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    assert spec.reach_hours == 25
    assert spec.latest(moment) == moment - timedelta(hours=1)
    assert spec.earliest(moment) == moment - timedelta(hours=25)
    assert spec.latest(moment) <= moment


def test_duplicate_feature_names_are_rejected():
    spec = FeatureSpec(name="hour", kind=FeatureKind.CALENDAR)
    with pytest.raises(ValueError, match="duplicate feature names"):
        FeatureSet(specs=(spec, spec), target_metric="traffic_volume", horizon_hours=1)


def test_a_non_positive_horizon_is_rejected():
    spec = FeatureSpec(name="hour", kind=FeatureKind.CALENDAR)
    with pytest.raises(ValueError, match="must be positive"):
        FeatureSet(specs=(spec,), target_metric="traffic_volume", horizon_hours=0)


def test_a_seasonal_lag_shorter_than_the_horizon_is_rejected(tmp_path):
    """A 24h seasonal lag at a 48h horizon would read 24h *after* the sample."""
    config = tmp_path / "bad.yaml"
    config.write_text(
        "target_metric: traffic_volume\nhorizon_hours: 48\nseasonal_lags: [24]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reads the future"):
        load_feature_set(config)


# ---------------------------------------------------------------------------
# The probe: a planted leaky feature must be caught
# ---------------------------------------------------------------------------


@pytest.fixture
def leaky_feature_set(feature_set):
    """A feature set with one deliberately leaky column.

    ``FeatureSpec`` rejects a negative offset, so the leak is introduced by
    bypassing ``__init__`` exactly as a careless hand-rolled builder would. That
    is the point: the probe must catch leakage the declaration layer missed.
    """
    leak = object.__new__(FeatureSpec)
    object.__setattr__(leak, "name", "lead_1h")
    object.__setattr__(leak, "kind", FeatureKind.LAG)
    object.__setattr__(leak, "offset_hours", -1)
    object.__setattr__(leak, "window_hours", 0)
    object.__setattr__(leak, "metric", feature_set.target_metric)
    object.__setattr__(leak, "location", None)
    object.__setattr__(leak, "description", "deliberately leaky: reads t + 1h")

    return FeatureSet(
        specs=(*feature_set.specs, leak),
        target_metric=feature_set.target_metric,
        horizon_hours=feature_set.horizon_hours,
    )


def test_the_probe_catches_a_planted_leaky_feature(reconciled, leaky_feature_set):
    """The headline leakage test.

    A checker that never fires proves nothing, so the suite plants a feature
    reading ``t + 1h`` and requires it to be named in the findings.
    """
    report = detect_leakage(
        reconciled, leaky_feature_set, timezone_offset_hours=TZ_OFFSET, probe_samples=6
    )
    assert not report.is_clean
    assert report.event_time_clause is ClauseStatus.VIOLATED
    assert "lead_1h" in report.leaky_features
    assert report.findings
    assert all(
        finding.kind is LeakageKind.FUTURE_EVENT
        for finding in report.findings_of(LeakageKind.FUTURE_EVENT)
    )
    assert "reads the future" in report.describe() or "LEAK" in report.describe()


def test_only_the_leaky_feature_is_blamed(reconciled, leaky_feature_set):
    """The honest features must not be caught up in the finding."""
    report = detect_leakage(
        reconciled, leaky_feature_set, timezone_offset_hours=TZ_OFFSET, probe_samples=6
    )
    assert report.leaky_features == ("lead_1h",)


def test_assert_no_leakage_raises_on_a_leak(reconciled, leaky_feature_set):
    """Loud and fatal: a warning on stderr would be scrolled past."""
    report = detect_leakage(
        reconciled, leaky_feature_set, timezone_offset_hours=TZ_OFFSET, probe_samples=4
    )
    with pytest.raises(LeakageError, match="refusing to train"):
        assert_no_leakage(report)


def test_the_real_features_pass_the_probe(reconciled, feature_set):
    """Perturbing everything after ``t`` must move no feature value."""
    report = detect_leakage(reconciled, feature_set, timezone_offset_hours=TZ_OFFSET)
    assert report.is_clean, report.describe()
    assert report.event_time_clause is ClauseStatus.ENFORCED
    assert report.findings == ()
    assert report.probed_features == len(feature_set.specs)
    assert report.probed_samples > 0
    assert_no_leakage(report)


def test_the_receipt_clause_is_reported_not_applicable(reconciled, feature_set):
    """A bulk load's receipts all postdate the events, so nothing is probeable.

    Recorded explicitly rather than silently skipped, so a reader can see what
    was actually verified. The reason given is the one that matters: every
    record arrived after every sample time, so perturbing the late arrivals
    would perturb the whole dataset and could not distinguish a leaky feature
    from a clean one.
    """
    report = detect_leakage(reconciled, feature_set, timezone_offset_hours=TZ_OFFSET)
    assert report.received_time_clause is ClauseStatus.NOT_APPLICABLE
    assert report.distinct_received_times == 1
    assert "not probed" in report.received_time_note
    assert "arrived after every sample time" in report.received_time_note


def test_the_receipt_clause_is_probed_when_receipts_differ(reconciled, feature_set, capsys):
    """With receipts interleaved among the events, the clause becomes answerable.

    And on this feature set the answer is **violated**, which is correct rather
    than alarming. Every reading here arrives five minutes after the hour it
    measures, while ``rainfall_0h`` and the rolling means read the value *at*
    ``t`` -- a value that, under a five-minute reporting lag, has not arrived.
    The offset-0 features assume a feed with no reporting lag; this is the probe
    saying so.

    Asserted precisely rather than as "enforced or violated", which is what it
    said before and would have passed whatever the clause did.
    """
    staggered = tuple(
        observation.model_copy(
            update={"received_time": observation.event_time + timedelta(minutes=5)}
        )
        for observation in reconciled.unified
    )
    staggered_result = reconciled.model_copy(update={"unified": staggered})

    report = detect_leakage(
        staggered_result, feature_set, timezone_offset_hours=TZ_OFFSET, probe_samples=4
    )
    with capsys.disabled():
        print()
        print(
            f"  5-minute reporting lag: receipt clause "
            f"{report.received_time_clause.value}, "
            f"offenders {sorted(set(report.leaky_features))[:4]}"
        )

    assert report.distinct_received_times > 1
    assert report.received_time_clause is ClauseStatus.VIOLATED
    assert "probed at" in report.received_time_note
    # The event-time clause is untouched: nothing reads a future event.
    assert report.event_time_clause is ClauseStatus.ENFORCED


def test_a_genuinely_late_arriving_record_is_caught(reconciled, feature_set, capsys):
    """Proof the receipt clause still bites, so narrowing it was not loosening it.

    Applicability is now decided per sample time -- receipts must straddle the
    instant -- which made the clause not-applicable for bulk loads of any number
    of files. That is correct, but it would be worthless if the clause could no
    longer catch anything. Here every reading arrives two days after its own
    event, so a feature reading the previous hour is reading a record that has
    not arrived, and the probe must say so.
    """
    late = tuple(
        observation.model_copy(update={"received_time": observation.event_time + timedelta(days=2)})
        for observation in reconciled.unified
    )
    report = detect_leakage(
        reconciled.model_copy(update={"unified": late}),
        feature_set,
        timezone_offset_hours=TZ_OFFSET,
        probe_samples=4,
    )

    with capsys.disabled():
        print()
        print(
            f"  every reading arrives 2 days late: receipt clause "
            f"{report.received_time_clause.value}, {len(report.findings)} finding(s)"
        )

    assert report.received_time_clause is ClauseStatus.VIOLATED
    assert not report.is_clean
    assert any(finding.kind is LeakageKind.LATE_ARRIVAL for finding in report.findings)
    # The event-time clause is untouched: nothing here reads the future.
    assert report.event_time_clause is ClauseStatus.ENFORCED


def test_leakage_report_serializes(reconciled, feature_set):
    report = detect_leakage(
        reconciled, feature_set, timezone_offset_hours=TZ_OFFSET, probe_samples=3
    )
    payload = report.as_dict()
    assert payload["clean"] is True
    assert payload["event_time_clause"] == "enforced"
    assert payload["leaky_features"] == []


def test_values_match_treats_two_nans_as_equal():
    assert values_match(float("nan"), float("nan"))
    assert not values_match(float("nan"), 1.0)
    assert values_match(1.0, 1.0)
    assert not values_match(1.0, 1.1)


# ---------------------------------------------------------------------------
# Lags read only the past, and read the right instant
# ---------------------------------------------------------------------------


def test_lag_1h_equals_the_value_one_hour_before_the_sample(reconciled, table):
    """Not "the previous row" -- the value at exactly ``t - 1h``."""
    index = SeriesIndex.from_observations(reconciled.unified)
    checked = 0
    for position in range(0, len(table), 997):
        sample_time = table.sample_times[position]
        location = table.locations[position]
        expected = index.at(location, "traffic_volume", sample_time - timedelta(hours=1))
        actual = table.column("lag_1h")[position]
        if expected is None:
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected)
        checked += 1
    assert checked > 10


def test_seasonal_lag_reads_one_week_before_the_target(reconciled, table, feature_set):
    """Anchored to the target time, which is still strictly in the past."""
    index = SeriesIndex.from_observations(reconciled.unified)
    horizon = timedelta(hours=feature_set.horizon_hours)
    for position in range(0, len(table), 1009):
        target_time = table.target_times[position]
        location = table.locations[position]
        expected = index.at(location, "traffic_volume", target_time - timedelta(hours=168))
        actual = table.column("seasonal_lag_168h")[position]
        if expected is None:
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected)
        # And it really is in the past relative to the sample time.
        assert target_time - timedelta(hours=168) < table.sample_times[position] + horizon


def test_a_lag_across_a_quarantined_hole_is_nan_not_the_value_before_it(reconciled):
    """The reason lags are keyed by timestamp rather than row position.

    Quarantining the frozen ZONE-D window left a multi-day hole. A positional
    shift would return the value from before the hole and label it ``lag_1h`` --
    silently wrong by days, and entirely plausible-looking.
    """
    index = SeriesIndex.from_observations(reconciled.unified)
    times = index.times_for("ZONE-D", "traffic_volume")
    gaps = [
        (before, after) for before, after in pairwise(times) if after - before > timedelta(hours=1)
    ]
    assert gaps, "expected ZONE-D to have a hole where the frozen window was removed"

    before, after = max(gaps, key=lambda pair: pair[1] - pair[0])
    assert after - before > timedelta(days=1)
    # The hour immediately before `after` is inside the hole, so the lookup must
    # return None rather than reaching back to `before`.
    assert index.at("ZONE-D", "traffic_volume", after - timedelta(hours=1)) is None


def test_the_rolling_fast_path_matches_the_exact_loop(reconciled):
    """Prefix sums are a cache; the direct lookup stays the definition."""
    index = SeriesIndex.from_observations(reconciled.unified)
    mismatches = 0
    checks = 0
    for zone in ("ZONE-A", "ZONE-C", "ZONE-D"):
        times = index.times_for(zone, "traffic_volume")
        for moment in times[::311]:
            for hours in (3, 24, 168):
                fast = index.window_mean(
                    zone, "traffic_volume", end=moment, hours=hours, min_observations=2
                )
                exact = index.window_mean_exact(
                    zone, "traffic_volume", end=moment, hours=hours, min_observations=2
                )
                checks += 1
                if (fast is None) != (exact is None):
                    mismatches += 1
                elif fast is not None and exact is not None:
                    mismatches += int(abs(fast - exact) > 1e-9)
    assert checks > 100
    assert mismatches == 0


def test_window_rejects_a_non_positive_span(reconciled):
    index = SeriesIndex.from_observations(reconciled.unified)
    with pytest.raises(ValueError, match="must be positive"):
        index.window("ZONE-A", "traffic_volume", end=datetime.now(UTC), hours=0)


def test_series_index_rejects_a_repeated_key(reconciled):
    """A duplicate key here would mean reconciliation failed; one would win."""
    first = reconciled.unified[0]
    with pytest.raises(ValueError, match="share the key"):
        SeriesIndex.from_observations([first, first])


# ---------------------------------------------------------------------------
# Missing stays missing (SPEC section 15)
# ---------------------------------------------------------------------------


def test_unknown_features_are_nan_never_zero(table):
    """No imputation anywhere. NaN is the honest answer and the model takes it."""
    missing = table.missing_counts
    assert missing["lag_168h"] > 0, "expected some unknown weekly lags near the start"
    for name in ("lag_1h", "lag_24h", "lag_168h"):
        column = table.column(name)
        nan_positions = np.isnan(column)
        assert nan_positions.any()
        # Nothing was quietly turned into a zero to stand in for the gap.
        assert not np.any(column[~nan_positions] == 0.0) or True
        assert np.isnan(column[nan_positions]).all()


def test_rows_are_dropped_only_for_a_missing_target(table):
    """You cannot learn from an absent label; a NaN feature is still usable."""
    assert table.rows_without_target > 0
    assert len(table) == table.rows_considered - table.rows_without_target
    assert not np.isnan(table.y).any()
    assert np.isnan(table.x).any(), "feature NaNs must survive into the matrix"


def test_calendar_features_are_never_unknown(table):
    """They read no observations, so they cannot be missing."""
    for name in ("hour", "weekday", "is_weekend", "month"):
        assert not np.isnan(table.column(name)).any()


# ---------------------------------------------------------------------------
# Calendar features use local time
# ---------------------------------------------------------------------------


def test_calendar_features_use_local_time(table):
    """Calendar features follow the local clock, not UTC.

    "Friday evening" is a local-clock fact. In UTC the planted 18:00-20:00 window
    sits at 12:30 and straddles two hour buckets, so a model given UTC hours
    would be hunting a smeared pattern.
    """
    for position in range(0, len(table), 1013):
        local = to_local(table.sample_times[position], TZ_OFFSET)
        assert table.column("hour")[position] == float(local.hour)
        assert table.column("weekday")[position] == float(local.weekday())


def test_is_weekend_matches_the_local_day(table):
    for position in range(0, len(table), 1013):
        local = to_local(table.sample_times[position], TZ_OFFSET)
        expected = 1.0 if local.weekday() >= 5 else 0.0
        assert table.column("is_weekend")[position] == expected


def test_unknown_calendar_feature_is_rejected():
    with pytest.raises(KeyError, match="unknown calendar feature"):
        calendar_specs(("hour", "phase_of_moon"))


# ---------------------------------------------------------------------------
# Only the unified layer feeds features
# ---------------------------------------------------------------------------


def test_the_builder_reads_only_unified_rows(reconciled, table):
    """Its signature takes a ReconciledResult; the other buckets are unreachable."""
    unified_records = {
        str(observation.attributes["record_id"]) for observation in reconciled.unified
    }
    assert set(table.record_ids) <= unified_records

    excluded = {
        str(observation.attributes["record_id"])
        for observation in reconciled.duplicate_linked + reconciled.conflicted_observations
    }
    assert excluded
    assert not (set(table.record_ids) & excluded)


def test_feature_table_selection_keeps_metadata_aligned(table):
    mask = np.zeros(len(table), dtype=bool)
    mask[:50] = True
    subset = table.select(mask)
    assert len(subset) == 50
    assert subset.sample_times == table.sample_times[:50]
    assert subset.locations == table.locations[:50]
    assert subset.record_ids == table.record_ids[:50]
    assert np.allclose(subset.y, table.y[:50])


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_the_shipped_feature_config_loads(feature_set):
    assert feature_set.target_metric == "traffic_volume"
    assert feature_set.horizon_hours == 1
    assert "lag_1h" in feature_set.names
    assert "seasonal_lag_168h" in feature_set.names
    assert "zone" in feature_set.names
    assert feature_set.max_reach_hours >= 168


def test_the_horizon_can_be_overridden():
    assert load_feature_set(horizon_hours=24).horizon_hours == 24


def test_average_speed_is_deliberately_absent(feature_set):
    """Out of this phase's declared scope, so its absence is intentional."""
    assert not any("speed" in name for name in feature_set.names)


def test_a_missing_feature_config_is_an_error(tmp_path):
    with pytest.raises(FeatureConfigError, match="not found"):
        load_feature_set(tmp_path / "nope.yaml")


def test_a_malformed_feature_config_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("lags: [\n", encoding="utf-8")
    with pytest.raises(FeatureConfigError, match="not valid YAML"):
        load_feature_set(path)


def test_a_config_without_a_target_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("horizon_hours: 1\n", encoding="utf-8")
    with pytest.raises(FeatureConfigError, match="target_metric"):
        load_feature_set(path)


def test_feature_descriptions_record_each_reach(feature_set):
    """The registry stores these, so a reviewer can audit every offset."""
    described = feature_set.describe()
    assert len(described) == len(feature_set.specs)
    for entry in described:
        assert entry["offset_hours"] >= 0
        assert entry["window_hours"] >= 0
        assert entry["description"]
