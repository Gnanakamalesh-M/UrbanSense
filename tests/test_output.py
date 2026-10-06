"""Tests for the prediction output: probability, severity, windows, sources.

The theme is that a number must not be able to claim more than it has earned.
Probabilities are checked against the base-rate reference as well as against
their uncalibrated selves; the validation reliability table is asserted to be
*labelled in-sample* rather than asserted to be flat; and the test-split
miscalibration is asserted to be **reported**, not asserted to be small.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.helpers import make_observation
from urbansense.config import load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.evaluation import temporal_split
from urbansense.features import build_feature_table, load_feature_set
from urbansense.features.builder import FeatureTable
from urbansense.ingestion import load_historical
from urbansense.prediction import (
    GradientForecaster,
    ProbabilityCalibrator,
    SourceLedger,
    build_predictions,
    congestion_windows,
    fit_thresholds,
)
from urbansense.prediction.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_table,
)
from urbansense.prediction.output import interval_confidence
from urbansense.preprocessing import load_location_registry
from urbansense.schemas import ProblemType
from urbansense.schemas.enums import Severity, SourceType
from urbansense.schemas.observation import Observation

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = REPO_ROOT / "data" / "sample" / "traffic_history.csv"
GROUND_TRUTH = REPO_ROOT / "data" / "sample" / "ground_truth.json"
TZ_OFFSET = 5.5


@pytest.fixture(scope="module")
def truth() -> dict:
    """The planted answers. Only tests read this."""
    return json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline():
    """Everything one prediction run needs, built once."""
    settings = load_settings()
    validated = validate_and_normalize(
        load_historical(HISTORY_CSV), settings=settings, registry=load_location_registry()
    )
    reconciled = reconcile(validated, settings=settings)
    feature_set = load_feature_set()
    split = temporal_split(
        build_feature_table(reconciled, feature_set, timezone_offset_hours=TZ_OFFSET)
    )
    thresholds = fit_thresholds(split.train, settings.problems[ProblemType.TRAFFIC].congestion)
    model = GradientForecaster().fit(split.train)
    band = model.calibrate_interval(split.validation)

    calibrator = ProbabilityCalibrator.from_residuals(split.train.y, model.predict(split.train))
    raw_validation = calibrator.raw(
        model.predict(split.validation),
        np.array([thresholds.by_zone[z] for z in split.validation.locations]),
    )
    outcomes_validation = thresholds.congested_mask(split.validation)
    calibrator.fit(raw_validation, outcomes_validation)

    return {
        "settings": settings,
        "reconciled": reconciled,
        "feature_set": feature_set,
        "split": split,
        "thresholds": thresholds,
        "model": model,
        "band": band,
        "calibrator": calibrator,
        "raw_validation": raw_validation,
        "outcomes_validation": outcomes_validation,
    }


def _raw_and_outcomes(
    pipeline: dict, table: FeatureTable
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Raw probabilities and observed congestion labels for a table."""
    thresholds = pipeline["thresholds"]
    raw = pipeline["calibrator"].raw(
        pipeline["model"].predict(table),
        np.array([thresholds.by_zone[z] for z in table.locations]),
    )
    return raw, thresholds.congested_mask(table)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_improves_the_brier_score_on_validation(pipeline, capsys):
    report = pipeline["calibrator"].report(
        pipeline["raw_validation"], pipeline["outcomes_validation"], dataset="validation"
    )
    with capsys.disabled():
        print()
        print(report.describe())
    assert report.improved, (
        f"calibrated brier {report.calibrated_brier:.4f} is worse than the raw "
        f"{report.raw_brier:.4f}"
    )
    assert report.beats_base_rate


def test_the_validation_table_is_labelled_in_sample(pipeline):
    """A flat table on the fitting split is arithmetic, not evidence.

    Asserting the gaps are zero would be asserting that isotonic regression
    works. What matters is that the report says where the number came from.
    """
    report = pipeline["calibrator"].report(
        pipeline["raw_validation"], pipeline["outcomes_validation"], dataset="validation"
    )
    assert report.in_sample
    assert "arithmetic, not evidence" in report.describe()


def test_the_test_split_miscalibration_is_reported(pipeline, capsys):
    """Asserted as *reported*, not as small.

    The calibrator is fitted at a 0.234 congestion rate and applied at 0.281,
    so it is under-confident by construction. The test that matters is that the
    output says so, with the shift quantified.
    """
    raw, outcomes = _raw_and_outcomes(pipeline, pipeline["split"].test)
    report = pipeline["calibrator"].report(raw, outcomes, dataset="test")
    with capsys.disabled():
        print()
        print(report.describe())

    assert not report.in_sample
    assert report.base_rate > 0.0
    assert abs(report.base_rate_shift) > 0.01
    text = report.describe()
    assert "under-confident" in text
    assert "the calibrator was fitted where congestion occurs" in text
    assert report.direction == "under-confident"


def test_the_probabilities_still_beat_the_base_rate_on_test(pipeline, capsys):
    """Worth something despite the drift -- but measured, not assumed."""
    raw, outcomes = _raw_and_outcomes(pipeline, pipeline["split"].test)
    report = pipeline["calibrator"].report(raw, outcomes, dataset="test")
    with capsys.disabled():
        print(
            f"\n  test brier: raw {report.raw_brier:.4f}  calibrated "
            f"{report.calibrated_brier:.4f}  base-rate-only {report.reference_brier:.4f}"
        )
    assert report.beats_base_rate
    assert report.improved


def test_probabilities_stay_in_the_unit_interval(pipeline):
    for table in (pipeline["split"].validation, pipeline["split"].test):
        raw, _ = _raw_and_outcomes(pipeline, table)
        calibrated = pipeline["calibrator"].calibrate(raw)
        assert np.all(raw >= 0.0) and np.all(raw <= 1.0)
        assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)
        assert not np.any(np.isnan(calibrated))


def test_the_reliability_table_accounts_for_every_row(pipeline):
    raw, outcomes = _raw_and_outcomes(pipeline, pipeline["split"].test)
    calibrated = pipeline["calibrator"].calibrate(raw)
    bins = reliability_table(calibrated, outcomes)
    assert sum(item.count for item in bins) == len(outcomes)
    assert all(item.count > 0 for item in bins), "empty bins must be omitted, not shown as zero"


def test_a_single_class_calibration_set_is_refused():
    """A constant presented as a calibrated probability would mislead."""
    calibrator = ProbabilityCalibrator(sigma=100.0)
    with pytest.raises(ValueError, match="only one outcome class"):
        calibrator.fit(np.array([0.2, 0.4, 0.6]), np.array([True, True, True]))


def test_an_unfitted_calibrator_says_so(pipeline):
    calibrator = ProbabilityCalibrator(sigma=100.0)
    assert not calibrator.is_fitted
    probabilities = calibrator.calibrate(np.array([0.2, 0.9]))
    assert np.allclose(probabilities, [0.2, 0.9]), "unfitted means pass-through, not invented"


def test_brier_and_calibration_error_against_hand_arithmetic():
    """The scoring maths, checked against numbers computed by hand."""
    probabilities = np.array([0.0, 1.0, 0.5, 0.5])
    outcomes = np.array([False, True, True, False])
    # (0-0)^2 + (1-1)^2 + (0.5-1)^2 + (0.5-0)^2 = 0.5, over 4 rows.
    assert brier_score(probabilities, outcomes) == pytest.approx(0.125)

    bins = reliability_table(np.array([0.05, 0.05, 0.95]), np.array([False, True, True]), bins=10)
    assert len(bins) == 2
    first = bins[0]
    assert first.count == 2
    assert first.mean_predicted == pytest.approx(0.05)
    assert first.observed_frequency == pytest.approx(0.5)
    assert first.gap == pytest.approx(0.45)
    # Weighted mean absolute gap: (2*0.45 + 1*0.05) / 3.
    assert expected_calibration_error(bins) == pytest.approx((2 * 0.45 + 1 * 0.05) / 3)


# ---------------------------------------------------------------------------
# Severity and the prediction record
# ---------------------------------------------------------------------------


def test_severity_is_monotonic_in_the_predicted_volume(pipeline):
    """Across real prediction records, not just the classifier in isolation."""
    table = pipeline["split"].test
    predicted = pipeline["model"].predict(table)
    records = build_predictions(
        table,
        predicted,
        thresholds=pipeline["thresholds"],
        calibrator=pipeline["calibrator"],
        model_version="Traffic-v1.1",
        timezone_offset_hours=TZ_OFFSET,
    )
    for zone in sorted({item.location_id for item in records}):
        rows = sorted(
            (item for item in records if item.location_id == zone),
            key=lambda item: item.predicted_volume,
        )
        ranks = [item.severity.rank for item in rows]
        assert ranks == sorted(ranks), f"severity is not monotonic within {zone}"


def test_probability_rises_with_the_predicted_volume(pipeline):
    """The probability and the volume come from one model, so they must agree."""
    table = pipeline["split"].test
    predicted = pipeline["model"].predict(table)
    records = build_predictions(
        table,
        predicted,
        thresholds=pipeline["thresholds"],
        calibrator=pipeline["calibrator"],
        model_version="Traffic-v1.1",
        timezone_offset_hours=TZ_OFFSET,
    )
    rows = sorted(
        (item for item in records if item.location_id == "ZONE-B"),
        key=lambda item: item.predicted_volume,
    )
    probabilities = [item.congestion_probability for item in rows]
    assert probabilities == sorted(probabilities), (
        "a higher predicted volume produced a lower congestion probability; the "
        "two are derived from the same prediction and cannot disagree"
    )


def test_confidence_is_an_interval_width_not_a_probability():
    assert interval_confidence(1000.0, 1000.0, 1000.0) == pytest.approx(1.0)
    assert interval_confidence(1000.0, 750.0, 1250.0) == pytest.approx(0.5)
    assert interval_confidence(1000.0, 0.0, 2000.0) == pytest.approx(0.0)
    assert np.isnan(interval_confidence(float("nan"), 1.0, 2.0))


def test_a_record_marks_an_uncalibrated_probability(pipeline):
    table = pipeline["split"].test.select(np.arange(len(pipeline["split"].test)) < 5)
    records = build_predictions(
        table,
        pipeline["model"].predict(table),
        thresholds=pipeline["thresholds"],
        calibrator=ProbabilityCalibrator(sigma=158.0),
        model_version="Traffic-v1.1",
        timezone_offset_hours=TZ_OFFSET,
    )
    assert all(not item.probability_is_calibrated for item in records)
    assert "UNCALIBRATED" in records[0].describe()


def test_a_record_carries_everything_spec_25_asks_for(pipeline):
    table = pipeline["split"].test.select(np.arange(len(pipeline["split"].test)) < 3)
    predicted = pipeline["model"].predict(table)
    records = build_predictions(
        table,
        predicted,
        thresholds=pipeline["thresholds"],
        calibrator=pipeline["calibrator"],
        model_version="Traffic-v1.1",
        interval_bounds=pipeline["band"].bounds(predicted, table.locations),
        timezone_offset_hours=TZ_OFFSET,
    )
    payload = records[0].as_dict()
    for field in (
        "location_id",
        "target_time",
        "predicted_volume",
        "congestion_probability",
        "severity",
        "interval",
        "model_version",
    ):
        assert field in payload
    assert "confidence" in payload["interval"]
    assert "not a probability" in str(payload["interval"]["confidence_note"])


def test_misaligned_predictions_are_refused(pipeline):
    table = pipeline["split"].test
    with pytest.raises(ValueError, match="misaligned"):
        build_predictions(
            table,
            np.zeros(len(table) - 1),
            thresholds=pipeline["thresholds"],
            calibrator=pipeline["calibrator"],
            model_version="Traffic-v1.1",
        )


# ---------------------------------------------------------------------------
# The planted Friday window
# ---------------------------------------------------------------------------


def test_the_planted_friday_window_is_predicted_congested(pipeline, truth, capsys):
    """Scored against ground truth, which only tests read."""
    planted = truth["patterns"]["friday_evening_congestion"]
    zone, weekday = planted["location_id"], planted["weekday"]
    table = pipeline["split"].test
    predicted = pipeline["model"].predict(table)
    records = build_predictions(
        table,
        predicted,
        thresholds=pipeline["thresholds"],
        calibrator=pipeline["calibrator"],
        model_version="Traffic-v1.1",
        timezone_offset_hours=TZ_OFFSET,
    )
    peak_rows = [
        item
        for item in records
        if item.location_id == zone
        and item.local_time.weekday() == weekday
        and planted["start_hour"] <= item.local_time.hour < planted["end_hour"]
    ]
    assert peak_rows, "the planted window is absent from the test split"

    congested = [item for item in peak_rows if item.is_congested]
    windows = congestion_windows(records)
    covering = [
        window
        for window in windows
        if window.location_id == zone
        and any(window.start_local <= item.local_time < window.end_local for item in peak_rows)
    ]

    with capsys.disabled():
        print(
            f"\n  planted {zone} Friday "
            f"{planted['start_hour']}-{planted['end_hour']}: "
            f"{len(congested)}/{len(peak_rows)} predicted congested, "
            f"{len(covering)} covering window(s)"
        )
        for window in covering[:3]:
            print(f"    {window.describe()}")

    assert len(congested) == len(peak_rows)
    assert covering, "no congestion window covers the planted peak"
    assert all(item.congestion_probability > 0.5 for item in peak_rows)


def test_the_friday_peak_is_more_severe_than_an_ordinary_evening(pipeline, truth, capsys):
    """What distinguishes the planted peak is severity, not the window itself.

    Most weekday evenings clear a threshold anchored to a quieter training
    period, so "a window exists" is weak evidence. The planted hour should be
    in a strictly higher severity band than the same hours on other weekdays.
    """
    planted = truth["patterns"]["friday_evening_congestion"]
    zone = planted["location_id"]
    table = pipeline["split"].test
    records = build_predictions(
        table,
        pipeline["model"].predict(table),
        thresholds=pipeline["thresholds"],
        calibrator=pipeline["calibrator"],
        model_version="Traffic-v1.1",
        timezone_offset_hours=TZ_OFFSET,
    )
    evening = [
        item
        for item in records
        if item.location_id == zone
        and planted["start_hour"] <= item.local_time.hour < planted["end_hour"]
    ]
    friday = [item for item in evening if item.local_time.weekday() == planted["weekday"]]
    other = [item for item in evening if item.local_time.weekday() != planted["weekday"]]
    assert friday and other

    friday_peak = max(item.ratio for item in friday)
    other_peak = max(item.ratio for item in other)
    other_congested = sum(1 for item in other if item.is_congested)

    with capsys.disabled():
        print(
            f"\n  {zone} 18-20 severity: Friday peak x{friday_peak:.2f}, "
            f"other weekdays peak x{other_peak:.2f}; "
            f"{other_congested}/{len(other)} non-Friday evening hours also clear "
            "the threshold, which is why severity rather than the window is the "
            "distinguishing signal"
        )

    assert friday_peak > other_peak
    assert max(item.severity.rank for item in friday) == Severity.SEVERE.rank


def test_windows_break_at_a_gap(pipeline):
    """Two congested hours either side of a hole are not one episode."""
    table = pipeline["split"].test
    records = build_predictions(
        table,
        pipeline["model"].predict(table),
        thresholds=pipeline["thresholds"],
        calibrator=pipeline["calibrator"],
        model_version="Traffic-v1.1",
        timezone_offset_hours=TZ_OFFSET,
    )
    zone_rows = sorted(
        (item for item in records if item.location_id == "ZONE-B" and item.is_congested),
        key=lambda item: item.target_time,
    )[:2]
    if len(zone_rows) < 2:
        pytest.skip("not enough congested hours to build the case")

    spaced = (
        zone_rows[0],
        # Same record moved a day later: congested, but not contiguous.
        type(zone_rows[1])(
            **{
                **zone_rows[1].__dict__,
                "target_time": zone_rows[1].target_time + timedelta(days=1),
                "local_time": zone_rows[1].local_time + timedelta(days=1),
            }
        ),
    )
    windows = congestion_windows(spaced)
    assert len(windows) == 2, "a gap must break the window rather than be bridged"


# ---------------------------------------------------------------------------
# Source-aware output
# ---------------------------------------------------------------------------


def test_the_demo_feed_reports_historical_only(pipeline, capsys):
    ledger = SourceLedger(pipeline["reconciled"].unified)
    table = pipeline["split"].test
    index = len(table) - 1
    breakdown = ledger.attribute(
        pipeline["feature_set"],
        location_id=table.locations[index],
        sample_time=table.sample_times[index],
    )
    with capsys.disabled():
        print()
        print(breakdown.describe())

    assert breakdown.by_type[SourceType.HISTORICAL] == pytest.approx(1.0)
    assert breakdown.is_single_source_type
    assert sum(breakdown.by_type.values()) == pytest.approx(1.0)
    assert "historical data only" in breakdown.describe()
    assert "not of influence" in breakdown.describe()


def test_a_hand_built_multi_source_case_sums_to_one(capsys):
    """The demo data has one stream, so this path needs building by hand.

    Without it, every multi-source line in the module would ship unexercised.
    """
    start = datetime(2026, 1, 8, 12, 0, tzinfo=UTC)
    observations: list[Observation] = []
    for index in range(6):
        observations.append(
            make_observation(
                1000.0 + index,
                event_time=start - timedelta(hours=index + 1),
                location_id="ZONE-M",
                source_id="sensor-hist",
                record_id=f"H-{index}",
            )
        )
    for index in range(3):
        observations.append(
            make_observation(
                2000.0 + index,
                event_time=start - timedelta(hours=index + 1),
                location_id="ZONE-M",
                source_id="feed-live",
                record_id=f"L-{index}",
            ).model_copy(update={"source_type": SourceType.REAL_TIME})
        )
    for index in range(1):
        observations.append(
            make_observation(
                5.0,
                event_time=start - timedelta(hours=index + 1),
                location_id="CITY",
                metric="rainfall",
                unit="mm",
                source_id="council-pdf",
                record_id=f"D-{index}",
            ).model_copy(update={"source_type": SourceType.DOCUMENT})
        )

    ledger = SourceLedger(observations)
    breakdown = ledger.attribute(load_feature_set(), location_id="ZONE-M", sample_time=start)

    with capsys.disabled():
        print()
        print(breakdown.describe())

    assert set(breakdown.by_type) == {
        SourceType.HISTORICAL,
        SourceType.REAL_TIME,
        SourceType.DOCUMENT,
    }
    assert sum(breakdown.by_type.values()) == pytest.approx(1.0)
    assert sum(item.share for item in breakdown.shares) == pytest.approx(1.0)
    assert not breakdown.is_single_source_type
    assert breakdown.observations_used == 10


def test_freshness_is_measured_against_the_prediction_time():
    start = datetime(2026, 1, 8, 12, 0, tzinfo=UTC)
    ledger = SourceLedger(
        [
            make_observation(
                900.0,
                event_time=start - timedelta(hours=30),
                location_id="ZONE-M",
                record_id="OLD",
            )
        ]
    )
    breakdown = ledger.attribute(load_feature_set(), location_id="ZONE-M", sample_time=start)
    assert breakdown.freshness == timedelta(hours=30)
    assert breakdown.recency.value == "historical"


def test_missing_input_windows_are_reported():
    """A prediction resting on fewer inputs is weaker, and says so."""
    start = datetime(2026, 1, 8, 12, 0, tzinfo=UTC)
    ledger = SourceLedger(
        [make_observation(900.0, event_time=start - timedelta(hours=1), location_id="ZONE-M")]
    )
    breakdown = ledger.attribute(load_feature_set(), location_id="ZONE-M", sample_time=start)
    assert breakdown.missing_inputs, "the rainfall windows found nothing and should say so"
    assert any("rainfall" in name for name in breakdown.missing_inputs)
