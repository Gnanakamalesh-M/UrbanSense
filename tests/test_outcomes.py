"""Tests for the outcome store and the error breakdown.

The error breakdown's numbers are checked against **arithmetic done by hand** on
a small fixture, not against another run of the same code. A slicing bug that
also appeared in the expected value would otherwise pass: the point of this
suite is that the slices are right, and the only independent authority on that
is a mean computed on paper.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from urbansense.config import load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.evaluation import (
    OutcomeStore,
    analyse_errors,
    build_outcomes,
    temporal_split,
)
from urbansense.evaluation.error_analysis import THIN_SLICE_ROWS
from urbansense.evaluation.outcomes import HEAVY_RAIN_MM, Outcome, rain_band, time_of_day
from urbansense.features import build_feature_table, load_feature_set
from urbansense.ingestion import load_historical
from urbansense.prediction import GradientForecaster, fit_thresholds
from urbansense.preprocessing import load_location_registry
from urbansense.schemas import ProblemType
from urbansense.schemas.enums import Severity

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = REPO_ROOT / "data" / "sample" / "traffic_history.csv"
GROUND_TRUTH = REPO_ROOT / "data" / "sample" / "ground_truth.json"
TZ_OFFSET = 5.5
MODEL_VERSION = "Traffic-v1.1"


def make_outcome(
    *,
    predicted: float,
    actual: float,
    zone: str = "ZONE-X",
    hour: int = 9,
    weekday: int = 0,
    rainfall: float | None = 0.0,
    severity: Severity = Severity.LOW,
) -> Outcome:
    """One hand-built outcome, for arithmetic that can be checked on paper."""
    target = datetime(2026, 6, 1, 0, tzinfo=UTC) + timedelta(days=weekday, hours=hour)
    return Outcome(
        prediction_id=f"{MODEL_VERSION}:{zone}:{target.isoformat()}",
        model_version=MODEL_VERSION,
        location_id=zone,
        target_time=target,
        local_hour=hour,
        weekday=weekday,
        predicted=predicted,
        actual=actual,
        probability=0.5,
        severity=severity,
        congested_predicted=False,
        congested_actual=False,
        rainfall_mm=rainfall,
        record_id="REC-HAND",
        recorded_at=datetime(2026, 7, 1, tzinfo=UTC),
    )


@pytest.fixture(scope="module")
def truth() -> dict:
    """The planted answers. Only tests read this."""
    return json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_outcomes():
    """Outcomes from the real pipeline on the test split."""
    settings = load_settings()
    validated = validate_and_normalize(
        load_historical(HISTORY_CSV), settings=settings, registry=load_location_registry()
    )
    reconciled = reconcile(validated, settings=settings)
    split = temporal_split(
        build_feature_table(reconciled, load_feature_set(), timezone_offset_hours=TZ_OFFSET)
    )
    thresholds = fit_thresholds(split.train, settings.problems[ProblemType.TRAFFIC].congestion)
    model = GradientForecaster().fit(split.train)
    rainfall = {
        observation.event_time: observation.value
        for observation in reconciled.unified
        if observation.metric == "rainfall"
        and observation.location_id == "CITY"
        and observation.value is not None
    }
    return build_outcomes(
        split.test,
        model.predict(split.test),
        model_version=MODEL_VERSION,
        thresholds=thresholds,
        rainfall_by_time=rainfall,
        timezone_offset_hours=TZ_OFFSET,
    )


# ---------------------------------------------------------------------------
# The store is append-only
# ---------------------------------------------------------------------------


def test_the_store_appends_and_never_rewrites(tmp_path):
    """Write twice; the first run's bytes must survive untouched."""
    store = OutcomeStore(tmp_path / "outcomes.jsonl")
    first = [make_outcome(predicted=100.0, actual=120.0, zone="ZONE-A")]
    store.append(first)
    after_first = store.path.read_bytes()

    second = [make_outcome(predicted=200.0, actual=190.0, zone="ZONE-B")]
    store.append(second)
    after_second = store.path.read_bytes()

    assert after_second.startswith(after_first), (
        "the second append rewrote or reordered the first run's rows"
    )
    assert len(store) == 2
    assert len(store.read()) == 2


def test_appending_nothing_creates_nothing(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.jsonl")
    assert store.append([]) == 0
    assert not store.path.exists()
    assert len(store) == 0


def test_an_outcome_round_trips(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.jsonl")
    original = make_outcome(
        predicted=1234.5,
        actual=1200.0,
        zone="ZONE-C",
        hour=19,
        weekday=4,
        rainfall=12.5,
        severity=Severity.HIGH,
    )
    store.append([original])
    (restored,) = store.read()

    assert restored.prediction_id == original.prediction_id
    assert restored.model_version == original.model_version
    assert restored.location_id == original.location_id
    assert restored.target_time == original.target_time
    assert restored.predicted == pytest.approx(original.predicted)
    assert restored.actual == pytest.approx(original.actual)
    assert restored.severity is Severity.HIGH
    assert restored.rainfall_mm == pytest.approx(12.5)
    assert restored.rain_band == "heavy"
    assert restored.is_friday_evening


def test_every_field_spec_28_names_is_stored(tmp_path):
    payload = make_outcome(predicted=100.0, actual=90.0).as_dict()
    for field in (
        "prediction_id",
        "predicted",
        "actual",
        "error",
        "location_id",
        "target_time",
        "conditions",
        "model_version",
    ):
        assert field in payload, f"SPEC section 28 requires {field}"
    for field in ("rainfall_mm", "rain_band", "local_hour", "weekday", "is_friday_evening"):
        assert field in payload["conditions"]


def test_the_error_is_signed_so_bias_stays_visible():
    over = make_outcome(predicted=120.0, actual=100.0)
    under = make_outcome(predicted=80.0, actual=100.0)
    assert over.error == pytest.approx(20.0)
    assert under.error == pytest.approx(-20.0)
    assert over.abs_error == under.abs_error == pytest.approx(20.0)


def test_unknown_rainfall_is_not_recorded_as_dry():
    """Missing stays missing. A dry bucket would flatter the dry numbers."""
    assert rain_band(None) == "unknown"
    assert rain_band(float("nan")) == "unknown"
    assert rain_band(0.0) == "dry"
    assert rain_band(5.0) == "light"
    assert rain_band(HEAVY_RAIN_MM + 0.1) == "heavy"
    unknown = make_outcome(predicted=1.0, actual=1.0, rainfall=None)
    assert unknown.rain_band == "unknown"
    assert unknown.as_dict()["conditions"]["rainfall_mm"] is None


def test_time_of_day_bands():
    assert time_of_day(0) == "night"
    assert time_of_day(5) == "night"
    assert time_of_day(6) == "morning"
    assert time_of_day(11) == "morning"
    assert time_of_day(12) == "afternoon"
    assert time_of_day(16) == "afternoon"
    assert time_of_day(17) == "evening"
    assert time_of_day(23) == "evening"


def test_misaligned_predictions_are_refused(real_outcomes):
    settings = load_settings()
    validated = validate_and_normalize(
        load_historical(HISTORY_CSV), settings=settings, registry=load_location_registry()
    )
    split = temporal_split(
        build_feature_table(
            reconcile(validated, settings=settings),
            load_feature_set(),
            timezone_offset_hours=TZ_OFFSET,
        )
    )
    with pytest.raises(ValueError, match="misaligned"):
        build_outcomes(split.test, np.zeros(len(split.test) - 1), model_version=MODEL_VERSION)


# ---------------------------------------------------------------------------
# The breakdown, against arithmetic done by hand
# ---------------------------------------------------------------------------


def test_the_rainfall_breakdown_matches_hand_computation():
    """Three dry rows and two heavy rows, with means computed on paper.

    dry:   |10-0| , |20-0| , |30-0|  -> MAE 20
    heavy: |100-0|, |200-0|          -> MAE 150
    overall: (10+20+30+100+200)/5    -> MAE 72
    """
    outcomes = [
        make_outcome(predicted=10.0, actual=0.0, rainfall=0.0),
        make_outcome(predicted=20.0, actual=0.0, rainfall=0.0),
        make_outcome(predicted=30.0, actual=0.0, rainfall=0.0),
        make_outcome(predicted=100.0, actual=0.0, rainfall=25.0),
        make_outcome(predicted=200.0, actual=0.0, rainfall=40.0),
    ]
    report = analyse_errors(outcomes, dataset="hand-built")

    assert report.overall.mae == pytest.approx(72.0)
    rain = report.breakdown("by rainfall")
    assert rain is not None
    assert rain.named("dry").mae == pytest.approx(20.0)
    assert rain.named("dry").rows == 3
    assert rain.named("heavy").mae == pytest.approx(150.0)
    assert rain.named("heavy").rows == 2
    assert rain.named("light") is None, "an empty band must be omitted, not shown as zero"


def test_the_zone_breakdown_matches_hand_computation():
    """ZONE-A errors 5 and 15 -> MAE 10; ZONE-B error 40 -> MAE 40."""
    outcomes = [
        make_outcome(predicted=105.0, actual=100.0, zone="ZONE-A"),
        make_outcome(predicted=85.0, actual=100.0, zone="ZONE-A"),
        make_outcome(predicted=140.0, actual=100.0, zone="ZONE-B"),
    ]
    zones = analyse_errors(outcomes, dataset="hand-built").breakdown("by zone")
    assert zones is not None
    assert zones.named("ZONE-A").mae == pytest.approx(10.0)
    assert zones.named("ZONE-B").mae == pytest.approx(40.0)


def test_the_rmse_and_relative_mae_match_hand_computation():
    """RMSE of errors 0 and 10 is sqrt(50); relative MAE is 5/100."""
    outcomes = [
        make_outcome(predicted=100.0, actual=100.0),
        make_outcome(predicted=110.0, actual=100.0),
    ]
    overall = analyse_errors(outcomes, dataset="hand-built").overall
    assert overall.mae == pytest.approx(5.0)
    assert overall.rmse == pytest.approx(np.sqrt(50.0))
    assert overall.mean_actual == pytest.approx(100.0)
    assert overall.relative_mae == pytest.approx(0.05)


def test_the_time_of_day_breakdown_matches_hand_computation():
    outcomes = [
        make_outcome(predicted=10.0, actual=0.0, hour=3),
        make_outcome(predicted=30.0, actual=0.0, hour=9),
        make_outcome(predicted=50.0, actual=0.0, hour=19),
        make_outcome(predicted=70.0, actual=0.0, hour=21),
    ]
    bands = analyse_errors(outcomes, dataset="hand-built").breakdown("by time of day")
    assert bands is not None
    assert bands.named("night").mae == pytest.approx(10.0)
    assert bands.named("morning").mae == pytest.approx(30.0)
    assert bands.named("evening").mae == pytest.approx(60.0)  # (50 + 70) / 2
    assert bands.named("afternoon") is None


def test_the_friday_evening_slice_matches_hand_computation():
    outcomes = [
        make_outcome(predicted=100.0, actual=0.0, weekday=4, hour=18),
        make_outcome(predicted=300.0, actual=0.0, weekday=4, hour=19),
        make_outcome(predicted=10.0, actual=0.0, weekday=4, hour=17),
        make_outcome(predicted=20.0, actual=0.0, weekday=2, hour=18),
    ]
    window = analyse_errors(outcomes, dataset="hand-built").breakdown("Friday evening window")
    assert window is not None
    assert window.named("Friday 18-20").rows == 2
    assert window.named("Friday 18-20").mae == pytest.approx(200.0)
    assert window.named("all other hours").rows == 2
    assert window.named("all other hours").mae == pytest.approx(15.0)


def test_the_weekday_breakdown_uses_local_weekdays():
    outcomes = [
        make_outcome(predicted=10.0, actual=0.0, weekday=0),
        make_outcome(predicted=40.0, actual=0.0, weekday=4),
    ]
    weekdays = analyse_errors(outcomes, dataset="hand-built").breakdown("by weekday")
    assert weekdays is not None
    assert weekdays.named("Mon").mae == pytest.approx(10.0)
    assert weekdays.named("Fri").mae == pytest.approx(40.0)


def test_an_empty_report_is_refused():
    """A table of NaNs reads like a healthy result."""
    with pytest.raises(ValueError, match="no outcomes"):
        analyse_errors([], dataset="nothing")


# ---------------------------------------------------------------------------
# Every required slice is present, on real data
# ---------------------------------------------------------------------------


def test_every_required_breakdown_is_present(real_outcomes):
    report = analyse_errors(real_outcomes, dataset="test")
    dimensions = {item.dimension for item in report.breakdowns}
    assert {
        "by rainfall",
        "by time of day",
        "by zone",
        "by weekday",
        "Friday evening window",
        "by predicted severity",
    } <= dimensions


def test_the_real_breakdown_covers_every_row(real_outcomes):
    """Each breakdown must partition the outcomes, losing none."""
    report = analyse_errors(real_outcomes, dataset="test")
    for breakdown in report.breakdowns:
        total = sum(item.rows for item in breakdown.slices)
        assert total == report.rows, f"{breakdown.dimension} covers {total} of {report.rows} rows"


def test_the_report_names_its_weakest_slices(real_outcomes, capsys):
    """Printed, not buried -- and the Friday window should be among them."""
    report = analyse_errors(real_outcomes, dataset="test")
    with capsys.disabled():
        print()
        print(report.render())

    weakest = report.weakest()
    assert weakest
    names = [item.name for item in weakest]
    assert "Friday 18-20" in names, (
        "the planted spike is the model's known weak point and should surface"
    )
    assert all(item.rows >= THIN_SLICE_ROWS for item in weakest)


def test_heavy_rain_is_harder_than_dry(real_outcomes, capsys):
    """SPEC section 28's own example. Reported whichever way it comes out."""
    report = analyse_errors(real_outcomes, dataset="test")
    rain = report.breakdown("by rainfall")
    assert rain is not None
    dry, heavy = rain.named("dry"), rain.named("heavy")
    assert dry is not None and heavy is not None

    with capsys.disabled():
        print(
            f"\n  dry   MAE {dry.mae:,.1f} ({dry.rows} rows, {dry.relative_mae:.1%})"
            f"\n  heavy MAE {heavy.mae:,.1f} ({heavy.rows} rows, {heavy.relative_mae:.1%})"
        )
    assert heavy.relative_mae > dry.relative_mae


def test_the_overall_figure_is_reported_last(real_outcomes):
    """The breakdowns are the product; the average is the footnote."""
    rendered = analyse_errors(real_outcomes, dataset="test").render()
    assert rendered.index("by rainfall") < rendered.index("overall MAE")
    assert "behave nothing like each other" in rendered


def test_thin_slices_are_labelled(real_outcomes):
    report = analyse_errors(real_outcomes, dataset="test")
    severity = report.breakdown("by predicted severity")
    assert severity is not None
    thin = [item for item in severity.slices if item.rows < THIN_SLICE_ROWS]
    if thin:
        assert "(thin)" in severity.render()


def test_the_report_serializes_for_the_registry(real_outcomes):
    payload = analyse_errors(real_outcomes, dataset="test").as_dict()
    assert payload["rows"] == len(real_outcomes)
    assert payload["model_version"] == MODEL_VERSION
    assert payload["overall"]["mae"] > 0
    assert payload["breakdowns"]
    assert payload["weakest_slices"]
    assert json.dumps(payload)  # must be JSON-serializable as-is


def test_outcomes_record_conditions_at_target_time(real_outcomes, truth):
    """Which is what makes the rainfall breakdown answerable at all."""
    wet = [item for item in real_outcomes if item.rain_band in ("light", "heavy")]
    assert wet, "the test period contains rain, so some outcomes must record it"
    assert truth["patterns"]["rain_traffic_effect"]["wet_hour_count"] > 0
    note = str(real_outcomes[0].as_dict()["conditions_note"])
    assert "not what the model saw" in note
    assert "never read back as inputs" in note
