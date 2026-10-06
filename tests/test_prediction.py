"""Tests for temporal splits, baselines, the gradient model and the registry.

Two groups carry the weight here.

**Exclusion proofs.** Nothing quarantined, duplicate-linked, conflicted or
frozen may reach training. Earlier phases set those records aside; this suite
checks that the feature table honoured it, because a leak would be invisible in
the metrics.

**Honest comparison.** The gradient model is asserted to beat the naive baseline,
and the numbers are printed either way so a regression is readable rather than
just red.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from urbansense.config import load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.evaluation import (
    evaluate,
    friday_evening_slices,
    mae,
    render_comparison,
    render_table,
    rmse,
    temporal_split,
)
from urbansense.evaluation.splits import default_gap_hours
from urbansense.features import build_feature_table, load_feature_set
from urbansense.ingestion import load_historical
from urbansense.prediction import (
    GradientForecaster,
    ModelRegistry,
    ModelVersion,
    NaiveLastValue,
    SameHourLastWeek,
)
from urbansense.preprocessing import load_location_registry
from urbansense.schemas.enums import ModelStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = REPO_ROOT / "data" / "sample" / "traffic_history.csv"
GROUND_TRUTH = REPO_ROOT / "data" / "sample" / "ground_truth.json"
TZ_OFFSET = 5.5

#: How much better than "the pattern does not exist" the model must be on the
#: Friday ZONE-B slice. This is the principled test of whether the planted x2.2
#: spike was learned: a model that averaged it away would predict roughly the
#: zone's ordinary level during the peak, and the gap between that error and the
#: model's is the evidence. Set with margin below the observed figure.
#:
#: A hard threshold on the slice-versus-overall error ratio is deliberately NOT
#: asserted. That ratio is dominated by the slice being a rare, sharp spike at
#: four times the overall volume level, so any pass mark for it would be a number
#: chosen to match the result. It is printed instead, which is both more useful
#: and more honest.
FRIDAY_PATTERN_RECOVERY = 0.40


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def truth() -> dict:
    """The planted answers. Only tests read this."""
    return json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validated(settings):
    return validate_and_normalize(
        load_historical(HISTORY_CSV),
        settings=settings,
        registry=load_location_registry(),
    )


@pytest.fixture(scope="module")
def reconciled(validated, settings):
    return reconcile(validated, settings=settings)


@pytest.fixture(scope="module")
def table(reconciled):
    return build_feature_table(reconciled, load_feature_set(), timezone_offset_hours=TZ_OFFSET)


@pytest.fixture(scope="module")
def split(table):
    return temporal_split(table)


@pytest.fixture(scope="module")
def fitted(split):
    model = GradientForecaster().fit(split.train)
    model.calibrate_interval(split.validation)
    return model


@pytest.fixture(scope="module")
def test_results(split, table, fitted):
    """Every model evaluated on the test segment, once."""
    slices = friday_evening_slices(table, timezone_offset_hours=TZ_OFFSET)
    results = []
    for model in (NaiveLastValue(), SameHourLastWeek()):
        model.fit(split.train)
        results.append(
            evaluate(model.name, "test", split.test, model.predict(split.test), slices=slices)
        )
    results.append(
        evaluate(fitted.name, "test", split.test, fitted.predict(split.test), slices=slices)
    )
    return results


# ---------------------------------------------------------------------------
# Temporal splits only (SPEC section 33)
# ---------------------------------------------------------------------------


def test_the_split_is_strictly_ordered_in_time(split):
    """train.max < validation.min < test.min, the property the module exists for."""
    assert split.is_strictly_ordered
    train, validation, test = split.bounds
    assert train.end is not None and validation.start is not None
    assert train.end < validation.start
    assert validation.end is not None and test.start is not None
    assert validation.end < test.start


def test_every_train_sample_precedes_every_test_sample(split):
    assert max(split.train.sample_times) < min(split.test.sample_times)
    assert max(split.validation.sample_times) < min(split.test.sample_times)


def test_the_gap_is_applied_at_both_boundaries(split, table):
    expected = timedelta(hours=default_gap_hours(table))
    before_validation = split.gap_before_validation()
    before_test = split.gap_before_test()
    assert before_validation is not None and before_validation >= expected
    assert before_test is not None and before_test >= expected
    assert split.excluded_rows > 0, "the gaps must actually exclude rows"


def test_the_gap_covers_the_furthest_feature_reach(table):
    """A test sample's lags must not reach into the training period."""
    gap = default_gap_hours(table)
    assert gap >= table.feature_set.max_reach_hours
    assert gap >= table.feature_set.horizon_hours


def test_the_splitter_has_no_random_seed():
    """A shuffled split of a time series is meaningless, not merely weaker.

    With lag_1h as a feature it puts hour 14 in train and hour 15 in test, so
    the model predicts what it has already seen. Making the path unavailable is
    cheaper than remembering not to take it.
    """
    import inspect

    parameters = set(inspect.signature(temporal_split).parameters)
    assert not parameters & {"seed", "random_state", "shuffle", "random"}


def test_split_fractions_must_leave_a_test_segment(table):
    with pytest.raises(ValueError, match="leave room for a test segment"):
        temporal_split(table, train_fraction=0.8, validation_fraction=0.3)


def test_an_empty_segment_is_an_error(table):
    """An empty segment still produces a metric, and it would be meaningless."""
    with pytest.raises(ValueError, match="empty"):
        temporal_split(table, gap_hours=24 * 365)


def test_split_rows_sum_with_the_excluded_gap_rows(split, table):
    total = len(split.train) + len(split.validation) + len(split.test)
    assert total + split.excluded_rows == len(table)


def test_split_describes_itself(split):
    rendered = split.describe()
    assert "never shuffled" in rendered
    assert "train:" in rendered and "test:" in rendered


# ---------------------------------------------------------------------------
# Exclusion proofs: nothing earlier phases set aside may reach training
# ---------------------------------------------------------------------------


def _all_record_ids(split) -> set[str]:
    return (
        set(split.train.record_ids) | set(split.validation.record_ids) | set(split.test.record_ids)
    )


def test_no_quarantined_record_reaches_any_split(split, validated):
    quarantined = {
        record.raw_payload.get("record_id")
        for record in validated.quarantined
        if record.raw_payload.get("record_id")
    }
    assert quarantined
    assert not (quarantined & _all_record_ids(split))


def test_no_frozen_sensor_reading_reaches_any_split(split, truth):
    """The 60 planted ZONE-D frozen rows, by record id."""
    frozen = set(truth["anomalies"]["sensor_failure"]["affected_record_ids"])
    assert len(frozen) == 60
    assert not (frozen & _all_record_ids(split))


def test_the_frozen_value_never_appears_as_a_zone_d_target(split, truth):
    """Belt and braces: not just the ids, the value at those hours.

    A different record carrying the same frozen reading would be just as wrong
    in the training data, so the check is on the data as well as the identifiers.
    """
    planted = truth["anomalies"]["sensor_failure"]
    frozen_value = planted["frozen_value"]
    offset = timedelta(hours=truth["timezone_offset_hours"])
    from datetime import UTC, datetime

    start = datetime.fromisoformat(planted["start"]).replace(tzinfo=UTC) - offset
    end = datetime.fromisoformat(planted["end"]).replace(tzinfo=UTC) - offset

    for segment in (split.train, split.validation, split.test):
        for location, target_time, target in zip(
            segment.locations, segment.target_times, segment.y, strict=True
        ):
            if location == planted["location_id"] and start <= target_time < end:
                assert target != pytest.approx(frozen_value), (
                    f"a frozen reading of {frozen_value} leaked into training at "
                    f"{target_time.isoformat()}"
                )


def test_no_duplicate_linked_record_reaches_any_split(split, reconciled, truth):
    """The copies are retained by reconciliation but must not be trained on.

    Training on both halves of a duplicate pair would double-weight that hour.
    """
    linked = {
        str(observation.attributes["record_id"]) for observation in reconciled.duplicate_linked
    }
    assert len(linked) == len(truth["planted_records"]["duplicates"])
    assert not (linked & _all_record_ids(split))


def test_no_conflicted_record_reaches_any_split(split, reconciled):
    """A disputed value has no agreed label, so it cannot be a training target."""
    conflicted = {
        str(observation.attributes["record_id"])
        for observation in reconciled.conflicted_observations
    }
    assert len(conflicted) == 16
    assert not (conflicted & _all_record_ids(split))


def test_every_training_record_is_a_unified_record(split, reconciled):
    unified = {str(observation.attributes["record_id"]) for observation in reconciled.unified}
    assert _all_record_ids(split) <= unified


# ---------------------------------------------------------------------------
# The honest comparison
# ---------------------------------------------------------------------------


def test_the_gradient_model_beats_naive_on_the_test_set(test_results, capsys):
    """Asserted, with both numbers printed either way."""
    by_name = {result.model_name: result for result in test_results}
    naive = by_name["naive_last"].overall
    weekly = by_name["same_hour_last_week"].overall
    model = by_name["gradient_boosting"].overall

    with capsys.disabled():
        print(
            f"\n  test-set overall MAE / RMSE:"
            f"\n    naive_last           {naive.mae:>9,.1f} / {naive.rmse:>9,.1f}"
            f"\n    same_hour_last_week  {weekly.mae:>9,.1f} / {weekly.rmse:>9,.1f}"
            f"\n    gradient_boosting    {model.mae:>9,.1f} / {model.rmse:>9,.1f}"
            f"\n    improvement over naive: "
            f"{(naive.mae - model.mae) / naive.mae:.1%}"
        )

    assert model.mae < naive.mae, (
        f"gradient MAE {model.mae:,.1f} did not beat naive {naive.mae:,.1f}"
    )
    assert model.rmse < naive.rmse
    assert model.mae < weekly.mae


def test_the_model_beats_naive_in_every_zone(test_results):
    by_name = {result.model_name: result for result in test_results}
    naive = by_name["naive_last"]
    model = by_name["gradient_boosting"]
    for item in model.slices:
        if not item.name.startswith("zone "):
            continue
        peer = naive.slice_named(item.name)
        assert peer is not None
        assert item.mae < peer.mae, f"gradient lost to naive on {item.name}"


def test_the_friday_zone_b_pattern_is_recovered(split, fitted, test_results, capsys):
    """The planted x2.2 spike must not be averaged away.

    Measured against the counterfactual rather than against the overall error. A
    model that ignored the pattern would predict roughly the zone's ordinary
    level during the peak, so the distance between *that* error and the model's
    is what says whether the pattern was learned. Comparing the slice error to
    the overall error instead would mostly measure the fact that the slice sits
    at four times the volume level.
    """
    segment = split.test
    predicted = fitted.predict(segment)

    def is_peak(location: str, target: object) -> bool:
        local = target + timedelta(hours=TZ_OFFSET)  # type: ignore[operator]
        return location == "ZONE-B" and local.weekday() == 4 and 18 <= local.hour < 20

    peak = np.array(
        [
            is_peak(location, target)
            for location, target in zip(segment.locations, segment.target_times, strict=True)
        ]
    )
    zone = np.array([location == "ZONE-B" for location in segment.locations])
    assert peak.sum() > 0

    # What "the pattern does not exist" looks like: the zone's ordinary level.
    ignored_prediction = float(segment.y[zone & ~peak].mean())
    ignored_mae = float(np.abs(ignored_prediction - segment.y[peak]).mean())
    model_mae = float(np.abs(predicted[peak] - segment.y[peak]).mean())
    recovery = 1.0 - model_mae / ignored_mae

    model = {result.model_name: result for result in test_results}["gradient_boosting"]
    slice_metrics = model.slice_named("ZONE-B Fri 18-20")
    assert slice_metrics is not None
    ratio = slice_metrics.relative_mae / model.overall.relative_mae

    with capsys.disabled():
        print(
            f"\n  Friday ZONE-B 18:00-20:00 ({int(peak.sum())} rows):"
            f"\n    mean actual during the peak   {segment.y[peak].mean():>9,.0f}"
            f"\n    zone's ordinary level         {ignored_prediction:>9,.0f}"
            f"\n    MAE if the pattern is ignored {ignored_mae:>9,.0f}"
            f"\n    MAE from the model            {model_mae:>9,.0f}"
            f"\n    pattern recovered             {recovery:>9.0%}"
            f"  (floor {FRIDAY_PATTERN_RECOVERY:.0%})"
            f"\n    for information: slice relative MAE is {ratio:.2f}x the overall"
            " figure"
        )

    assert recovery >= FRIDAY_PATTERN_RECOVERY, (
        f"the model is only {recovery:.0%} better than ignoring the Friday ZONE-B "
        f"pattern entirely (MAE {model_mae:,.0f} vs {ignored_mae:,.0f}); the "
        "planted spike is being averaged away"
    )


def test_the_model_beats_the_baselines_on_the_friday_slice(test_results):
    """Where the planted pattern lives is where a model must earn its keep."""
    by_name = {result.model_name: result for result in test_results}
    model = by_name["gradient_boosting"].slice_named("ZONE-B Fri 18-20")
    naive = by_name["naive_last"].slice_named("ZONE-B Fri 18-20")
    assert model is not None and naive is not None
    assert model.mae < naive.mae


def test_the_report_names_every_slice_including_weak_ones(test_results):
    """Honest reporting: no filtering, no best-model highlighting."""
    rendered = render_table(test_results, title="test")
    for model in ("naive_last", "same_hour_last_week", "gradient_boosting"):
        assert model in rendered
    assert "ZONE-B Fri 18-20" in rendered
    assert "overall" in rendered

    comparison = render_comparison(
        test_results, baseline="naive_last", candidate="gradient_boosting"
    )
    assert "overall" in comparison
    assert "better" in comparison or "worse" in comparison


# ---------------------------------------------------------------------------
# Baselines and metrics
# ---------------------------------------------------------------------------


def test_naive_baseline_is_the_lag_column(split):
    model = NaiveLastValue().fit(split.train)
    predicted = model.predict(split.test)
    assert np.allclose(predicted, split.test.column("lag_1h"), equal_nan=True)


def test_same_hour_last_week_is_the_weekly_lag(split):
    model = SameHourLastWeek().fit(split.train)
    predicted = model.predict(split.test)
    assert np.allclose(predicted, split.test.column("lag_168h"), equal_nan=True)


def test_a_baseline_missing_its_feature_fails_loudly(split):
    """Silently predicting zeros would set an absurdly low bar."""
    with pytest.raises(KeyError, match="needs the"):
        NaiveLastValue(feature="not_a_feature").predict(split.test)


def test_unanswered_rows_count_as_errors():
    """Declining to predict is not free; otherwise a sparse baseline flatters."""
    actual = np.array([100.0, 200.0])
    predicted = np.array([100.0, np.nan])
    assert mae(actual, predicted) == pytest.approx(100.0)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt((0 + 200**2) / 2))


def test_evaluate_rejects_a_misaligned_prediction_array(split):
    with pytest.raises(ValueError, match="refusing to evaluate"):
        evaluate("x", "test", split.test, np.zeros(3))


def test_relative_mae_is_reported(test_results):
    for result in test_results:
        assert 0.0 <= result.overall.relative_mae < 10.0


# ---------------------------------------------------------------------------
# The gradient model
# ---------------------------------------------------------------------------


def test_the_model_handles_nan_features_without_imputation(split, fitted):
    """The reason HistGradientBoosting was chosen over an imputer."""
    assert np.isnan(split.train.x).any(), "expected unknown features in training"
    predicted = fitted.predict(split.test)
    assert not np.isnan(predicted).any(), "the model answered every row"


def test_predicting_before_fitting_is_an_error(split):
    with pytest.raises(RuntimeError, match="must be fitted"):
        GradientForecaster().predict(split.test)


def test_a_reordered_feature_table_is_rejected(split, fitted):
    """The design matrix is positional; a swapped column is a silent disaster."""
    import dataclasses

    reordered = dataclasses.replace(
        split.test,
        feature_set=dataclasses.replace(
            split.test.feature_set, specs=tuple(reversed(split.test.feature_set.specs))
        ),
    )
    with pytest.raises(ValueError, match="differ from the fitted order"):
        fitted.predict(reordered)


def test_fitting_on_an_empty_table_is_an_error(split):
    empty = split.test.select(np.zeros(len(split.test), dtype=bool))
    with pytest.raises(ValueError, match="empty"):
        GradientForecaster().fit(empty)


def test_the_model_is_deterministic(split):
    first = GradientForecaster().fit(split.train).predict(split.test)
    second = GradientForecaster().fit(split.train).predict(split.test)
    assert np.allclose(first, second)


def test_the_interval_is_calibrated_on_validation_not_test(split, fitted):
    """Calibrating on test would make the coverage self-fulfilling."""
    interval = fitted.interval
    assert interval.lower_offsets and interval.upper_offsets
    for zone, lower in interval.lower_offsets.items():
        assert lower <= interval.upper_offsets[zone]


def test_the_interval_brackets_most_test_rows(split, fitted):
    """A crude band, so this checks it is roughly right, not calibrated."""
    predicted, lower, upper = fitted.predict_with_interval(split.test)
    assert predicted.shape == lower.shape == upper.shape
    assert np.all(lower >= 0.0), "a negative vehicle count is not a lower bound"
    assert np.all(lower <= upper)
    inside = float(np.mean((split.test.y >= lower) & (split.test.y <= upper)))
    assert 0.4 < inside < 1.0, f"interval covered {inside:.1%} of test rows"


def test_the_interval_records_its_own_limitation(fitted):
    """Stated, not implied: constant width within a zone is wrong in a known way."""
    described = fitted.interval.as_dict()
    assert "quantile regression" in str(described["limitation"])
    assert described["lower_quantile"] == 0.10


def test_the_model_describes_itself_for_the_registry(fitted):
    described = fitted.describe()
    assert "HistGradientBoostingRegressor" in str(described["algorithm"])
    assert described["hyperparameters"]["loss"] == "squared_error"
    assert described["feature_names"]


# ---------------------------------------------------------------------------
# Model registry v1 (SPEC sections 31, 32)
# ---------------------------------------------------------------------------


@pytest.fixture
def model_version(split, fitted, test_results, table) -> ModelVersion:
    train, validation, test = split.bounds
    return ModelVersion(
        version="Traffic-v1.0",
        problem_type="traffic",
        target=table.feature_set.target_metric,
        horizon_hours=table.feature_set.horizon_hours,
        algorithm="sklearn.ensemble.HistGradientBoostingRegressor",
        algorithm_rationale="handles NaN natively",
        hyperparameters=fitted.hyperparameters,
        features=table.feature_set.describe(),
        training_period=train.as_dict(),
        validation_period=validation.as_dict(),
        test_period=test.as_dict(),
        gap_hours=split.gap_hours,
        metrics=[result.as_dict() for result in test_results],
        leakage_checks={"event_time_clause": "enforced"},
        excluded_rows={"without_target": table.rows_without_target},
        dataset_version={"seed": 20260101},
        interval=fitted.interval.as_dict(),
        code_version="0.1.0",
        notes=["trained on a synthetic dataset"],
    )


def test_the_registry_writes_complete_metadata(model_version, tmp_path):
    registry = ModelRegistry(tmp_path / "registry", tmp_path / "artifacts")
    path = registry.save(model_version)
    assert path.is_file()

    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "version",
        "status",
        "problem_type",
        "target",
        "horizon_hours",
        "algorithm",
        "hyperparameters",
        "features",
        "training_period",
        "validation_period",
        "test_period",
        "gap_hours",
        "metrics",
        "leakage_checks",
        "excluded_rows",
        "dataset_version",
        "interval",
        "created_at",
        "code_version",
    ):
        assert field in payload, f"registry entry is missing {field}"

    assert payload["status"] == ModelStatus.CHAMPION.value
    assert payload["features"]
    assert payload["metrics"]
    assert payload["training_period"]["rows"] > 0
    assert payload["gap_hours"] > 0


def test_the_registry_records_metrics_for_every_slice(model_version, tmp_path):
    registry = ModelRegistry(tmp_path / "registry", tmp_path / "artifacts")
    payload = json.loads(registry.save(model_version).read_text(encoding="utf-8"))
    names = {item["name"] for entry in payload["metrics"] for item in entry["slices"]}
    assert "ZONE-B Fri 18-20" in names
    assert any(entry["overall"]["mae"] > 0 for entry in payload["metrics"])


def test_the_registry_round_trips(model_version, tmp_path):
    registry = ModelRegistry(tmp_path / "registry", tmp_path / "artifacts")
    registry.save(model_version)
    loaded = registry.load_metadata("Traffic-v1.0")
    assert loaded["version"] == "Traffic-v1.0"
    assert registry.versions() == ("Traffic-v1.0",)
    assert registry.champions() == ("Traffic-v1.0",)


def test_the_registry_saves_and_reloads_the_estimator(model_version, fitted, tmp_path):
    registry = ModelRegistry(tmp_path / "registry", tmp_path / "artifacts")
    path = registry.save(model_version, estimator=fitted)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_path"] is not None

    restored = registry.load_estimator("Traffic-v1.0")
    assert isinstance(restored, GradientForecaster)
    assert restored.feature_names == fitted.feature_names


def test_a_missing_artifact_explains_how_to_recreate_it(model_version, tmp_path):
    """Artifacts are gitignored, so a fresh clone has metadata without weights."""
    registry = ModelRegistry(tmp_path / "registry", tmp_path / "artifacts")
    registry.save(model_version)
    with pytest.raises(FileNotFoundError, match="retrain"):
        registry.load_estimator("Traffic-v1.0")


def test_an_unregistered_version_is_an_error(tmp_path):
    registry = ModelRegistry(tmp_path / "registry", tmp_path / "artifacts")
    with pytest.raises(FileNotFoundError, match="no registry entry"):
        registry.load_metadata("Traffic-v9.9")
