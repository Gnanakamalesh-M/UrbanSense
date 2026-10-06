"""Tests for candidate updates, promotion rules and the champion/challenger registry.

The failure this suite is built around is a **leaky challenger**: one that saw
its own evaluation horizon, won its comparison because of it, and got promoted.
That failure is flattering rather than loud, so the horizon is asserted three
independent ways -- on the filter, on the fit's recorded target end, and through
the leakage probe.

The promotion rules are tested with hand-built challengers whose outcome is
decided by construction: one that is plainly worse must be rejected, one that is
plainly better must be accepted. Testing only against the real replay would make
the rules' behaviour contingent on whether the data happened to cooperate.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from urbansense.adaptation import (
    AdaptiveReplay,
    CandidateStatus,
    CandidateStore,
    ChallengerError,
    LineageError,
    PromotionRules,
    ReplayResult,
    TrainingWindow,
    assert_single_champion,
    available_at,
    candidate_from_event,
    decide,
    evaluate_challenger,
    evaluation_window,
    lineage,
    load_adaptation_config,
    promote,
    register_challenger,
    reject,
    render_lineage,
    train_challenger,
    version_name,
    window_rows,
)
from urbansense.adaptation.candidates import ChallengerEvaluation, RuleOutcome
from urbansense.config import load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.data_integrity.unified import ReconciledResult
from urbansense.drift.detectors import DriftKind, DriftSignal, DriftStatus
from urbansense.drift.events import DriftEvent
from urbansense.evaluation import friday_evening_slices
from urbansense.features import build_feature_table, detect_leakage, load_feature_set
from urbansense.features.builder import FeatureTable
from urbansense.features.leakage import ClauseStatus
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.ingestion.result import IngestionResult, IngestionStats
from urbansense.prediction import GradientForecaster, ModelRegistry, ModelVersion
from urbansense.preprocessing import load_location_registry
from urbansense.schemas.enums import ModelStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "sample"
HISTORY_CSV = DATA_DIR / "traffic_history.csv"
ZONES_CSV = DATA_DIR / "zones.csv"
GROUND_TRUTH = DATA_DIR / "ground_truth.json"
TZ_OFFSET = 5.5

TRAIN_DAYS = 75
REFERENCE_DAYS = 60
REPLAY_END = datetime(2026, 6, 25, tzinfo=UTC)
DECISION = datetime(2026, 4, 2, tzinfo=UTC)


@pytest.fixture(scope="module")
def truth() -> dict:
    """The planted answers. Only tests read this."""
    return json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline():
    settings = load_settings()
    validated = validate_and_normalize(
        load_historical(HISTORY_CSV, registry=ZoneRegistry.from_csv(ZONES_CSV)),
        settings=settings,
        registry=load_location_registry(),
    )
    reconciled = reconcile(validated, settings=settings)
    table = build_feature_table(reconciled, load_feature_set(), timezone_offset_hours=TZ_OFFSET)
    earliest = min(table.sample_times)
    train_end = earliest + timedelta(days=TRAIN_DAYS)
    champion = GradientForecaster().fit(_window(table, earliest, train_end))
    return {
        "reconciled": reconciled,
        "table": table,
        "train_end": train_end,
        "reference_end": train_end + timedelta(days=REFERENCE_DAYS),
        "champion": champion,
        "config": load_adaptation_config(),
    }


def _window(table: FeatureTable, start: datetime, end: datetime) -> FeatureTable:
    mask = np.array([start <= moment < end for moment in table.sample_times], dtype=bool)
    return table.select(mask)


def _firing_event(moment: datetime) -> DriftEvent:
    """A hand-built firing event, for candidate construction."""
    return DriftEvent(
        detected_at=moment,
        window_start=moment - timedelta(days=7),
        window_end=moment,
        signals=(
            DriftSignal(
                kind=DriftKind.PREDICTION,
                subject="prediction",
                status=DriftStatus.DRIFTED,
                statistic=0.09,
                threshold=0.03,
                reference_rows=4000,
                recent_rows=650,
            ),
        ),
        episode_id="EP-TEST",
    )


# ---------------------------------------------------------------------------
# A challenger can never see its own evaluation horizon
# ---------------------------------------------------------------------------


def test_available_at_filters_on_the_target_time_too(pipeline):
    """The subtle wrong answer is filtering on the sample time alone.

    A row sampled an hour before the decision carries a target from after it,
    so a sample-time filter hands a challenger "trained up to Thursday" a
    Friday measurement.
    """
    table = pipeline["table"]
    available = available_at(table, DECISION)

    assert len(available) > 0
    assert max(available.sample_times) < DECISION
    assert max(available.target_times) < DECISION, (
        "a target at or after the decision time reached the training data"
    )

    # The stricter filter must actually drop rows a sample-only filter keeps.
    sample_only = sum(1 for moment in table.sample_times if moment < DECISION)
    assert len(available) < sample_only


def test_a_challenger_never_trains_on_the_decision_horizon(pipeline, capsys):
    fit = train_challenger(
        pipeline["table"],
        decision_time=DECISION,
        window=TrainingWindow(name="expanding"),
        reconciled=pipeline["reconciled"],
        timezone_offset_hours=TZ_OFFSET,
    )
    with capsys.disabled():
        print()
        print(f"  {fit.describe()}")

    assert fit.respects_horizon
    assert fit.training_end < DECISION
    assert fit.target_end < DECISION
    assert fit.leakage_clean


def test_the_leakage_probe_runs_on_the_challengers_own_feature_set(pipeline):
    """So a challenger cannot inherit a clean bill of health."""
    fit = train_challenger(
        pipeline["table"],
        decision_time=DECISION,
        window=TrainingWindow(name="expanding"),
        reconciled=pipeline["reconciled"],
        timezone_offset_hours=TZ_OFFSET,
    )
    assert fit.leakage is not None
    assert fit.leakage.is_clean
    assert fit.as_dict()["leakage"] is not None


def test_the_evaluation_window_starts_after_the_gap(pipeline):
    rules = pipeline["config"].candidates
    evaluation = evaluation_window(
        pipeline["table"], decision_time=DECISION, gap=rules.gap, span=rules.evaluation_span
    )
    assert len(evaluation) > 0
    assert min(evaluation.sample_times) >= DECISION + rules.gap
    assert max(evaluation.sample_times) < DECISION + rules.gap + rules.evaluation_span


def test_training_and_evaluation_rows_never_overlap(pipeline):
    """The gap's whole purpose, checked on real row identities."""
    rules = pipeline["config"].candidates
    fit = train_challenger(
        pipeline["table"],
        decision_time=DECISION,
        window=TrainingWindow(name="expanding"),
        probe_leakage=False,
    )
    training = window_rows(
        pipeline["table"], decision_time=DECISION, window=TrainingWindow(name="expanding")
    )
    evaluation = evaluation_window(
        pipeline["table"], decision_time=DECISION, gap=rules.gap, span=rules.evaluation_span
    )
    assert not set(training.record_ids) & set(evaluation.record_ids)
    assert fit.target_end < min(evaluation.sample_times)


def test_a_recent_window_trains_on_less_than_an_expanding_one(pipeline):
    expanding = window_rows(
        pipeline["table"], decision_time=DECISION, window=TrainingWindow(name="expanding")
    )
    recent = window_rows(
        pipeline["table"],
        decision_time=DECISION,
        window=TrainingWindow(name="recent_60d", days=60),
    )
    assert 0 < len(recent) < len(expanding)
    assert min(recent.sample_times) > min(expanding.sample_times)


def test_too_little_data_refuses_to_train(pipeline):
    """A model fitted on 40 rows would lose its comparison for the wrong reason."""
    earliest = min(pipeline["table"].sample_times)
    with pytest.raises(ChallengerError, match="rows available"):
        train_challenger(
            pipeline["table"],
            decision_time=earliest + timedelta(days=2),
            window=TrainingWindow(name="expanding"),
            probe_leakage=False,
        )


# ---------------------------------------------------------------------------
# Candidate records carry the drift context
# ---------------------------------------------------------------------------


def test_a_candidate_records_the_drift_that_prompted_it():
    event = _firing_event(DECISION)
    candidate = candidate_from_event(event, decision_time=DECISION, champion_version="Traffic-v1.1")
    assert candidate.episode_id == "EP-TEST"
    assert candidate.drift_kinds == (DriftKind.PREDICTION,)
    assert candidate.drift_subject == "prediction"
    assert candidate.drift_magnitude == pytest.approx(3.0)
    assert candidate.champion_version == "Traffic-v1.1"
    assert candidate.status is CandidateStatus.RAISED

    payload = candidate.as_dict()
    assert payload["drift"]["kinds"] == ["prediction"]
    assert payload["drift"]["magnitude_vs_gate"] == pytest.approx(3.0)
    assert payload["decision_time"] == DECISION.isoformat()


def test_a_candidate_cannot_be_raised_without_drift():
    """A retrain with no justification recorded is the thing to prevent."""
    quiet = DriftEvent(
        detected_at=DECISION,
        window_start=DECISION - timedelta(days=7),
        window_end=DECISION,
        signals=(
            DriftSignal(
                kind=DriftKind.PREDICTION,
                subject="prediction",
                status=DriftStatus.STABLE,
                statistic=0.01,
                threshold=0.03,
                reference_rows=4000,
                recent_rows=650,
            ),
        ),
    )
    with pytest.raises(ValueError, match="did not fire"):
        candidate_from_event(quiet, decision_time=DECISION, champion_version="Traffic-v1.1")


def test_the_candidate_store_appends(tmp_path):
    store = CandidateStore(tmp_path)
    first = candidate_from_event(
        _firing_event(DECISION), decision_time=DECISION, champion_version="Traffic-v1.1"
    )
    store.append([first])
    after_first = store.path.read_bytes()
    second = candidate_from_event(
        _firing_event(DECISION + timedelta(days=28)),
        decision_time=DECISION + timedelta(days=28),
        champion_version="Traffic-v1.2",
    )
    store.append([second])

    assert store.path.read_bytes().startswith(after_first)
    rows = store.read()
    assert len(rows) == 2
    assert rows[0]["candidate_id"] != rows[1]["candidate_id"]
    assert "recorded_at" in rows[0]


# ---------------------------------------------------------------------------
# Promotion rules, on hand-built challengers
# ---------------------------------------------------------------------------


def _evaluation(
    *,
    version: str,
    champion_mae: float,
    challenger_mae: float,
    rules: PromotionRules,
    rows: int = 2500,
    slice_results: tuple[tuple[str, int, float, float], ...] = (),
    leakage_clean: bool = True,
) -> ChallengerEvaluation:
    """A hand-built evaluation whose verdict follows from its numbers."""
    improvement = 1.0 - (challenger_mae / champion_mae)
    outcomes = [
        RuleOutcome(
            rule="evaluation rows",
            passed=rows >= rules.min_evaluation_rows,
            detail=f"{rows} rows",
        ),
        RuleOutcome(
            rule="MAE improvement",
            passed=improvement >= rules.min_relative_mae_improvement,
            detail=f"{improvement:+.2%} against {rules.min_relative_mae_improvement:+.2%}",
        ),
        RuleOutcome(rule="training horizon", passed=True, detail="respected"),
        RuleOutcome(
            rule="leakage probe",
            passed=leakage_clean,
            detail="clean" if leakage_clean else "FAILED",
        ),
    ]
    for name, slice_rows, champion, challenger in slice_results:
        regression = (challenger / champion) - 1.0
        outcomes.append(
            RuleOutcome(
                rule=f"protected slice {name}",
                passed=(
                    True
                    if slice_rows < rules.min_slice_rows
                    else regression <= rules.max_slice_regression
                ),
                detail=f"{slice_rows} rows, {regression:+.1%}",
            )
        )
    return ChallengerEvaluation(
        version=version,
        training_window="expanding",
        training_rows=15000,
        training_end=DECISION - timedelta(hours=1),
        evaluation_rows=rows,
        champion_mae=champion_mae,
        challenger_mae=challenger_mae,
        champion_rmse=champion_mae * 1.6,
        challenger_rmse=challenger_mae * 1.6,
        slice_results=slice_results,
        leakage_clean=leakage_clean,
        rules=tuple(outcomes),
    )


def test_a_clearly_better_challenger_is_promoted(pipeline, capsys):
    rules = pipeline["config"].promotion
    candidate = candidate_from_event(
        _firing_event(DECISION), decision_time=DECISION, champion_version="Traffic-v1.1"
    )
    evaluation = _evaluation(
        version="Traffic-v1.2-expanding",
        champion_mae=300.0,
        challenger_mae=240.0,
        rules=rules,
        slice_results=(("ZONE-B Fri 18-20", 20, 3000.0, 2000.0),),
    )
    decided = decide(candidate, [evaluation], rules=rules, evaluation_rows=2500)

    with capsys.disabled():
        print()
        print(f"  {decided.describe()[:400]}")

    assert decided.status is CandidateStatus.PROMOTED
    assert decided.promoted_version == "Traffic-v1.2-expanding"
    assert evaluation.passed


def test_a_worse_challenger_is_rejected_and_the_champion_kept(pipeline, capsys):
    rules = pipeline["config"].promotion
    candidate = candidate_from_event(
        _firing_event(DECISION), decision_time=DECISION, champion_version="Traffic-v1.1"
    )
    evaluation = _evaluation(
        version="Traffic-v1.2-expanding",
        champion_mae=300.0,
        challenger_mae=330.0,
        rules=rules,
    )
    decided = decide(candidate, [evaluation], rules=rules, evaluation_rows=2500)

    with capsys.disabled():
        print()
        print(f"  {decided.describe()[:400]}")

    assert decided.status is CandidateStatus.REJECTED
    assert decided.promoted_version is None
    assert any("champion kept" in note for note in decided.notes)
    assert any("closest was" in note for note in decided.notes)


def test_a_marginal_challenger_is_rejected_with_the_gap_stated(pipeline):
    """A rejection that says how close it came settles the question."""
    rules = pipeline["config"].promotion
    candidate = candidate_from_event(
        _firing_event(DECISION), decision_time=DECISION, champion_version="Traffic-v1.1"
    )
    # 1.8% improvement against a 3% requirement.
    evaluation = _evaluation(
        version="Traffic-v1.2-expanding",
        champion_mae=300.0,
        challenger_mae=294.6,
        rules=rules,
    )
    decided = decide(candidate, [evaluation], rules=rules, evaluation_rows=2500)

    assert decided.status is CandidateStatus.REJECTED
    reasons = " ".join(decided.notes)
    assert "+1.80%" in reasons
    assert "+3.00%" in reasons


def test_a_challenger_that_regresses_a_protected_slice_is_rejected(pipeline):
    """Even when its overall error improves."""
    rules = pipeline["config"].promotion
    candidate = candidate_from_event(
        _firing_event(DECISION), decision_time=DECISION, champion_version="Traffic-v1.1"
    )
    evaluation = _evaluation(
        version="Traffic-v1.2-expanding",
        champion_mae=300.0,
        challenger_mae=240.0,
        rules=rules,
        slice_results=(("ZONE-B Fri 18-20", 20, 3000.0, 4500.0),),
    )
    decided = decide(candidate, [evaluation], rules=rules, evaluation_rows=2500)

    assert decided.status is CandidateStatus.REJECTED
    assert not evaluation.passed
    failures = [rule.rule for rule in evaluation.failures]
    assert any("protected slice" in rule for rule in failures)


def test_a_leaky_challenger_is_rejected(pipeline):
    """The most dangerous outcome available, so it is a hard rule."""
    rules = pipeline["config"].promotion
    candidate = candidate_from_event(
        _firing_event(DECISION), decision_time=DECISION, champion_version="Traffic-v1.1"
    )
    evaluation = _evaluation(
        version="Traffic-v1.2-expanding",
        champion_mae=300.0,
        challenger_mae=120.0,  # suspiciously good
        rules=rules,
        leakage_clean=False,
    )
    decided = decide(candidate, [evaluation], rules=rules, evaluation_rows=2500)

    assert decided.status is CandidateStatus.REJECTED
    assert any("leakage" in rule.rule for rule in evaluation.failures)


def test_a_thin_protected_slice_abstains_rather_than_blocking(pipeline):
    """A safeguard two observations can swing is a coin toss."""
    rules = pipeline["config"].promotion
    candidate = candidate_from_event(
        _firing_event(DECISION), decision_time=DECISION, champion_version="Traffic-v1.1"
    )
    evaluation = _evaluation(
        version="Traffic-v1.2-expanding",
        champion_mae=300.0,
        challenger_mae=240.0,
        rules=rules,
        # Two rows, and much worse -- but too thin to judge.
        slice_results=(("ZONE-B Fri 18-20", 2, 3000.0, 9000.0),),
    )
    decided = decide(candidate, [evaluation], rules=rules, evaluation_rows=2500)
    assert decided.status is CandidateStatus.PROMOTED


def test_too_few_evaluation_rows_is_inconclusive_not_rejected(pipeline):
    """Nothing judged and found wanting is not the same as nothing to judge."""
    rules = pipeline["config"].promotion
    candidate = candidate_from_event(
        _firing_event(DECISION), decision_time=DECISION, champion_version="Traffic-v1.1"
    )
    decided = decide(candidate, [], rules=rules, evaluation_rows=12)

    assert decided.status is CandidateStatus.INCONCLUSIVE
    assert decided.promoted_version is None
    assert any("not a rejection" in note for note in decided.notes)


def test_the_winner_comes_from_the_passing_set_only(pipeline):
    """A better score cannot buy its way past a failed rule."""
    rules = pipeline["config"].promotion
    candidate = candidate_from_event(
        _firing_event(DECISION), decision_time=DECISION, champion_version="Traffic-v1.1"
    )
    scores_better_but_fails = _evaluation(
        version="Traffic-v1.2-expanding",
        champion_mae=300.0,
        challenger_mae=200.0,
        rules=rules,
        slice_results=(("ZONE-B Fri 18-20", 20, 3000.0, 5000.0),),
    )
    passes = _evaluation(
        version="Traffic-v1.2-recent_60d",
        champion_mae=300.0,
        challenger_mae=260.0,
        rules=rules,
        slice_results=(("ZONE-B Fri 18-20", 20, 3000.0, 2800.0),),
    )
    decided = decide(
        candidate, [scores_better_but_fails, passes], rules=rules, evaluation_rows=2500
    )

    assert decided.status is CandidateStatus.PROMOTED
    assert decided.promoted_version == "Traffic-v1.2-recent_60d"
    assert decided.best is scores_better_but_fails, "the better score is still reported"


def test_every_rule_outcome_is_recorded_including_passes(pipeline):
    rules = pipeline["config"].promotion
    evaluation = _evaluation(
        version="Traffic-v1.2-expanding",
        champion_mae=300.0,
        challenger_mae=330.0,
        rules=rules,
    )
    assert len(evaluation.rules) >= 4
    assert any(rule.passed for rule in evaluation.rules)
    assert any(not rule.passed for rule in evaluation.rules)
    rendered = evaluation.describe()
    assert "PASS" in rendered and "FAIL" in rendered


def test_evaluate_challenger_scores_both_models_on_the_same_rows(pipeline):
    """An A/B, not two differently-scoped measurements."""
    rules = pipeline["config"]
    fit = train_challenger(
        pipeline["table"],
        decision_time=DECISION,
        window=TrainingWindow(name="expanding"),
        probe_leakage=False,
    )
    evaluation = evaluation_window(
        pipeline["table"],
        decision_time=DECISION,
        gap=rules.candidates.gap,
        span=rules.candidates.evaluation_span,
    )
    scored = evaluate_challenger(
        pipeline["champion"],
        fit,
        evaluation,
        rules=rules.promotion,
        version="Traffic-v1.2-expanding",
        timezone_offset_hours=TZ_OFFSET,
    )
    assert scored.evaluation_rows == len(evaluation)
    assert scored.champion_mae > 0 and scored.challenger_mae > 0
    assert scored.relative_improvement == pytest.approx(
        1.0 - scored.challenger_mae / scored.champion_mae
    )


# ---------------------------------------------------------------------------
# Exactly one champion, always
# ---------------------------------------------------------------------------


def _base_version(version: str = "Traffic-v1.1") -> ModelVersion:
    """A minimal registry entry, for lineage tests."""
    return ModelVersion(
        version=version,
        problem_type="traffic",
        target="traffic_volume",
        horizon_hours=1,
        algorithm="sklearn.ensemble.HistGradientBoostingRegressor",
        algorithm_rationale="handles NaN natively",
        hyperparameters={"max_iter": 300},
        features=[{"name": "lag_1h"}],
        training_period={"start": "2025-10-01", "end": "2025-12-14", "rows": 7000},
        validation_period={"start": "2025-12-15", "end": "2026-02-12", "rows": 2900},
        test_period={"start": "2026-02-20", "end": "2026-06-30", "rows": 4500},
        gap_hours=168,
        metrics=[{"overall": {"mae": 264.4}}],
        leakage_checks={"clean": True},
        excluded_rows={},
        dataset_version={"seed": 20260101},
        interval={"method": "residual band"},
        status=ModelStatus.CHAMPION,
    )


@pytest.fixture
def registry(tmp_path) -> ModelRegistry:
    """A registry holding one champion."""
    store = ModelRegistry(tmp_path / "registry", tmp_path / "artifacts")
    store.save(_base_version())
    return store


def test_promotion_leaves_exactly_one_champion(registry):
    registry.save(
        _base_version("Traffic-v1.2-expanding").__class__(
            **{
                **{
                    k: getattr(_base_version("Traffic-v1.2-expanding"), k)
                    for k in _base_version().__dataclass_fields__
                },
                "status": ModelStatus.CHALLENGER,
            }
        )
    )
    promoted, archived = promote(registry, "Traffic-v1.2-expanding")

    assert promoted == "Traffic-v1.2-expanding"
    assert archived == ("Traffic-v1.1",)
    assert registry.champions() == ("Traffic-v1.2-expanding",)
    assert assert_single_champion(registry) == "Traffic-v1.2-expanding"


def test_a_sequence_of_promotions_keeps_one_champion(registry, capsys):
    """After any number of promotions and rejections, exactly one serves."""
    for index in (2, 3, 4):
        version = f"Traffic-v1.{index}-expanding"
        entry = _base_version(version)
        registry.save(
            type(entry)(
                **{
                    **{k: getattr(entry, k) for k in entry.__dataclass_fields__},
                    "status": ModelStatus.CHALLENGER,
                }
            )
        )
        if index == 3:
            reject(registry, version)
        else:
            promote(registry, version)

    with capsys.disabled():
        print()
        print("  " + render_lineage(registry).replace("\n", "\n  "))

    assert registry.champions() == ("Traffic-v1.4-expanding",)
    statuses = {entry.version: entry.status for entry in lineage(registry)}
    assert statuses["Traffic-v1.1"] == ModelStatus.ARCHIVED.value
    assert statuses["Traffic-v1.2-expanding"] == ModelStatus.ARCHIVED.value
    assert statuses["Traffic-v1.3-expanding"] == ModelStatus.REJECTED.value
    assert statuses["Traffic-v1.4-expanding"] == ModelStatus.CHAMPION.value


def test_rejected_and_archived_versions_stay_on_disk(registry):
    entry = _base_version("Traffic-v1.2-expanding")
    registry.save(
        type(entry)(
            **{
                **{k: getattr(entry, k) for k in entry.__dataclass_fields__},
                "status": ModelStatus.CHALLENGER,
            }
        )
    )
    reject(registry, "Traffic-v1.2-expanding")

    assert registry.metadata_path("Traffic-v1.2-expanding").is_file()
    metadata = registry.load_metadata("Traffic-v1.2-expanding")
    assert metadata["status"] == ModelStatus.REJECTED.value
    assert metadata["metrics"], "a rejected version keeps its metrics"


def test_promoting_an_unregistered_version_is_refused(registry):
    with pytest.raises(LineageError, match="not in the registry"):
        promote(registry, "Traffic-v9.9-nonexistent")


def test_two_champions_is_detected(registry):
    """The invariant is checked, not assumed."""
    entry = _base_version("Traffic-v1.2-expanding")
    registry.save(entry)  # also CHAMPION
    with pytest.raises(LineageError, match="2 champions"):
        assert_single_champion(registry)


def test_no_champion_is_detected(registry):
    registry.set_status("Traffic-v1.1", ModelStatus.ARCHIVED)
    with pytest.raises(LineageError, match="no champion"):
        assert_single_champion(registry)


def test_a_challenger_entry_records_its_full_lineage(registry, tmp_path):
    path = register_challenger(
        registry,
        version="Traffic-v1.2-expanding",
        base=_base_version(),
        candidate_id="CAND-20260402T000000",
        drift_episode_id="EP-20260312T173000",
        supersedes="Traffic-v1.1",
        training_rows=17347,
        training_start="2025-09-30T17:30:00+00:00",
        training_end="2026-04-01T23:30:00+00:00",
        training_window="expanding",
        metrics=[{"overall": {"mae": 238.1}}],
        leakage={"clean": True},
        decision={"status": "promoted", "relative_improvement": 0.2208},
    )
    metadata = json.loads(path.read_text(encoding="utf-8"))

    # Every SPEC section 31 field.
    for field in (
        "version",
        "training_period",
        "features",
        "algorithm",
        "hyperparameters",
        "metrics",
        "dataset_version",
        "created_at",
        "status",
    ):
        assert field in metadata, f"SPEC section 31 requires {field}"

    assert metadata["status"] == ModelStatus.CHALLENGER.value
    assert metadata["training_period"]["rows"] == 17347
    assert metadata["adaptation"]["candidate_id"] == "CAND-20260402T000000"
    assert metadata["adaptation"]["drift_episode_id"] == "EP-20260312T173000"
    assert metadata["adaptation"]["supersedes"] == "Traffic-v1.1"
    assert metadata["adaptation"]["decision"]["status"] == "promoted"


def test_version_names_say_which_window_they_came_from():
    assert version_name("Traffic-v1", index=2, window="expanding") == ("Traffic-v1.2-expanding")
    assert version_name("Traffic-v1", index=3, window="recent_60d") == ("Traffic-v1.3-recent_60d")


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def replay_result(pipeline):
    replay = AdaptiveReplay(
        table=pipeline["table"],
        reconciled=pipeline["reconciled"],
        champion=pipeline["champion"],
        reference=_window(pipeline["table"], pipeline["train_end"], pipeline["reference_end"]),
        config=pipeline["config"],
        training_end=pipeline["train_end"],
        timezone_offset_hours=TZ_OFFSET,
    )
    return replay.run(
        scenario="drift_history",
        start=pipeline["reference_end"],
        end=REPLAY_END,
        slices=friday_evening_slices(pipeline["table"], timezone_offset_hours=TZ_OFFSET),
    )


def test_the_replay_scores_both_arms_on_identical_rows(replay_result):
    """Otherwise the comparison is two runs, not an A/B."""
    assert replay_result.static.rows == replay_result.adaptive.rows
    assert replay_result.static.rows > 0
    assert any("identical rows" in note for note in replay_result.notes)


def test_the_replay_reports_static_against_adaptive_honestly(replay_result, capsys):
    """Printed whichever way it comes out, slices included."""
    with capsys.disabled():
        print()
        print("  " + replay_result.render().replace("\n", "\n  "))

    rendered = replay_result.render()
    assert "static MAE" in rendered and "adaptive MAE" in rendered
    # Whichever way it went, the verdict is stated in words.
    assert ("adaptation reduced overall MAE" in rendered) or ("adaptation did NOT help" in rendered)
    # And the slices that got worse are named, not hidden behind the headline.
    assert ("no slice got worse" in rendered) or ("it was worse on" in rendered)


def test_the_replay_raises_candidates_and_decides_each_one(replay_result, capsys):
    statuses = [item.status for item in replay_result.candidates]
    with capsys.disabled():
        print(f"\n  {len(statuses)} candidate(s):")
        for candidate in replay_result.candidates:
            print(
                f"    {candidate.raised_at:%Y-%m-%d}  {candidate.status.value}"
                + (f" -> {candidate.promoted_version}" if candidate.promoted_version else "")
            )

    assert statuses, "the replay raised no candidates"
    assert CandidateStatus.RAISED not in statuses, "every candidate must be decided"
    assert all(
        status
        in (
            CandidateStatus.PROMOTED,
            CandidateStatus.REJECTED,
            CandidateStatus.INCONCLUSIVE,
        )
        for status in statuses
    )


def test_every_candidate_tried_both_windows_and_recorded_every_rule(replay_result):
    """The rules were applied in full, not short-circuited on the first failure.

    Deliberately not asserting that the replay produces a rejection. Whether a
    challenger wins depends on the data, and with the prediction-drift gate the
    first candidate arrives late enough that it does -- an earlier version
    using the faster performance gate raised a premature candidate at
    2026-03-05 and correctly rejected it. The reject path is tested with
    hand-built challengers above, where the outcome follows from construction
    rather than from whether the dataset happened to cooperate.
    """
    assert replay_result.candidates
    for candidate in replay_result.candidates:
        if candidate.status is CandidateStatus.INCONCLUSIVE:
            continue
        windows = {evaluation.training_window for evaluation in candidate.evaluations}
        assert windows == {"expanding", "recent_60d"}, f"{candidate.candidate_id} tried {windows}"
        for evaluation in candidate.evaluations:
            rules = {rule.rule for rule in evaluation.rules}
            assert "MAE improvement" in rules
            assert "training horizon" in rules
            assert "leakage probe" in rules
            assert any(rule.startswith("protected slice") for rule in rules)


def test_every_challenger_in_the_replay_respected_its_horizon(replay_result):
    """The guarantee that makes every number in the run meaningful."""
    for candidate in replay_result.candidates:
        for evaluation in candidate.evaluations:
            assert evaluation.training_end < candidate.decision_time, (
                f"{evaluation.version} trained to {evaluation.training_end} for a "
                f"decision at {candidate.decision_time}"
            )
            assert evaluation.leakage_clean


def test_the_adaptive_arm_serves_more_than_one_model(replay_result):
    assert len(replay_result.adaptive.models_used) > 1
    assert len(replay_result.static.models_used) == 1


def test_the_replay_is_deterministic(pipeline):
    """Evolution across runs is meaningless if the runs disagree."""

    def run() -> object:
        replay = AdaptiveReplay(
            table=pipeline["table"],
            reconciled=pipeline["reconciled"],
            champion=pipeline["champion"],
            reference=_window(pipeline["table"], pipeline["train_end"], pipeline["reference_end"]),
            config=pipeline["config"],
            training_end=pipeline["train_end"],
            timezone_offset_hours=TZ_OFFSET,
        )
        return replay.run(
            scenario="determinism",
            start=pipeline["reference_end"],
            # A shorter replay: determinism does not need the whole series, and
            # this keeps the test affordable.
            end=pipeline["reference_end"] + timedelta(days=70),
        )

    first, second = run(), run()
    assert [e.detected_at for e in first.events] == [e.detected_at for e in second.events]
    assert [e.is_drift for e in first.events] == [e.is_drift for e in second.events]
    assert [c.candidate_id for c in first.candidates] == [c.candidate_id for c in second.candidates]
    assert [c.status for c in first.candidates] == [c.status for c in second.candidates]
    assert first.static.overall.mae == pytest.approx(second.static.overall.mae, rel=1e-12)
    assert first.adaptive.overall.mae == pytest.approx(second.adaptive.overall.mae, rel=1e-12)


# ---------------------------------------------------------------------------
# Scenario runs are isolated from each other
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def merged_table(pipeline):
    """The history and the July holdout as one stream, validated once.

    The `july_holdout` scenario's table. Built here because two defects were
    only visible with it: the registry inherited state when a second scenario
    ran, and the leakage probe changed its verdict once a second file was
    merged in.
    """
    settings = load_settings()
    coordinates = ZoneRegistry.from_csv(ZONES_CSV)
    parts = [
        load_historical(path, registry=coordinates)
        for path in (HISTORY_CSV, DATA_DIR / "traffic_holdout.csv")
    ]
    combined = IngestionResult(
        observations=tuple(o for part in parts for o in part.observations),
        rejected=tuple(r for part in parts for r in part.rejected),
        stats=IngestionStats(
            rows_read=sum(part.stats.rows_read for part in parts),
            observations=sum(part.stats.observations for part in parts),
            rejected=sum(part.stats.rejected for part in parts),
            missing_values=sum(part.stats.missing_values for part in parts),
            source_path=HISTORY_CSV,
        ),
    )
    assert combined.accounts_for_every_row
    validated = validate_and_normalize(
        combined, settings=settings, registry=load_location_registry()
    )
    reconciled = reconcile(validated, settings=settings)
    return {
        "reconciled": reconciled,
        "table": build_feature_table(
            reconciled, load_feature_set(), timezone_offset_hours=TZ_OFFSET
        ),
    }


def _short_replay(
    pipeline: dict,
    *,
    scenario: str,
    table: FeatureTable | None = None,
    reconciled: ReconciledResult | None = None,
) -> tuple[AdaptiveReplay, ReplayResult]:
    """A deliberately short replay, for tests about bookkeeping not metrics."""
    replay = AdaptiveReplay(
        table=table if table is not None else pipeline["table"],
        reconciled=reconciled if reconciled is not None else pipeline["reconciled"],
        champion=pipeline["champion"],
        reference=_window(pipeline["table"], pipeline["train_end"], pipeline["reference_end"]),
        config=pipeline["config"],
        training_end=pipeline["train_end"],
        timezone_offset_hours=TZ_OFFSET,
    )
    result = replay.run(
        scenario=scenario,
        start=pipeline["reference_end"],
        end=pipeline["reference_end"] + timedelta(days=70),
    )
    return replay, result


def test_running_both_scenarios_and_repeating_one_keeps_one_champion(
    pipeline, merged_table, tmp_path, capsys
):
    """The sequence that crashed, run end to end against one registry root.

    The failure was inheritance: a second run re-seeded its starting champion
    into a root that still held the previous run's champion, and the next
    promotion raised ``2 champions`` naming a version the run had not reached.
    Four recordings against a single root -- both scenarios, and each repeated
    -- must each end with exactly one champion and raise nothing.
    """
    root = tmp_path / "adaptation"
    history_replay, history_result = _short_replay(pipeline, scenario="drift_history")
    holdout_replay, holdout_result = _short_replay(
        pipeline,
        table=merged_table["table"],
        reconciled=merged_table["reconciled"],
        scenario="july_holdout",
    )

    sequence = [
        ("drift_history", history_replay, history_result),
        ("july_holdout", holdout_replay, holdout_result),
        ("drift_history", history_replay, history_result),
        ("july_holdout", holdout_replay, holdout_result),
    ]

    seen: list[tuple[str, Path, str]] = []
    for label, replay, result in sequence:
        # No try/except: an inherited champion raised LineageError here, so the
        # absence of an exception is half of what this test asserts.
        registry_path, champion = replay.record(result, root=root)
        store = ModelRegistry(registry_path, registry_path.parent / "artifacts")
        assert store.champions() == (champion,)
        assert assert_single_champion(store) == champion
        seen.append((label, registry_path, champion))

    with capsys.disabled():
        print("\n  four recordings against one root:")
        for label, path, champion in seen:
            print(f"    {label:<14} -> {champion}  ({path.parent.name})")

    # Each run got its own directory, which is what makes inheritance impossible.
    assert len({path for _, path, _ in seen}) == len(seen)


def test_a_run_directory_holds_only_that_runs_versions(pipeline, tmp_path):
    """Isolation by construction, not by cleaning up afterwards."""
    root = tmp_path / "adaptation"
    replay, result = _short_replay(pipeline, scenario="drift_history")

    first_path, _ = replay.record(result, root=root)
    second_path, _ = replay.record(result, root=root)

    assert first_path != second_path
    first = ModelRegistry(first_path, first_path.parent / "artifacts")
    second = ModelRegistry(second_path, second_path.parent / "artifacts")
    assert first.versions() == second.versions(), "same run, same versions"
    # And neither sees the other's files.
    assert len(first.versions()) == len(list(first_path.glob("*.json")))


def test_promote_demotes_every_other_champion(registry):
    """Promotion means demoting whatever else holds the title.

    The old implementation archived only a named incumbent, so a registry that
    already held a stale champion ended with two and failed its own
    post-condition several steps later.
    """
    for version in ("Traffic-v1.2-expanding", "Traffic-v1.3-expanding"):
        entry = _base_version(version)
        registry.save(entry)  # saved as CHAMPION, deliberately
    assert len(registry.champions()) == 3, "three champions, as the broken state had"

    promoted, archived = promote(registry, "Traffic-v1.3-expanding")

    assert promoted == "Traffic-v1.3-expanding"
    assert set(archived) == {"Traffic-v1.1", "Traffic-v1.2-expanding"}
    assert registry.champions() == ("Traffic-v1.3-expanding",)


# ---------------------------------------------------------------------------
# The leakage verdict cannot depend on later data
# ---------------------------------------------------------------------------


def test_the_probe_verdict_is_identical_whatever_later_data_exists(pipeline, merged_table, capsys):
    """The same challenger, judged against two tables that differ only by a suffix.

    This was the second defect: the expanding-window challenger at 2026-03-12
    came out clean against the history and FAILED against history+holdout, with
    237 findings. Nothing about the challenger changed -- merging a second CSV
    gave the stream two load instants instead of one, which flipped the
    receipt-time clause from not-applicable to probed, and then every record
    counted as a late arrival.
    """
    decision = datetime(2026, 3, 12, tzinfo=UTC)
    window = TrainingWindow(name="expanding")

    verdicts = {}
    for label, source in (
        ("history only", pipeline),
        ("history+holdout", merged_table),
    ):
        fit = train_challenger(
            source["table"],
            decision_time=decision,
            window=window,
            reconciled=source["reconciled"],
            timezone_offset_hours=TZ_OFFSET,
        )
        assert fit.leakage is not None
        verdicts[label] = (
            fit.leakage.is_clean,
            fit.leakage.event_time_clause,
            fit.leakage.received_time_clause,
            len(fit.leakage.findings),
        )

    with capsys.disabled():
        print("\n  same challenger, two tables:")
        for label, (clean, event, receipt, findings) in verdicts.items():
            print(
                f"    {label:<16} {'CLEAN' if clean else 'FAILED':<7} "
                f"event={event.value}  receipt={receipt.value}  findings={findings}"
            )

    assert verdicts["history only"] == verdicts["history+holdout"], (
        "the probe's verdict changed because of data after the decision time"
    )
    assert all(clean for clean, *_ in verdicts.values())


def test_merging_a_second_file_adds_no_duplicates_or_conflicts(pipeline, merged_table):
    """Rules out the other explanations before accepting the receipt one.

    If the holdout had overlapped the history, the merged stream would carry new
    duplicate or conflicting keys and the extra findings could have been a real
    leak. It does not: both counts are identical, so the only difference between
    the two tables is the number of load instants.
    """
    history = pipeline["reconciled"]
    merged = merged_table["reconciled"]

    assert len(merged.unified) > len(history.unified), "the holdout did add rows"
    assert len(merged.duplicate_linked) == len(history.duplicate_linked)
    assert len(merged.conflicted_observations) == len(history.conflicted_observations)


def test_the_receipt_clause_is_unprobeable_on_any_bulk_load(pipeline, merged_table):
    """One file or two, a bulk load's receipts all postdate the events.

    The count of distinct receipt instants differs between the two tables -- that
    is the input that used to change the verdict -- and the clause status must
    not.
    """
    reports = [
        detect_leakage(source["reconciled"], load_feature_set(), timezone_offset_hours=TZ_OFFSET)
        for source in (pipeline, merged_table)
    ]
    assert reports[0].distinct_received_times != reports[1].distinct_received_times
    assert all(report.received_time_clause is ClauseStatus.NOT_APPLICABLE for report in reports)
    assert all("not probed" in report.received_time_note for report in reports)
    assert all(report.is_clean for report in reports)


def test_the_replay_result_serializes(replay_result):
    payload = replay_result.as_dict()
    assert payload["scenario"] == "drift_history"
    assert payload["static"]["rows"] == payload["adaptive"]["rows"]
    assert "relative_improvement" in payload
    assert "regressed_slices" in payload
    assert json.dumps(payload)


def test_the_drifted_zones_improve_most(replay_result, truth, capsys):
    """Scored against ground truth: adaptation should help where the shift was.

    ZONE-A and ZONE-B carry the planted level shift; ZONE-C and ZONE-D do not.
    If adaptation helped the control zones as much as the drifted ones, it would
    be fitting noise rather than tracking a change.
    """
    drifted = set(truth["anomalies"]["baseline_drift"]["location_ids"])
    controls = set(truth["anomalies"]["baseline_drift"]["control_location_ids"])

    gains: dict[str, float] = {}
    for item in replay_result.adaptive.slices:
        baseline = replay_result.static.slice_named(item.name)
        if baseline is None or baseline.mae <= 0 or not item.name.startswith("zone "):
            continue
        gains[item.name.removeprefix("zone ")] = 1.0 - (item.mae / baseline.mae)

    with capsys.disabled():
        print("\n  per-zone improvement from adaptation:")
        for zone, gain in sorted(gains.items()):
            tag = "planted" if zone in drifted else "control"
            print(f"    {zone}  {gain:+.1%}  ({tag})")

    best_drifted = max(gains[zone] for zone in drifted if zone in gains)
    best_control = max(gains[zone] for zone in controls if zone in gains)
    assert best_drifted > best_control, (
        "adaptation helped a control zone more than a drifted one, which would "
        "suggest it is fitting noise rather than tracking the shift"
    )
