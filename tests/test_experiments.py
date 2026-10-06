"""The three-arm experiment (SPEC sections 38, 39).

The experiment exists to answer a question Phase 7 did not: adaptive retraining
beat a frozen model, but nobody had tried the cheap alternative of retraining on
a calendar. These tests guard the properties that make the comparison worth
believing rather than the numbers it happens to produce.

Four things matter here, and each has a test:

**The arms are scored on identical rows.** Not "the same number of rows" --
the same records, in the same order. If the arms were evaluated on even slightly
different slices, the error figures would not be comparable and the whole
experiment would be decoration.

**Retrains happen where each arm says they do.** The scheduled arm retrains on
the calendar and nowhere else; the adaptive arm retrains only when the promotion
rules accept a challenger. If the counts drifted from the reported cost, the
cost column would be fiction.

**The run is deterministic for a given seed list.** A result that moves between
runs cannot be pre-registered.

**The report does not assume adaptive wins.** The generator is fed hand-built
results in which adaptive loses outright and in which it ties, and must say so.
This is the test that would have caught a report template written around the
conclusion its author expected.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from urbansense.adaptation.replay import ArmResult, ReplayResult, RetrainCost
from urbansense.config import CongestionConfig, load_settings
from urbansense.evaluation.metrics import SliceMetrics
from urbansense.experiments.config import (
    ADAPTIVE,
    ARM_ORDER,
    SCHEDULED,
    STATIC,
    ExperimentConfig,
    ExperimentConfigError,
    load_experiment_config,
)
from urbansense.experiments.metrics import (
    Aggregate,
    ArmMetrics,
    ClassificationScore,
    within_one_sd,
)
from urbansense.experiments.report import (
    Verdict,
    build_report,
    decide,
    summarize_arm,
    write_report,
)
from urbansense.experiments.runner import ExperimentRunner, SeedRun
from urbansense.prediction.congestion import CongestionThresholds
from urbansense.schemas import ProblemType

# A short replay. The properties under test -- identical rows, retrain
# bookkeeping, determinism -- do not need the full stream, and the full stream
# costs minutes per seed.
REPLAY_DAYS = 45


def _config(**overrides: object) -> ExperimentConfig:
    """The pre-registered config, narrowed for a fast test run."""
    base = load_experiment_config()
    defaults: dict[str, object] = {"seeds": [20260101], "replay_days": REPLAY_DAYS}
    return base.model_copy(update={**defaults, **overrides})


@pytest.fixture(scope="module")
def single_seed(tmp_path_factory: pytest.TempPathFactory) -> SeedRun:
    """One full seed run, shared by the tests that only need to inspect it.

    Module-scoped because generating and replaying a seed takes the better part
    of a minute, and nothing in these tests mutates the result.
    """
    workspace = tmp_path_factory.mktemp("experiment")
    runner = ExperimentRunner(config=_config(), workspace=workspace)
    return runner.run_seed(20260101)


class TestPreRegistration:
    """The committed configuration is the experiment's pre-registration."""

    def test_the_committed_config_loads_and_names_three_arms(self) -> None:
        """The file under configs/ is the one the experiment actually reads."""
        config = load_experiment_config()
        assert tuple(config.arms) == ARM_ORDER
        assert config.pre_registered
        assert config.retrain_every == timedelta(days=config.retrain_every_days)

    def test_duplicate_seeds_are_rejected(self) -> None:
        """The same dataset twice would narrow the spread without adding evidence."""
        payload = {**load_experiment_config().model_dump(), "seeds": [1, 1]}
        with pytest.raises(ValidationError, match="duplicate seeds"):
            ExperimentConfig.model_validate(payload)

    def test_a_single_arm_is_rejected(self) -> None:
        """One arm is not a comparison."""
        payload = {**load_experiment_config().model_dump(), "arms": [STATIC]}
        with pytest.raises(ValidationError, match="at least two arms"):
            ExperimentConfig.model_validate(payload)

    def test_a_malformed_config_file_raises_the_packages_own_error(self, tmp_path: Path) -> None:
        """The loader wraps pydantic's error, so callers catch one exception type."""
        bad = tmp_path / "experiment.yaml"
        bad.write_text("seeds: [1, 1]", encoding="utf-8")
        with pytest.raises(ExperimentConfigError):
            load_experiment_config(bad)


class TestIdenticalEvaluationRows:
    """All three arms must be measured on exactly the same records."""

    def test_all_three_arms_score_the_identical_rows(self, single_seed: SeedRun) -> None:
        """Same records, same order -- not merely the same count.

        This is the property that makes the arms comparable at all, and it holds
        by construction: the replay loop predicts three times on the one window
        slice it already accumulated, rather than each arm walking the stream
        itself.
        """
        arms = single_seed.replay.arms
        assert set(arms) == set(ARM_ORDER)

        record_ids = single_seed.replay.record_ids
        assert len(record_ids) == len(set(record_ids)), "a record was scored twice"
        assert len(record_ids) == single_seed.rows > 0

        for name, arm in arms.items():
            assert arm.rows == single_seed.rows, f"{name} scored a different number of rows"
            assert arm.predictions.shape == single_seed.replay.actual.shape, (
                f"{name} produced a prediction count that does not match the targets"
            )

    def test_the_arms_predictions_differ_even_though_the_rows_do_not(
        self, single_seed: SeedRun
    ) -> None:
        """Identical rows must not mean identical predictions.

        A guard against the failure mode the previous test cannot see: if a
        refactor pointed all three arms at one model, every row assertion would
        still pass and the experiment would silently compare a model with
        itself.
        """
        static = single_seed.replay.static.predictions
        scheduled = single_seed.replay.arms[SCHEDULED].predictions
        adaptive = single_seed.replay.adaptive.predictions
        assert not np.array_equal(static, scheduled)
        assert not np.array_equal(static, adaptive)

    def test_every_arm_is_scored_against_the_same_observed_targets(
        self, single_seed: SeedRun
    ) -> None:
        """The targets come from the stream once, not per arm."""
        actual = single_seed.replay.actual
        assert actual.size == single_seed.rows
        assert not np.isnan(actual).any(), "a missing target reached the evaluation set"


class TestRetrainBookkeeping:
    """Each arm retrains where its own rules say, and the cost reflects it."""

    def test_the_static_arm_never_retrains(self, single_seed: SeedRun) -> None:
        """A frozen champion is the point of the arm."""
        cost = single_seed.replay.static.cost
        assert cost == RetrainCost(retrains=0, fits=0, training_rows=0)
        assert len(set(single_seed.replay.static.models_used)) == 1

    def test_the_scheduled_arm_retrains_on_the_calendar_and_nowhere_else(
        self, single_seed: SeedRun
    ) -> None:
        """One deployment per elapsed interval, one fit each.

        Deployed unconditionally, so retrains and fits are equal -- unlike the
        adaptive arm, which trains challengers it then rejects.
        """
        config = _config()
        arm = single_seed.replay.arms[SCHEDULED]
        span = single_seed.replay.scored_to - single_seed.replay.scored_from
        # One deployment per fully elapsed interval. The bound is a range rather
        # than an equality because the scored span is measured from the first
        # and last rows that carried a target, which need not land exactly on
        # the replay's requested boundaries.
        expected = int(span / config.retrain_every)

        assert expected <= arm.cost.retrains <= expected + 1, (
            f"{arm.cost.retrains} retrain(s) over a {span.days}-day replay "
            f"does not match a {config.retrain_every_days}-day calendar"
        )
        assert arm.cost.fits == arm.cost.retrains, (
            "the scheduled arm deploys every model it trains, so fits must equal retrains"
        )
        assert arm.cost.training_rows > 0

    def test_the_adaptive_arm_retrains_only_on_a_promotion(self, single_seed: SeedRun) -> None:
        """Fits may exceed retrains; retrains may not exceed promotions.

        The gap between the two is the adaptive arm's distinguishing cost: it
        pays to train challengers that the promotion rules then reject.
        """
        arm = single_seed.replay.adaptive
        promotions = [item for item in single_seed.replay.candidates if item.promoted_version]
        assert arm.cost.retrains == len(promotions)
        assert arm.cost.fits >= arm.cost.retrains

    def test_training_rows_are_counted_per_fit_not_per_deployment(
        self, single_seed: SeedRun
    ) -> None:
        """Rejected challengers still cost their training rows.

        Counting only deployed models would understate the adaptive arm's cost
        in exactly the direction that flatters it.
        """
        adaptive = single_seed.replay.adaptive
        if adaptive.cost.fits > adaptive.cost.retrains:
            assert adaptive.cost.training_rows > 0


class TestDeterminism:
    """A pre-registered experiment has to produce the same answer twice."""

    def test_the_same_seed_gives_identical_metrics_and_costs(self, tmp_path: Path) -> None:
        """Two independent runs of one seed agree exactly."""
        config = _config()
        first = ExperimentRunner(config=config, workspace=tmp_path / "a").run_seed(20260101)
        second = ExperimentRunner(config=config, workspace=tmp_path / "b").run_seed(20260101)

        assert first.rows == second.rows
        assert first.replay.record_ids == second.replay.record_ids
        for name in ARM_ORDER:
            left, right = first.metrics[name], second.metrics[name]
            assert left.mae == pytest.approx(right.mae, rel=1e-12)
            assert left.rmse == pytest.approx(right.rmse, rel=1e-12)
            assert left.brier == pytest.approx(right.brier, rel=1e-12)
            assert left.cost == right.cost

    def test_the_workspace_is_used_and_the_committed_dataset_is_not_touched(
        self, tmp_path: Path
    ) -> None:
        """Generation goes to the workspace; data/sample is a fixture.

        Other phases' tests assert against the committed dataset. Regenerating
        it here with a different seed would break them in a way that looks
        unrelated to this module.
        """
        before = Path("data/sample/traffic_history.csv").read_bytes()
        runner = ExperimentRunner(config=_config(), workspace=tmp_path)
        runner.run_seed(20260101)
        assert list(tmp_path.glob("seed-*")), "nothing was generated into the workspace"
        assert Path("data/sample/traffic_history.csv").read_bytes() == before


def _metrics(
    arm: str,
    *,
    seed: int,
    mae: float,
    slice_mae: float,
    f1_hits: int = 80,
    retrains: int = 0,
    fits: int = 0,
    rows: int = 0,
) -> ArmMetrics:
    """A hand-built arm result, for driving the report with known numbers."""
    return ArmMetrics(
        arm=arm,
        seed=seed,
        rows=1_000,
        mae=mae,
        rmse=mae * 1.6,
        slices=(
            SliceMetrics(
                name="ZONE-A Fri 18-20",
                rows=10,
                mae=slice_mae,
                rmse=slice_mae * 1.5,
                coverage=1.0,
                mean_actual=5_000.0,
            ),
        ),
        congestion=ClassificationScore(
            true_positive=f1_hits,
            false_positive=100 - f1_hits,
            false_negative=100 - f1_hits,
            true_negative=800,
        ),
        brier=0.08,
        reference_brier=0.2,
        cost=RetrainCost(retrains=retrains, fits=fits, training_rows=rows),
        models_used=(f"{arm}-v1",),
    )


def _congestion_config() -> CongestionConfig:
    """The traffic congestion block, which the problem config is required to define."""
    found = load_settings().problems[ProblemType.TRAFFIC].congestion
    assert found is not None, "the traffic problem config must define a congestion block"
    return found


def _stub_run(seed: int, metrics: dict[str, ArmMetrics]) -> SeedRun:
    """A SeedRun carrying hand-built metrics and the minimum around them.

    The report serializes a run's replay and thresholds, so both have to exist;
    neither is read for the properties these tests assert, so both are empty
    rather than generated.
    """
    empty = SliceMetrics(
        name="overall", rows=0, mae=float("nan"), rmse=float("nan"), coverage=0.0, mean_actual=0.0
    )
    arm = ArmResult(name=STATIC, overall=empty, slices=(), rows=0, models_used=())
    moment = datetime(2026, 6, 1, tzinfo=UTC)
    return SeedRun(
        seed=seed,
        metrics=metrics,
        replay=ReplayResult(
            scenario=f"stub-{seed}",
            events=(),
            candidates=(),
            static=arm,
            adaptive=arm,
            scored_from=moment,
            scored_to=moment,
        ),
        rows=1_000,
        thresholds=CongestionThresholds(
            by_zone={"ZONE-A": 5_000.0},
            config=_congestion_config(),
            fitted_on="train",
            rows=0,
            first_sample=moment.isoformat(),
            last_sample=moment.isoformat(),
        ),
    )


def _verdict_from(rows: dict[str, list[tuple[float, float]]]) -> Verdict:
    """Decide between arms given per-arm ``(mae, slice_mae)`` pairs per seed.

    Goes straight to :func:`decide` rather than through a full report, because
    these tests are about the ranking and the tie language, and a hand-built
    :class:`SeedRun` would need a replay and a threshold set it never reads.
    """
    summaries = {
        arm: summarize_arm(
            arm,
            [
                _metrics(arm, seed=index, mae=mae, slice_mae=slice_mae)
                for index, (mae, slice_mae) in enumerate(values)
            ],
        )
        for arm, values in rows.items()
    }
    return decide(summaries)


class TestTheReportDoesNotAssumeAdaptiveWins:
    """Fed a result where adaptive loses, the generator must say adaptive lost.

    The point of the experiment is that its conclusion was not decided in
    advance. A report template built around "adaptation helps" would pass every
    other test in this file and still be worthless.
    """

    def test_a_clear_adaptive_loss_is_reported_as_a_loss(self) -> None:
        """Scheduled is clearly better; the verdict ranks it first and says so."""
        verdict = _verdict_from(
            {
                STATIC: [(300.0, 1200.0), (302.0, 1210.0)],
                SCHEDULED: [(200.0, 600.0), (201.0, 610.0)],
                ADAPTIVE: [(260.0, 900.0), (261.0, 910.0)],
            }
        )
        assert verdict.ranking == (SCHEDULED, ADAPTIVE, STATIC)
        assert verdict.best == SCHEDULED
        assert not verdict.is_tie(SCHEDULED, ADAPTIVE)
        assert SCHEDULED in verdict.conclusion

    def test_a_gap_inside_the_spread_is_reported_as_a_tie(self) -> None:
        """Noise is not a finding.

        Two arms whose means differ by less than the seed-to-seed spread are
        reported as not distinguished, in words, rather than ranked as though
        the ordering meant something.
        """
        verdict = _verdict_from(
            {
                SCHEDULED: [(200.0, 600.0), (240.0, 640.0)],
                ADAPTIVE: [(205.0, 610.0), (245.0, 650.0)],
            }
        )
        assert verdict.is_tie(SCHEDULED, ADAPTIVE)
        assert "not distinguished" in verdict.conclusion

    def test_an_adaptive_win_is_also_reported_plainly(self) -> None:
        """The generator is not biased against adaptation either.

        The counterpart to the loss test: a report that could only ever report a
        tie or a loss would be as broken as one that could only report a win.
        """
        verdict = _verdict_from(
            {
                SCHEDULED: [(300.0, 900.0), (301.0, 905.0)],
                ADAPTIVE: [(200.0, 600.0), (201.0, 610.0)],
            }
        )
        assert verdict.best == ADAPTIVE
        assert not verdict.is_tie(SCHEDULED, ADAPTIVE)

    def test_regressions_are_listed_for_the_winning_arm_too(self) -> None:
        """An arm can win overall and still lose the hours that matter.

        Here adaptive has the lower overall error but the worse Friday-evening
        slice, and the report has to surface that rather than let the headline
        stand alone.
        """
        seeds = [
            _stub_run(
                seed,
                {
                    SCHEDULED: _metrics(SCHEDULED, seed=seed, mae=300.0, slice_mae=600.0),
                    ADAPTIVE: _metrics(ADAPTIVE, seed=seed, mae=200.0, slice_mae=1500.0),
                },
            )
            for seed in (1, 2)
        ]
        report = build_report(_config(seeds=[1, 2]), seeds, now=datetime(2026, 10, 4, tzinfo=UTC))

        assert report.verdict.best == ADAPTIVE
        assert "ZONE-A Fri 18-20" in report.regressions(ADAPTIVE)
        assert report.regressions(SCHEDULED) == ()

        rendered = report.render()
        assert "ZONE-A Fri 18-20" in rendered
        assert "Limitations" in rendered
        assert "Synthetic data" in rendered

    def test_an_empty_report_is_refused(self) -> None:
        """A table of n/a would read like a finished experiment."""
        with pytest.raises(ValueError, match="no seed runs"):
            build_report(_config(), [])

    def test_the_written_report_carries_the_pre_registration(self, tmp_path: Path) -> None:
        """A result without the parameters fixed in advance is not evidence."""
        config = _config(seeds=[1])
        seeds = [_stub_run(1, {ADAPTIVE: _metrics(ADAPTIVE, seed=1, mae=200.0, slice_mae=600.0)})]
        report = build_report(config, seeds, now=datetime(2026, 10, 4, tzinfo=UTC))
        json_path, markdown_path = write_report(report, tmp_path)

        assert json_path.is_file() and markdown_path.is_file()
        payload = json_path.read_text(encoding="utf-8")
        assert config.pre_registered in payload
        assert "pre_registered_config" in payload
        assert "limitations" in payload


class TestAggregationIsHonest:
    """The spread has to mean what it says."""

    def test_one_seed_has_no_spread_rather_than_zero_spread(self) -> None:
        """nan, not 0.0: one dataset carries no information about variation."""
        single = Aggregate.of([200.0])
        assert single.mean == pytest.approx(200.0)
        assert np.isnan(single.sd)
        assert "n=1" in single.describe() or "+/-" not in single.describe()

    def test_precision_is_nan_when_nothing_was_called(self) -> None:
        """An arm that called no congestion has undefined precision, not perfect precision."""
        silent = ClassificationScore(
            true_positive=0, false_positive=0, false_negative=50, true_negative=950
        )
        assert np.isnan(silent.precision)
        assert silent.recall == pytest.approx(0.0)

    def test_without_a_measured_spread_nothing_is_called_a_difference(self) -> None:
        """Single-seed figures are "too close to call", never a finding.

        The default runs toward caution rather than toward a headline: with no
        spread measured, a 1-point gap between two one-dataset numbers is not
        evidence of anything, so it is reported as undistinguished rather than
        as a win.
        """
        assert within_one_sd(Aggregate.of([200.0]), Aggregate.of([201.0]))
        assert within_one_sd(Aggregate.of([200.0, 240.0]), Aggregate.of([205.0, 245.0]))

    def test_a_gap_larger_than_the_spread_is_a_difference(self) -> None:
        """Caution is the default, not the only answer."""
        assert not within_one_sd(Aggregate.of([200.0, 205.0]), Aggregate.of([400.0, 405.0]))

    def test_a_missing_figure_is_never_tied_with_a_present_one(self) -> None:
        """An arm that produced no number did not draw with one that did."""
        assert not within_one_sd(Aggregate.of([]), Aggregate.of([200.0, 240.0]))


class TestNoLeakage:
    """Every arm trains only on data available before its decision time."""

    def test_no_challenger_trains_on_data_at_or_after_its_decision_time(
        self, single_seed: SeedRun
    ) -> None:
        """The horizon gap holds for every model trained during the replay.

        The scheduled arm reuses the adaptive arm's training function, its
        ``available_at`` filter and its gap exactly; only what happens
        afterwards differs. So this assertion covers both, and any leakage
        advantage would apply to them equally -- which is what makes the
        comparison fair rather than merely close.
        """
        checked = 0
        for candidate in single_seed.replay.candidates:
            for evaluation in candidate.evaluations:
                assert evaluation.training_end < candidate.decision_time, (
                    f"{evaluation.version} trained on data at or after its decision time"
                )
                assert evaluation.leakage_clean, f"{evaluation.version} failed the leakage probe"
                checked += 1
        if not single_seed.replay.candidates:
            pytest.skip("no candidate was raised on this seed, so nothing was trained to check")
        assert checked > 0

    def test_the_evaluation_window_starts_after_the_reference_period(
        self, single_seed: SeedRun
    ) -> None:
        """Scoring begins past training and reference; neither is evaluated on.

        The frozen champion's training segment and the drift detector's
        reference window both precede the first scored instant, so no arm is
        measured on rows that shaped it.
        """
        assert single_seed.replay.scored_from < single_seed.replay.scored_to
        train_end = datetime.fromisoformat(single_seed.thresholds.last_sample)
        assert single_seed.replay.scored_from > train_end, (
            "scoring began inside the period the champion and the congestion "
            "threshold were fitted on"
        )

    def test_the_replay_exposes_one_shared_set_of_scored_records(
        self, single_seed: SeedRun
    ) -> None:
        """The record ids live on the result, not per arm.

        Structural rather than numeric: there is one list of scored records
        because there was one walk through the stream. An arm cannot have been
        scored on a different set, because no other set exists.
        """
        assert len(single_seed.replay.record_ids) == single_seed.replay.actual.size
        assert len(single_seed.replay.locations) == single_seed.replay.actual.size
