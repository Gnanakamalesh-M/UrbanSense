"""Tests for drift detection, scored against the planted change point.

A drift detector has two failure modes and only one of them is obvious. Missing
a real shift is bad; firing on a quiet stretch is worse, because a detector that
cries wolf gets muted and then misses the real one silently. So every detection
test here is paired with a false-alarm count, and the central test compares the
demo dataset against a **drift-free copy of itself** -- same seed, the level
shift simply switched off. Asserting quiet on the first half of a drifted series
only shows the detector is not firing yet; asserting quiet on a stream with
nothing in it is the real claim.

That comparison is what established the phase's main finding, and it is
counterintuitive enough to be worth a test of its own: feature and performance
drift fire on *both* streams because this generator has annual seasonality, and
only prediction drift separates a changed city from a model that is merely out
of season.

`ground_truth.json` is read here and nowhere in `src/`. The detectors find the
change point by watching the error rise, the same way a deployment would.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from urbansense.config import load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.drift import (
    PSI_MAJOR,
    DriftKind,
    DriftMonitor,
    DriftStatus,
    DriftStore,
    EpisodeTracker,
    MonitorConfig,
    detection_delay,
    false_alarm_count,
    feature_drift,
    ks_test,
    performance_drift,
    prediction_drift,
    psi,
    sigma_gate,
)
from urbansense.drift.events import DriftEvent
from urbansense.features import build_feature_table, load_feature_set
from urbansense.features.builder import FeatureTable
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.prediction import GradientForecaster
from urbansense.preprocessing import load_location_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "sample"
HISTORY_CSV = DATA_DIR / "traffic_history.csv"
ZONES_CSV = DATA_DIR / "zones.csv"
GROUND_TRUTH = DATA_DIR / "ground_truth.json"
TZ_OFFSET = 5.5

#: Longest acceptable gap between the planted change and the first *qualifying*
#: firing. Two weekly evaluations: the first may land before enough drifted data
#: has accumulated for the output distribution to move.
MAX_DETECTION_DELAY = timedelta(days=14)

#: Families allowed to raise a candidate. Prediction drift alone, because it is
#: the only one that stayed silent on the drift-free stream.
REQUIRED = frozenset({DriftKind.PREDICTION})

#: Boundaries for the replay. The reference window begins exactly where the
#: training period ends, so it is entirely out of sample -- an overlapping
#: reference understates the reference error and floods the run with false
#: alarms, which is what an earlier version of this suite did.
TRAIN_DAYS = 75
REFERENCE_DAYS = 60
REPLAY_END = datetime(2026, 6, 25, tzinfo=UTC)


@pytest.fixture(scope="module")
def truth() -> dict:
    """The planted answers. Only tests read this."""
    return json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def table() -> FeatureTable:
    settings = load_settings()
    validated = validate_and_normalize(
        load_historical(HISTORY_CSV, registry=ZoneRegistry.from_csv(ZONES_CSV)),
        settings=settings,
        registry=load_location_registry(),
    )
    return build_feature_table(
        reconcile(validated, settings=settings),
        load_feature_set(),
        timezone_offset_hours=TZ_OFFSET,
    )


def _window(table: FeatureTable, start: datetime, end: datetime) -> FeatureTable:
    mask = np.array([start <= moment < end for moment in table.sample_times], dtype=bool)
    return table.select(mask)


def _replay(table: FeatureTable) -> dict:
    """Monitor a table with the standard boundaries, derived from its own span.

    Boundaries come from the data rather than hard-coded dates, so the drifted
    and drift-free streams are monitored identically and a regenerated dataset
    cannot silently change what is being tested.
    """
    earliest = min(table.sample_times)
    train_end = earliest + timedelta(days=TRAIN_DAYS)
    reference_end = train_end + timedelta(days=REFERENCE_DAYS)
    champion = GradientForecaster().fit(_window(table, earliest, train_end))
    monitor = DriftMonitor(
        reference=_window(table, train_end, reference_end),
        predict=champion.predict,
        training_end=train_end,
    )
    gates = monitor.calibrate()
    events, candidates = monitor.replay(table, start=reference_end, end=REPLAY_END)
    return {
        "monitor": monitor,
        "gates": gates,
        "events": events,
        "candidates": candidates,
        "train_end": train_end,
        "reference_end": reference_end,
    }


@pytest.fixture(scope="module")
def replayed(table):
    """A full monitored replay of the history, with a pre-drift champion."""
    return _replay(table)


# ---------------------------------------------------------------------------
# The planted change is found, and nothing before it is
# ---------------------------------------------------------------------------


def test_the_planted_drift_is_detected_with_a_short_delay(replayed, truth, capsys):
    """Scored against ground truth, which only tests read.

    Two delays are printed because they differ and both are honest: the first
    firing of any kind, and the first firing that would actually raise a
    candidate. Specificity costs about a week here.
    """
    planted = truth["anomalies"]["baseline_drift"]
    change_point = datetime.fromisoformat(planted["start_date"]).replace(tzinfo=UTC)
    any_delay = detection_delay(replayed["events"], change_point=change_point)
    acting_delay = detection_delay(replayed["events"], change_point=change_point, required=REQUIRED)

    with capsys.disabled():
        print(
            f"\n  planted shift x{planted['volume_multiplier']} in "
            f"{', '.join(planted['location_ids'])} from {planted['start_date']}"
        )
        print(f"  first firing of any kind:  {any_delay.days if any_delay else None} days")
        print(
            f"  first acting firing:       "
            f"{acting_delay.days if acting_delay else None} days "
            f"(prediction drift, the only family that does not also fire on "
            f"seasonal degradation)"
        )

    assert any_delay is not None, "the planted drift was never detected"
    assert acting_delay is not None, "no firing qualified to raise a candidate"
    assert acting_delay <= MAX_DETECTION_DELAY, f"acted {acting_delay.days} days after the change"
    assert any_delay <= acting_delay, "a qualifying firing cannot precede any firing"


def test_no_false_alarm_before_the_planted_change(replayed, truth, capsys):
    """The count that decides whether anyone will trust this detector."""
    change_point = datetime.fromisoformat(
        truth["anomalies"]["baseline_drift"]["start_date"]
    ).replace(tzinfo=UTC)
    stable = [event for event in replayed["events"] if event.detected_at < change_point]
    any_alarms = false_alarm_count(replayed["events"], before=change_point)
    acting_alarms = false_alarm_count(replayed["events"], before=change_point, required=REQUIRED)

    with capsys.disabled():
        print(f"\n  stable stretch: {len(stable)} evaluation(s) before {change_point:%Y-%m-%d}")
        print(f"    firings of any kind:  {any_alarms}")
        print(f"    firings that would act: {acting_alarms}")
        for event in stable:
            print(f"    {event.describe()[:100]}")

    assert stable, "the replay did not cover any pre-drift period"
    assert acting_alarms == 0, f"{acting_alarms} false alarm(s) would have retrained"
    assert any_alarms == 0, f"{any_alarms} false alarm(s) on a stable stretch"


def test_the_error_detector_is_the_one_that_finds_it_first(replayed, truth, capsys):
    """Recorded because it is the opposite of the intuitive expectation.

    Distributional feature drift is the headline idea in SPEC section 29, but on
    a level shift the model's own error moves first and most cleanly.
    """
    change_point = datetime.fromisoformat(
        truth["anomalies"]["baseline_drift"]["start_date"]
    ).replace(tzinfo=UTC)
    first = next(
        event
        for event in replayed["events"]
        if event.is_drift and event.detected_at >= change_point
    )
    kinds = {signal.kind for signal in first.fired}
    with capsys.disabled():
        print(
            f"\n  first firing {first.detected_at:%Y-%m-%d}: "
            f"{', '.join(sorted(kind.value for kind in kinds))}"
        )
    assert DriftKind.PERFORMANCE in kinds


def test_every_detector_family_fires_at_some_point(replayed):
    """All three of SPEC section 29's families are exercised, not just one."""
    fired = {signal.kind for event in replayed["events"] for signal in event.fired}
    assert fired == {DriftKind.FEATURE, DriftKind.PREDICTION, DriftKind.PERFORMANCE}


def test_the_gates_come_from_the_reference_period(replayed, capsys):
    gates = replayed["gates"]
    with capsys.disabled():
        print()
        print(f"  {gates.describe()}")
    assert gates.established
    assert gates.feature_psi == PSI_MAJOR
    assert gates.prediction_psi > 0
    # Above 1.0: a window that is *better* than the reference is not drift.
    assert gates.performance_ratio > 1.0
    assert "out-of-sample" in str(gates.as_dict()["reference_error_basis"])


# ---------------------------------------------------------------------------
# A stream with no drift in it at all
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stable_table(tmp_path_factory) -> FeatureTable:
    """The demo dataset with its level shift switched off, and nothing else.

    Only the drift is changed. Shortening the timeline as well would move the
    planted missing-data windows outside the generated period, and changing more
    than one thing would weaken the comparison: this should be the same stream
    with the one effect removed.
    """
    from urbansense.synthetic import load_generator_config
    from urbansense.synthetic.generator import generate_to_directory

    base = load_generator_config()
    config = base.model_copy(update={"drift": base.drift.model_copy(update={"enabled": False})})
    out = tmp_path_factory.mktemp("stable")
    generate_to_directory(config, out, force=True)

    settings = load_settings()
    validated = validate_and_normalize(
        load_historical(
            out / "traffic_history.csv", registry=ZoneRegistry.from_csv(out / "zones.csv")
        ),
        settings=settings,
        registry=load_location_registry(),
    )
    return build_feature_table(
        reconcile(validated, settings=settings),
        load_feature_set(),
        timezone_offset_hours=config.timezone_offset_hours,
    )


def test_a_stream_with_no_planted_drift_raises_no_candidate(stable_table, capsys):
    """The claim that matters: nothing planted, nothing acted on.

    Firings are *expected* here and are not a failure. The champion trains on
    autumn and the stream runs to midsummer, so its error genuinely rises --
    that is seasonal degradation, correctly reported. What must not happen is a
    retrain, and no firing on this stream qualifies to raise one.
    """
    run = _replay(stable_table)
    events = run["events"]
    fired = [event for event in events if event.is_drift]
    qualifying = [event for event in events if event.qualifies(REQUIRED)]

    with capsys.disabled():
        print(f"\n  drift-free dataset: {len(stable_table):,} rows, {len(events)} evaluation(s)")
        print(f"    firings of any kind: {len(fired)}  (seasonal degradation, expected)")
        print(f"    firings that would act: {len(qualifying)}")
        print(f"    candidates raised: {len(run['candidates'])}")

    assert events, "the stable replay evaluated nothing"
    assert not qualifying, (
        f"{len(qualifying)} firing(s) on a drift-free stream would have retrained"
    )
    assert not run["candidates"]


def test_prediction_drift_is_the_family_that_discriminates(stable_table, replayed, capsys):
    """The phase's main finding, asserted rather than only documented.

    Feature and performance drift cannot tell a changed city from a model out of
    season. Prediction drift can. If a future change made feature drift the
    gate, this test is what would catch it.
    """
    stable = _replay(stable_table)

    def counts(events) -> dict[DriftKind, int]:
        return {
            kind: sum(1 for event in events for signal in event.fired if signal.kind is kind)
            for kind in DriftKind
        }

    drifted_counts = counts(replayed["events"])
    stable_counts = counts(stable["events"])

    with capsys.disabled():
        print(f"\n  {'family':<20}{'drifted':>10}{'drift-free':>13}")
        for kind in DriftKind:
            print(f"  {kind.value:<20}{drifted_counts[kind]:>10}{stable_counts[kind]:>13}")

    # Prediction drift is silent on the stream with nothing planted in it.
    assert stable_counts[DriftKind.PREDICTION] == 0
    assert drifted_counts[DriftKind.PREDICTION] > 0

    # The other two fire on both, which is why they cannot be the gate.
    assert stable_counts[DriftKind.FEATURE] > 0
    assert stable_counts[DriftKind.PERFORMANCE] > 0


# ---------------------------------------------------------------------------
# Inconclusive is an outcome, not silence
# ---------------------------------------------------------------------------


def test_a_thin_reference_reports_insufficient_rather_than_quiet(table):
    """A three-sigma gate from two numbers is not a gate."""
    earliest = min(table.sample_times)
    train_end = earliest + timedelta(days=TRAIN_DAYS)
    champion = GradientForecaster().fit(_window(table, earliest, train_end))
    # Two weeks of reference: out of sample, but fewer windows than
    # min_reference_windows, so no gate can be estimated.
    reference = _window(table, train_end, train_end + timedelta(days=14))
    monitor = DriftMonitor(reference=reference, predict=champion.predict, training_end=train_end)
    gates = monitor.calibrate()

    assert not gates.established
    assert np.isnan(gates.prediction_psi)
    assert np.isnan(gates.performance_ratio)
    assert "NOT established" in gates.describe()

    event = monitor.evaluate(table, at=datetime(2026, 4, 1, tzinfo=UTC))
    statuses = {signal.kind: signal.status for signal in event.signals}
    assert statuses[DriftKind.PREDICTION] is DriftStatus.INSUFFICIENT_REFERENCE
    assert statuses[DriftKind.PERFORMANCE] is DriftStatus.INSUFFICIENT_REFERENCE
    assert event.inconclusive


def test_an_in_sample_reference_is_refused(table):
    """The mistake that produced 17 false alarms on a drift-free dataset.

    An overlapping reference is partly in-sample, so the reference error is
    understated and every later window looks drifted. Checked at construction
    because by the time the false alarms appear the cause is far away.
    """
    earliest = min(table.sample_times)
    train_end = earliest + timedelta(days=TRAIN_DAYS)
    champion = GradientForecaster().fit(_window(table, earliest, train_end))
    overlapping = _window(
        table, train_end - timedelta(days=45), train_end + timedelta(days=REFERENCE_DAYS)
    )
    with pytest.raises(ValueError, match="before the training end"):
        DriftMonitor(reference=overlapping, predict=champion.predict, training_end=train_end)


def test_a_sparse_window_is_reported_as_such(table, replayed):
    monitor = replayed["monitor"]
    # An instant long before the data starts has nothing behind it.
    event = monitor.evaluate(table, at=min(table.sample_times) - timedelta(days=30))
    assert not event.is_drift
    assert event.inconclusive
    assert event.signals[0].status is DriftStatus.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# Unmonitorable features are refused structurally
# ---------------------------------------------------------------------------


def test_a_feature_averaged_over_the_window_is_refused(table):
    """The bug that a first attempt at this phase shipped with.

    roll_mean_168h is a one-week mean compared across one-week windows, so it
    is near-constant within any window and its PSI reads 5 or 6 on perfectly
    stable data. The guard is structural, so a config change cannot bring the
    false alarms back.
    """
    config = MonitorConfig(watched_features=("roll_mean_24h", "roll_mean_168h"))
    rejected = dict(config.unmonitorable(table))
    assert "roll_mean_168h" in rejected
    assert "near-constant" in rejected["roll_mean_168h"]
    assert config.monitorable(table) == ("roll_mean_24h",)


def test_a_zero_inflated_covariate_is_refused(table):
    config = MonitorConfig(watched_features=("rainfall_0h",))
    rejected = dict(config.unmonitorable(table))
    assert "rainfall_0h" in rejected
    assert "zero-inflated" in rejected["rainfall_0h"]
    assert config.monitorable(table) == ()


def test_an_unknown_feature_is_reported_not_ignored(table):
    config = MonitorConfig(watched_features=("not_a_feature",))
    rejected = dict(config.unmonitorable(table))
    assert "not_a_feature" in rejected
    assert "not a column" in rejected["not_a_feature"]


# ---------------------------------------------------------------------------
# Episodes and the cooldown
# ---------------------------------------------------------------------------


def test_consecutive_firings_become_one_episode(replayed, capsys):
    """Episodes count the firings that could act, not every firing.

    Feature and performance drift fire a week earlier here, and those events
    are logged -- but an episode begins when something qualifies to raise a
    candidate, which is what an episode is for.
    """
    monitor = replayed["monitor"]
    episodes = monitor.tracker.episodes
    firings = sum(1 for event in replayed["events"] if event.is_drift)
    qualifying = sum(1 for event in replayed["events"] if event.qualifies(REQUIRED))

    with capsys.disabled():
        print(
            f"\n  {firings} firing(s), {qualifying} qualifying, collapsed into "
            f"{len(episodes)} episode(s):"
        )
        for episode in episodes:
            print(f"    {episode.describe()}")

    assert len(episodes) == 1, "one continuous shift should be one episode"
    assert episodes[0].events == qualifying
    assert len(replayed["candidates"]) < qualifying, (
        "the cooldown must suppress repeat candidates within an episode"
    )


def test_the_cooldown_spaces_candidates_out(replayed):
    candidates = replayed["candidates"]
    assert len(candidates) >= 2
    gaps = [later - earlier for earlier, later in pairwise(candidates)]
    assert all(gap >= timedelta(days=28) for gap in gaps), f"gaps were {gaps}"


def test_a_quiet_evaluation_closes_an_open_episode():
    """So a second shift later is a second episode, not a continuation."""
    tracker = EpisodeTracker(cooldown=timedelta(days=28))
    moment = datetime(2026, 1, 1, tzinfo=UTC)

    firing = _event(moment, fired=True)
    _, raised = tracker.observe(firing)
    assert raised
    assert tracker.open_episode is not None

    quiet = _event(moment + timedelta(days=7), fired=False)
    tracker.observe(quiet)
    assert tracker.open_episode is None
    assert tracker.episodes[0].closed_at == quiet.detected_at

    again = _event(moment + timedelta(days=14), fired=True)
    _, raised_again = tracker.observe(again)
    assert raised_again, "a new episode must be able to raise a candidate immediately"
    assert len(tracker.episodes) == 2


def _event(moment: datetime, *, fired: bool) -> DriftEvent:
    """A hand-built event, firing or not.

    Built from a prediction-drift signal because that is the family allowed to
    raise a candidate; these tests exercise the episode and cooldown machinery,
    not the family gating, which has its own test above.
    """
    from urbansense.drift.detectors import DriftSignal

    signal = DriftSignal(
        kind=DriftKind.PREDICTION,
        subject="prediction",
        status=DriftStatus.DRIFTED if fired else DriftStatus.STABLE,
        statistic=2.5 if fired else 1.0,
        threshold=1.8,
        reference_rows=1000,
        recent_rows=600,
    )
    return DriftEvent(
        detected_at=moment,
        window_start=moment - timedelta(days=7),
        window_end=moment,
        signals=(signal,),
    )


def test_an_episode_within_its_cooldown_raises_no_candidate():
    tracker = EpisodeTracker(cooldown=timedelta(days=28))
    moment = datetime(2026, 1, 1, tzinfo=UTC)
    raised = []
    for week in range(6):
        _, flag = tracker.observe(_event(moment + timedelta(days=7 * week), fired=True))
        raised.append(flag)
    assert raised[0] is True
    assert raised[1:4] == [False, False, False]
    assert raised[4] is True, "the cooldown should elapse after 28 days"
    assert tracker.episodes[0].candidates_raised == 2


# ---------------------------------------------------------------------------
# The store is append-only
# ---------------------------------------------------------------------------


def test_the_drift_store_appends(tmp_path, replayed):
    store = DriftStore(tmp_path)
    first = store.append_events(replayed["events"][:3])
    bytes_after_first = store.events_path.read_bytes()
    second = store.append_events(replayed["events"][3:6])

    assert first == 3 and second == 3
    assert store.events_path.read_bytes().startswith(bytes_after_first)
    assert len(store.read_events()) == 6

    store.append_episodes(replayed["monitor"].tracker.episodes)
    episodes = store.read_episodes()
    assert episodes
    assert "episode_id" in episodes[0]


def test_an_event_record_carries_every_spec_29_field(replayed):
    event = next(item for item in replayed["events"] if item.is_drift)
    payload = event.as_dict()
    for field in ("detected_at", "kinds", "magnitude", "episode_id", "signals"):
        assert field in payload, f"SPEC section 29 requires {field}"
    assert payload["is_drift"]
    signal = payload["signals"][0]
    for field in ("kind", "subject", "status", "statistic", "threshold"):
        assert field in signal


def test_appending_nothing_creates_nothing(tmp_path):
    store = DriftStore(tmp_path / "empty")
    assert store.append_events([]) == 0
    assert not store.events_path.exists()


# ---------------------------------------------------------------------------
# The statistics themselves
# ---------------------------------------------------------------------------


def test_psi_is_near_zero_for_the_same_distribution():
    generator = np.random.default_rng(11)
    reference = generator.normal(1000, 100, 4000)
    recent = generator.normal(1000, 100, 700)
    assert psi(reference, recent) < 0.05


def test_psi_grows_with_the_size_of_a_shift():
    generator = np.random.default_rng(11)
    reference = generator.normal(1000, 100, 4000)
    values = [
        psi(reference, generator.normal(1000 * factor, 100, 700))
        for factor in (1.0, 1.05, 1.18, 1.5)
    ]
    assert values == sorted(values)
    assert values[2] > PSI_MAJOR, "the planted x1.18 shape must be detectable"


def test_psi_bins_come_from_the_reference_not_the_window():
    """Otherwise the statistic adapts to the shift it exists to notice.

    A binning recomputed on each recent window would place its own quantile
    edges and report near-zero for any shift.
    """
    generator = np.random.default_rng(3)
    reference = generator.normal(1000, 100, 4000)
    shifted = generator.normal(1500, 100, 700)
    # Self-comparison of the shifted window is near zero; against the
    # reference it is large. If edges came from the window, both would be small.
    assert psi(shifted, shifted) < 0.01
    assert psi(reference, shifted) > 1.0


def test_psi_needs_enough_data():
    generator = np.random.default_rng(5)
    assert np.isnan(psi(generator.normal(0, 1, 10), generator.normal(0, 1, 500)))


def test_psi_refuses_a_degenerate_reference():
    """A reference with too few distinct quantiles cannot be binned.

    A constant collapses every interior edge onto one value and a binary
    feature onto two, so most bins are empty and the sum of logs reflects the
    epsilon floor rather than the data. An earlier version of the guard only
    rejected the fully constant case and let a binary feature through.
    """
    assert np.isnan(psi(np.full(500, 7.0), np.linspace(0, 10, 500)))
    binary = np.concatenate([np.zeros(250), np.ones(250)])
    assert np.isnan(psi(binary, np.ones(500)))


def test_the_ks_effect_size_is_the_gate_not_the_p_value(capsys):
    """On a large sample a trivial difference is still 'significant'."""
    generator = np.random.default_rng(17)
    reference = generator.normal(1000, 100, 20000)
    barely = generator.normal(1004, 100, 20000)
    result = ks_test(reference, barely)
    with capsys.disabled():
        print(
            f"\n  a 0.4% mean shift on 20,000 rows: D={result.statistic:.4f}, "
            f"p={result.p_value:.2e}"
        )
    assert result.tested
    assert result.statistic < 0.15, "the effect size should call this immaterial"
    assert "the D statistic is the gate" in str(result.as_dict()["note"])


def test_feature_drift_reports_both_statistics():
    generator = np.random.default_rng(23)
    reference = generator.normal(1000, 100, 3000)
    shifted = generator.normal(1180, 100, 700)
    signal = feature_drift("roll_mean_24h", reference, shifted)
    assert signal.fired
    assert signal.kind is DriftKind.FEATURE
    assert signal.detail is not None
    assert signal.detail["ks"]["tested"]
    assert signal.detail["ks_agrees"]
    assert signal.magnitude > 1.0


def test_performance_drift_needs_an_out_of_sample_reference():
    """The mistake that would make this detector useless."""
    signal = performance_drift(
        300.0, reference_error=float("nan"), gate_ratio=1.8, recent_rows=600, reference_rows=4000
    )
    assert signal.status is DriftStatus.INSUFFICIENT_REFERENCE
    assert not signal.fired
    assert signal.detail is not None
    assert "no usable reference" in str(signal.detail["reason"])


def test_performance_drift_fires_on_a_worse_error_only():
    better = performance_drift(
        100.0, reference_error=180.0, gate_ratio=1.8, recent_rows=600, reference_rows=4000
    )
    worse = performance_drift(
        400.0, reference_error=180.0, gate_ratio=1.8, recent_rows=600, reference_rows=4000
    )
    assert better.status is DriftStatus.STABLE
    assert worse.status is DriftStatus.DRIFTED
    assert worse.detail is not None
    assert worse.detail["reference_is_out_of_sample"]


def test_prediction_drift_uses_the_supplied_gate():
    generator = np.random.default_rng(29)
    reference = generator.normal(3000, 400, 3000)
    recent = generator.normal(3150, 400, 700)
    index = psi(reference, recent)
    loose = prediction_drift(reference, recent, gate=index * 2)
    tight = prediction_drift(reference, recent, gate=index * 0.5)
    assert not loose.fired
    assert tight.fired
    assert tight.detail is not None
    assert "reference-period variability" in str(tight.detail["gate_source"])


def test_sigma_gate_is_calibrated_on_the_reference_only():
    """The measured stable values must produce the documented gate."""
    stable_ratios = [1.67, 1.68, 1.61, 1.76, 1.62, 1.67]
    gate = sigma_gate(stable_ratios)
    assert gate == pytest.approx(1.8287, abs=1e-4)
    assert gate > max(stable_ratios), "the gate must sit above the stable maximum"


def test_sigma_gate_refuses_a_thin_reference():
    assert np.isnan(sigma_gate([1.6, 1.7, 1.65]))
    assert not np.isnan(sigma_gate([1.6, 1.7, 1.65, 1.68]))


def test_sigma_gate_respects_its_floor():
    """A gate below 1.0 would fire on a window better than the reference."""
    assert sigma_gate([0.5, 0.5, 0.5, 0.5], floor=1.0) == 1.0
