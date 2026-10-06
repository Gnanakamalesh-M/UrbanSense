"""Tests for pattern discovery, evolution and interaction.

Scored against `ground_truth.json`, which only tests read. The point of this
phase is that discovery *finds* the planted effects without being told where to
look, so every headline test checks both halves: the effect is recovered at
roughly the planted size, **and** nothing is reported where nothing was planted.

False discoveries matter as much as discoveries here. A scan over 672 cells that
reported ten patterns would be worthless even if one of them were the real
Friday effect, so the spurious-discovery test prints the test count and the FDR
level alongside the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urbansense.config import load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.ingestion import load_historical
from urbansense.patterns import (
    ObservationFrame,
    PatternKind,
    PatternStatus,
    PatternStore,
    benjamini_hochberg,
    discover_patterns,
    fit_rainfall_effect,
    median_ratio,
)

# Aliased: pytest would collect anything named test_* at module level as a test.
from urbansense.patterns import test_trend as trend_test
from urbansense.patterns.evolution import classify
from urbansense.patterns.models import InteractionVerdict, WindowStrength
from urbansense.preprocessing import load_location_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = REPO_ROOT / "data" / "sample" / "traffic_history.csv"
GROUND_TRUTH = REPO_ROOT / "data" / "sample" / "ground_truth.json"
TZ_OFFSET = 5.5

#: Tolerance on the recovered Friday effect against the planted 2.2. Generous
#: because the recovery runs through a median ratio and a control adjustment, each
#: of which carries sampling noise.
FRIDAY_TOLERANCE = 0.25

#: Tolerance on the recovered rain slope against the planted 0.03.
RAIN_TOLERANCE = 0.30

#: Tolerance on the recovered drift against the planted 1.18.
DRIFT_TOLERANCE = 0.20


@pytest.fixture(scope="module")
def truth() -> dict:
    """The planted answers. Only tests read this."""
    return json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def settings():
    return load_settings()


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
def discovered(reconciled):
    return discover_patterns(reconciled, timezone_offset_hours=TZ_OFFSET)


# ---------------------------------------------------------------------------
# The Friday ZONE-B pattern, found rather than looked up
# ---------------------------------------------------------------------------


def test_the_friday_zone_b_pattern_is_discovered(discovered, truth, capsys):
    """Found from data alone, at roughly the planted strength."""
    planted = truth["patterns"]["friday_evening_congestion"]
    found = discovered.of_kind(PatternKind.SPATIOTEMPORAL)

    with capsys.disabled():
        print(
            f"\n  spatio-temporal scan: {discovered.stats.tests_run} hypotheses, "
            f"FDR q={discovered.stats.fdr_level}, {len(found)} discovered"
        )
        for pattern in found:
            print(
                f"    {pattern.scope.describe()}: "
                f"{pattern.evidence.effect.describe()} "
                f"(planted x{planted['volume_multiplier']})"
            )

    assert len(found) == 1, f"expected one pattern, found {[p.pattern_id for p in found]}"
    pattern = found[0]
    assert pattern.scope.location_id == planted["location_id"]
    assert pattern.scope.weekday == planted["weekday"]
    assert set(pattern.scope.hours) == set(range(planted["start_hour"], planted["end_hour"]))

    expected = planted["volume_multiplier"]
    assert pattern.strength == pytest.approx(expected, rel=FRIDAY_TOLERANCE), (
        f"recovered x{pattern.strength:.3f} against planted x{expected}"
    )
    assert pattern.is_significant
    assert pattern.evidence.effect.lower > 1.0


def test_the_friday_pattern_records_what_it_controlled(discovered):
    """A claim without its controls cannot be judged."""
    pattern = discovered.of_kind(PatternKind.SPATIOTEMPORAL)[0]
    controls = " ".join(pattern.evidence.controls)
    assert "hour of day" in controls
    assert "zone level" in controls
    assert "weekday" in controls
    # And what was *not* controlled is stated too.
    assert any("rainfall is not controlled" in note for note in pattern.evidence.notes)


def test_the_raw_lift_is_larger_than_the_reported_one(discovered):
    """The control adjustment is doing real work.

    Friday is busier than Tuesday in every zone, so the within-zone lift
    overstates what is specific to ZONE-B. If the reported figure equalled the
    raw one, the adjustment would be a no-op.
    """
    pattern = discovered.of_kind(PatternKind.SPATIOTEMPORAL)[0]
    note = next(n for n in pattern.evidence.notes if "raw within-zone lift" in n)
    raw = float(note.split("x")[1].split(";")[0])
    assert raw > pattern.strength
    assert raw / pattern.strength > 1.05


def test_the_scan_adjusts_p_values_for_multiple_testing(discovered):
    pattern = discovered.of_kind(PatternKind.SPATIOTEMPORAL)[0]
    assert pattern.evidence.adjusted_p_value is not None
    assert pattern.evidence.adjusted_p_value >= pattern.evidence.p_value


# ---------------------------------------------------------------------------
# No spurious discoveries where nothing was planted
# ---------------------------------------------------------------------------


def test_no_spurious_pattern_in_the_unplanted_zones(discovered, truth, capsys):
    """ZONE-C and ZONE-D had no spatio-temporal effect planted.

    This matters more than finding the real one: a scan that reported ten
    patterns would be worthless even with the true one among them.
    """
    planted_zone = truth["patterns"]["friday_evening_congestion"]["location_id"]
    control_zones = truth["patterns"]["friday_evening_congestion"]["control_location_ids"]
    found = discovered.of_kind(PatternKind.SPATIOTEMPORAL)
    spurious = [p for p in found if p.scope.location_id != planted_zone]

    rate = len(spurious) / len(found) if found else 0.0
    with capsys.disabled():
        print(
            f"\n  false-discovery report:"
            f"\n    hypotheses tested      {discovered.stats.tests_run}"
            f"\n    FDR level              q={discovered.stats.fdr_level}"
            f"\n    expected by chance     ~{discovered.stats.tests_run * 0.05:.0f}"
            " at an uncorrected p<0.05"
            f"\n    patterns reported      {len(found)}"
            f"\n    unplanted zones        {', '.join(control_zones)}"
            f"\n    spurious discoveries   {len(spurious)}"
            f"\n    realized FDR           {rate:.3f}"
        )

    assert not spurious, (
        f"spurious spatio-temporal patterns: "
        f"{[(p.scope.location_id, p.scope.weekday, p.scope.hours) for p in spurious]}"
    )


def test_no_level_shift_in_the_unplanted_zones(discovered, truth):
    """ZONE-C and ZONE-D did not drift, so nothing should be reported there."""
    drift = truth["anomalies"]["baseline_drift"]
    shifts = discovered.of_kind(PatternKind.LEVEL_SHIFT)
    reported = {p.scope.location_id for p in shifts}
    for zone in drift["control_location_ids"]:
        assert zone not in reported, f"a level shift was reported for {zone}"


def test_the_benjamini_hochberg_gate_rejects_pure_noise():
    """672 uniform p-values must yield no discoveries.

    The guard against the scan's own machinery: if the correction let noise
    through, every finding above would be suspect.
    """
    import random

    generator = random.Random(7)
    noise = [generator.random() for _ in range(672)]
    assert not any(benjamini_hochberg(noise, level=0.05))


# ---------------------------------------------------------------------------
# The rain effect
# ---------------------------------------------------------------------------


def test_the_rain_effect_is_discovered(discovered, truth, capsys):
    planted = truth["patterns"]["rain_traffic_effect"]
    found = discovered.of_kind(PatternKind.CONDITIONAL)

    with capsys.disabled():
        for pattern in found:
            print(
                f"\n  rain effect: {pattern.evidence.effect.describe()} per mm "
                f"(planted {planted['volume_coefficient']:+.4f})"
            )

    assert len(found) == 1
    pattern = found[0]
    expected = planted["volume_coefficient"]
    assert pattern.strength > 0, "the planted effect is positive"
    assert pattern.strength == pytest.approx(expected, rel=RAIN_TOLERANCE)
    assert pattern.evidence.effect.excludes_null
    assert pattern.evidence.effect.lower > 0


def test_the_rain_baseline_excludes_rain(discovered):
    """Normalizing by a baseline that already contains rain biases the slope."""
    pattern = discovered.of_kind(PatternKind.CONDITIONAL)[0]
    controls = " ".join(pattern.evidence.controls)
    assert "dry hours only" in controls
    assert "saturation" in controls


def test_the_heavy_rain_fit_is_reported_alongside(discovered):
    """Saturation is visible rather than hidden by the pre-committed cut."""
    pattern = discovered.of_kind(PatternKind.CONDITIONAL)[0]
    assert any("slope over all rain" in note for note in pattern.evidence.notes)


def test_the_rain_slope_is_robust_to_the_light_rain_cut(reconciled):
    """The pre-committed cut is not doing the work.

    If the light-rain and all-rain estimates disagreed wildly, the headline
    number would be an artifact of where the cut was placed.
    """
    frame = ObservationFrame.from_result(reconciled, timezone_offset_hours=TZ_OFFSET)
    fit = fit_rainfall_effect(frame, "traffic_volume")
    assert fit.light.tested and fit.full.tested
    assert fit.light.effect.value == pytest.approx(fit.full.effect.value, rel=0.25)


# ---------------------------------------------------------------------------
# The drift, as a level shift
# ---------------------------------------------------------------------------


def test_the_drift_is_discovered_as_a_level_shift(discovered, truth, capsys):
    planted = truth["anomalies"]["baseline_drift"]
    shifts = discovered.of_kind(PatternKind.LEVEL_SHIFT)

    with capsys.disabled():
        print(
            f"\n  level shifts (planted x{planted['volume_multiplier']} in "
            f"{', '.join(planted['location_ids'])} from {planted['start_date']}):"
        )
        for pattern in shifts:
            print(
                f"    {pattern.scope.location_id}: "
                f"{pattern.evidence.effect.describe()} at "
                f"{pattern.pattern_id.rsplit('-', 2)[-2:]} -> {pattern.status.value}"
            )

    reported = {p.scope.location_id for p in shifts}
    for zone in planted["location_ids"]:
        assert zone in reported, f"no level shift discovered in {zone}"

    expected = planted["volume_multiplier"]
    for pattern in shifts:
        if pattern.scope.location_id not in planted["location_ids"]:
            continue
        assert pattern.strength == pytest.approx(expected, rel=DRIFT_TOLERANCE)
        assert pattern.kind is PatternKind.LEVEL_SHIFT


def test_the_change_point_is_near_the_planted_date(discovered, truth):
    planted_month = truth["anomalies"]["baseline_drift"]["start_date"][:7]
    for pattern in discovered.of_kind(PatternKind.LEVEL_SHIFT):
        month = "-".join(pattern.pattern_id.rsplit("-", 2)[-2:])
        year, number = (int(part) for part in month.split("-"))
        planted_year, planted_number = (int(part) for part in planted_month.split("-"))
        months_apart = abs((year - planted_year) * 12 + number - planted_number)
        assert months_apart <= 1, (
            f"{pattern.scope.location_id} change-point {month} is more than a "
            f"month from the planted {planted_month}"
        )


def test_at_least_one_drifted_zone_is_marked_strengthening(discovered, truth):
    """SPEC section 22's status for a pattern that grows.

    Only one of the two drifted zones reaches STRENGTHENING, because the
    per-window level measure compares a zone against the others -- and the other
    drifted zone rose too, which dilutes the apparent trend. The level-shift
    *kind* carries the finding either way; the status is the weaker signal.
    """
    planted = truth["anomalies"]["baseline_drift"]["location_ids"]
    statuses = {p.scope.location_id: p.status for p in discovered.of_kind(PatternKind.LEVEL_SHIFT)}
    assert any(statuses.get(zone) is PatternStatus.STRENGTHENING for zone in planted)


def test_the_level_shift_states_its_identifying_assumption(discovered):
    """Difference-in-differences needs an unaffected reference, and says so."""
    pattern = discovered.of_kind(PatternKind.LEVEL_SHIFT)[0]
    notes = " ".join(pattern.evidence.notes)
    assert "assumes at least some zones are unaffected" in notes
    assert "seasonality" in " ".join(pattern.evidence.controls)


# ---------------------------------------------------------------------------
# Interaction (SPEC section 23)
# ---------------------------------------------------------------------------


def test_the_friday_rain_interaction_is_tested_and_not_overclaimed(discovered, capsys):
    """The planted data has no interaction on the log scale.

    The generator composes effects multiplicatively, so a SUPPORTED verdict here
    would be a spurious finding -- and with only sixteen wet Friday evenings, a
    point-estimate test would produce one.
    """
    assert discovered.interactions, "the interaction was not tested at all"

    with capsys.disabled():
        for item in discovered.interactions:
            print()
            print(f"  {item.describe()}")

    for item in discovered.interactions:
        assert item.verdict is InteractionVerdict.NO_EVIDENCE, (
            f"{item.interaction_id} returned {item.verdict.value}; the planted "
            "data composes multiplicatively, so there is no interaction to find "
            "on the log scale"
        )
        assert item.interaction_effect is not None
        assert not item.interaction_effect.excludes_null
        assert item.interaction_effect.lower <= 0.0 <= item.interaction_effect.upper


def test_the_interaction_names_its_scale(discovered):
    """A claim of no interaction is meaningless until the scale is stated."""
    for item in discovered.interactions:
        assert "log" in item.scale
        assert "multiplicatively" in item.scale


def test_the_interaction_reports_its_power(discovered):
    """A null result from sixteen observations is not evidence of absence."""
    for item in discovered.interactions:
        assert item.min_cell_count > 0
        assert set(item.cell_counts) == {
            "both",
            "first_only",
            "second_only",
            "neither",
        }
        notes = " ".join(item.notes)
        assert "bounds the power" in notes
        assert "absence of evidence" in notes


def test_no_interaction_is_reported_as_supported(discovered):
    assert discovered.supported_interactions == ()


# ---------------------------------------------------------------------------
# Evolution (SPEC section 22)
# ---------------------------------------------------------------------------


def test_tested_patterns_carry_a_strength_series(discovered):
    """Evolution is measurable only if strength is tracked per window."""
    tracked = [
        p
        for p in discovered.patterns
        if p.kind
        in (
            PatternKind.SPATIOTEMPORAL,
            PatternKind.CONDITIONAL,
            PatternKind.LEVEL_SHIFT,
        )
    ]
    assert tracked
    for pattern in tracked:
        assert len(pattern.strength_by_window) >= 6
        assert pattern.historical_strength is not None
        assert pattern.recent_strength is not None
        assert pattern.status_reason


def test_unmeasurable_windows_are_recorded_not_filled():
    """Interpolating a gap would manufacture the trend being looked for."""
    windows = (
        WindowStrength(window="2026-01", strength=2.0, n_treatment=20, n_control=50, measured=True),
        WindowStrength(window="2026-02", strength=None, n_treatment=1, n_control=3, measured=False),
        WindowStrength(window="2026-03", strength=2.1, n_treatment=20, n_control=50, measured=True),
        WindowStrength(window="2026-04", strength=2.0, n_treatment=20, n_control=50, measured=True),
    )
    summary = classify(windows)
    assert summary.windows[1].strength is None
    assert summary.windows[1].measured is False


def test_a_reversing_series_is_conflicting():
    """SPEC section 22's CONFLICTING: evidence that disagrees with itself."""
    windows = tuple(
        WindowStrength(
            window=f"2026-{month:02d}",
            strength=strength,
            n_treatment=20,
            n_control=50,
            measured=True,
        )
        for month, strength in enumerate([1.5, 0.6, 1.4, 0.7, 1.6, 0.5], start=1)
    )
    assert classify(windows).status is PatternStatus.CONFLICTING


def test_a_rising_series_is_strengthening():
    windows = tuple(
        WindowStrength(
            window=f"2026-{month:02d}",
            strength=strength,
            n_treatment=20,
            n_control=50,
            measured=True,
        )
        for month, strength in enumerate([1.1, 1.2, 1.3, 1.4, 1.6, 1.8], start=1)
    )
    summary = classify(windows)
    assert summary.status is PatternStatus.STRENGTHENING
    assert "upward trend" in summary.reason


def test_a_flat_series_is_stable_even_when_the_last_window_jumps():
    """A status that flipped on one noisy window would carry no information."""
    windows = tuple(
        WindowStrength(
            window=f"2026-{month:02d}",
            strength=strength,
            n_treatment=20,
            n_control=50,
            measured=True,
        )
        for month, strength in enumerate([2.0, 2.1, 1.9, 2.0, 2.1, 2.3], start=1)
    )
    assert classify(windows).status is PatternStatus.STABLE


def test_a_short_series_is_insufficient_history():
    windows = (
        WindowStrength(window="2026-01", strength=2.0, n_treatment=20, n_control=50, measured=True),
        WindowStrength(window="2026-02", strength=2.1, n_treatment=20, n_control=50, measured=True),
    )
    summary = classify(windows)
    assert summary.status is PatternStatus.INSUFFICIENT_HISTORY
    assert "at least three" in summary.reason


def test_descriptive_profiles_say_why_they_are_not_tracked(discovered):
    """Not "insufficient history", which would imply more data would help."""
    profiles = discovered.of_kind(PatternKind.HOURLY_PROFILE)
    assert profiles
    for pattern in profiles:
        assert "not tracked" in pattern.status_reason


# ---------------------------------------------------------------------------
# Profiles and hotspots are emitted as context
# ---------------------------------------------------------------------------


def test_profiles_and_hotspots_are_discovered(discovered):
    """Context the other findings are measured against (SPEC section 21)."""
    assert discovered.of_kind(PatternKind.HOURLY_PROFILE)
    assert discovered.of_kind(PatternKind.WEEKDAY_PROFILE)
    assert discovered.of_kind(PatternKind.HOTSPOT)

    hourly = discovered.of_kind(PatternKind.HOURLY_PROFILE)[0]
    assert hourly.strength > 2.0, "a diurnal cycle should be a large effect"


def test_descriptive_profiles_are_excluded_from_the_discovery_count(discovered):
    """They were never part of the multiple-testing family.

    Folding them into the count would overstate what survived correction.
    """
    assert discovered.stats.discoveries < len(discovered.patterns)
    assert discovered.stats.discoveries == 4


def test_a_hotspot_excludes_itself_from_its_baseline(discovered):
    pattern = discovered.of_kind(PatternKind.HOTSPOT)[0]
    assert "excluded from its own baseline" in " ".join(pattern.evidence.controls)


# ---------------------------------------------------------------------------
# Determinism, provenance of the run, and the store
# ---------------------------------------------------------------------------


def test_discovery_is_deterministic(reconciled):
    """Evolution cannot be tracked across runs that disagree with each other."""
    first = discover_patterns(reconciled, timezone_offset_hours=TZ_OFFSET)
    second = discover_patterns(reconciled, timezone_offset_hours=TZ_OFFSET)

    assert [p.pattern_id for p in first.patterns] == [p.pattern_id for p in second.patterns]
    assert [p.status for p in first.patterns] == [p.status for p in second.patterns]
    for left, right in zip(first.patterns, second.patterns, strict=True):
        assert left.strength == pytest.approx(right.strength, rel=1e-12)
        assert left.evidence.effect.lower == pytest.approx(
            right.evidence.effect.lower, rel=1e-12, nan_ok=True
        )
    assert [i.verdict for i in first.interactions] == [i.verdict for i in second.interactions]


def test_the_bootstrap_is_seeded(discovered):
    assert discovered.stats.bootstrap_seed != 0
    left = median_ratio([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 2.0, 2.0], seed=5)
    right = median_ratio([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 2.0, 2.0], seed=5)
    assert left.lower == right.lower and left.upper == right.upper


def test_discovery_reads_only_the_unified_layer(discovered, reconciled):
    """The duplicate-linked and conflicted buckets are unreachable from here."""
    assert discovered.stats.source_rows == len(reconciled.unified)
    assert reconciled.duplicate_linked
    assert reconciled.conflicted_observations


def test_unknown_readings_are_dropped_not_imputed(discovered, reconciled):
    assert discovered.frame.dropped_unknown > 0
    assert len(discovered.frame) == (
        discovered.stats.source_rows - discovered.frame.dropped_unknown
    )


def test_the_run_header_states_the_multiple_testing_context(discovered):
    """One pattern out of 672 tests is a different claim from one out of ten."""
    header = discovered.header()
    assert str(discovered.stats.tests_run) in header
    assert "FDR" in header
    assert "seed" in header


def test_the_store_writes_patterns_with_the_run_header(discovered, tmp_path):
    store = PatternStore(tmp_path)
    target = store.write(discovered)
    assert target is not None and target.exists()

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["run"]["tests_run"] == discovered.stats.tests_run
    assert payload["run"]["fdr_level"] == discovered.stats.fdr_level
    assert payload["run"]["bootstrap_seed"] == discovered.stats.bootstrap_seed
    assert len(payload["patterns"]) == len(discovered.patterns)
    assert payload["interactions"]
    assert payload["notes"]

    first = payload["patterns"][0]
    for field in (
        "pattern_id",
        "kind",
        "scope",
        "first_detected",
        "last_observed",
        "frequency_per_week",
        "confidence",
        "historical_strength",
        "recent_strength",
        "status",
        "strength_by_window",
        "evidence",
    ):
        assert field in first, f"pattern record is missing {field}"


def test_each_run_writes_a_new_report(discovered, tmp_path):
    from datetime import UTC, datetime

    store = PatternStore(tmp_path)
    first = store.write(discovered, now=datetime(2026, 10, 2, 9, 0, tzinfo=UTC))
    second = store.write(discovered, now=datetime(2026, 10, 2, 10, 0, tzinfo=UTC))
    assert first != second
    assert len(store.reports()) == 2


def test_the_trend_test_needs_enough_windows():
    assert not trend_test([1.0, 2.0]).tested
    assert trend_test([1.0, 1.1, 1.2, 1.4, 1.6]).direction == "up"
