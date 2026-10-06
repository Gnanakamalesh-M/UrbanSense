"""Tests for anomaly detection and candidate explanations (SPEC section 24).

Two halves, and the second matters as much as the first. A detector that flags
everything finds every planted defect, so these tests check both that the
planted anomalies are caught **and** that the recurring Friday peak -- which is
large, real and entirely expected -- is left alone.

The population being scanned is stated explicitly, because it is the subtlest
thing here: zero planted point anomalies survive into the unified layer. Phase 2
quarantines all of them on range and frozen-run checks before discovery runs,
which is correct pipeline behaviour but makes the unified layer the wrong place
to score the statistical detector. Scoring therefore runs on the pre-quarantine
population; discovery itself still reads the unified layer only.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from urbansense.anomaly import (
    ExplanationKind,
    detect_anomalies,
    explain,
    pre_quarantine_population,
)
from urbansense.config import load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.ingestion import load_historical
from urbansense.patterns import PatternKind, discover_patterns
from urbansense.preprocessing import load_location_registry
from urbansense.schemas.enums import SourceStatus, ValidationStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = REPO_ROOT / "data" / "sample" / "traffic_history.csv"
GROUND_TRUTH = REPO_ROOT / "data" / "sample" / "ground_truth.json"
TZ_OFFSET = 5.5
VOLUME = "traffic_volume"

#: Floor on frozen-reading recall. Not 1.0 on purpose: the sensor froze at 316
#: vehicles/hour, which is a *plausible* overnight figure, so the overnight
#: readings in the run are genuinely indistinguishable from real data on value
#: alone. Phase 2's frozen-run detector catches those by repetition; a
#: distributional detector should not be expected to.
MIN_FROZEN_RECALL = 0.50


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
def held(validated):
    """The observations inside quarantine records, where parsing succeeded."""
    return tuple(
        record.observation for record in validated.quarantined if record.observation is not None
    )


@pytest.fixture(scope="module")
def population(validated, held):
    """The population Phase 2 saw, where the planted defects still exist."""
    return pre_quarantine_population(validated.valid, held)


@pytest.fixture(scope="module")
def patterns(validated, settings):
    reconciled = reconcile(validated, settings=settings)
    return discover_patterns(reconciled, timezone_offset_hours=TZ_OFFSET).patterns


@pytest.fixture(scope="module")
def result(population, patterns, validated):
    return detect_anomalies(
        population,
        patterns=patterns,
        source_health=validated.source_health,
        timezone_offset_hours=TZ_OFFSET,
        metric=VOLUME,
    )


def _record_ids(observations) -> dict[str, object]:
    """Record id to observation, for scoring against ground truth."""
    return {
        str(obs.attributes.get("record_id")): obs
        for obs in observations
        if obs.attributes.get("record_id") is not None
    }


# ---------------------------------------------------------------------------
# The planted defects are found
# ---------------------------------------------------------------------------


def test_the_planted_negative_volumes_are_flagged(result, truth, population, capsys):
    planted = {
        item["record_id"]
        for item in truth["planted_records"]["invalid"]
        if item["kind"] == "negative_value"
    }
    present = planted & set(_record_ids(population))
    assert present, "the planted negatives are not in the scanned population"

    found = present & result.record_ids
    with capsys.disabled():
        print(
            f"\n  negative volumes: {len(found)}/{len(present)} flagged "
            f"(population {result.observations_scanned:,} readings)"
        )
    assert found == present, f"missed {sorted(present - found)}"


def test_the_frozen_sensor_window_is_partly_flagged(result, truth, population, capsys):
    """A floor, not perfection -- and the reason is recorded.

    316 vehicles/hour is implausible at 09:00 and unremarkable at 03:00, so a
    detector working from the value distribution alone cannot catch the whole
    run. The readings it misses are the overnight ones. Phase 2's frozen-run
    detector is what catches those, by repetition rather than by magnitude.
    """
    frozen = truth["anomalies"]["sensor_failure"]
    planted = set(frozen["affected_record_ids"]) & set(_record_ids(population))
    assert planted

    found = planted & result.record_ids
    recall = len(found) / len(planted)
    with capsys.disabled():
        print(
            f"\n  frozen window at {frozen['frozen_value']:,.0f} in "
            f"{frozen['location_id']}: {len(found)}/{len(planted)} flagged "
            f"(recall {recall:.2f}); the misses are overnight hours where that "
            "value is plausible"
        )
    assert recall >= MIN_FROZEN_RECALL


def test_a_hand_built_spike_is_detected_with_an_explanation(population, patterns):
    """An injected spike, so detection is exercised on a value we chose.

    The planted defects all arrive via Phase 1, so without this the detector is
    only ever tested against one generator's idea of an anomaly.
    """
    victim = next(
        obs
        for obs in population
        if obs.metric == VOLUME
        and obs.value is not None
        and obs.validation_status is ValidationStatus.VALID
    )
    spike = victim.model_copy(update={"value": victim.value * 12.0})
    injected = [spike if obs is victim else obs for obs in population]

    detected = detect_anomalies(
        injected, patterns=patterns, timezone_offset_hours=TZ_OFFSET, metric=VOLUME
    )
    flagged = {item.observation_id: item for item in detected.anomalies}
    assert spike.observation_id in flagged

    anomaly = flagged[spike.observation_id]
    assert anomaly.explanations, "an anomaly with no candidates at all"
    assert anomaly.value == pytest.approx(victim.value * 12.0)
    assert anomaly.deviation > detected.threshold
    assert anomaly.group_size > 0


# ---------------------------------------------------------------------------
# The recurring peak is not an anomaly
# ---------------------------------------------------------------------------


def test_the_recurring_friday_peak_is_not_flagged(result, truth, population, capsys):
    """Expected is not anomalous.

    The Friday ZONE-B evening peak is 2.2x the zone's usual volume -- a bigger
    departure than most of the planted defects. It is not an anomaly, because
    within its own weekday-and-hour comparison group it *is* the norm. This test
    is the reason detection conditions on (zone, metric, weekday, hour) rather
    than on a zone-wide distribution.
    """
    planted = set(truth["patterns"]["friday_evening_congestion"]["affected_record_ids"]) & set(
        _record_ids(population)
    )
    assert planted, "the Friday peak rows are not in the scanned population"

    flagged = planted & result.record_ids
    with capsys.disabled():
        print(
            f"\n  recurring Friday peak: {len(flagged)}/{len(planted)} flagged "
            "(x2.2 above the zone's usual volume, and expected)"
        )
    assert not flagged, f"the recurring peak was reported as anomalous: {sorted(flagged)}"


def test_the_overall_flag_rate_stays_low(result, capsys):
    """A detector that flags 5% of a city's readings is not usable."""
    with capsys.disabled():
        print(
            f"\n  flagged {len(result.anomalies)} of {result.observations_scanned:,} "
            f"({result.rate:.2%}), threshold {result.threshold} MAD-sigma"
        )
        for kind, count in result.by_leading_explanation.items():
            print(f"    leading candidate {kind}: {count}")
    assert result.rate < 0.01


# ---------------------------------------------------------------------------
# Never an error
# ---------------------------------------------------------------------------


def test_no_anomaly_is_classified_as_an_error(result):
    """SPEC section 24: do not automatically classify an anomaly as an error."""
    assert result.anomalies
    for anomaly in result.anomalies:
        assert anomaly.is_error is None
        assert anomaly.status is ValidationStatus.SUSPECT
        assert anomaly.as_dict()["is_error"] is None
        assert "not classified" in str(anomaly.as_dict()["is_error_note"])


def test_the_anomaly_record_has_no_way_to_set_an_error_verdict(result):
    """Structural, not a convention a later phase can forget."""
    anomaly = result.anomalies[0]
    with pytest.raises((AttributeError, TypeError)):
        anomaly.is_error = True  # type: ignore[misc]


def test_every_anomaly_carries_at_least_one_candidate(result):
    for anomaly in result.anomalies:
        assert anomaly.explanations
        assert anomaly.leading_explanation is anomaly.explanations[0]
        supports = [item.support for item in anomaly.explanations]
        assert supports == sorted(supports, reverse=True)


# ---------------------------------------------------------------------------
# The explanation layer, exercised directly
# ---------------------------------------------------------------------------


def test_unexplained_is_a_stated_outcome(population):
    """An empty list would read as "nothing to see"."""
    observation = next(obs for obs in population if obs.value is not None)
    candidates = explain(observation)
    assert len(candidates) == 1
    assert candidates[0].kind is ExplanationKind.UNEXPLAINED
    assert "no rainfall" in candidates[0].detail


def test_rain_is_offered_as_a_candidate(population):
    observation = next(obs for obs in population if obs.value is not None)
    candidates = explain(observation, rain_mm=14.0)
    kinds = {item.kind for item in candidates}
    assert ExplanationKind.RAINFALL in kinds
    assert ExplanationKind.UNEXPLAINED not in kinds


def test_a_repeated_value_suggests_a_sensor_without_concluding_one(population):
    observation = next(obs for obs in population if obs.value is not None)
    candidates = explain(observation, repeated_neighbours=5)
    leading = candidates[0]
    assert leading.kind is ExplanationKind.SENSOR_SUSPECT
    assert "rarely does" in leading.detail
    # A candidate, not a verdict: support stays below certainty.
    assert leading.support < 1.0


def test_an_unhealthy_source_is_offered_as_a_candidate(population):
    observation = next(obs for obs in population if obs.value is not None)
    candidates = explain(observation, source_health=SourceStatus.FAILED)
    assert candidates[0].kind is ExplanationKind.SENSOR_SUSPECT
    assert "failed" in candidates[0].detail


def test_a_covering_pattern_is_offered_as_a_candidate(population, patterns):
    """A reading inside a known regularity is recognized as such."""
    pattern = next(p for p in patterns if p.kind is PatternKind.SPATIOTEMPORAL)
    hour = next(iter(pattern.scope.hours))
    observation = next(
        obs
        for obs in population
        if obs.location_id == pattern.scope.location_id
        and obs.metric == pattern.scope.metric
        and (obs.event_time + timedelta(hours=TZ_OFFSET)).weekday() == pattern.scope.weekday
        and (obs.event_time + timedelta(hours=TZ_OFFSET)).hour == hour
    )
    candidates = explain(observation, patterns=patterns, timezone_offset_hours=TZ_OFFSET)
    matched = [item for item in candidates if item.kind is ExplanationKind.RECURRING_PATTERN]
    assert matched
    assert matched[0].pattern_id == pattern.pattern_id


def test_candidates_are_not_mutually_exclusive(population):
    """Several things can be true at once, so support does not sum to one."""
    observation = next(obs for obs in population if obs.value is not None)
    candidates = explain(observation, rain_mm=12.0, source_health=SourceStatus.WARNING)
    assert len(candidates) >= 2
    assert sum(item.support for item in candidates) != pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The population helper
# ---------------------------------------------------------------------------


def test_the_pre_quarantine_population_is_a_superset_without_duplicates(
    validated, held, population
):
    assert len(population) == len(validated.valid) + len(held)
    ids = [obs.observation_id for obs in population]
    assert len(set(ids)) == len(ids)


def test_the_population_helper_does_not_mutate_the_pipeline(validated):
    """It is a measurement tool, not a pipeline stage."""
    before = len(validated.valid)
    pre_quarantine_population(validated.valid, ())
    assert len(validated.valid) == before


def test_an_empty_population_yields_no_anomalies():
    result = detect_anomalies([], metric=VOLUME)
    assert result.anomalies == ()
    assert result.rate == 0.0
