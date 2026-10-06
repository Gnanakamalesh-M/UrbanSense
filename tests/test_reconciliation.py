"""Tests for duplicates, conflicts, reconciliation and the unified layer.

Scored against `ground_truth.json`, which only tests read — the AST-based
source-tree check in `test_data_integrity.py` keeps that true for the new
modules automatically.

Two headline tests report precision and recall: one over the 12 planted
duplicates, one over the 8 planted conflicts. Both assert precision as well as
recall, because recall alone is trivially gamed by calling everything a
duplicate.

The hand-built groups matter as much as the planted ones. The demo data happens
to contain no cross-source agreement and no three-way dispute, so those paths
would ship entirely unexercised if the only fixtures were the generated file.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.helpers import make_observation, make_validated
from urbansense.config import load_settings
from urbansense.data_integrity import (
    ConflictGroup,
    GroupKind,
    ObservationKey,
    QuarantineWriter,
    explain_provenance,
    group_has_conflict,
    reconcile,
    validate_and_normalize,
    values_agree,
)
from urbansense.ingestion import load_historical
from urbansense.preprocessing import load_location_registry
from urbansense.schemas.enums import (
    AggregationLevel,
    ReconciliationPolicy,
    ReconciliationState,
    SourceType,
    ValidationStatus,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = REPO_ROOT / "data" / "sample" / "traffic_history.csv"
GROUND_TRUTH = REPO_ROOT / "data" / "sample" / "ground_truth.json"
BASE_TIME = datetime(2024, 6, 7, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures: one full validate + reconcile run for the module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def truth() -> dict:
    """The planted answers. Only tests may read this."""
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(observations) -> set[str]:
    return {str(obs.attributes["record_id"]) for obs in observations}


# ---------------------------------------------------------------------------
# The invariant: every observation in exactly one place
# ---------------------------------------------------------------------------


def test_every_observation_is_in_exactly_one_place(reconciled, validated):
    """Three-way partition of the valid set, exact.

    If this fails, records have gone missing between validation and
    reconciliation, which the project forbids outright.
    """
    assert reconciled.accounts_for_every_observation
    assert reconciled.stats.observations_in == len(validated.valid)
    assert (
        len(reconciled.unified)
        + len(reconciled.duplicate_linked)
        + len(reconciled.conflicted_observations)
    ) == len(validated.valid)


def test_the_buckets_are_disjoint(reconciled):
    """Counts matching is necessary but not sufficient.

    A record appearing in two buckets while another vanished would still add
    up, so set disjointness is asserted separately.
    """
    assert reconciled.observation_ids_are_disjoint


def test_the_expected_split_on_the_demo_file(reconciled):
    assert len(reconciled.unified) == 58_900
    assert len(reconciled.duplicate_linked) == 12
    assert len(reconciled.conflicted_observations) == 16
    assert reconciled.stats.observations_in == 58_928


def test_no_conflicted_observation_reaches_the_unified_layer(reconciled):
    """A disputed slot is a visible hole, not a guess (SPEC section 10)."""
    unified_ids = {obs.observation_id for obs in reconciled.unified}
    for group in reconciled.conflicts:
        for observation_id in group.observation_ids:
            assert observation_id not in unified_ids


# ---------------------------------------------------------------------------
# Duplicates: scored against the planted set (SPEC section 9)
# ---------------------------------------------------------------------------


def test_duplicate_precision_and_recall(reconciled, truth, capsys):
    """All 12 planted duplicates found, with nothing else called a duplicate.

    Scored against SAME_SOURCE_REPEAT groups only, so a cross-source agreement
    can never inflate the duplicate score with records that were never
    duplicates in section 9's sense.
    """
    planted = truth["planted_records"]["duplicates"]
    expected_pairs = {frozenset({item["record_id"], item["duplicate_of"]}) for item in planted}

    detected_pairs = {
        frozenset(reference.record_id for reference in group.references)
        for group in reconciled.same_source_groups
    }

    caught = expected_pairs & detected_pairs
    missed = expected_pairs - detected_pairs
    false_positives = detected_pairs - expected_pairs

    recall = len(caught) / len(expected_pairs)
    precision = len(caught) / len(detected_pairs) if detected_pairs else 0.0

    with capsys.disabled():
        print(
            f"\n  duplicate scoring over {len(expected_pairs)} planted pairs:"
            f"\n    recall    {recall:.3f}  ({len(caught)}/{len(expected_pairs)} found)"
            f"\n    precision {precision:.3f}  ({len(caught)}/{len(detected_pairs)} "
            "detected were planted)"
        )

    assert not missed, f"planted duplicates not found: {[sorted(p) for p in missed]}"
    assert not false_positives, (
        f"{len(false_positives)} groups were called duplicates but were not planted: "
        f"{[sorted(p) for p in false_positives]}"
    )
    assert recall == 1.0
    assert precision == 1.0


def test_duplicates_are_linked_to_their_representative(reconciled, truth):
    planted = {
        frozenset({item["record_id"], item["duplicate_of"]})
        for item in truth["planted_records"]["duplicates"]
    }
    by_record = {str(obs.attributes["record_id"]): obs for obs in reconciled.duplicate_linked}
    unified_by_id = {obs.observation_id: obs for obs in reconciled.unified}

    assert len(by_record) == 12
    for record_id, observation in by_record.items():
        assert observation.validation_status is ValidationStatus.DUPLICATE
        assert observation.duplicate_of is not None
        representative = unified_by_id[observation.duplicate_of]
        pair = frozenset({record_id, str(representative.attributes["record_id"])})
        assert pair in planted, f"{record_id} linked to an unplanted partner"


def test_both_source_records_still_exist(reconciled, truth):
    """Deleting the copy would be the obvious implementation and is wrong.

    A mistaken duplicate judgement must stay recoverable (SPEC section 9).
    """
    everywhere = (
        _records(reconciled.unified)
        | _records(reconciled.duplicate_linked)
        | _records(reconciled.conflicted_observations)
    )
    for item in truth["planted_records"]["duplicates"]:
        assert item["record_id"] in everywhere
        assert item["duplicate_of"] in everywhere


def test_duplicate_values_agree(reconciled):
    """A group is only a duplicate group because its members agree."""
    for group in reconciled.duplicate_groups:
        values = [reference.value for reference in group.references]
        for other in values[1:]:
            assert values_agree(values[0], other)


def test_representative_choice_is_deterministic(validated, settings):
    first = reconcile(validated, settings=settings)
    second = reconcile(validated, settings=settings)
    assert [g.representative_id for g in first.duplicate_groups] == [
        g.representative_id for g in second.duplicate_groups
    ]


def test_unified_row_records_its_duplicate_group(reconciled):
    collapsed = [obs for obs in reconciled.unified if obs.attributes.get("duplicate_count", 0) > 0]
    assert len(collapsed) == 12
    for observation in collapsed:
        assert observation.attributes["duplicate_group_id"].startswith("DUP-")
        assert len(observation.attributes["source_references"]) == 2


# ---------------------------------------------------------------------------
# Conflicts: scored against the planted set (SPEC section 10)
# ---------------------------------------------------------------------------


def test_conflict_precision_and_recall(reconciled, truth, capsys):
    """All 8 planted conflicts marked, and no false conflicts."""
    planted = truth["planted_records"]["conflicts"]
    expected_pairs = {frozenset({item["record_id"], item["conflicts_with"]}) for item in planted}
    detected_pairs = {
        frozenset(candidate.record_id for candidate in group.candidates)
        for group in reconciled.conflicts
    }

    caught = expected_pairs & detected_pairs
    missed = expected_pairs - detected_pairs
    false_positives = detected_pairs - expected_pairs

    recall = len(caught) / len(expected_pairs)
    precision = len(caught) / len(detected_pairs) if detected_pairs else 0.0

    with capsys.disabled():
        print(
            f"\n  conflict scoring over {len(expected_pairs)} planted pairs:"
            f"\n    recall    {recall:.3f}  ({len(caught)}/{len(expected_pairs)} found)"
            f"\n    precision {precision:.3f}  ({len(caught)}/{len(detected_pairs)} "
            "detected were planted)"
        )

    assert not missed, f"planted conflicts not found: {[sorted(p) for p in missed]}"
    assert not false_positives, (
        f"{len(false_positives)} false conflicts: {[sorted(p) for p in false_positives]}"
    )
    assert recall == 1.0
    assert precision == 1.0


def test_both_conflicting_values_are_kept(reconciled, truth):
    """The disagreement is the information; nothing is overwritten."""
    by_pair = {
        frozenset(c.record_id for c in group.candidates): group for group in reconciled.conflicts
    }
    for item in truth["planted_records"]["conflicts"]:
        group = by_pair[frozenset({item["record_id"], item["conflicts_with"]})]
        values = sorted(c.value for c in group.candidates if c.value is not None)
        assert values == sorted([item["original_value"], item["conflicting_value"]])


def test_conflicting_records_are_linked_both_ways(reconciled, truth):
    by_record = {
        str(obs.attributes["record_id"]): obs for obs in reconciled.conflicted_observations
    }
    for item in truth["planted_records"]["conflicts"]:
        left = by_record[item["record_id"]]
        right = by_record[item["conflicts_with"]]
        assert left.validation_status is ValidationStatus.CONFLICT
        assert right.validation_status is ValidationStatus.CONFLICT
        assert right.observation_id in left.conflicts_with
        assert left.observation_id in right.conflicts_with


def test_conflicts_record_policy_and_unresolved_state(reconciled):
    """No automatic winner in this phase, and the record says so."""
    assert reconciled.conflicts
    for group in reconciled.conflicts:
        assert group.policy is ReconciliationPolicy.MANUAL_REVIEW
        assert group.state is ReconciliationState.UNRESOLVED
        assert "no winner chosen" in group.detail
    assert len(reconciled.unresolved_conflicts) == len(reconciled.conflicts)


def test_planted_conflicts_are_cross_source(reconciled):
    """A sensor and the warehouse disagreeing is a cross-source dispute."""
    assert len(reconciled.cross_source_conflicts) == 8
    for group in reconciled.conflicts:
        assert len(group.source_ids) == 2
        assert group.spread is not None and group.spread > 0


# ---------------------------------------------------------------------------
# Cross-source agreement: hand-built, absent from the demo data
# ---------------------------------------------------------------------------


@pytest.fixture
def cross_source_pair():
    """Two sources reporting the same value for the same observation."""
    return [
        make_observation(8200.0, event_time=BASE_TIME, source_id="sensor-B07", record_id="REC-A"),
        make_observation(
            8200.0, event_time=BASE_TIME, source_id="city-warehouse", record_id="REC-B"
        ),
    ]


def test_cross_source_agreement_shares_one_observation_key(cross_source_pair):
    """Same key is what makes them the same observation to begin with."""
    left, right = cross_source_pair
    assert ObservationKey.of(left) == ObservationKey.of(right)
    assert left.source_id != right.source_id


def test_cross_source_agreement_collapses_to_one_unified_row(cross_source_pair, settings):
    result = reconcile(make_validated(cross_source_pair), settings=settings)
    assert len(result.unified) == 1, "two agreeing sources must not yield two rows"
    assert len(result.duplicate_groups) == 1
    assert result.conflicts == ()


def test_cross_source_agreement_keeps_both_source_references(cross_source_pair, settings):
    """Corroboration must not be thrown away by the collapse."""
    result = reconcile(make_validated(cross_source_pair), settings=settings)
    unified = result.unified[0]

    references = unified.attributes["source_references"]
    assert len(references) == 2
    assert {reference["source_id"] for reference in references} == {
        "sensor-B07",
        "city-warehouse",
    }
    assert {reference["record_id"] for reference in references} == {"REC-A", "REC-B"}
    assert sum(1 for reference in references if reference["is_representative"]) == 1

    assert set(unified.attributes["contributing_source_ids"]) == {
        "sensor-B07",
        "city-warehouse",
    }
    assert set(unified.attributes["contributing_source_types"]) == {SourceType.HISTORICAL.value}


def test_cross_source_agreement_is_reported_as_its_own_kind(cross_source_pair, settings):
    """It must not be counted as a same-source repeat.

    Conflating them would let corroboration inflate the planted-duplicate
    score with records that were never duplicates in section 9's sense.
    """
    result = reconcile(make_validated(cross_source_pair), settings=settings)
    group = result.duplicate_groups[0]

    assert group.kind is GroupKind.CROSS_SOURCE_AGREEMENT
    assert group.kind is not GroupKind.SAME_SOURCE_REPEAT
    assert result.by_group_kind == {
        GroupKind.SAME_SOURCE_REPEAT: 0,
        GroupKind.CROSS_SOURCE_AGREEMENT: 1,
    }
    assert result.stats.cross_source_agreements == 1
    assert result.stats.same_source_repeats == 0
    assert result.same_source_groups == ()
    assert len(result.cross_source_groups) == 1
    assert "Corroboration, not redundancy" in group.detail


def test_cross_source_agreement_keeps_the_other_record_linked(cross_source_pair, settings):
    """The non-representative member is linked, not deleted."""
    result = reconcile(make_validated(cross_source_pair), settings=settings)
    assert len(result.duplicate_linked) == 1

    linked = result.duplicate_linked[0]
    assert linked.validation_status is ValidationStatus.DUPLICATE
    assert linked.duplicate_of == result.unified[0].observation_id
    assert result.accounts_for_every_observation


def test_same_source_repeat_is_reported_as_its_own_kind(settings):
    """The mirror of the test above, so the two kinds cannot be swapped."""
    pair = [
        make_observation(8200.0, event_time=BASE_TIME, source_id="sensor-B07", record_id="REC-A"),
        make_observation(8200.0, event_time=BASE_TIME, source_id="sensor-B07", record_id="REC-B"),
    ]
    result = reconcile(make_validated(pair), settings=settings)
    assert result.duplicate_groups[0].kind is GroupKind.SAME_SOURCE_REPEAT
    assert result.by_group_kind == {
        GroupKind.SAME_SOURCE_REPEAT: 1,
        GroupKind.CROSS_SOURCE_AGREEMENT: 0,
    }


# ---------------------------------------------------------------------------
# What is NOT the same observation (SPEC section 10)
# ---------------------------------------------------------------------------


def test_same_value_at_a_different_time_is_not_a_duplicate(settings):
    pair = [
        make_observation(8200.0, event_time=BASE_TIME, record_id="REC-A"),
        make_observation(8200.0, event_time=BASE_TIME + timedelta(hours=1), record_id="REC-B"),
    ]
    result = reconcile(make_validated(pair), settings=settings)
    assert len(result.unified) == 2
    assert result.duplicate_groups == ()
    assert result.duplicate_linked == ()


def test_same_value_at_a_different_location_is_not_a_duplicate(settings):
    pair = [
        make_observation(8200.0, event_time=BASE_TIME, location_id="ZONE-A", record_id="REC-A"),
        make_observation(8200.0, event_time=BASE_TIME, location_id="ZONE-B", record_id="REC-B"),
    ]
    result = reconcile(make_validated(pair), settings=settings)
    assert len(result.unified) == 2
    assert result.duplicate_groups == ()


def test_same_value_for_a_different_metric_is_not_a_duplicate(settings):
    pair = [
        make_observation(45.0, event_time=BASE_TIME, metric="traffic_volume", record_id="REC-A"),
        make_observation(45.0, event_time=BASE_TIME, metric="average_speed", record_id="REC-B"),
    ]
    result = reconcile(make_validated(pair), settings=settings)
    assert len(result.unified) == 2


def test_different_unit_is_not_a_conflict(settings):
    """2050 vehicles/15min and 8200 vehicles/hour are not competing claims.

    They are different measurements until normalization brings them into one
    unit, and comparing them before that is exactly the mistake SPEC section 11
    warns about.
    """
    pair = [
        make_observation(8200.0, event_time=BASE_TIME, unit="vehicles/hour", record_id="REC-A"),
        make_observation(2050.0, event_time=BASE_TIME, unit="vehicles/15min", record_id="REC-B"),
    ]
    result = reconcile(make_validated(pair), settings=settings)
    assert result.conflicts == ()
    assert len(result.unified) == 2


def test_different_measurement_duration_is_not_a_conflict(settings):
    left = make_observation(8200.0, event_time=BASE_TIME, record_id="REC-A")
    right = make_observation(2050.0, event_time=BASE_TIME, record_id="REC-B").model_copy(
        update={"measurement_duration": timedelta(minutes=15)}
    )
    result = reconcile(make_validated([left, right]), settings=settings)
    assert result.conflicts == ()
    assert len(result.unified) == 2


def test_different_aggregation_level_is_not_a_conflict(settings):
    left = make_observation(8200.0, event_time=BASE_TIME, record_id="REC-A")
    right = make_observation(900.0, event_time=BASE_TIME, record_id="REC-B").model_copy(
        update={"aggregation_level": AggregationLevel.QUARTER_HOUR}
    )
    result = reconcile(make_validated([left, right]), settings=settings)
    assert result.conflicts == ()
    assert len(result.unified) == 2


# ---------------------------------------------------------------------------
# Materiality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "agree"),
    [
        (8200.0, 8200.0, True),
        (8200.0, 8280.0, True),  # ~1%, within the 2% tolerance
        (8200.0, 10004.0, False),  # 22%, the planted divergence
        (0.0, 0.4, True),  # absolute floor protects near-zero readings
        (0.0, 5.0, False),
        (None, None, True),  # both say nothing was measured
        (None, 8200.0, False),  # a gap and a reading genuinely disagree
    ],
)
def test_value_agreement_boundaries(left, right, agree):
    assert values_agree(left, right) is agree


def test_a_small_divergence_is_a_duplicate_not_a_conflict(settings):
    pair = [
        make_observation(8200.0, event_time=BASE_TIME, source_id="sensor-B07", record_id="REC-A"),
        make_observation(
            8280.0, event_time=BASE_TIME, source_id="city-warehouse", record_id="REC-B"
        ),
    ]
    result = reconcile(make_validated(pair), settings=settings)
    assert result.conflicts == ()
    assert len(result.duplicate_groups) == 1


def test_tolerance_is_configurable(settings):
    pair = [
        make_observation(8200.0, event_time=BASE_TIME, source_id="sensor-B07", record_id="REC-A"),
        make_observation(
            8280.0, event_time=BASE_TIME, source_id="city-warehouse", record_id="REC-B"
        ),
    ]
    strict = reconcile(
        make_validated(pair),
        settings=settings,
        relative_tolerance=0.001,
        absolute_tolerance=0.0,
    )
    assert len(strict.conflicts) == 1


def test_one_dissenter_makes_the_whole_group_a_conflict(settings):
    """Not a duplicate pair plus a dissenter.

    Splitting it that way would pick a winner by majority vote, which SPEC
    section 10 forbids -- and a majority is not evidence, since two feeds can
    share one upstream fault.
    """
    members = [
        make_observation(8200.0, event_time=BASE_TIME, source_id="sensor-A", record_id="REC-A"),
        make_observation(8200.0, event_time=BASE_TIME, source_id="sensor-B", record_id="REC-B"),
        make_observation(5000.0, event_time=BASE_TIME, source_id="sensor-C", record_id="REC-C"),
    ]
    result = reconcile(make_validated(members), settings=settings)

    assert len(result.conflicts) == 1
    assert result.duplicate_groups == ()
    assert len(result.conflicted_observations) == 3
    assert result.unified == ()
    assert len(result.conflicts[0].candidates) == 3
    assert result.accounts_for_every_observation


def test_group_has_conflict_detects_any_disagreeing_pair():
    agreeing = [
        make_observation(100.0, event_time=BASE_TIME, record_id="A"),
        make_observation(100.0, event_time=BASE_TIME, record_id="B"),
    ]
    assert not group_has_conflict(agreeing)
    agreeing.append(make_observation(500.0, event_time=BASE_TIME, record_id="C"))
    assert group_has_conflict(agreeing)


# ---------------------------------------------------------------------------
# Unified layer and source views (SPEC sections 19, 20)
# ---------------------------------------------------------------------------


def test_every_unified_row_keeps_source_references(reconciled):
    """A combined figure a user cannot decompose is one they cannot audit."""
    for observation in reconciled.unified[:500]:
        references = observation.attributes["source_references"]
        assert references
        assert all(reference["source_id"] for reference in references)
        assert observation.attributes["contributing_source_types"]


def test_singletons_also_carry_a_source_reference(reconciled):
    """So a consumer never branches on 'did this collapse?' to find origin."""
    singleton = next(
        obs for obs in reconciled.unified if obs.attributes.get("duplicate_count") == 0
    )
    references = singleton.attributes["source_references"]
    assert len(references) == 1
    assert references[0]["is_representative"] is True


def test_unified_rows_keep_their_source_type(reconciled):
    for observation in reconciled.unified[:200]:
        assert observation.source_type is SourceType.HISTORICAL


def test_source_views_partition_the_unified_layer(reconciled):
    """Separate views come before the combined one (SPEC section 20)."""
    total = (
        len(reconciled.historical_view)
        + len(reconciled.real_time_view)
        + len(reconciled.document_view)
    )
    assert total == len(reconciled.unified)
    assert len(reconciled.historical_view) == 58_900
    assert reconciled.real_time_view == ()
    assert reconciled.document_view == ()


def test_by_source_type_counts(reconciled):
    assert reconciled.by_source_type == {SourceType.HISTORICAL: 58_900}


def test_real_time_view_is_populated_for_a_real_time_source(settings):
    """The view exists now so the layer's shape does not change later."""
    observation = make_observation(100.0, event_time=BASE_TIME, record_id="REC-A")
    live = observation.model_copy(update={"source_type": SourceType.REAL_TIME})
    result = reconcile(make_validated([live]), settings=settings)
    assert len(result.real_time_view) == 1
    assert result.historical_view == ()


def test_unified_layer_is_ordered_by_event_time(reconciled):
    times = [obs.event_time for obs in reconciled.unified]
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# Provenance (SPEC section 7)
# ---------------------------------------------------------------------------


def test_provenance_for_a_singleton(reconciled):
    singleton = next(
        obs for obs in reconciled.unified if obs.attributes.get("duplicate_count") == 0
    )
    chain = explain_provenance(singleton.observation_id, reconciled)
    assert chain is not None
    assert chain.placement == "unified"
    assert chain.duplicate_group is None
    assert chain.conflict_group is None
    assert chain.source_row is not None
    assert chain.observation.provenance.pipeline_version
    assert "when it happened" in chain.describe()


def test_provenance_for_a_duplicate_names_the_whole_group(reconciled, truth):
    item = truth["planted_records"]["duplicates"][0]
    chain = explain_provenance(item["record_id"], reconciled)
    assert chain is not None
    assert chain.placement == "duplicate_linked"
    assert chain.duplicate_group is not None
    assert chain.duplicate_group.kind is GroupKind.SAME_SOURCE_REPEAT
    assert {r.record_id for r in chain.duplicate_group.references} == {
        item["record_id"],
        item["duplicate_of"],
    }
    assert chain.representative is not None
    rendered = chain.describe()
    assert "duplicate group" in rendered
    assert item["duplicate_of"] in rendered


def test_provenance_for_a_conflict_shows_every_candidate(reconciled, truth):
    item = truth["planted_records"]["conflicts"][0]
    chain = explain_provenance(item["record_id"], reconciled)
    assert chain is not None
    assert chain.placement == "conflicted"
    assert chain.conflict_group is not None
    assert chain.is_unresolved_conflict
    assert len(chain.conflict_peers) == 1

    rendered = chain.describe()
    assert "CONFLICT" in rendered
    assert "no winner chosen" in rendered
    assert str(int(item["conflicting_value"])) in rendered


def test_provenance_accepts_an_observation_id_or_a_record_id(reconciled, truth):
    """A user holding a feed export knows the record id, not ours."""
    item = truth["planted_records"]["duplicates"][0]
    by_record = explain_provenance(item["record_id"], reconciled)
    assert by_record is not None
    by_observation = explain_provenance(by_record.observation.observation_id, reconciled)
    assert by_observation is not None
    assert by_observation.observation == by_record.observation


def test_provenance_returns_none_for_an_unknown_id(reconciled):
    assert explain_provenance("REC-does-not-exist", reconciled) is None


def test_provenance_is_serializable(reconciled, truth):
    item = truth["planted_records"]["conflicts"][0]
    chain = explain_provenance(item["record_id"], reconciled)
    assert chain is not None
    payload = json.loads(json.dumps(chain.as_dict()))
    assert payload["record_id"] == item["record_id"]
    assert payload["conflict"]["resolved"] is False
    assert len(payload["conflict"]["candidates"]) == 2


# ---------------------------------------------------------------------------
# Review queue (SPEC sections 10, 18)
# ---------------------------------------------------------------------------


def test_conflicts_are_written_with_every_candidate(reconciled, tmp_path):
    writer = QuarantineWriter(tmp_path)
    target = writer.write_conflicts(reconciled.conflicts, now=datetime(2026, 10, 2, tzinfo=UTC))
    assert target is not None and target.exists()
    assert target.parent.name == "conflicts"

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert len(payload) == 8
    for group in payload:
        assert group["resolved"] is False
        assert group["state"] == "unresolved"
        assert group["policy"] == "manual_review"
        assert len(group["candidates"]) == 2
        # No candidate is marked as chosen, and the record carries no field
        # that would let a reader infer one. (The prose deliberately says "no
        # winner chosen", so this checks structure rather than text.)
        assert not {"winner", "chosen", "resolved_value", "selected"} & group.keys()
        for candidate in group["candidates"]:
            assert not {"winner", "chosen", "selected"} & candidate.keys()
            assert candidate["value"] is not None


def test_no_conflict_file_when_there_are_no_conflicts(tmp_path):
    assert QuarantineWriter(tmp_path).write_conflicts([]) is None
    assert not list(tmp_path.iterdir())


def test_conflict_queue_is_separate_from_the_suspicious_queue(reconciled, tmp_path):
    """'Unsure about this value' must not read as 'could not decide which'."""
    writer = QuarantineWriter(tmp_path)
    conflicts = writer.write_conflicts(reconciled.conflicts)
    notes = writer.write_review_notes([], {})
    assert conflicts is not None
    assert conflicts.parent.name == "conflicts"
    assert notes is None


def test_each_run_writes_a_new_conflict_file(reconciled, tmp_path):
    writer = QuarantineWriter(tmp_path)
    first = writer.write_conflicts(
        reconciled.conflicts, now=datetime(2026, 10, 2, 9, 0, tzinfo=UTC)
    )
    second = writer.write_conflicts(
        reconciled.conflicts, now=datetime(2026, 10, 2, 10, 0, tzinfo=UTC)
    )
    assert first != second
    assert first is not None and second is not None


# ---------------------------------------------------------------------------
# Phase 2 behaviour is unchanged, and raw data is untouched
# ---------------------------------------------------------------------------


def test_reconciliation_does_not_resurrect_quarantined_records(reconciled, validated):
    """Quarantined stays quarantined; reconciliation only sees the valid set."""
    quarantined = {
        record.raw_payload.get("record_id")
        for record in validated.quarantined
        if record.raw_payload.get("record_id")
    }
    everywhere = (
        _records(reconciled.unified)
        | _records(reconciled.duplicate_linked)
        | _records(reconciled.conflicted_observations)
    )
    assert not (quarantined & everywhere)


def test_raw_files_are_unchanged(settings):
    before_hash = _sha256(HISTORY_CSV)
    before_mtime = HISTORY_CSV.stat().st_mtime_ns

    validated = validate_and_normalize(
        load_historical(HISTORY_CSV),
        settings=settings,
        registry=load_location_registry(),
    )
    reconcile(validated, settings=settings)

    assert _sha256(HISTORY_CSV) == before_hash
    assert HISTORY_CSV.stat().st_mtime_ns == before_mtime


def test_reconciliation_writes_nothing_by_itself(
    reconciled, validated, settings, tmp_path, monkeypatch
):
    """Persisting the review queue is the caller's decision."""
    monkeypatch.chdir(tmp_path)
    result = reconcile(validated, settings=settings)
    assert result.conflicts
    assert list(tmp_path.iterdir()) == []


def test_reconciliation_is_deterministic(validated, settings):
    first = reconcile(validated, settings=settings)
    second = reconcile(validated, settings=settings)
    assert first.stats == second.stats
    assert [g.group_id for g in first.duplicate_groups] == [
        g.group_id for g in second.duplicate_groups
    ]
    assert [g.conflict_id for g in first.conflicts] == [g.conflict_id for g in second.conflicts]
    assert [o.observation_id for o in first.unified] == [o.observation_id for o in second.unified]


def test_observation_key_describes_itself(reconciled):
    group = reconciled.conflicts[0]
    rendered = group.key.describe()
    assert group.key.metric in rendered
    assert group.key.location_id in rendered
    assert group.key.unit in rendered
    assert group.key.event_time.isoformat() in rendered


def test_observation_key_is_usable_as_a_grouping_key(reconciled):
    """Equality and hashing must be by value, since it keys the grouping dict.

    An identity-based key would put every observation in its own group and no
    duplicate or conflict would ever be found.
    """
    key = reconciled.conflicts[0].key
    twin = ObservationKey(
        location_id=key.location_id,
        event_time=key.event_time,
        metric=key.metric,
        unit=key.unit,
        measurement_duration=key.measurement_duration,
        aggregation_level=key.aggregation_level,
    )
    assert twin == key
    assert hash(twin) == hash(key)
    assert len({key, twin}) == 1

    different = ObservationKey(
        location_id=key.location_id,
        event_time=key.event_time + timedelta(hours=1),
        metric=key.metric,
        unit=key.unit,
        measurement_duration=key.measurement_duration,
        aggregation_level=key.aggregation_level,
    )
    assert different != key


def test_conflict_group_exposes_its_key(reconciled):
    group = reconciled.conflicts[0]
    assert isinstance(group, ConflictGroup)
    assert group.key == ObservationKey(
        location_id=group.key.location_id,
        event_time=group.key.event_time,
        metric=group.key.metric,
        unit=group.key.unit,
        measurement_duration=group.key.measurement_duration,
        aggregation_level=group.key.aggregation_level,
    )


def test_counts_in_stats_match_the_collections(reconciled):
    stats = reconciled.stats
    assert stats.unified == len(reconciled.unified)
    assert stats.duplicate_linked == len(reconciled.duplicate_linked)
    assert stats.conflicted == len(reconciled.conflicted_observations)
    assert stats.duplicate_groups == len(reconciled.duplicate_groups)
    assert stats.conflict_groups == len(reconciled.conflicts)
    assert stats.same_source_repeats == len(reconciled.same_source_groups)
    assert stats.cross_source_agreements == len(reconciled.cross_source_groups)


def test_input_count_matches_the_file(reconciled, validated):
    with HISTORY_CSV.open(encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    assert validated.stats.rows_in == rows
    assert reconciled.stats.observations_in == len(validated.valid)
