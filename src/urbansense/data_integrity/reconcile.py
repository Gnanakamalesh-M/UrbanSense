"""The reconciliation pipeline (SPEC sections 2, 9, 10, 19).

Takes the validated output of Phase 2 and produces the unified data layer,
implementing the DETECT DUPLICATES/CONFLICTS -> RECONCILE -> UNIFIED DATA
stretch of the lifecycle.

**One grouping pass, three outcomes.** Duplicates and conflicts are the same
question -- "do these records describe the same observation?" -- with different
answers once the values are compared. Running two independent passes would risk
them disagreeing about what a group is, so there is a single grouping by
:class:`~urbansense.data_integrity.keys.ObservationKey` and then a decision per
group:

* values agree, one source  -> duplicate repeat, collapsed and linked
* values agree, many sources -> corroboration, collapsed with all references
* any material disagreement  -> conflict, every candidate kept, none unified

**Nothing is deleted, overwritten or guessed.** Duplicate records survive with
``duplicate_of`` set. Conflicting records survive with ``conflicts_with`` set
and go to review with no winner chosen. A disputed slot simply has no unified
row -- an honest hole rather than an invented value.

**Phase 2's detectors are untouched.** This is a separate entry point;
``validate_and_normalize`` is not modified and its judgements are taken as
given.
"""

from __future__ import annotations

from datetime import UTC, datetime

from urbansense.config.models import Settings
from urbansense.data_integrity.conflicts import (
    ConflictGroup,
    build_conflict_group,
    group_has_conflict,
)
from urbansense.data_integrity.duplicates import (
    DuplicateGroup,
    GroupKind,
    build_duplicate_group,
)
from urbansense.data_integrity.keys import group_by_key
from urbansense.data_integrity.result import ValidatedResult
from urbansense.data_integrity.unified import ReconciledResult, ReconciliationStats
from urbansense.schemas.enums import ReconciliationState, ValidationStatus
from urbansense.schemas.observation import Observation


def _with_source_references(observation: Observation, group: DuplicateGroup) -> Observation:
    """Attach a duplicate group's source references to its representative.

    This is what makes the collapse auditable: the unified row names every
    source record behind it, so a combined figure can always be decomposed back
    into the streams that produced it (SPEC section 19).
    """
    attributes = dict(observation.attributes)
    attributes["duplicate_group_id"] = group.group_id
    attributes["duplicate_group_kind"] = group.kind.value
    attributes["source_references"] = [reference.as_dict() for reference in group.references]
    attributes["contributing_source_ids"] = list(group.source_ids)
    attributes["contributing_source_types"] = list(group.source_types)
    attributes["duplicate_count"] = len(group.duplicate_ids)
    return observation.model_copy(update={"attributes": attributes})


def _as_singleton(observation: Observation) -> Observation:
    """Give an ungrouped observation its own one-entry source reference.

    Every unified row carries ``source_references`` whether or not it was part
    of a group, so a consumer never has to branch on "did this collapse?" to
    find out where a value came from.
    """
    attributes = dict(observation.attributes)
    attributes["source_references"] = [
        {
            "observation_id": observation.observation_id,
            "record_id": str(observation.attributes.get("record_id", observation.observation_id)),
            "source_id": observation.source_id,
            "source_type": observation.source_type.value,
            "value": observation.value,
            "received_time": observation.received_time.isoformat(),
            "is_representative": True,
        }
    ]
    attributes["contributing_source_ids"] = [observation.source_id]
    attributes["contributing_source_types"] = [observation.source_type.value]
    attributes["duplicate_count"] = 0
    return observation.model_copy(update={"attributes": attributes})


def _as_duplicate(observation: Observation, group: DuplicateGroup) -> Observation:
    """Link a duplicate record to its representative, keeping the record.

    Marked ``DUPLICATE`` and pointed at the representative rather than dropped:
    if the duplicate judgement is later found wrong, this record is still here
    to recover (SPEC section 9).
    """
    attributes = dict(observation.attributes)
    attributes["duplicate_group_id"] = group.group_id
    attributes["duplicate_group_kind"] = group.kind.value
    attributes["duplicate_of_record_id"] = next(
        (reference.record_id for reference in group.references if reference.is_representative),
        None,
    )
    return observation.model_copy(
        update={
            "validation_status": ValidationStatus.DUPLICATE,
            "duplicate_of": group.representative_id,
            "attributes": attributes,
        }
    )


def _as_conflicted(observation: Observation, group: ConflictGroup) -> Observation:
    """Mark a record as conflicting and link it to the others.

    Every member keeps its own value. Nothing is overwritten, because the
    disagreement *is* the information (SPEC section 10).
    """
    attributes = dict(observation.attributes)
    attributes["conflict_id"] = group.conflict_id
    attributes["reconciliation_policy"] = group.policy.value
    attributes["reconciliation_state"] = group.state.value
    attributes["conflicting_values"] = [
        {"source_id": c.source_id, "value": c.value, "unit": c.unit} for c in group.candidates
    ]
    return observation.model_copy(
        update={
            "validation_status": ValidationStatus.CONFLICT,
            "conflicts_with": tuple(
                observation_id
                for observation_id in group.observation_ids
                if observation_id != observation.observation_id
            ),
            "attributes": attributes,
        }
    )


def reconcile(
    validated_result: ValidatedResult,
    *,
    settings: Settings,
    now: datetime | None = None,
    relative_tolerance: float | None = None,
    absolute_tolerance: float | None = None,
) -> ReconciledResult:
    """Reconcile a validated result into the unified data layer.

    Args:
        validated_result: Output of ``validate_and_normalize``. Only its
            ``valid`` observations are reconciled; quarantined records stay
            quarantined and are not resurrected here.
        settings: Configuration supplying the materiality thresholds.
        now: Detection timestamp. Defaults to the data's own receipt time so a
            run over a historical file is reproducible rather than depending on
            when it was processed.
        relative_tolerance: Overrides the configured relative tolerance.
        absolute_tolerance: Overrides the configured absolute tolerance.

    Returns:
        A :class:`ReconciledResult` in which every input observation appears in
        exactly one of ``unified``, ``duplicate_linked`` or
        ``conflicted_observations``.
    """
    reconciliation = settings.reconciliation
    relative = (
        relative_tolerance
        if relative_tolerance is not None
        else reconciliation.conflict_relative_tolerance
    )
    absolute = (
        absolute_tolerance
        if absolute_tolerance is not None
        else reconciliation.conflict_absolute_tolerance
    )
    detected_at = now if now is not None else _default_reference(validated_result)

    unified: list[Observation] = []
    duplicate_linked: list[Observation] = []
    conflicted: list[Observation] = []
    duplicate_groups: list[DuplicateGroup] = []
    conflict_groups: list[ConflictGroup] = []

    for key, members in group_by_key(list(validated_result.valid)).items():
        if len(members) == 1:
            unified.append(_as_singleton(members[0]))
            continue

        if group_has_conflict(members, relative_tolerance=relative, absolute_tolerance=absolute):
            conflict = build_conflict_group(
                key, members, detected_at=detected_at, policy=reconciliation.conflict_policy
            )
            conflict_groups.append(conflict)
            # No member reaches the unified layer: with no winner chosen, the
            # slot is a visible hole rather than an invented value.
            conflicted.extend(_as_conflicted(observation, conflict) for observation in members)
            continue

        duplicates = build_duplicate_group(key, members, detected_at=detected_at)
        duplicate_groups.append(duplicates)
        for observation in members:
            if observation.observation_id == duplicates.representative_id:
                unified.append(_with_source_references(observation, duplicates))
            else:
                duplicate_linked.append(_as_duplicate(observation, duplicates))

    # Deterministic output order, independent of dict iteration or row order.
    unified.sort(key=lambda obs: (obs.event_time, obs.location_id, obs.metric))
    duplicate_linked.sort(key=lambda obs: obs.observation_id)
    conflicted.sort(key=lambda obs: obs.observation_id)
    duplicate_groups.sort(key=lambda group: group.group_id)
    conflict_groups.sort(key=lambda group: group.conflict_id)

    kinds = [group.kind for group in duplicate_groups]

    return ReconciledResult(
        unified=tuple(unified),
        duplicate_linked=tuple(duplicate_linked),
        conflicts=tuple(conflict_groups),
        duplicate_groups=tuple(duplicate_groups),
        conflicted_observations=tuple(conflicted),
        stats=ReconciliationStats(
            observations_in=len(validated_result.valid),
            unified=len(unified),
            duplicate_linked=len(duplicate_linked),
            conflicted=len(conflicted),
            duplicate_groups=len(duplicate_groups),
            same_source_repeats=sum(1 for k in kinds if k is GroupKind.SAME_SOURCE_REPEAT),
            cross_source_agreements=sum(1 for k in kinds if k is GroupKind.CROSS_SOURCE_AGREEMENT),
            conflict_groups=len(conflict_groups),
            source_path=validated_result.stats.source_path,
        ),
    )


def _default_reference(validated_result: ValidatedResult) -> datetime:
    """Pick a detection timestamp, preferring the data's own receipt time."""
    for observation in validated_result.valid:
        return observation.received_time
    return datetime.now(UTC)


__all__ = ["ReconciliationState", "reconcile"]
