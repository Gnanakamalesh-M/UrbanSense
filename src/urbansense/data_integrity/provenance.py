"""Answering "where did this value come from?" (SPEC section 7).

    The system must allow users to determine: Where did this value come from?
    Never silently lose the origin of a value.

The schema already guarantees each record carries its own ``Provenance``.
Reconciliation adds a second question that one record cannot answer alone: a
unified row may stand for several source records, and a conflicting row has
peers asserting different values. Both are part of the honest answer, so
:func:`explain_provenance` assembles the whole chain:

* the record itself, its source and the original feed row;
* every source record behind it, when it represents a collapsed group;
* the record it duplicates, when it is a duplicate;
* the peers it disagrees with and their values, when it conflicts;
* the reconciliation policy and state, so "we have not decided" is visible
  rather than implied by absence.

Lookup works for a record in any of the three buckets. A provenance tool that
only resolved unified rows would fail for exactly the records a user is most
likely to ask about -- the ones that were linked or disputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from urbansense.data_integrity.conflicts import ConflictGroup
from urbansense.data_integrity.duplicates import DuplicateGroup
from urbansense.data_integrity.unified import ReconciledResult
from urbansense.schemas.observation import Observation
from urbansense.schemas.provenance import DocumentProvenance

#: Where a record ended up after reconciliation.
PLACEMENTS = ("unified", "duplicate_linked", "conflicted")


def origin_of(observation: Observation) -> str:
    """Describe where a value came from, whatever provenance shape it carries.

    A sensor reading's origin is its feed; a document-derived value's origin is
    the document, page and the text it was read from. Handling only the direct
    shape would fail on exactly the records whose origin is hardest to trace --
    which is the opposite of what this module is for (SPEC sections 7, 34).
    """
    provenance = observation.provenance
    if isinstance(provenance, DocumentProvenance):
        where = f" page {provenance.page}" if provenance.page is not None else ""
        if provenance.section:
            where += f" section {provenance.section}"
        return (
            f"{provenance.document_name}{where} "
            f"({provenance.extraction_kind.value}, "
            f"confidence {provenance.extraction_confidence:.0%})"
        )
    return provenance.origin


@dataclass(frozen=True, slots=True)
class ProvenanceChain:
    """The full origin story of one observation.

    Attributes:
        observation: The record asked about.
        placement: Which bucket it ended up in -- ``unified``,
            ``duplicate_linked`` or ``conflicted``.
        source_row: The feed row exactly as received, when retained.
        duplicate_group: The group it belongs to, if any.
        conflict_group: The disagreement it is part of, if any.
        representative: The logical observation it points at, when it is a
            duplicate.
        conflict_peers: The other records in its conflict, when it conflicts.
    """

    observation: Observation
    placement: str
    source_row: dict[str, str] | None = None
    duplicate_group: DuplicateGroup | None = None
    conflict_group: ConflictGroup | None = None
    representative: Observation | None = None
    conflict_peers: tuple[Observation, ...] = field(default_factory=tuple)

    @property
    def record_id(self) -> str:
        """The source's own reading id."""
        return str(self.observation.attributes.get("record_id", self.observation.observation_id))

    @property
    def is_unresolved_conflict(self) -> bool:
        """Whether this value is disputed and no decision has been made."""
        if self.conflict_group is None:
            return False
        from urbansense.schemas.enums import ReconciliationState

        return self.conflict_group.state is ReconciliationState.UNRESOLVED

    def as_dict(self) -> dict[str, Any]:
        """Serializable form."""
        observation = self.observation
        payload: dict[str, Any] = {
            "observation_id": observation.observation_id,
            "record_id": self.record_id,
            "placement": self.placement,
            "value": observation.value,
            "unit": observation.unit,
            "original_value": observation.original_value,
            "original_unit": observation.original_unit,
            "metric": observation.metric,
            "location_id": observation.location_id,
            "event_time": observation.event_time.isoformat(),
            "source_time": (
                None if observation.source_time is None else observation.source_time.isoformat()
            ),
            "received_time": observation.received_time.isoformat(),
            "source_id": observation.source_id,
            "source_type": observation.source_type.value,
            "validation_status": observation.validation_status.value,
            "provenance": {
                "origin": origin_of(observation),
                "ingested_at": observation.provenance.ingested_at.isoformat(),
                "pipeline_version": observation.provenance.pipeline_version,
                "notes": observation.provenance.notes,
            },
            "source_row": self.source_row,
        }
        if self.duplicate_group is not None:
            payload["duplicate_group"] = {
                "group_id": self.duplicate_group.group_id,
                "kind": self.duplicate_group.kind.value,
                "representative_id": self.duplicate_group.representative_id,
                "detail": self.duplicate_group.detail,
                "references": [
                    reference.as_dict() for reference in self.duplicate_group.references
                ],
            }
        if self.conflict_group is not None:
            payload["conflict"] = self.conflict_group.as_dict()
        return payload

    def describe(self) -> str:
        """Render the chain as readable text."""
        observation = self.observation
        value = "missing" if observation.value is None else f"{observation.value:g}"
        lines = [
            f"{self.record_id} ({observation.observation_id})",
            f"  value         {value} {observation.unit}",
        ]
        if observation.original_unit is not None:
            original = (
                "missing"
                if observation.original_value is None
                else f"{observation.original_value:g}"
            )
            lines.append(
                f"  as reported   {original} {observation.original_unit} (normalized above)"
            )
        lines.extend(
            [
                f"  measures      {observation.metric} at {observation.location_id}",
                f"  event_time    {observation.event_time.isoformat()}  (when it happened)",
                f"  source_time   "
                f"{'-' if observation.source_time is None else observation.source_time.isoformat()}"
                "  (what the source asserted)",
                f"  received_time {observation.received_time.isoformat()}  (when we were told)",
                f"  source        {observation.source_id} ({observation.source_type.value} stream)",
                f"  pipeline      {observation.provenance.pipeline_version}",
                f"  status        {observation.validation_status.value}",
                f"  placement     {self.placement}",
            ]
        )
        if observation.provenance.notes:
            lines.append(f"  provenance    {observation.provenance.notes}")
        if self.source_row is not None:
            lines.append(f"  source row    {self.source_row}")

        if self.duplicate_group is not None:
            duplicates = self.duplicate_group
            lines.append(f"  duplicate group {duplicates.group_id} ({duplicates.kind.value})")
            for reference in duplicates.references:
                marker = "representative" if reference.is_representative else "linked copy"
                reference_value = "missing" if reference.value is None else f"{reference.value:g}"
                lines.append(
                    f"    - {reference.record_id} from {reference.source_id} "
                    f"({reference.source_type}) = {reference_value}  [{marker}]"
                )
            lines.append(f"    {duplicates.detail}")

        if self.conflict_group is not None:
            conflict = self.conflict_group
            lines.append(
                f"  CONFLICT {conflict.conflict_id} "
                f"({conflict.state.value}, policy {conflict.policy.value})"
            )
            for candidate in conflict.candidates:
                candidate_value = "missing" if candidate.value is None else f"{candidate.value:g}"
                lines.append(
                    f"    - {candidate.record_id} from {candidate.source_id} "
                    f"({candidate.source_type}) = {candidate_value} {candidate.unit}"
                )
            lines.append("    no winner chosen; every candidate retained")

        return "\n".join(lines)


def _index(result: ReconciledResult) -> dict[str, tuple[Observation, str]]:
    """Index every observation by id and by the source's record id."""
    index: dict[str, tuple[Observation, str]] = {}
    for placement, observations in (
        ("unified", result.unified),
        ("duplicate_linked", result.duplicate_linked),
        ("conflicted", result.conflicted_observations),
    ):
        for observation in observations:
            index[observation.observation_id] = (observation, placement)
            record_id = observation.attributes.get("record_id")
            if isinstance(record_id, str):
                index.setdefault(record_id, (observation, placement))
    return index


def explain_provenance(identifier: str, result: ReconciledResult) -> ProvenanceChain | None:
    """Assemble the full origin story of one observation.

    Args:
        identifier: An ``observation_id`` or the source's own ``record_id``.
            Both are accepted because a user holding a feed export knows the
            latter, not the former.
        result: The reconciliation result to look in.

    Returns:
        The chain, or ``None`` when the identifier is not in this result.
    """
    found = _index(result).get(identifier)
    if found is None:
        return None
    observation, placement = found

    stored_row = observation.attributes.get("source_row")
    source_row = (
        {str(key): str(value) for key, value in stored_row.items()}
        if isinstance(stored_row, dict)
        else None
    )

    duplicate_group = next(
        (
            group
            for group in result.duplicate_groups
            if observation.observation_id == group.representative_id
            or observation.observation_id in group.duplicate_ids
        ),
        None,
    )
    conflict_group = next(
        (
            group
            for group in result.conflicts
            if observation.observation_id in group.observation_ids
        ),
        None,
    )

    representative: Observation | None = None
    if duplicate_group is not None and observation.duplicate_of is not None:
        representative = next(
            (
                candidate
                for candidate in result.unified
                if candidate.observation_id == duplicate_group.representative_id
            ),
            None,
        )

    peers: tuple[Observation, ...] = ()
    if conflict_group is not None:
        peers = tuple(
            candidate
            for candidate in result.conflicted_observations
            if candidate.observation_id in conflict_group.observation_ids
            and candidate.observation_id != observation.observation_id
        )

    return ProvenanceChain(
        observation=observation,
        placement=placement,
        source_row=source_row,
        duplicate_group=duplicate_group,
        conflict_group=conflict_group,
        representative=representative,
        conflict_peers=peers,
    )
