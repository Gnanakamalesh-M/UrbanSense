"""Duplicate detection and linking (SPEC section 9).

    Do not simply delete duplicates. Instead:
    Duplicate detected -> link related observations -> preserve source records
                       -> use one logical observation where appropriate

All four steps happen here, and the first is the one that constrains the rest.
Deleting the extra copy would be the obvious implementation and is exactly
wrong: if the duplicate judgement later turns out to be mistaken, there is
nothing left to recover. So every source record survives; one member is marked
the *logical* observation and the others point at it via ``duplicate_of``.

Two kinds of group are distinguished, because they mean different things:

**SAME_SOURCE_REPEAT** -- one source reported the same measurement twice. A
pipeline or retry artefact, and the second copy adds no information.

**CROSS_SOURCE_AGREEMENT** -- two sources independently reported the same
measurement and agree. That is *corroboration*, not redundancy: it is evidence
the value is right, and the unified row keeps both references so the
corroboration is not thrown away by the collapse.

Keeping them apart matters for scoring too. A detector that reported
corroboration as a duplicate repeat would inflate its own duplicate count with
records that were never duplicates in the sense section 9 means.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from urbansense.data_integrity.keys import ObservationKey
from urbansense.schemas.observation import Observation


class GroupKind(StrEnum):
    """Why a set of records collapses to one logical observation."""

    #: One source reported the same measurement more than once.
    SAME_SOURCE_REPEAT = "same_source_repeat"
    #: Independent sources reported the same measurement and agree.
    CROSS_SOURCE_AGREEMENT = "cross_source_agreement"


@dataclass(frozen=True, slots=True)
class SourceReference:
    """One source record behind a logical observation.

    Attributes:
        observation_id: The record.
        record_id: The source's own reading id.
        source_id: Which source produced it.
        source_type: Which stream it arrived on.
        value: Its value, retained so the collapse stays auditable.
        received_time: When this system was told.
        is_representative: Whether this is the record the unified layer carries.
    """

    observation_id: str
    record_id: str
    source_id: str
    source_type: str
    value: float | None
    received_time: datetime
    is_representative: bool

    def as_dict(self) -> dict[str, object]:
        """Serializable form, stored on the unified observation."""
        return {
            "observation_id": self.observation_id,
            "record_id": self.record_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "value": self.value,
            "received_time": self.received_time.isoformat(),
            "is_representative": self.is_representative,
        }


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """A set of records describing one observation, in agreement.

    Attributes:
        group_id: Stable identifier.
        kind: Why they collapse -- a repeat, or corroboration.
        key: The observation they all measure.
        representative_id: The record the unified layer carries.
        duplicate_ids: The other members, which keep their own records and
            point at the representative.
        references: Every member, with its source and value.
        detected_at: When the group was detected.
        detail: Human-readable summary.
    """

    group_id: str
    kind: GroupKind
    key: ObservationKey
    representative_id: str
    duplicate_ids: tuple[str, ...]
    references: tuple[SourceReference, ...]
    detected_at: datetime
    detail: str

    @property
    def size(self) -> int:
        """Number of source records in the group."""
        return len(self.references)

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Sources involved, sorted."""
        return tuple(sorted({reference.source_id for reference in self.references}))

    @property
    def source_types(self) -> tuple[str, ...]:
        """Streams involved, sorted."""
        return tuple(sorted({reference.source_type for reference in self.references}))


def choose_representative(members: list[Observation]) -> Observation:
    """Pick the logical observation for a group.

    The earliest record wins, ordered by ``(received_time, event_time,
    observation_id)``. Earliest-received is the natural choice -- the first
    report is the original and any later copy is the repeat -- but on real data
    the whole group often shares one ``received_time`` (it arrives in a single
    ingestion run), so the ``observation_id`` tie-break is what actually makes
    the choice deterministic rather than dependent on row order.

    The choice is not a judgement about which value is *correct*: every member
    agrees within tolerance, which is why the group is a duplicate group and
    not a conflict.
    """
    return min(members, key=lambda obs: (obs.received_time, obs.event_time, obs.observation_id))


def build_duplicate_group(
    key: ObservationKey,
    members: list[Observation],
    *,
    detected_at: datetime,
) -> DuplicateGroup:
    """Build the group record for a set of agreeing observations."""
    representative = choose_representative(members)
    source_ids = {observation.source_id for observation in members}
    kind = (
        GroupKind.SAME_SOURCE_REPEAT if len(source_ids) == 1 else GroupKind.CROSS_SOURCE_AGREEMENT
    )

    references = tuple(
        SourceReference(
            observation_id=observation.observation_id,
            record_id=str(observation.attributes.get("record_id", observation.observation_id)),
            source_id=observation.source_id,
            source_type=observation.source_type.value,
            value=observation.value,
            received_time=observation.received_time,
            is_representative=observation.observation_id == representative.observation_id,
        )
        for observation in members
    )

    representative_record = str(
        representative.attributes.get("record_id", representative.observation_id)
    )
    if kind is GroupKind.SAME_SOURCE_REPEAT:
        detail = (
            f"{len(members)} records from {representative.source_id} report the same "
            f"measurement for {key.describe()}. Collapsed to one logical observation; "
            "every source record retained and linked."
        )
    else:
        detail = (
            f"{len(members)} records from {', '.join(sorted(source_ids))} independently "
            f"agree on {key.describe()}. Corroboration, not redundancy: collapsed to one "
            "logical observation with all source references retained."
        )

    return DuplicateGroup(
        group_id=f"DUP-{representative_record}",
        kind=kind,
        key=key,
        representative_id=representative.observation_id,
        duplicate_ids=tuple(
            observation.observation_id
            for observation in members
            if observation.observation_id != representative.observation_id
        ),
        references=references,
        detected_at=detected_at,
        detail=detail,
    )
