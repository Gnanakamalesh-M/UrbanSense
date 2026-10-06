"""Conflict detection (SPEC section 10).

When two sources describe the same observation and disagree:

    Historical = 7900, Real-time = 8200, Document = 8100

the system records the disagreement and keeps every value. It does **not** pick
one. The spec is explicit on both halves:

    do not automatically overwrite one with another
    ...
    Never silently hide conflicts.

The reason is not squeamishness. "Real-time is fresher" and "historical is
authoritative" are both *policies*, and a policy that discards the values it
rejects destroys the only evidence that it was the wrong policy. So a conflict
is detected, recorded with all its candidates, and routed to review with its
state marked unresolved.

Note what is *not* a conflict. Two records only conflict if they are claims
about the same observation, which :class:`~urbansense.data_integrity.keys.ObservationKey`
defines: same time, place, metric, unit, duration and aggregation. A 15-minute
count and an hourly count at the same instant are different measurements and
both stand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from urbansense.data_integrity.keys import ObservationKey
from urbansense.schemas.enums import ReconciliationPolicy, ReconciliationState
from urbansense.schemas.observation import Observation

#: Relative difference at or under which two values are taken to agree.
DEFAULT_RELATIVE_TOLERANCE = 0.02

#: Absolute difference at or under which two values agree regardless of ratio.
#: Without a floor, two near-zero readings differing by a rounding step would
#: look like a 100% disagreement and manufacture conflicts out of quiet hours.
DEFAULT_ABSOLUTE_TOLERANCE = 0.5


def values_agree(
    left: float | None,
    right: float | None,
    *,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
) -> bool:
    """Whether two values are the same measurement within tolerance.

    Two missing values agree -- both say "nothing was measured here", which is
    a consistent claim. A missing value and a present one do **not** agree:
    that is a real disagreement about whether anything was observed, and
    treating it as agreement would let a gap silently absorb a reading.

    Args:
        left: First value, or ``None`` for missing.
        right: Second value, or ``None`` for missing.
        relative_tolerance: Permitted relative difference.
        absolute_tolerance: Permitted absolute difference, as a floor.

    Returns:
        ``True`` when the values agree.
    """
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False

    difference = abs(left - right)
    allowed = max(absolute_tolerance, relative_tolerance * max(abs(left), abs(right)))
    return difference <= allowed


@dataclass(frozen=True, slots=True)
class ConflictCandidate:
    """One side of a disagreement, kept in full.

    Attributes:
        observation_id: The record.
        record_id: The source's own reading id.
        source_id: Which source asserted this value.
        source_type: Which stream it arrived on, so a reviewer can see whether
            the dispute is historical-versus-real-time or within one stream.
        value: The asserted value. Retained verbatim -- the whole point is that
            no candidate is discarded.
        unit: Unit of the value.
        received_time: When this system was told.
    """

    observation_id: str
    record_id: str
    source_id: str
    source_type: str
    value: float | None
    unit: str
    received_time: datetime


@dataclass(frozen=True, slots=True)
class ConflictGroup:
    """A set of records that describe one observation and disagree.

    Attributes:
        conflict_id: Stable identifier for the group.
        key: The observation all members claim to measure.
        candidates: Every asserted value, with its source.
        observation_ids: Members, in deterministic order.
        policy: How this conflict is meant to be settled.
        state: Whether it has been settled. Always ``UNRESOLVED`` in this
            phase -- there is no automatic winner.
        detected_at: When the conflict was detected.
        cross_source: Whether the disagreement spans different sources. A
            single source disagreeing with itself at the same key is a
            different problem (a clock or pipeline fault) and worth
            distinguishing for the reviewer.
        detail: Human-readable summary.
    """

    conflict_id: str
    key: ObservationKey
    candidates: tuple[ConflictCandidate, ...]
    observation_ids: tuple[str, ...]
    policy: ReconciliationPolicy
    state: ReconciliationState
    detected_at: datetime
    cross_source: bool
    detail: str

    @property
    def spread(self) -> float | None:
        """Difference between the largest and smallest asserted value."""
        present = [c.value for c in self.candidates if c.value is not None]
        if len(present) < 2:
            return None
        return max(present) - min(present)

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Sources involved, sorted."""
        return tuple(sorted({c.source_id for c in self.candidates}))

    def as_dict(self) -> dict[str, object]:
        """Serializable form for the review queue."""
        return {
            "conflict_id": self.conflict_id,
            "key": self.key.as_dict(),
            "description": self.key.describe(),
            "policy": self.policy.value,
            "state": self.state.value,
            "resolved": self.state is ReconciliationState.RESOLVED,
            "detected_at": self.detected_at.isoformat(),
            "cross_source": self.cross_source,
            "spread": self.spread,
            "detail": self.detail,
            # Every candidate, with no winner marked. A reviewer decides.
            "candidates": [
                {
                    "observation_id": candidate.observation_id,
                    "record_id": candidate.record_id,
                    "source_id": candidate.source_id,
                    "source_type": candidate.source_type,
                    "value": candidate.value,
                    "unit": candidate.unit,
                    "received_time": candidate.received_time.isoformat(),
                }
                for candidate in self.candidates
            ],
        }


def group_has_conflict(
    members: list[Observation],
    *,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
) -> bool:
    """Whether any pair in the group materially disagrees.

    A single material disagreement makes the **whole** group a conflict, even
    when most members agree. Splitting it into "the agreeing majority plus a
    dissenter" would be picking a winner by vote, which SPEC section 10
    forbids -- and the majority is not evidence, since two feeds can share one
    upstream fault.
    """
    for index, left in enumerate(members):
        for right in members[index + 1 :]:
            if not values_agree(
                left.value,
                right.value,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            ):
                return True
    return False


def build_conflict_group(
    key: ObservationKey,
    members: list[Observation],
    *,
    detected_at: datetime,
    policy: ReconciliationPolicy = ReconciliationPolicy.MANUAL_REVIEW,
) -> ConflictGroup:
    """Record a disagreement, keeping every candidate value."""
    candidates = tuple(
        ConflictCandidate(
            observation_id=observation.observation_id,
            record_id=str(observation.attributes.get("record_id", observation.observation_id)),
            source_id=observation.source_id,
            source_type=observation.source_type.value,
            value=observation.value,
            unit=observation.unit,
            received_time=observation.received_time,
        )
        for observation in members
    )
    cross_source = len({candidate.source_id for candidate in candidates}) > 1

    rendered = ", ".join(
        f"{candidate.source_id}={'missing' if candidate.value is None else f'{candidate.value:g}'}"
        for candidate in candidates
    )
    scope = "different sources" if cross_source else "the same source"
    detail = (
        f"{len(candidates)} records from {scope} disagree about "
        f"{key.describe()}: {rendered}. All values retained; no winner chosen "
        f"({policy.value})."
    )

    return ConflictGroup(
        conflict_id=f"CFL-{candidates[0].record_id}",
        key=key,
        candidates=candidates,
        observation_ids=tuple(observation.observation_id for observation in members),
        policy=policy,
        state=ReconciliationState.UNRESOLVED,
        detected_at=detected_at,
        cross_source=cross_source,
        detail=detail,
    )
