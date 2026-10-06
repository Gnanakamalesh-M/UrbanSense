"""The reconciled result and the unified data layer (SPEC sections 19, 20).

    Only after individual sources are validated and reconciled should they be
    combined. The unified layer must retain source references.

Both halves are enforced here. :class:`ReconciledResult` is produced by
reconciliation, so nothing reaches ``unified`` before being reconciled, and
every unified row carries ``source_references`` naming each source record
behind it. A combined figure a user cannot decompose into its contributing
streams is a figure they cannot audit.

Section 20 adds an ordering requirement -- "Always provide separate views
before presenting the unified result" -- so the source-specific views are
first-class properties beside the unified one rather than something a caller
has to reconstruct with a filter.

The partition is a three-way split of the input, and
:attr:`ReconciledResult.accounts_for_every_observation` asserts it: every valid
observation is in ``unified``, or linked as a duplicate, or inside a conflict
group. Nothing synthetic is added to any bucket, which is what keeps the
invariant a simple equality rather than a bookkeeping exercise.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from urbansense.data_integrity.conflicts import ConflictGroup
from urbansense.data_integrity.duplicates import DuplicateGroup, GroupKind
from urbansense.schemas.enums import SourceType
from urbansense.schemas.observation import Observation


class ReconciliationStats(BaseModel):
    """Counters for one reconciliation run.

    Attributes:
        observations_in: Valid observations received from validation, so the
            invariant is checked against that stage's output rather than
            against our own bookkeeping.
        unified: Logical observations in the unified layer.
        duplicate_linked: Records linked to a representative and kept.
        conflicted: Records inside unresolved conflict groups.
        duplicate_groups: Groups that collapsed to one logical observation.
        same_source_repeats: Groups where one source reported twice.
        cross_source_agreements: Groups where independent sources corroborated.
        conflict_groups: Disagreements recorded for review.
        source_path: File the data came from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    observations_in: int = 0
    unified: int = 0
    duplicate_linked: int = 0
    conflicted: int = 0
    duplicate_groups: int = 0
    same_source_repeats: int = 0
    cross_source_agreements: int = 0
    conflict_groups: int = 0
    source_path: Path | None = None


class ReconciledResult(BaseModel):
    """Everything one reconciliation run produced.

    Attributes:
        unified: The unified data layer -- one logical observation per
            phenomenon, each retaining its source references. Excludes
            unresolved conflicts, so a disputed slot is a visible hole rather
            than a guess.
        duplicate_linked: Duplicate records, retained with ``duplicate_of``
            set. Preserved deliberately: a mistaken duplicate judgement must be
            recoverable (SPEC section 9).
        conflicts: Recorded disagreements, every candidate value kept.
        duplicate_groups: The groups that collapsed.
        stats: Counters for the run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    unified: tuple[Observation, ...] = ()
    duplicate_linked: tuple[Observation, ...] = ()
    conflicts: tuple[ConflictGroup, ...] = ()
    duplicate_groups: tuple[DuplicateGroup, ...] = ()
    conflicted_observations: tuple[Observation, ...] = ()
    stats: ReconciliationStats = Field(default_factory=ReconciliationStats)

    # ------------------------------------------------------------------
    # The invariant
    # ------------------------------------------------------------------

    @property
    def accounts_for_every_observation(self) -> bool:
        """Whether every input observation is in exactly one place.

        If this is false, records have been lost between validation and
        reconciliation, which the project forbids outright.
        """
        return self.stats.observations_in == (
            len(self.unified) + len(self.duplicate_linked) + len(self.conflicted_observations)
        )

    @property
    def observation_ids_are_disjoint(self) -> bool:
        """Whether the three buckets share no observation.

        The count matching is necessary but not sufficient: a record appearing
        in two buckets while another vanished would still add up.
        """
        unified = {obs.observation_id for obs in self.unified}
        linked = {obs.observation_id for obs in self.duplicate_linked}
        conflicted = {obs.observation_id for obs in self.conflicted_observations}
        total = len(unified) + len(linked) + len(conflicted)
        return len(unified | linked | conflicted) == total

    # ------------------------------------------------------------------
    # Source-specific views come first (SPEC section 20)
    # ------------------------------------------------------------------

    def source_view(self, source_type: SourceType) -> tuple[Observation, ...]:
        """Unified rows whose representative arrived on one stream."""
        return tuple(obs for obs in self.unified if obs.source_type is source_type)

    @property
    def historical_view(self) -> tuple[Observation, ...]:
        """Historical observations (SPEC section 20)."""
        return self.source_view(SourceType.HISTORICAL)

    @property
    def real_time_view(self) -> tuple[Observation, ...]:
        """Current observations (SPEC section 20)."""
        return self.source_view(SourceType.REAL_TIME)

    @property
    def document_view(self) -> tuple[Observation, ...]:
        """Document-derived observations (SPEC section 20).

        Empty until document ingestion lands; the view exists now so the
        unified layer's shape does not change when it does.
        """
        return self.source_view(SourceType.DOCUMENT)

    @property
    def by_source_type(self) -> dict[SourceType, int]:
        """Unified row counts per contributing stream."""
        counts = Counter(obs.source_type for obs in self.unified)
        return dict(sorted(counts.items(), key=lambda pair: pair[0].value))

    # ------------------------------------------------------------------
    # Group reporting
    # ------------------------------------------------------------------

    @property
    def by_group_kind(self) -> dict[GroupKind, int]:
        """Duplicate group counts per kind.

        Reported separately so corroboration can never be counted as a
        same-source repeat, which would inflate the duplicate score with
        records that were never duplicates in section 9's sense.
        """
        counts = Counter(group.kind for group in self.duplicate_groups)
        return {kind: counts.get(kind, 0) for kind in GroupKind}

    @property
    def same_source_groups(self) -> tuple[DuplicateGroup, ...]:
        """Groups where one source reported the same measurement twice."""
        return tuple(
            group for group in self.duplicate_groups if group.kind is GroupKind.SAME_SOURCE_REPEAT
        )

    @property
    def cross_source_groups(self) -> tuple[DuplicateGroup, ...]:
        """Groups where independent sources corroborated each other."""
        return tuple(
            group
            for group in self.duplicate_groups
            if group.kind is GroupKind.CROSS_SOURCE_AGREEMENT
        )

    @property
    def unresolved_conflicts(self) -> tuple[ConflictGroup, ...]:
        """Conflicts still awaiting a human decision."""
        from urbansense.schemas.enums import ReconciliationState

        return tuple(
            group for group in self.conflicts if group.state is ReconciliationState.UNRESOLVED
        )

    @property
    def cross_source_conflicts(self) -> tuple[ConflictGroup, ...]:
        """Conflicts where the disagreement spans different sources."""
        return tuple(group for group in self.conflicts if group.cross_source)

    def __len__(self) -> int:
        """Number of logical observations in the unified layer."""
        return len(self.unified)
