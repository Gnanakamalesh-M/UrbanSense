"""Data integrity engine (SPEC sections 8-10, 14-19).

Validation, sensor-failure detection, outlier detection, source health, the
quarantine workflow, and -- from Phase 3 -- duplicate and conflict handling,
reconciliation and the unified data layer. Where
:mod:`urbansense.preprocessing` reshapes, this package *judges*, and the
division matters: a record is set aside here for a stated reason, never as a
side effect of a transform.

Four rules shape everything in it:

**Nothing questionable is destroyed.** Invalid values are not clipped, outliers
are not corrected, gaps are not filled, duplicates are not deleted and
conflicts are not overwritten. Records that cannot be used are written to
``quarantine/`` with their payload exactly as received (SPEC section 18).

**Unusual is not invalid.** An impossible value (negative traffic, 480 km/h) is
not a measurement and is quarantined. A merely surprising value could be a
festival, an accident or a diversion, so it is flagged ``SUSPECT`` and **kept**
in the usable stream for investigation (SPEC section 16).

**Disagreement is information.** When two sources contradict each other the
system records the dispute with every candidate value and routes it to review.
It does not pick a winner: "real-time is fresher" and "historical is
authoritative" are both policies, and a policy that discards the values it
rejects destroys the evidence that it was the wrong policy (SPEC section 10).

**Nothing is lost silently.** Validation partitions every input row into usable
or quarantined; reconciliation partitions every usable observation into
unified, duplicate-linked or conflicted. Both partitions are asserted by
``accounts_for_*`` properties on their results.
"""

from urbansense.data_integrity.conflicts import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
    ConflictCandidate,
    ConflictGroup,
    build_conflict_group,
    group_has_conflict,
    values_agree,
)
from urbansense.data_integrity.duplicates import (
    DuplicateGroup,
    GroupKind,
    SourceReference,
    build_duplicate_group,
    choose_representative,
)
from urbansense.data_integrity.keys import ObservationKey, group_by_key
from urbansense.data_integrity.outliers import (
    DEFAULT_MIN_GROUP_SIZE,
    DEFAULT_THRESHOLD,
    OutlierFlag,
    detect_outliers,
)
from urbansense.data_integrity.pipeline import validate_and_normalize
from urbansense.data_integrity.provenance import ProvenanceChain, explain_provenance
from urbansense.data_integrity.quarantine import (
    CONFLICTS_SUBDIR,
    REVIEW_SUBDIR,
    QuarantineWriter,
)
from urbansense.data_integrity.ranges import RangeViolation, check_range
from urbansense.data_integrity.reconcile import reconcile
from urbansense.data_integrity.result import ValidatedResult, ValidationStats
from urbansense.data_integrity.sensors import (
    DEFAULT_FAILED_GAP,
    DEFAULT_IGNORED_FROZEN_VALUES,
    DEFAULT_MIN_FROZEN_RUN,
    DEFAULT_WARNING_GAP,
    DEFAULT_WARNING_MISSING_RATE,
    FrozenRun,
    SourceHealth,
    assess_source_health,
    detect_frozen_runs,
)
from urbansense.data_integrity.unified import ReconciledResult, ReconciliationStats

__all__ = [
    "CONFLICTS_SUBDIR",
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_FAILED_GAP",
    "DEFAULT_IGNORED_FROZEN_VALUES",
    "DEFAULT_MIN_FROZEN_RUN",
    "DEFAULT_MIN_GROUP_SIZE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WARNING_GAP",
    "DEFAULT_WARNING_MISSING_RATE",
    "REVIEW_SUBDIR",
    "ConflictCandidate",
    "ConflictGroup",
    "DuplicateGroup",
    "FrozenRun",
    "GroupKind",
    "ObservationKey",
    "OutlierFlag",
    "ProvenanceChain",
    "QuarantineWriter",
    "RangeViolation",
    "ReconciledResult",
    "ReconciliationStats",
    "SourceHealth",
    "SourceReference",
    "ValidatedResult",
    "ValidationStats",
    "assess_source_health",
    "build_conflict_group",
    "build_duplicate_group",
    "check_range",
    "choose_representative",
    "detect_frozen_runs",
    "detect_outliers",
    "explain_provenance",
    "group_by_key",
    "group_has_conflict",
    "reconcile",
    "validate_and_normalize",
    "values_agree",
]
