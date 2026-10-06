"""The pattern discovery entry point (SPEC sections 21, 22, 23).

Runs every analysis over the unified layer and returns one result carrying the
patterns, the interaction verdicts and the bookkeeping needed to judge them.

**Nothing here reads the planted answers.** No module in this package imports or
names ``ground_truth.json``; the AST-based source-tree test enforces it. Every
threshold is a stated default, chosen for the kind of effect worth reporting
rather than fitted to what the data happens to contain.

Order matters in one place: interactions are tested against the spatio-temporal
patterns that discovery actually *found*, so the interaction stage has no
independent knowledge of where to look. If the scan finds nothing, nothing is
tested, which is the correct behaviour rather than a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from urbansense.data_integrity.unified import ReconciledResult
from urbansense.patterns.conditional import discover_conditional
from urbansense.patterns.evolution import track_all
from urbansense.patterns.frame import ObservationFrame
from urbansense.patterns.interaction import discover_interactions
from urbansense.patterns.levelshift import discover_level_shifts
from urbansense.patterns.models import (
    DiscoveryStats,
    InteractionTest,
    InteractionVerdict,
    Pattern,
    PatternKind,
    PatternStatus,
)
from urbansense.patterns.spatiotemporal import (
    DEFAULT_FDR_LEVEL,
    DEFAULT_MIN_LOCALIZED_LIFT,
    discover_spatiotemporal,
)
from urbansense.patterns.statistics import DEFAULT_BOOTSTRAP_SEED
from urbansense.patterns.temporal import discover_hotspots, discover_profiles

#: Metric discovery runs over by default. Traffic only in v1.
DEFAULT_METRIC = "traffic_volume"


@dataclass(frozen=True)
class DiscoveryResult:
    """Everything one discovery run produced.

    Attributes:
        patterns: Discovered patterns, with evolution status attached.
        interactions: Interaction verdicts, one per candidate pair.
        stats: Bookkeeping -- tests run, FDR level, discoveries, seed.
        frame: The view discovery read, kept so a caller can inspect the same
            data the findings came from.
        generated_at: When the run happened.
    """

    patterns: tuple[Pattern, ...]
    interactions: tuple[InteractionTest, ...]
    stats: DiscoveryStats
    frame: ObservationFrame
    generated_at: datetime
    _by_kind: dict[PatternKind, tuple[Pattern, ...]] = field(default_factory=dict, repr=False)

    def of_kind(self, kind: PatternKind) -> tuple[Pattern, ...]:
        """Patterns of one kind."""
        cached = self._by_kind.get(kind)
        if cached is None:
            cached = tuple(p for p in self.patterns if p.kind is kind)
            self._by_kind[kind] = cached
        return cached

    def for_zone(self, location_id: str) -> tuple[Pattern, ...]:
        """Patterns scoped to one zone."""
        return tuple(p for p in self.patterns if p.scope.location_id == location_id)

    @property
    def by_status(self) -> dict[PatternStatus, int]:
        """Pattern counts per evolution status."""
        counts: dict[PatternStatus, int] = {}
        for pattern in self.patterns:
            counts[pattern.status] = counts.get(pattern.status, 0) + 1
        return dict(sorted(counts.items(), key=lambda pair: pair[0].value))

    @property
    def by_kind(self) -> dict[PatternKind, int]:
        """Pattern counts per kind."""
        counts: dict[PatternKind, int] = {}
        for pattern in self.patterns:
            counts[pattern.kind] = counts.get(pattern.kind, 0) + 1
        return dict(sorted(counts.items(), key=lambda pair: pair[0].value))

    @property
    def supported_interactions(self) -> tuple[InteractionTest, ...]:
        """Interactions the evidence actually supports.

        Separated from the rest because a list of "interactions tested" reads as
        a list of interactions found unless the verdicts are kept apart.
        """
        return tuple(
            item for item in self.interactions if item.verdict is InteractionVerdict.SUPPORTED
        )

    def header(self) -> str:
        """The multiple-testing header a reader needs before the findings.

        Printed before any pattern, because "one pattern found" means something
        very different out of ten hypotheses than out of 672.
        """
        return (
            f"{self.stats.discoveries} pattern(s) from {self.stats.tests_run} "
            f"hypotheses at FDR q={self.stats.fdr_level}; "
            f"{self.stats.cells_too_sparse} cell(s) too sparse to test; "
            f"{self.stats.source_rows:,} unified rows; seed "
            f"{self.stats.bootstrap_seed}"
        )


def discover_patterns(
    result: ReconciledResult,
    *,
    metric: str = DEFAULT_METRIC,
    timezone_offset_hours: float = 0.0,
    fdr_level: float = DEFAULT_FDR_LEVEL,
    min_localized_lift: float = DEFAULT_MIN_LOCALIZED_LIFT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    now: datetime | None = None,
    include_profiles: bool = True,
) -> DiscoveryResult:
    """Discover patterns over the unified layer.

    Args:
        result: Reconciliation output. Only ``result.unified`` is read -- the
            duplicate-linked and conflicted buckets are not reachable from here.
        metric: Metric to analyse.
        timezone_offset_hours: Source's local offset. Hours and weekdays are
            local, because "Friday evening" is a local-clock fact.
        fdr_level: Benjamini-Hochberg level for the spatio-temporal scan.
        min_localized_lift: Zone-specific lift a cell must show to be reported.
        seed: Bootstrap seed, so a run is reproducible and evolution can be
            tracked across runs.
        now: Timestamp for the run. Defaults to now.
        include_profiles: Whether to emit the descriptive hourly, weekday,
            seasonal and hotspot profiles alongside the tested findings.

    Returns:
        A :class:`DiscoveryResult`.
    """
    frame = ObservationFrame.from_result(result, timezone_offset_hours=timezone_offset_hours)

    spatiotemporal, tests_run, sparse = discover_spatiotemporal(
        frame,
        metric,
        fdr_level=fdr_level,
        min_localized_lift=min_localized_lift,
        seed=seed,
    )
    conditional = discover_conditional(frame, metric)
    level_shifts = discover_level_shifts(frame, metric, seed=seed)

    tested = [*spatiotemporal, *conditional, *level_shifts]
    descriptive = (
        [*discover_profiles(frame, metric), *discover_hotspots(frame, metric)]
        if include_profiles
        else []
    )

    # Evolution is tracked for every pattern, so a descriptive profile also
    # carries its history -- a diurnal shape that changed shape is itself a
    # finding worth seeing.
    patterns = track_all([*tested, *descriptive], frame)
    patterns.sort(key=lambda item: (item.kind.value, item.pattern_id))

    interactions = discover_interactions(frame, metric, list(spatiotemporal))

    return DiscoveryResult(
        patterns=tuple(patterns),
        interactions=tuple(interactions),
        stats=DiscoveryStats(
            tests_run=tests_run,
            fdr_level=fdr_level,
            # Counts only the hypothesis-tested findings. The descriptive
            # profiles were not part of the multiple-testing family, so folding
            # them into this number would overstate what survived correction.
            discoveries=len(tested),
            cells_too_sparse=sparse,
            bootstrap_seed=seed,
            source_rows=frame.source_rows,
        ),
        frame=frame,
        generated_at=now if now is not None else datetime.now(UTC),
    )
