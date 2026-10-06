"""Pattern discovery and evolution (SPEC sections 21, 22, 23).

    UrbanSense must discover patterns rather than only predict predefined
    targets.

So this package *finds* regularities rather than being told where to look. It
scans the unified layer, decides what is real, and reports strength with its
evidence attached.

Four properties hold throughout:

**Nothing here reads the planted answers.** No module imports or names
``ground_truth.json``, enforced by an AST-based source-tree test. Every threshold
is a stated default chosen for the size of effect worth reporting, not fitted to
what the data happens to contain.

**Effect sizes, not just p-values.** With thousands of hours per zone a 1.01x
difference is significant and worthless, so strength and certainty stay separate
numbers.

**Confounds are controlled and named.** Each pattern records what was held
constant -- and, as importantly, what was not. The spatio-temporal scan holds hour
and zone fixed and divides out the weekday effect every zone shares; the
level-shift detector divides out the seasonal movement the whole city shares.

**Multiple testing is corrected.** Scanning 672 cells at p<0.05 produces about 34
false positives by construction, so Benjamini-Hochberg controls the expected
share of false discoveries and the test count is reported beside the findings.
"""

from urbansense.patterns.conditional import (
    ConditionalFit,
    discover_conditional,
    fit_rainfall_effect,
)
from urbansense.patterns.discovery import (
    DEFAULT_METRIC,
    DiscoveryResult,
    discover_patterns,
)
from urbansense.patterns.evolution import (
    EvolutionSummary,
    classify,
    track_all,
    track_evolution,
)
from urbansense.patterns.frame import (
    CITY_LOCATION,
    RAINFALL_METRIC,
    WET_THRESHOLD_MM,
    ObservationFrame,
    Reading,
)
from urbansense.patterns.interaction import test_interaction
from urbansense.patterns.levelshift import (
    ShiftCandidate,
    discover_level_shifts,
    scan_change_points,
)
from urbansense.patterns.models import (
    DiscoveryStats,
    EffectSize,
    Evidence,
    InteractionTest,
    InteractionVerdict,
    Pattern,
    PatternKind,
    PatternScope,
    PatternStatus,
    WindowStrength,
)
from urbansense.patterns.spatiotemporal import (
    DEFAULT_FDR_LEVEL,
    DEFAULT_MIN_LOCALIZED_LIFT,
    discover_spatiotemporal,
)
from urbansense.patterns.statistics import (
    DEFAULT_BOOTSTRAP_SEED,
    ComparisonResult,
    SlopeResult,
    TrendResult,
    adjusted_p_values,
    benjamini_hochberg,
    compare_groups,
    fit_slope,
    median_ratio,
    test_trend,
)
from urbansense.patterns.store import DEFAULT_PATTERN_DIR, PatternStore
from urbansense.patterns.temporal import discover_hotspots, discover_profiles

__all__ = [
    "CITY_LOCATION",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_FDR_LEVEL",
    "DEFAULT_METRIC",
    "DEFAULT_MIN_LOCALIZED_LIFT",
    "DEFAULT_PATTERN_DIR",
    "RAINFALL_METRIC",
    "WET_THRESHOLD_MM",
    "ComparisonResult",
    "ConditionalFit",
    "DiscoveryResult",
    "DiscoveryStats",
    "EffectSize",
    "Evidence",
    "EvolutionSummary",
    "InteractionTest",
    "InteractionVerdict",
    "ObservationFrame",
    "Pattern",
    "PatternKind",
    "PatternScope",
    "PatternStatus",
    "PatternStore",
    "Reading",
    "ShiftCandidate",
    "SlopeResult",
    "TrendResult",
    "WindowStrength",
    "adjusted_p_values",
    "benjamini_hochberg",
    "classify",
    "compare_groups",
    "discover_conditional",
    "discover_hotspots",
    "discover_level_shifts",
    "discover_patterns",
    "discover_profiles",
    "discover_spatiotemporal",
    "fit_rainfall_effect",
    "fit_slope",
    "median_ratio",
    "scan_change_points",
    "test_interaction",
    "test_trend",
    "track_all",
    "track_evolution",
]
