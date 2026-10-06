"""Feature engineering for the prediction engine (SPEC sections 21, 25, 33).

Every feature must be computable from information available *strictly before*
the prediction target time, and that is enforced two independent ways rather
than trusted:

**Declared.** Each :class:`~urbansense.features.specs.FeatureSpec` states how far
back it reads, and a negative offset raises at construction. A forward-looking
feature cannot be declared through the normal path.

**Probed.** :func:`~urbansense.features.leakage.detect_leakage` perturbs the data
a feature should not be able to see, rebuilds the feature, and flags any value
that moved. Declarations describe intent; the probe measures behaviour, and
catches a hand-rolled feature that ignores its own declaration.

Two further rules, inherited from the earlier phases:

**Only the unified layer.** :func:`~urbansense.features.builder.build_feature_table`
takes a ``ReconciledResult`` and reads ``result.unified`` alone. Quarantined,
duplicate-linked and conflicted records are not reachable from its arguments, so
they cannot enter training by mistake.

**Missing stays missing.** An unknown feature is ``NaN``, never zero and never
imputed. Lags are looked up by timestamp rather than row position, because the
data has real holes where quarantined records were removed and a positional shift
would silently read across them.
"""

from urbansense.features.builder import (
    DEFAULT_MIN_WINDOW_OBSERVATIONS,
    FeatureTable,
    build_feature_table,
    compute_features,
)
from urbansense.features.calendar import (
    CALENDAR_FEATURES,
    WEEKEND_DAYS,
    calendar_specs,
    compute_calendar,
    to_local,
)
from urbansense.features.leakage import (
    DEFAULT_PROBE_SAMPLES,
    ClauseStatus,
    LeakageError,
    LeakageFinding,
    LeakageKind,
    LeakageReport,
    assert_no_leakage,
    detect_leakage,
    values_match,
)
from urbansense.features.loader import (
    DEFAULT_FEATURES_PATH,
    FeatureConfigError,
    load_feature_set,
)
from urbansense.features.series import SeriesIndex
from urbansense.features.specs import FeatureKind, FeatureSet, FeatureSpec

__all__ = [
    "CALENDAR_FEATURES",
    "DEFAULT_FEATURES_PATH",
    "DEFAULT_MIN_WINDOW_OBSERVATIONS",
    "DEFAULT_PROBE_SAMPLES",
    "WEEKEND_DAYS",
    "ClauseStatus",
    "FeatureConfigError",
    "FeatureKind",
    "FeatureSet",
    "FeatureSpec",
    "FeatureTable",
    "LeakageError",
    "LeakageFinding",
    "LeakageKind",
    "LeakageReport",
    "SeriesIndex",
    "assert_no_leakage",
    "build_feature_table",
    "calendar_specs",
    "compute_calendar",
    "compute_features",
    "detect_leakage",
    "load_feature_set",
    "to_local",
    "values_match",
]
