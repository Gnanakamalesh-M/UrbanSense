"""Normalization of ingested observations (SPEC sections 11, 12, 13).

Schema, time, unit and location normalization. Both the original and the
normalized representation are retained -- values are never compared without
understanding their units, and an old report uploaded today stays historical.

**Normalization reshapes; it never judges.** Nothing here decides a record is
invalid or quarantines anything. A conversion that cannot be performed raises,
and the caller (the integrity pipeline) turns that into a quarantine record
with a stated reason. Keeping that line means a quarantined record is always
quarantined deliberately, rather than as a side effect of a failed conversion
somewhere in the middle of a transform.

The one thing normalization refuses to do is guess. An unknown unit is not
coerced to the canonical one, and a location that merely *resembles* a known
zone is not matched to it -- both become review items instead, because a wrong
guess here is invisible afterwards.
"""

from urbansense.preprocessing.loader import (
    DEFAULT_REGISTRY_PATH,
    LocationRegistryError,
    load_location_registry,
)
from urbansense.preprocessing.locations import (
    MAX_SUGGESTIONS,
    SUGGESTION_THRESHOLD,
    LocationMatch,
    LocationRegistry,
    MatchKind,
    ZoneEntry,
    normalize_name,
)
from urbansense.preprocessing.recency import (
    DEFAULT_CURRENT_WITHIN,
    DEFAULT_RECENT_WITHIN,
    arrival_lag,
    classify_recency,
    is_historical,
)
from urbansense.preprocessing.units import (
    Normalized,
    UnitError,
    are_compatible,
    canonicalize_unit,
    convert,
    dimension_of,
    is_known_unit,
    known_units,
    normalize_value,
)

__all__ = [
    "DEFAULT_CURRENT_WITHIN",
    "DEFAULT_RECENT_WITHIN",
    "DEFAULT_REGISTRY_PATH",
    "MAX_SUGGESTIONS",
    "SUGGESTION_THRESHOLD",
    "LocationMatch",
    "LocationRegistry",
    "LocationRegistryError",
    "MatchKind",
    "Normalized",
    "UnitError",
    "ZoneEntry",
    "are_compatible",
    "arrival_lag",
    "canonicalize_unit",
    "classify_recency",
    "convert",
    "dimension_of",
    "is_historical",
    "is_known_unit",
    "known_units",
    "load_location_registry",
    "normalize_name",
    "normalize_value",
]
