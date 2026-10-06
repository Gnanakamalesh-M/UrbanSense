"""Grouping model columns into readable factors (SPEC section 27).

The model has nineteen columns. A person asking why a prediction is high wants
six factors, not a ranked list of lag offsets -- section 27's own example shows
"Historical traffic 38%, Rainfall 24%, Peak hour 19%", not ``seasonal_lag_24h``.

The mapping lives in configuration so that adding a feature forces a decision
about how to explain it. The loader **rejects a feature that matches no group**,
and one that matches two: a silently ungrouped feature would simply vanish from
every explanation while still moving every prediction, which is the most
dangerous way for this layer to be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

from urbansense.features.loader import DEFAULT_FEATURES_PATH, FeatureConfigError


@dataclass(frozen=True)
class FactorGroups:
    """Feature-name patterns grouped under readable factor names.

    Attributes:
        patterns: Factor name to the glob patterns it claims.
    """

    patterns: dict[str, tuple[str, ...]]

    @property
    def names(self) -> tuple[str, ...]:
        """Factor names, in configuration order."""
        return tuple(self.patterns)

    def factor_for(self, feature: str) -> str:
        """The factor a feature belongs to.

        Raises:
            FeatureConfigError: if the feature matches no factor or several. An
                unmapped feature would disappear from explanations while still
                influencing predictions, and an ambiguous one would be counted
                under whichever group happened to be iterated first.
        """
        matched = [
            name
            for name, globs in self.patterns.items()
            if any(fnmatch(feature, pattern) for pattern in globs)
        ]
        if not matched:
            raise FeatureConfigError(
                f"feature {feature!r} belongs to no factor group. Add it to the "
                "'factors' block in the feature configuration -- an ungrouped "
                "feature would move predictions without ever appearing in an "
                "explanation."
            )
        if len(matched) > 1:
            raise FeatureConfigError(
                f"feature {feature!r} matches several factor groups {matched}; "
                "the groups must partition the features."
            )
        return matched[0]

    def assign(self, features: tuple[str, ...]) -> dict[str, str]:
        """Map every feature to its factor, validating the whole set at once."""
        return {feature: self.factor_for(feature) for feature in features}

    def grouped(self, features: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        """Factor name to the features it covers, factors with none omitted."""
        assignment = self.assign(features)
        out: dict[str, list[str]] = {name: [] for name in self.patterns}
        for feature, factor in assignment.items():
            out[factor].append(feature)
        return {name: tuple(items) for name, items in out.items() if items}


def load_factor_groups(path: Path | None = None) -> FactorGroups:
    """Load the factor grouping from the feature configuration.

    Strict like every other loader in the project: a malformed block is an
    error, never a silent default.

    Raises:
        FeatureConfigError: if the file is absent, unparseable, or the
            ``factors`` block is missing or malformed.
    """
    target = path if path is not None else DEFAULT_FEATURES_PATH
    if not target.is_file():
        raise FeatureConfigError(f"feature configuration not found: {target}")
    try:
        raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FeatureConfigError(f"{target} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise FeatureConfigError(f"{target} must contain a mapping at the top level")

    block = raw.get("factors")
    if not isinstance(block, dict) or not block:
        raise FeatureConfigError(
            f"{target}: a non-empty 'factors' mapping is required so every feature "
            "can be explained under a readable name"
        )

    patterns: dict[str, tuple[str, ...]] = {}
    for name, globs in block.items():
        if not isinstance(name, str) or not name:
            raise FeatureConfigError(f"{target}: factor names must be non-empty strings")
        if not isinstance(globs, list) or not globs:
            raise FeatureConfigError(
                f"{target}: factor {name!r} must list at least one feature pattern"
            )
        for pattern in globs:
            if not isinstance(pattern, str) or not pattern:
                raise FeatureConfigError(
                    f"{target}: factor {name!r} has a non-string pattern {pattern!r}"
                )
        patterns[name] = tuple(globs)

    return FactorGroups(patterns=patterns)
