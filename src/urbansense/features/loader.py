"""Loading a feature set from configuration (SPEC sections 3, 39).

Features are declared in YAML so the model's inputs are a reviewable artifact
rather than a list buried in code. The loader is strict, like every other config
loader in the project: an unknown calendar feature or a malformed block is an
error, never a silent default. A feature quietly dropped would train a model on
less than the config asked for, and the metrics would look fine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from urbansense.features.calendar import calendar_specs
from urbansense.features.specs import FeatureKind, FeatureSet, FeatureSpec

#: Default feature configuration shipped with the repository.
DEFAULT_FEATURES_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "features" / "traffic.yaml"
)


class FeatureConfigError(RuntimeError):
    """Raised when a feature configuration is missing or malformed."""


def load_feature_set(path: Path | None = None, *, horizon_hours: int | None = None) -> FeatureSet:
    """Load and validate a feature set.

    Args:
        path: YAML file to load. Defaults to the shipped traffic features.
        horizon_hours: Overrides the file's horizon, for training a second model
            at a different lead time without editing the config.

    Returns:
        A validated :class:`FeatureSet`.

    Raises:
        FeatureConfigError: if the file is absent, unparseable, not a mapping, or
            declares something invalid.
    """
    target = path if path is not None else DEFAULT_FEATURES_PATH

    if not target.is_file():
        raise FeatureConfigError(f"feature configuration not found: {target}")
    try:
        raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FeatureConfigError(f"{target} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise FeatureConfigError(
            f"{target} must contain a mapping at the top level, got "
            f"{type(raw).__name__ if raw is not None else 'nothing'}"
        )

    target_metric = raw.get("target_metric")
    if not isinstance(target_metric, str) or not target_metric:
        raise FeatureConfigError(f"{target}: 'target_metric' must be a non-empty string")

    horizon = horizon_hours if horizon_hours is not None else raw.get("horizon_hours")
    if not isinstance(horizon, int):
        raise FeatureConfigError(f"{target}: 'horizon_hours' must be an integer, got {horizon!r}")

    specs: list[FeatureSpec] = []

    try:
        specs.extend(calendar_specs(tuple(raw.get("calendar", []) or [])))
    except KeyError as exc:
        raise FeatureConfigError(f"{target}: {exc}") from exc

    for lag in raw.get("lags", []) or []:
        if not isinstance(lag, int):
            raise FeatureConfigError(f"{target}: lag {lag!r} must be an integer of hours")
        specs.append(
            FeatureSpec(
                name=f"lag_{lag}h",
                kind=FeatureKind.LAG,
                offset_hours=lag,
                metric=target_metric,
                description=f"target value {lag}h before the sample time",
            )
        )

    for lag in raw.get("seasonal_lags", []) or []:
        if not isinstance(lag, int):
            raise FeatureConfigError(f"{target}: seasonal lag {lag!r} must be an integer of hours")
        # Anchored to the TARGET time rather than the sample time: "the value at
        # this time last week". The offset from the sample time is therefore
        # lag - horizon, which stays positive for any sensible pairing. A lag
        # shorter than the horizon yields a negative offset and FeatureSpec
        # rejects it, which is correct -- that feature would read the future.
        specs.append(
            FeatureSpec(
                name=f"seasonal_lag_{lag}h",
                kind=FeatureKind.LAG,
                offset_hours=lag - horizon,
                metric=target_metric,
                description=(
                    f"target value {lag}h before the target time "
                    f"({lag - horizon}h before the sample time)"
                ),
            )
        )

    for window in raw.get("rolling_means", []) or []:
        if not isinstance(window, int):
            raise FeatureConfigError(
                f"{target}: rolling window {window!r} must be an integer of hours"
            )
        specs.append(
            FeatureSpec(
                name=f"roll_mean_{window}h",
                kind=FeatureKind.ROLLING,
                offset_hours=0,
                window_hours=window,
                metric=target_metric,
                description=f"mean target over the {window}h ending at the sample time",
            )
        )

    for entry in raw.get("covariates", []) or []:
        if not isinstance(entry, dict):
            raise FeatureConfigError(f"{target}: each covariate must be a mapping")
        try:
            metric = str(entry["metric"])
            location = str(entry["location"])
        except KeyError as exc:
            raise FeatureConfigError(
                f"{target}: a covariate needs 'metric' and 'location' ({exc})"
            ) from exc
        offset = int(entry.get("offset_hours", 0))
        window = int(entry.get("window_hours", 0))
        suffix = f"roll_{window}h" if window else f"{offset}h"
        specs.append(
            FeatureSpec(
                name=f"{metric}_{suffix}",
                kind=FeatureKind.COVARIATE,
                offset_hours=offset,
                window_hours=window,
                metric=metric,
                location=location,
                description=(
                    f"{metric} at {location}, "
                    + (f"mean over {window}h ending" if window else f"{offset}h before")
                    + " the sample time"
                ),
            )
        )

    if raw.get("include_zone", False):
        specs.append(
            FeatureSpec(
                name="zone",
                kind=FeatureKind.SPATIAL,
                description="categorical code for the location",
            )
        )

    try:
        return FeatureSet(specs=tuple(specs), target_metric=target_metric, horizon_hours=horizon)
    except ValueError as exc:
        raise FeatureConfigError(f"invalid feature configuration in {target}: {exc}") from exc
