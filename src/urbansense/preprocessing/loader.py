"""Loading the location registry from configuration (SPEC sections 13, 39).

Mirrors the strict loading in :mod:`urbansense.config.loader`: a missing file,
a malformed document or an ambiguous alias is an error, never a silent default.
An empty registry that loaded "successfully" would send every row to review and
look like a data problem rather than a configuration one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from urbansense.preprocessing.locations import LocationRegistry

#: Default registry shipped with the repository.
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "locations" / "chennai_zones.yaml"
)


class LocationRegistryError(RuntimeError):
    """Raised when the location registry is missing, malformed or ambiguous."""


def load_location_registry(path: Path | None = None) -> LocationRegistry:
    """Load and validate the canonical location registry.

    Args:
        path: YAML file to load. Defaults to the shipped registry.

    Returns:
        A validated :class:`LocationRegistry`.

    Raises:
        LocationRegistryError: if the file is absent, unparseable, not a
            mapping, declares no zones, or contains an ambiguous alias.
    """
    target = path if path is not None else DEFAULT_REGISTRY_PATH

    if not target.is_file():
        raise LocationRegistryError(f"location registry not found: {target}")
    try:
        raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LocationRegistryError(f"{target} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise LocationRegistryError(
            f"{target} must contain a mapping at the top level, got "
            f"{type(raw).__name__ if raw is not None else 'nothing'}"
        )

    try:
        registry = LocationRegistry.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError and friends
        raise LocationRegistryError(f"invalid location registry in {target}: {exc}") from exc

    if not registry.zones:
        raise LocationRegistryError(
            f"{target} declares no zones; an empty registry would send every "
            "observation to location review"
        )
    return registry
