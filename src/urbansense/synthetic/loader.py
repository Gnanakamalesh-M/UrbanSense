"""Loading generator configuration from YAML (SPEC sections 3, 39).

Kept separate from :mod:`urbansense.synthetic.config` so the models stay
importable without any filesystem concern, and mirrors the strict loading in
:mod:`urbansense.config.loader`: a malformed or unknown key is an error, never
a silent default. A generator config that quietly dropped its Friday-evening
block would produce a dataset with no pattern in it, and every later phase's
test would then be asserting against nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from urbansense.synthetic.config import GeneratorConfig

#: Default generator config shipped with the repository.
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "synthetic" / "traffic_demo.yaml"
)


class GeneratorConfigError(RuntimeError):
    """Raised when a generator configuration is missing or malformed."""


def load_generator_config(path: Path | None = None, *, seed: int | None = None) -> GeneratorConfig:
    """Load and validate a generator configuration.

    Args:
        path: YAML file to load. Defaults to the shipped demo config.
        seed: Overrides the file's seed, for generating alternative datasets
            without editing the config.

    Returns:
        A validated :class:`GeneratorConfig`.

    Raises:
        GeneratorConfigError: if the file is absent, unparseable, not a mapping,
            or fails validation.
    """
    target = path if path is not None else DEFAULT_CONFIG_PATH

    if not target.is_file():
        raise GeneratorConfigError(f"generator configuration not found: {target}")
    try:
        raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GeneratorConfigError(f"{target} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise GeneratorConfigError(
            f"{target} must contain a mapping at the top level, got "
            f"{type(raw).__name__ if raw is not None else 'nothing'}"
        )

    if seed is not None:
        raw["seed"] = seed

    try:
        return GeneratorConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError and friends
        raise GeneratorConfigError(f"invalid generator configuration in {target}: {exc}") from exc
