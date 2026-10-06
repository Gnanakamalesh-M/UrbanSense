"""YAML configuration loading (SPEC sections 3, 39).

``base.yaml`` holds run-wide settings and names which problem files to load.
Each problem lives in its own file under ``configs/problems/``. Loading is
strict: a named problem file that is missing, or a problem file whose declared
type disagrees with its filename, is an error. Silent skipping would mean a run
quietly covering less than the operator thinks it does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from urbansense.config.models import ProblemConfig, Settings
from urbansense.schemas.enums import ProblemType

#: Default location of the configuration directory, relative to the repo root.
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

#: Files in ``configs/problems/`` starting with this prefix are templates and
#: documentation, never loaded as active problems.
TEMPLATE_PREFIX = "_"


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed or inconsistent."""


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``.

    Raises:
        ConfigError: if the file is absent, unparseable, or not a mapping.
    """
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} must contain a mapping at the top level, got {type(raw).__name__}"
        )
    return raw


def load_problem(path: Path) -> ProblemConfig:
    """Load and validate a single problem configuration file.

    Raises:
        ConfigError: if the file is malformed, or if its ``problem_type`` does
            not match its filename -- a mismatch means one of the two is a typo,
            and guessing which would silently configure the wrong domain.
    """
    data = _read_yaml(path)
    try:
        problem = ProblemConfig.model_validate(data)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(f"invalid problem configuration in {path}: {exc}") from exc

    expected = path.stem
    if problem.problem_type.value != expected:
        raise ConfigError(
            f"{path}: declares problem_type={problem.problem_type.value!r} but is named "
            f"{expected!r}. Rename the file or fix the field -- one of them is a typo."
        )
    return problem


def load_settings(
    config_dir: Path | None = None,
    *,
    env: str | None = None,
    base_filename: str = "base.yaml",
) -> Settings:
    """Load ``base.yaml`` plus every problem it enables.

    Args:
        config_dir: Directory holding ``base.yaml`` and ``problems/``. Defaults
            to the repo's ``configs/``.
        env: Overrides the ``env`` field from the file, for CI or tests.
        base_filename: Name of the base file, for alternative environments.

    Returns:
        A validated :class:`Settings`.

    Raises:
        ConfigError: on any missing file, malformed document, unknown problem
            name, or inconsistent problem declaration.
    """
    directory = config_dir if config_dir is not None else DEFAULT_CONFIG_DIR
    base = _read_yaml(directory / base_filename)

    requested = base.pop("problems", []) or []
    if not isinstance(requested, list) or not all(isinstance(name, str) for name in requested):
        raise ConfigError(
            f"{directory / base_filename}: 'problems' must be a list of problem names, "
            f"e.g. ['traffic']."
        )

    # Resolve and check the whole list before touching the filesystem, so a
    # malformed list is reported as such rather than as a missing file.
    problem_types: list[ProblemType] = []
    for name in requested:
        try:
            problem_type = ProblemType(name)
        except ValueError as exc:
            known = ", ".join(sorted(member.value for member in ProblemType))
            raise ConfigError(
                f"unknown problem {name!r} requested in {base_filename}; known types: {known}."
            ) from exc
        if problem_type in problem_types:
            raise ConfigError(f"problem {name!r} is listed more than once in {base_filename}.")
        problem_types.append(problem_type)

    problems: dict[ProblemType, ProblemConfig] = {
        problem_type: load_problem(directory / "problems" / f"{problem_type.value}.yaml")
        for problem_type in problem_types
    }

    if env is not None:
        base["env"] = env

    try:
        return Settings.model_validate({**base, "problems": problems})
    except Exception as exc:
        raise ConfigError(f"invalid configuration in {directory / base_filename}: {exc}") from exc


def available_problem_files(config_dir: Path | None = None) -> tuple[Path, ...]:
    """List loadable problem files, excluding templates.

    Useful for discovering what *could* be enabled without enabling it.
    """
    directory = (config_dir if config_dir is not None else DEFAULT_CONFIG_DIR) / "problems"
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(p for p in directory.glob("*.yaml") if not p.name.startswith(TEMPLATE_PREFIX))
    )
