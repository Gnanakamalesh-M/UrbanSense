"""Configuration layer (SPEC sections 3, 39).

Everything problem-specific lives in YAML so that adding flood or waste later is
a configuration change rather than a code change. Traffic is the only problem
enabled in v1, and nothing in the codebase special-cases it.
"""

from urbansense.config.loader import (
    ConfigError,
    available_problem_files,
    load_problem,
    load_settings,
)
from urbansense.config.models import (
    CongestionConfig,
    MetricConfig,
    PathsConfig,
    ProblemConfig,
    Settings,
    TemporalConfig,
    ValidationConfig,
)

__all__ = [
    "ConfigError",
    "CongestionConfig",
    "MetricConfig",
    "PathsConfig",
    "ProblemConfig",
    "Settings",
    "TemporalConfig",
    "ValidationConfig",
    "available_problem_files",
    "load_problem",
    "load_settings",
]
