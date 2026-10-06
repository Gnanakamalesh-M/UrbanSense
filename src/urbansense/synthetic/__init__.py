"""Synthetic data generation with known ground truth (SPEC sections 37, 38).

This package exists so later phases can be *scored*, not merely demonstrated. A
phase claiming to discover a Friday congestion pattern, detect a drift event or
quarantine an invalid record is only testable against data where the answer was
planted on purpose -- so every injected effect is recorded in
``ground_truth.json`` beside the data, with its parameters and the exact record
ids involved.

The generated CSV carries only what a real source would send. Labels live in the
ground-truth file alone: a label column in the data would be a leakage channel,
and a phase that could read the answer off its input would be credited with a
discovery it never made.
"""

from urbansense.synthetic.config import (
    DriftConfig,
    FridayEveningConfig,
    GeneratorConfig,
    IntegrityInjectionConfig,
    MissingDataConfig,
    MissingWindowConfig,
    RainEffectConfig,
    SensorFailureConfig,
    ZoneConfig,
)
from urbansense.synthetic.generator import (
    GENERATOR_VERSION,
    GROUND_TRUTH_FILENAME,
    HISTORY_FILENAME,
    HOLDOUT_FILENAME,
    GeneratedDataset,
    generate,
    generate_to_directory,
)
from urbansense.synthetic.ground_truth import GroundTruth
from urbansense.synthetic.loader import (
    DEFAULT_CONFIG_PATH,
    GeneratorConfigError,
    load_generator_config,
)
from urbansense.synthetic.writer import (
    CSV_COLUMNS,
    FORBIDDEN_COLUMN_PATTERNS,
    RawRow,
    WriteRefused,
    write_rows,
)

__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_CONFIG_PATH",
    "FORBIDDEN_COLUMN_PATTERNS",
    "GENERATOR_VERSION",
    "GROUND_TRUTH_FILENAME",
    "HISTORY_FILENAME",
    "HOLDOUT_FILENAME",
    "DriftConfig",
    "FridayEveningConfig",
    "GeneratedDataset",
    "GeneratorConfig",
    "GeneratorConfigError",
    "GroundTruth",
    "IntegrityInjectionConfig",
    "MissingDataConfig",
    "MissingWindowConfig",
    "RainEffectConfig",
    "RawRow",
    "SensorFailureConfig",
    "WriteRefused",
    "ZoneConfig",
    "generate",
    "generate_to_directory",
    "load_generator_config",
    "write_rows",
]
