"""Break prediction errors down and say where the model is weak (SPEC section 28).

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/error_report.py
    python scripts/error_report.py --no-write
    python scripts/error_report.py --dataset holdout

Runs the pipeline, predicts the test split, records every prediction/actual pair
in the append-only outcome store, and prints the breakdown: by rainfall, time of
day, zone, weekday, the Friday evening window and predicted severity, ending
with the weakest slices.

The July **holdout** is available as a second, stricter dataset. It is a month
the generator reserved and nothing has trained on, so comparing the two reports
shows whether the test split was flattering.

Breakdowns print before the overall figure on purpose. An aggregate MAE averages
over conditions that behave nothing like each other, and a reader who sees it
first will anchor on it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.config import ConfigError, load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.evaluation import (
    DEFAULT_OUTCOME_DIR,
    OutcomeStore,
    analyse_errors,
    build_outcomes,
    temporal_split,
)
from urbansense.features import build_feature_table, load_feature_set
from urbansense.features.builder import FeatureTable
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.prediction import (
    GradientForecaster,
    ProbabilityCalibrator,
    fit_thresholds,
)
from urbansense.preprocessing import LocationRegistryError, load_location_registry
from urbansense.schemas import ProblemType
from urbansense.schemas.observation import Observation
from urbansense.synthetic.generator import HISTORY_FILENAME, HOLDOUT_FILENAME, ZONES_FILENAME

DEFAULT_DATA_DIR = Path("data") / "sample"
DEFAULT_VERSION = "Traffic-v1.1"
RAINFALL_METRIC = "rainfall"
CITY = "CITY"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Report where the traffic model fails.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/error_report.py\n"
            "  python scripts/error_report.py --no-write\n"
            "  python scripts/error_report.py --dataset holdout\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"history CSV (default: {DEFAULT_DATA_DIR / HISTORY_FILENAME})",
    )
    parser.add_argument(
        "--dataset",
        choices=("test", "holdout", "both"),
        default="test",
        help="which dataset to report on (default: test)",
    )
    parser.add_argument(
        "--version", default=DEFAULT_VERSION, help="model version to label outcomes with"
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=DEFAULT_OUTCOME_DIR / "outcomes.jsonl",
        help="append-only outcome log",
    )
    parser.add_argument(
        "--weakest", type=int, default=5, help="weakest slices to rank (default: 5)"
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print the report without appending outcomes"
    )
    return parser


def _rainfall_by_time(
    observations: Sequence[Observation],
) -> dict[datetime, float]:
    """City rainfall keyed by event time, for the conditions column.

    Hours with no reading are simply absent, so the outcome records ``None``
    rather than a fabricated zero.
    """
    return {
        item.event_time: item.value
        for item in observations
        if item.metric == RAINFALL_METRIC and item.location_id == CITY and item.value is not None
    }


def main(argv: list[str] | None = None) -> int:
    """Predict, record and report. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    source = args.input if args.input is not None else DEFAULT_DATA_DIR / HISTORY_FILENAME
    if not source.is_file():
        print(f"error: {source} not found.", file=sys.stderr)
        print(
            "hint: generate the dataset first:\n"
            "  python scripts/generate_demo_data.py --out data/sample",
            file=sys.stderr,
        )
        return 1

    try:
        settings = load_settings()
        registry = load_location_registry()
    except (ConfigError, LocationRegistryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    congestion = settings.problems[ProblemType.TRAFFIC].congestion
    if congestion is None:
        print("error: the traffic problem config defines no congestion block.", file=sys.stderr)
        return 2

    zones_path = source.parent / ZONES_FILENAME
    coordinates = (
        ZoneRegistry.from_csv(zones_path) if zones_path.is_file() else ZoneRegistry.empty()
    )

    try:
        from urbansense.synthetic import load_generator_config

        timezone_offset = load_generator_config().timezone_offset_hours
    except Exception:
        timezone_offset = 0.0

    print(f"loading {source}")
    validated = validate_and_normalize(
        load_historical(source, registry=coordinates), settings=settings, registry=registry
    )
    reconciled = reconcile(validated, settings=settings)
    feature_set = load_feature_set()
    table = build_feature_table(reconciled, feature_set, timezone_offset_hours=timezone_offset)
    split = temporal_split(table)

    thresholds = fit_thresholds(split.train, congestion)
    model = GradientForecaster().fit(split.train)
    calibrator = ProbabilityCalibrator.from_residuals(split.train.y, model.predict(split.train))
    validation_thresholds = np.array(
        [thresholds.by_zone[zone] for zone in split.validation.locations], dtype=np.float64
    )
    calibrator.fit(
        calibrator.raw(model.predict(split.validation), validation_thresholds),
        thresholds.congested_mask(split.validation),
    )

    rainfall = _rainfall_by_time(reconciled.unified)
    store = OutcomeStore(args.outcomes)

    datasets: list[tuple[str, FeatureTable]] = []
    if args.dataset in ("test", "both"):
        datasets.append(("test", split.test))
    if args.dataset in ("holdout", "both"):
        holdout_path = source.parent / HOLDOUT_FILENAME
        if not holdout_path.is_file():
            print(f"warning: {holdout_path} not found; skipping the holdout.", file=sys.stderr)
        else:
            holdout_validated = validate_and_normalize(
                load_historical(holdout_path, registry=coordinates),
                settings=settings,
                registry=registry,
            )
            holdout_reconciled = reconcile(holdout_validated, settings=settings)
            holdout_table = build_feature_table(
                holdout_reconciled, feature_set, timezone_offset_hours=timezone_offset
            )
            rainfall.update(_rainfall_by_time(holdout_reconciled.unified))
            datasets.append(("holdout", holdout_table))

    for name, dataset in datasets:
        predicted = model.predict(dataset)
        zone_thresholds = np.array(
            [thresholds.by_zone.get(zone, float("nan")) for zone in dataset.locations],
            dtype=np.float64,
        )
        probabilities = calibrator.calibrate(calibrator.raw(predicted, zone_thresholds))
        outcomes = build_outcomes(
            dataset,
            predicted,
            model_version=args.version,
            thresholds=thresholds,
            probabilities=probabilities.tolist(),
            rainfall_by_time=rainfall,
            timezone_offset_hours=timezone_offset,
        )

        print("\n" + "=" * 72)
        print(analyse_errors(outcomes, dataset=name, model_version=args.version).render())
        print("=" * 72)

        if not args.no_write:
            written = store.append(outcomes)
            print(f"\nappended {written:,} outcomes to {store.path} (now {len(store):,} rows)")

    if args.no_write:
        print("\n--no-write: nothing appended")
    else:
        print("the outcome log is append-only; earlier runs are never rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
