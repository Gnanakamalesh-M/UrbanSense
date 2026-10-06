"""Predict congestion for one zone and date (SPEC sections 25, 26, 27).

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/predict_demo.py
    python scripts/predict_demo.py --zone ZONE-B --date 2026-05-15
    python scripts/predict_demo.py --zone ZONE-C --no-explain

Runs ingest -> validate -> reconcile -> features -> leakage check -> temporal
split -> fit -> calibrate -> predict, then prints per hour: predicted volume,
calibrated congestion probability, severity, the congestion window, interval
tightness, the contributing factors and the source/freshness breakdown.

Everything fitted -- the model, the congestion thresholds, the probability
calibrator and the interval band -- uses train and validation only. The chosen
date comes from the test split, which is scored and never fitted on.

**It exits non-zero if the leakage check fails.** An explanation of a prediction
that saw the future is more dangerous than no explanation, because it is
convincing.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.config import ConfigError, load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.evaluation import temporal_split
from urbansense.explainability import FactorAttributor, load_factor_groups
from urbansense.features import (
    assert_no_leakage,
    build_feature_table,
    detect_leakage,
    load_feature_set,
)
from urbansense.features.leakage import LeakageError
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.prediction import (
    GradientForecaster,
    ProbabilityCalibrator,
    SourceLedger,
    build_predictions,
    congestion_windows,
    fit_thresholds,
)
from urbansense.preprocessing import LocationRegistryError, load_location_registry
from urbansense.schemas import ProblemType
from urbansense.synthetic.generator import HISTORY_FILENAME, ZONES_FILENAME

DEFAULT_DATA_DIR = Path("data") / "sample"
DEFAULT_ZONE = "ZONE-B"
DEFAULT_VERSION = "Traffic-v1.1"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Predict congestion for one zone and date.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/predict_demo.py\n"
            "  python scripts/predict_demo.py --zone ZONE-B --date 2026-05-15\n"
            "  python scripts/predict_demo.py --zone ZONE-C --no-explain\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"history CSV (default: {DEFAULT_DATA_DIR / HISTORY_FILENAME})",
    )
    parser.add_argument("--zone", default=DEFAULT_ZONE, help=f"zone (default: {DEFAULT_ZONE})")
    parser.add_argument(
        "--date",
        default=None,
        help="local date as YYYY-MM-DD (default: the last Friday in the test split)",
    )
    parser.add_argument(
        "--version", default=DEFAULT_VERSION, help="model version to label predictions with"
    )
    parser.add_argument(
        "--top-factors", type=int, default=4, help="factors to print per hour (default: 4)"
    )
    parser.add_argument(
        "--no-explain", action="store_true", help="skip the per-hour factor attribution"
    )
    parser.add_argument(
        "--no-sources", action="store_true", help="skip the source and freshness breakdown"
    )
    return parser


def _pick_date(local_times: list[datetime], zone_rows: int) -> date:
    """Default to the last Friday available, falling back to the last day.

    Friday because the planted pattern lives there, so the default run shows
    the interesting case rather than a quiet Tuesday.
    """
    fridays = [moment.date() for moment in local_times if moment.weekday() == 4]
    if fridays:
        return max(fridays)
    if not local_times:
        raise SystemExit(f"no rows for this zone ({zone_rows} candidates)")
    return max(moment.date() for moment in local_times)


def main(argv: list[str] | None = None) -> int:
    """Predict, explain and print. Returns a process exit code."""
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

    # The source's local offset, read from the dataset config rather than
    # assumed: "Friday evening" is a local-clock fact.
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

    print("checking for leakage before predicting anything...")
    report = detect_leakage(reconciled, feature_set, timezone_offset_hours=timezone_offset)
    try:
        assert_no_leakage(report)
    except LeakageError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 3
    print(f"  {report.describe().splitlines()[0]}")

    table = build_feature_table(reconciled, feature_set, timezone_offset_hours=timezone_offset)
    split = temporal_split(table)

    thresholds = fit_thresholds(split.train, congestion)
    model = GradientForecaster().fit(split.train)
    band = model.calibrate_interval(split.validation)

    calibrator = ProbabilityCalibrator.from_residuals(split.train.y, model.predict(split.train))
    validation_thresholds = np.array(
        [thresholds.by_zone[zone] for zone in split.validation.locations], dtype=np.float64
    )
    raw_validation = calibrator.raw(model.predict(split.validation), validation_thresholds)
    validation_outcomes = thresholds.congested_mask(split.validation)
    calibrator.fit(raw_validation, validation_outcomes)

    print()
    print(thresholds.describe())
    print()
    print(calibrator.report(raw_validation, validation_outcomes, dataset="validation").describe())

    # Everything above was fitted on train/validation. The rows below come from
    # the test split, which is scored and never fitted on.
    offset = timedelta(hours=timezone_offset)
    local_times = [
        split.test.target_times[index] + offset
        for index in range(len(split.test))
        if split.test.locations[index] == args.zone
    ]
    if not local_times:
        print(f"\nerror: no test rows for {args.zone}.", file=sys.stderr)
        print(f"  zones available: {sorted(set(split.test.locations))}", file=sys.stderr)
        return 4

    chosen = (
        date.fromisoformat(args.date) if args.date else _pick_date(local_times, len(local_times))
    )
    mask = np.array(
        [
            split.test.locations[index] == args.zone
            and (split.test.target_times[index] + offset).date() == chosen
            for index in range(len(split.test))
        ],
        dtype=bool,
    )
    if not mask.any():
        print(f"\nerror: no test rows for {args.zone} on {chosen}.", file=sys.stderr)
        print(
            f"  available dates: {min(local_times).date()} to {max(local_times).date()}",
            file=sys.stderr,
        )
        return 4

    subset = split.test.select(mask)
    predicted = model.predict(subset)
    attributor = (
        None
        if args.no_explain
        else FactorAttributor.from_forecaster(
            model, groups=load_factor_groups(), background=split.validation.x[:500]
        )
    )
    predictions = build_predictions(
        subset,
        predicted,
        thresholds=thresholds,
        calibrator=calibrator,
        model_version=args.version,
        interval_bounds=band.bounds(predicted, subset.locations),
        timezone_offset_hours=timezone_offset,
        ledger=None if args.no_sources else SourceLedger(reconciled.unified),
        attributor=attributor,
        explain_rows=attributor is not None,
    )

    print("\n" + "=" * 72)
    print(f"{args.zone} on {chosen} ({len(predictions)} hours, model {args.version})")
    if attributor is not None:
        print(f"explanations: {attributor.method.value}")
    print("=" * 72)

    print("\n  hour   predicted   probability   severity    interval tightness")
    for item in sorted(predictions, key=lambda row: row.target_time):
        print(
            f"  {item.local_time:%H:%M}  {item.predicted_volume:>10,.0f}   "
            f"{item.congestion_probability:>10.1%}   {item.severity.value:<10}  "
            f"{item.confidence:>10.1%}"
        )

    windows = congestion_windows(predictions)
    print("\nexpected congestion windows")
    if not windows:
        print("  none: no hour is predicted above the zone's threshold")
    for window in windows:
        print(f"  {window.describe()}")
    if windows:
        print(
            "  note: the threshold is anchored to the training period, so windows "
            "are common on later weekday evenings. Severity is what separates an "
            "ordinary peak from an unusual one."
        )

    peak = max(predictions, key=lambda row: row.predicted_volume)
    print("\npeak hour")
    print("  " + peak.describe().replace("\n", "\n  "))
    if peak.explanation is not None:
        print()
        print("  " + peak.explanation.describe(limit=args.top_factors).replace("\n", "\n  "))
    if peak.sources is not None:
        print()
        print("  " + peak.sources.describe().replace("\n", "\n  "))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
