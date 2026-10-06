"""Generate the synthetic demo dataset (SPEC section 37).

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/generate_demo_data.py --out data/sample --force

Writes ``traffic_history.csv``, ``traffic_holdout.csv``, ``zones.csv`` and
``ground_truth.json``. Output is deterministic: the same seed reproduces the
files byte for byte, which is what makes the committed sample verifiable.

Existing files are **not** replaced unless ``--force`` is given. Raw data is
immutable by project rule, and a careless rerun is the likeliest way to destroy
a dataset someone meant to keep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run directly from a source checkout, so the
# script works without `pip install -e .` first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.synthetic import (
    GeneratorConfigError,
    WriteRefused,
    generate_to_directory,
    load_generator_config,
)

DEFAULT_OUT = Path("data") / "sample"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "Generate demo data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/generate_demo_data.py\n"
            "  python scripts/generate_demo_data.py --out data/sample --force\n"
            "  python scripts/generate_demo_data.py --seed 7 --out data/raw/alt\n"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="generator config YAML (default: configs/synthetic/traffic_demo.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the seed from the config, to produce an alternative dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing files (without this, existing data is left untouched)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the dataset. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        config = load_generator_config(args.config, seed=args.seed)
    except GeneratorConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        dataset = generate_to_directory(config, args.out, force=args.force)
    except WriteRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: pass --force to replace the existing dataset.", file=sys.stderr)
        return 1

    truth = dataset.ground_truth
    friday = truth.friday_evening
    drift = truth.drift
    failure = truth.sensor_failure
    planted = truth.planted_records

    print(f"wrote {dataset.history_path}  ({dataset.history_rows:,} rows)")
    print(f"wrote {dataset.holdout_path}  ({dataset.holdout_rows:,} rows, held out)")
    print(f"wrote {dataset.zones_path}")
    print(f"wrote {dataset.ground_truth_path}")
    print()
    print(f"seed {truth.seed} -- regenerating with it reproduces these files exactly")
    print()
    print("planted ground truth:")
    print(
        f"  {friday.weekday_name} {friday.start_hour:02d}:00-{friday.end_hour:02d}:00 "
        f"in {friday.location_id}: x{friday.volume_multiplier} volume"
    )
    print(f"  rain effect: +{truth.rain_effect.volume_coefficient:.0%} volume per mm, capped")
    print(
        f"  drift from {drift.start_date}: x{drift.volume_multiplier} in "
        f"{', '.join(drift.location_ids)}"
    )
    print(
        f"  sensor failure: {failure.location_id} frozen at {failure.frozen_value:g} "
        f"from {failure.start} to {failure.end}"
    )
    print(f"  missing values: {len(truth.missing_data.all_record_ids):,} (empty cells, never zero)")
    print(
        f"  planted defects: {len(planted.duplicates)} duplicates, "
        f"{len(planted.conflicts)} conflicts, {len(planted.invalid)} invalid"
    )
    print()
    print("labels live in ground_truth.json only; the CSV carries no answer columns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
