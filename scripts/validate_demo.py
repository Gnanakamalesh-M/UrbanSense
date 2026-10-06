"""Validate and normalize a traffic CSV (SPEC sections 11-18).

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/validate_demo.py --input data/sample/traffic_history.csv
    python scripts/validate_demo.py --no-write        # summary only

Prints the valid count, quarantined counts by reason, the suspicious count, the
recency breakdown and a source-health table; writes quarantined records to
``quarantine/`` with their original payloads, and review notes for retained-but-
flagged rows to ``quarantine/review/``.

Ingestion runs **without** problem config on purpose, so it does only
representability checks and this stage owns all semantic normalization and
validation -- metric aliases, units, locations, ranges.

Exits non-zero if the row-count invariant fails, because a validation run that
has silently lost records is worse than one that crashed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.config import ConfigError, load_settings
from urbansense.data_integrity import QuarantineWriter, validate_and_normalize
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.preprocessing import LocationRegistryError, load_location_registry
from urbansense.synthetic.generator import HISTORY_FILENAME, ZONES_FILENAME

DEFAULT_DATA_DIR = Path("data") / "sample"
DEFAULT_QUARANTINE_DIR = Path("quarantine")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate and normalize a traffic CSV into canonical observations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/validate_demo.py\n"
            "  python scripts/validate_demo.py --input data/sample/traffic_history.csv\n"
            "  python scripts/validate_demo.py --no-write\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"CSV to validate (default: {DEFAULT_DATA_DIR / HISTORY_FILENAME})",
    )
    parser.add_argument(
        "--locations",
        type=Path,
        default=None,
        help="location registry YAML (default: configs/locations/chennai_zones.yaml)",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=DEFAULT_QUARANTINE_DIR,
        help=f"where to write quarantined records (default: {DEFAULT_QUARANTINE_DIR})",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the summary without writing anything to disk",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        metavar="N",
        help="show N example quarantined records per reason (default: 3; 0 to hide)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate the file. Returns a process exit code."""
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
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        registry = load_location_registry(args.locations)
    except LocationRegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    zones_path = source.parent / ZONES_FILENAME
    coordinates = (
        ZoneRegistry.from_csv(zones_path) if zones_path.is_file() else ZoneRegistry.empty()
    )

    # No settings: ingestion stays at representability so this stage owns all
    # semantic validation, and the bogus-unit row reaches the unit validator.
    ingestion = load_historical(source, registry=coordinates)
    result = validate_and_normalize(ingestion, settings=settings, registry=registry)
    stats = result.stats

    print(f"validated {source}")
    print(f"  rows in:       {stats.rows_in:,}")
    print(f"  valid:         {len(result.valid):,}")
    print(f"    clean:       {len(result.clean):,}")
    print(f"    suspicious:  {len(result.suspicious):,}  (flagged, retained)")
    print(f"  quarantined:   {len(result.quarantined):,}")
    print(f"    from ingest: {stats.carried_forward:,}  (unparseable rows)")
    print(f"  missing values: {stats.missing_values:,} (still missing, never zero)")
    print(f"  unit conversions: {stats.normalized_units:,}")

    if not result.accounts_for_every_row:
        print(
            f"\nerror: {stats.rows_in} rows in but "
            f"{len(result.valid)} valid + {len(result.quarantined)} quarantined "
            "- records were lost",
            file=sys.stderr,
        )
        return 3
    print(f"  every row accounted for: yes ({len(result.valid)} + {len(result.quarantined)})")

    if result.by_reason:
        print("\nquarantined by reason:")
        for reason, count in result.by_reason.items():
            print(f"  {reason.value:26s} {count:5,}")

    if result.by_recency:
        print("\nrecency (how old each reading was when it arrived):")
        for recency, count in result.by_recency.items():
            print(f"  {recency.value:26s} {count:5,}")

    print("\nsource health:")
    print(f"  {'source':<16} {'status':<8} {'obs':>7} {'missing':>8} {'frozen':>7}  detail")
    for source_id, health in sorted(result.source_health.items()):
        print(
            f"  {source_id:<16} {health.status.value:<8} {health.observation_count:>7,} "
            f"{health.missing_count:>8,} {health.frozen_observation_count:>7,}  "
            f"{health.detail[:58]}"
        )

    if result.frozen_runs:
        print("\nfrozen sensor windows:")
        for run in result.frozen_runs:
            print(
                f"  {run.location_id} {run.metric}: {run.value:g} repeated "
                f"{run.length}x over {run.duration} "
                f"({run.start.isoformat()} -> {run.end.isoformat()})"
            )

    if args.show_examples > 0 and result.quarantined:
        print("\nexample quarantined records (payloads kept exactly as received):")
        shown: dict[str, int] = {}
        for record in result.quarantined:
            key = record.reason.value
            if shown.get(key, 0) >= args.show_examples:
                continue
            shown[key] = shown.get(key, 0) + 1
            print(f"  [{key}] {record.raw_payload.get('record_id', '?')}")
            print(f"      payload: {record.raw_payload}")
            print(f"      why:     {record.detail[:110]}")

    if args.no_write:
        print("\n--no-write: nothing written to disk.")
        return 0

    writer = QuarantineWriter(args.quarantine_dir)
    quarantine_file = writer.write_records(result.quarantined)
    review_file = writer.write_review_notes(result.suspicious, result.outlier_flags)

    print()
    if quarantine_file is not None:
        print(f"wrote {len(result.quarantined)} quarantined records to {quarantine_file}")
    else:
        print("nothing to quarantine.")
    if review_file is not None:
        print(
            f"wrote {len(result.suspicious)} review notes to {review_file} "
            "(these rows remain in the valid set)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
