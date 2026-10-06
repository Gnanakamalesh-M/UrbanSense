"""Ingest the demo dataset into canonical observations (SPEC sections 4, 18).

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/ingest_demo.py
    python scripts/ingest_demo.py --input data/sample/traffic_history.csv
    python scripts/ingest_demo.py --write-quarantine

Reads the input **read-only** and never modifies it. Rows that cannot become
observations are reported, and with ``--write-quarantine`` are written to
``quarantine/`` with their original payloads intact -- the ingestion library
itself writes nothing, so that decision stays here with the caller.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.config import ConfigError, load_settings
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.synthetic.generator import (
    HISTORY_FILENAME,
    ZONES_FILENAME,
)

DEFAULT_DATA_DIR = Path("data") / "sample"
DEFAULT_QUARANTINE_DIR = Path("quarantine")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Ingest a traffic CSV into canonical observations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/ingest_demo.py\n"
            "  python scripts/ingest_demo.py --input data/sample/traffic_history.csv\n"
            "  python scripts/ingest_demo.py --write-quarantine\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"directory holding the generated dataset (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"explicit CSV to ingest (default: <data-dir>/{HISTORY_FILENAME})",
    )
    parser.add_argument(
        "--write-quarantine",
        action="store_true",
        help=f"write rejected rows to {DEFAULT_QUARANTINE_DIR}/ as JSON",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=DEFAULT_QUARANTINE_DIR,
        help=f"where to write rejected rows (default: {DEFAULT_QUARANTINE_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ingest the file. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    source = args.input if args.input is not None else args.data_dir / HISTORY_FILENAME
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

    zones_path = source.parent / ZONES_FILENAME
    registry = ZoneRegistry.from_csv(zones_path) if zones_path.is_file() else ZoneRegistry.empty()

    result = load_historical(source, settings=settings, registry=registry)
    stats = result.stats

    print(f"ingested {source}")
    print(f"  rows read:     {stats.rows_read:,}")
    print(f"  observations:  {stats.observations:,}")
    print(f"  rejected:      {stats.rejected:,}")
    print(f"  missing values: {stats.missing_values:,} (represented as missing, never as zero)")

    if not result.accounts_for_every_row:
        # Would mean ingestion lost data silently, which the project forbids.
        print("error: rows were lost during ingestion", file=sys.stderr)
        return 3
    print("  every row accounted for: yes")

    if result.observations:
        by_metric = Counter(obs.metric for obs in result.observations)
        print("\nobservations by metric:")
        for metric, count in sorted(by_metric.items()):
            print(f"  {metric:<18} {count:>8,}")

        first = result.observations[0]
        print("\nthe three timestamps on the first observation:")
        print(f"  event_time    {first.event_time.isoformat()}  (when it happened)")
        assert first.source_time is not None
        print(f"  source_time   {first.source_time.isoformat()}  (what the source asserted)")
        print(f"  received_time {first.received_time.isoformat()}  (when we read it)")
        print(f"  provenance    {first.provenance.notes}")

    if result.rejected:
        by_reason = Counter(record.reason.value for record in result.rejected)
        print("\nrejected rows by reason:")
        for reason, count in sorted(by_reason.items()):
            print(f"  {reason:<22} {count:>4}")

        if args.write_quarantine:
            args.quarantine_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            target = args.quarantine_dir / f"ingestion-{source.stem}-{stamp}.json"
            payload = [record.model_dump(mode="json") for record in result.rejected]
            target.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"\nwrote {len(payload)} quarantined rows to {target}")
            print("payloads are preserved exactly as received, for review.")
        else:
            print("\npass --write-quarantine to persist these with their payloads.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
