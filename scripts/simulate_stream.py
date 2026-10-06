"""Replay the held-out period as a live observation stream (SPEC section 37).

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/simulate_stream.py --limit 20            # fast, no waiting
    python scripts/simulate_stream.py --speed 60 --limit 50  # a minute a second
    python scripts/simulate_stream.py --speed 1              # real time

The held-out period sits *after* the historical range, so nothing replayed here
has been trained on. Each observation is stamped with its ``received_time`` as
it is emitted, which is what makes this a live source rather than a slow file
read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.config import ConfigError, load_settings
from urbansense.ingestion import StreamSimulator, ZoneRegistry
from urbansense.synthetic.generator import (
    HOLDOUT_FILENAME,
    ZONES_FILENAME,
)

DEFAULT_DATA_DIR = Path("data") / "sample"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Replay held-out observations as a simulated live stream.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/simulate_stream.py --limit 20\n"
            "  python scripts/simulate_stream.py --speed 60 --limit 50\n"
            "  python scripts/simulate_stream.py --speed 1 --interval 2\n"
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
        help=f"explicit CSV to replay (default: <data-dir>/{HOLDOUT_FILENAME})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="wall-clock seconds between observations at --speed 1 (default: 1.0)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help=(
            "speed-up factor: 1 is real time, 60 replays a minute per second, "
            "0 is fast mode with no waiting at all (default: 0)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="stop after this many observations; 0 replays the whole file (default: 20)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the closing summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Replay the stream. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    source = args.input if args.input is not None else args.data_dir / HOLDOUT_FILENAME
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

    simulator = StreamSimulator(
        source,
        settings=settings,
        registry=registry,
        interval_seconds=args.interval,
        speed=args.speed,
    )

    mode = (
        "fast (no waiting)"
        if simulator.delay_seconds == 0
        else (f"{simulator.delay_seconds:.3f}s between observations")
    )
    if not args.quiet:
        print(f"replaying {source.name} -- {mode}")
        print(f"{'event_time (UTC)':<22} {'zone':<8} {'metric':<16} {'value':>10}  received")
        print("-" * 86)

    limit = args.limit if args.limit > 0 else None
    emitted = 0
    try:
        for observation in simulator.observations(limit=limit):
            emitted += 1
            if args.quiet:
                continue
            # "missing" rather than a number: absence is never shown as zero.
            value = "missing" if observation.value is None else f"{observation.value:,.1f}"
            print(
                f"{observation.event_time:%Y-%m-%d %H:%M}      "
                f"{observation.location_id:<8} {observation.metric:<16} "
                f"{value:>10}  {observation.received_time:%H:%M:%S}"
            )
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)

    print()
    print(f"emitted {emitted:,} observations from {simulator.rows_read:,} rows")
    if simulator.rejected:
        print(f"rejected {len(simulator.rejected):,} unparseable rows (payloads preserved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
