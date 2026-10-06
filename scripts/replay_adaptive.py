"""Replay a scenario and compare the static model against the adaptive system.

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/replay_adaptive.py
    python scripts/replay_adaptive.py --scenario july_holdout
    python scripts/replay_adaptive.py --no-write --step-days 7

Prints the drift events in time order, each candidate update's decision with
every rule outcome, the registry lineage, and the static-versus-adaptive table.

Two scenarios:

``drift_history``
    The champion trains on the first 75 days and is then frozen. The reference
    window follows it, strictly out of sample. The rest of the history replays
    weekly, so the planted 2026-03 level shift is genuinely unseen -- which is
    what makes it detectable at all.

``july_holdout``
    The same, carried into the held-out July file. Nothing has ever trained on
    that month, so it is the stricter of the two.

**Nothing here reads ``ground_truth.json``.** The replay finds the change the
way a deployment would: by noticing its own error rise, and then checking
whether the model's output distribution moved with it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.adaptation import (
    AdaptiveReplay,
    CandidateStore,
    load_adaptation_config,
    promotion_summary,
    render_lineage,
)
from urbansense.config import ConfigError, load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.drift import DEFAULT_DRIFT_DIR, DriftStore, summarize
from urbansense.evaluation import friday_evening_slices
from urbansense.features import build_feature_table, load_feature_set
from urbansense.features.builder import FeatureTable
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.ingestion.result import IngestionResult, IngestionStats
from urbansense.prediction import GradientForecaster, ModelRegistry
from urbansense.preprocessing import LocationRegistryError, load_location_registry
from urbansense.synthetic.generator import HISTORY_FILENAME, HOLDOUT_FILENAME, ZONES_FILENAME

DEFAULT_DATA_DIR = Path("data") / "sample"

#: Days the frozen champion trains on, and days of out-of-sample reference that
#: follow. The reference must start where training ends: an overlapping
#: reference is partly in-sample, which understates the reference error and
#: turns a stable stream into a stream of false alarms.
DEFAULT_TRAIN_DAYS = 75
DEFAULT_REFERENCE_DAYS = 60

SCENARIOS = ("drift_history", "july_holdout")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Replay a scenario and compare static against adaptive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/replay_adaptive.py\n"
            "  python scripts/replay_adaptive.py --scenario july_holdout\n"
            "  python scripts/replay_adaptive.py --no-write --step-days 7\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"history CSV (default: {DEFAULT_DATA_DIR / HISTORY_FILENAME})",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="drift_history",
        help="which scenario to replay (default: drift_history)",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=DEFAULT_TRAIN_DAYS,
        help=f"days the frozen champion trains on (default: {DEFAULT_TRAIN_DAYS})",
    )
    parser.add_argument(
        "--reference-days",
        type=int,
        default=DEFAULT_REFERENCE_DAYS,
        help=(
            f"days of out-of-sample reference after training (default: {DEFAULT_REFERENCE_DAYS})"
        ),
    )
    parser.add_argument(
        "--step-days", type=int, default=None, help="replay step (default: from config)"
    )
    parser.add_argument(
        "--show-events",
        type=int,
        default=40,
        help="drift evaluations to print (default: 40)",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=DEFAULT_DRIFT_DIR.parent, help="reports root"
    )
    parser.add_argument("--no-write", action="store_true", help="print without writing any log")
    return parser


def _window(table: FeatureTable, start: datetime, end: datetime) -> FeatureTable:
    """Rows whose sample time falls in ``[start, end)``."""
    mask = np.array([start <= moment < end for moment in table.sample_times], dtype=bool)
    return table.select(mask)


def main(argv: list[str] | None = None) -> int:
    """Replay, decide and report. Returns a process exit code."""
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
        config = load_adaptation_config()
    except (ConfigError, LocationRegistryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.step_days is not None:
        config = config.model_copy(
            update={
                "monitoring": config.monitoring.model_copy(
                    update={"step_days": args.step_days, "window_days": args.step_days}
                )
            }
        )

    zones_path = source.parent / ZONES_FILENAME
    coordinates = (
        ZoneRegistry.from_csv(zones_path) if zones_path.is_file() else ZoneRegistry.empty()
    )

    try:
        from urbansense.synthetic import load_generator_config

        timezone_offset = load_generator_config().timezone_offset_hours
    except Exception:
        timezone_offset = 0.0

    paths = [source]
    if args.scenario == "july_holdout":
        holdout = source.parent / HOLDOUT_FILENAME
        if not holdout.is_file():
            print(f"error: {holdout} not found.", file=sys.stderr)
            return 1
        paths.append(holdout)

    print(f"loading {', '.join(str(path) for path in paths)}")
    # Ingested files are merged *before* validation, so the stream is one
    # timeline validated and reconciled once. Merging after validation would
    # mean splicing two results together and the row-accounting invariant --
    # every input row usable or quarantined -- would no longer hold.
    ingested = [load_historical(path, registry=coordinates) for path in paths]
    combined = IngestionResult(
        observations=tuple(
            observation for result in ingested for observation in result.observations
        ),
        rejected=tuple(record for result in ingested for record in result.rejected),
        stats=IngestionStats(
            rows_read=sum(result.stats.rows_read for result in ingested),
            observations=sum(result.stats.observations for result in ingested),
            rejected=sum(result.stats.rejected for result in ingested),
            missing_values=sum(result.stats.missing_values for result in ingested),
            source_path=paths[0],
        ),
    )
    if not combined.accounts_for_every_row:
        print("error: ingestion lost rows while merging the scenario files.", file=sys.stderr)
        return 3

    validated = validate_and_normalize(combined, settings=settings, registry=registry)
    reconciled = reconcile(validated, settings=settings)
    rows_in = validated.stats.rows_in
    feature_set = load_feature_set()
    table = build_feature_table(reconciled, feature_set, timezone_offset_hours=timezone_offset)
    print(
        f"  {rows_in:,} rows -> {len(reconciled.unified):,} unified -> "
        f"{len(table):,} feature rows "
        f"({min(table.sample_times):%Y-%m-%d} to {max(table.sample_times):%Y-%m-%d})"
    )

    earliest = min(table.sample_times)
    train_end = earliest + timedelta(days=args.train_days)
    reference_end = train_end + timedelta(days=args.reference_days)
    champion = GradientForecaster().fit(_window(table, earliest, train_end))
    print(
        f"\nfrozen champion: trained {earliest:%Y-%m-%d} to {train_end:%Y-%m-%d} "
        f"({len(_window(table, earliest, train_end)):,} rows)"
    )
    print(
        f"reference window: {train_end:%Y-%m-%d} to {reference_end:%Y-%m-%d}, "
        "strictly out of sample"
    )
    print()
    print(config.promotion.describe())

    replay = AdaptiveReplay(
        table=table,
        reconciled=reconciled,
        champion=champion,
        reference=_window(table, train_end, reference_end),
        config=config,
        training_end=train_end,
        timezone_offset_hours=timezone_offset,
    )
    result = replay.run(
        scenario=args.scenario,
        start=reference_end,
        end=max(table.sample_times),
        slices=friday_evening_slices(table, timezone_offset_hours=timezone_offset),
    )

    print("\n" + "=" * 72)
    print(f"drift monitoring -- {args.scenario}")
    print("=" * 72)
    print(f"  {result.gates['summary'] if 'summary' in result.gates else ''}".rstrip())
    for key in ("feature_psi", "prediction_psi", "performance_ratio", "reference_error"):
        print(f"  {key:<20} {result.gates.get(key)}")
    print(
        f"  acting families: "
        f"{', '.join(kind.value for kind in config.monitoring.candidate_kinds)}  "
        "(the others are logged as leading indicators)"
    )
    print()
    events = result.events[: max(0, args.show_events)]
    print(summarize(events))
    if len(result.events) > len(events):
        print(f"  ... {len(result.events) - len(events)} more evaluation(s)")

    print("\n" + "=" * 72)
    print("candidate updates")
    print("=" * 72)
    print(promotion_summary(result.candidates))
    for candidate in result.candidates:
        print()
        print("  " + candidate.describe().replace("\n", "\n  "))

    print("\n" + "=" * 72)
    print(result.render())
    print("=" * 72)

    if args.no_write:
        print("\n--no-write: nothing written")
        return 0

    # Each run writes to its own fresh registry directory, so replaying a
    # scenario twice -- or the other scenario -- cannot inherit a champion.
    # Versions also stay out of models/registry/: a replay is an offline
    # experiment, and nine challengers per run would bury the model that is
    # actually serving while making its champion invariant describe an
    # experiment rather than a deployment.
    registry_path, champion_now = replay.record(result, root=args.reports_dir / "adaptation")
    store = ModelRegistry(registry_path, registry_path.parent / "artifacts")
    print()
    print(render_lineage(store))
    print(f"  exactly one champion, as required: {champion_now}")
    print(f"  written to {registry_path}")

    drift_store = DriftStore(args.reports_dir / "drift")
    written_events = drift_store.append_events(result.events)
    written_episodes = drift_store.append_episodes(result.episodes)
    candidate_store = CandidateStore(args.reports_dir / "adaptation")
    written_candidates = candidate_store.append(result.candidates)

    print(
        f"\nappended {written_events} event(s) and {written_episodes} episode(s) to "
        f"{drift_store.root}"
    )
    print(f"appended {written_candidates} candidate(s) to {candidate_store.path}")
    print("both logs are append-only; earlier runs are never rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
