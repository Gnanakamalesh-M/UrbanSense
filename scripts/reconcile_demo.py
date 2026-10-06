"""Validate, reconcile and build the unified data layer (SPEC sections 9, 10, 19).

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/reconcile_demo.py --input data/sample/traffic_history.csv
    python scripts/reconcile_demo.py --no-write              # summary only
    python scripts/reconcile_demo.py --explain REC-065665    # one record's origin

Runs validation, then reconciliation. Prints duplicates linked (by group kind),
conflicts recorded, unified rows and the source-specific views, and writes the
conflict groups to ``quarantine/conflicts/`` with every candidate value and no
winner chosen.

Exits non-zero if the partition invariant fails, because a run that has
silently lost observations is worse than one that crashed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.config import ConfigError, load_settings
from urbansense.data_integrity import (
    QuarantineWriter,
    explain_provenance,
    reconcile,
    validate_and_normalize,
)
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.preprocessing import LocationRegistryError, load_location_registry
from urbansense.synthetic.generator import HISTORY_FILENAME, ZONES_FILENAME

DEFAULT_DATA_DIR = Path("data") / "sample"
DEFAULT_QUARANTINE_DIR = Path("quarantine")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Reconcile validated observations into the unified data layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/reconcile_demo.py\n"
            "  python scripts/reconcile_demo.py --no-write\n"
            "  python scripts/reconcile_demo.py --explain REC-065665\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"CSV to reconcile (default: {DEFAULT_DATA_DIR / HISTORY_FILENAME})",
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
        help=f"where to write the conflict queue (default: {DEFAULT_QUARANTINE_DIR})",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the summary without writing anything to disk",
    )
    parser.add_argument(
        "--explain",
        metavar="RECORD_ID",
        default=None,
        help="print the full provenance chain for one record id, then exit",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=2,
        metavar="N",
        help="show N example duplicate groups and conflicts (default: 2; 0 to hide)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Reconcile the file. Returns a process exit code."""
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

    validated = validate_and_normalize(
        load_historical(source, registry=coordinates),
        settings=settings,
        registry=registry,
    )
    result = reconcile(validated, settings=settings)
    stats = result.stats

    # One record's origin story, then stop.
    if args.explain is not None:
        chain = explain_provenance(args.explain, result)
        if chain is None:
            print(f"error: {args.explain} is not in the reconciled set.", file=sys.stderr)
            print(
                "hint: it may have been quarantined during validation; see\n"
                "  python scripts/validate_demo.py",
                file=sys.stderr,
            )
            return 1
        print(chain.describe())
        return 0

    print(f"reconciled {source}")
    print(f"  validated input:  {stats.observations_in:,} observations")
    print(f"  unified:          {len(result.unified):,}")
    print(f"  duplicate-linked: {len(result.duplicate_linked):,}  (retained, not deleted)")
    print(
        f"  conflicted:       {len(result.conflicted_observations):,} "
        f"in {len(result.conflicts)} group(s)  (every value kept)"
    )

    if not result.accounts_for_every_observation or not result.observation_ids_are_disjoint:
        print(
            f"\nerror: {stats.observations_in} observations in but "
            f"{len(result.unified)} unified + {len(result.duplicate_linked)} linked + "
            f"{len(result.conflicted_observations)} conflicted - observations were lost",
            file=sys.stderr,
        )
        return 3
    print(
        f"  every observation in exactly one place: yes "
        f"({len(result.unified)} + {len(result.duplicate_linked)} + "
        f"{len(result.conflicted_observations)})"
    )

    print("\nduplicate groups by kind:")
    for kind, count in result.by_group_kind.items():
        print(f"  {kind.value:26s} {count:5,}")

    # Source-specific views come before the unified result (SPEC section 20).
    print("\nsource-specific views:")
    print(f"  {'historical':26s} {len(result.historical_view):5,}")
    print(f"  {'real_time':26s} {len(result.real_time_view):5,}")
    print(f"  {'document':26s} {len(result.document_view):5,}")
    print(f"  {'-> unified':26s} {len(result.unified):5,}")

    if args.show_examples > 0 and result.duplicate_groups:
        print("\nexample duplicate groups (all source records retained):")
        for group in result.duplicate_groups[: args.show_examples]:
            print(f"  [{group.kind.value}] {group.group_id} - {group.key.describe()}")
            for reference in group.references:
                marker = "representative" if reference.is_representative else "linked copy"
                value = "missing" if reference.value is None else f"{reference.value:g}"
                print(
                    f"      {reference.record_id} from {reference.source_id} = {value}  [{marker}]"
                )

    if args.show_examples > 0 and result.conflicts:
        print("\nexample conflicts (no winner chosen):")
        for conflict in result.conflicts[: args.show_examples]:
            print(f"  {conflict.conflict_id} - {conflict.key.describe()}")
            for candidate in conflict.candidates:
                value = "missing" if candidate.value is None else f"{candidate.value:g}"
                print(
                    f"      {candidate.record_id} from {candidate.source_id} "
                    f"({candidate.source_type}) = {value} {candidate.unit}"
                )
            print(f"      state: {conflict.state.value}, policy: {conflict.policy.value}")

    if args.no_write:
        print("\n--no-write: nothing written to disk.")
        return 0

    writer = QuarantineWriter(args.quarantine_dir)
    conflict_file = writer.write_conflicts(result.conflicts)

    print()
    if conflict_file is not None:
        print(
            f"wrote {len(result.conflicts)} conflict group(s) to {conflict_file}\n"
            "every candidate value retained; a reviewer decides."
        )
    else:
        print("no conflicts to review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
