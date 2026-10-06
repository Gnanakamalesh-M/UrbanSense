"""Discover patterns and scan for anomalies (SPEC sections 21, 22, 23, 24).

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/discover_patterns.py
    python scripts/discover_patterns.py --no-write
    python scripts/discover_patterns.py --kind spatiotemporal --min-strength 1.5

Runs ingest -> validate -> reconcile -> discover -> track evolution -> test
interactions -> detect anomalies, then writes the pattern report to
``reports/patterns/``.

The multiple-testing header prints **before** any finding, and is not
suppressible. A list of patterns without the number of hypotheses behind it
cannot be judged: one pattern out of 672 tests is a different claim from one out
of ten, and printing the findings first invites the reader to believe the
stronger one.

Nothing here reads ``ground_truth.json``. Discovery is scored against the
planted answers in the test suite, which is the only place that file is opened;
this script sees the same unified layer a real deployment would.

Anomalies are scanned over the **pre-quarantine** population -- the valid
observations plus the ones held in quarantine records. That is a reporting
choice with a reason: Phase 2 quarantines every planted defect before discovery
runs, so scanning the unified layer alone would print an anomaly count of
roughly zero and conceal what the detector can do. Discovery itself still reads
the unified layer only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.anomaly import (
    DEFAULT_ANOMALY_DIR,
    AnomalyResult,
    detect_anomalies,
    pre_quarantine_population,
    write_anomaly_report,
)
from urbansense.config import ConfigError, load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.patterns import (
    DEFAULT_FDR_LEVEL,
    DEFAULT_METRIC,
    DEFAULT_MIN_LOCALIZED_LIFT,
    DEFAULT_PATTERN_DIR,
    DiscoveryResult,
    Pattern,
    PatternKind,
    PatternStore,
    discover_patterns,
)
from urbansense.preprocessing import LocationRegistryError, load_location_registry
from urbansense.synthetic.generator import HISTORY_FILENAME, ZONES_FILENAME

DEFAULT_DATA_DIR = Path("data") / "sample"

#: Kinds a reader can filter on, as lower-case names.
KIND_CHOICES = tuple(kind.value for kind in PatternKind)

#: The kinds that went through a hypothesis test and the FDR correction, and
#: so are the ones the header's discovery count refers to. The profiles and
#: hotspots are printed separately: they are the context these findings were
#: measured against, and listing them together would imply the scan reported
#: a dozen discoveries when it reported four.
TESTED_KINDS = frozenset(
    {
        PatternKind.SPATIOTEMPORAL,
        PatternKind.CONDITIONAL,
        PatternKind.LEVEL_SHIFT,
    }
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Discover traffic patterns and scan for anomalies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/discover_patterns.py\n"
            "  python scripts/discover_patterns.py --no-write\n"
            "  python scripts/discover_patterns.py --kind spatiotemporal "
            "--min-strength 1.5\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"history CSV (default: {DEFAULT_DATA_DIR / HISTORY_FILENAME})",
    )
    parser.add_argument(
        "--metric", default=DEFAULT_METRIC, help=f"metric to analyse (default: {DEFAULT_METRIC})"
    )
    parser.add_argument(
        "--kind",
        action="append",
        choices=KIND_CHOICES,
        default=None,
        help="only print patterns of this kind; repeatable (default: all)",
    )
    parser.add_argument(
        "--min-strength",
        type=float,
        default=None,
        help="only print patterns whose effect size is at least this large",
    )
    parser.add_argument(
        "--fdr-level",
        type=float,
        default=DEFAULT_FDR_LEVEL,
        help=f"Benjamini-Hochberg level for the scan (default: {DEFAULT_FDR_LEVEL})",
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=DEFAULT_MIN_LOCALIZED_LIFT,
        help=(
            "zone-specific lift a cell must show to be reported "
            f"(default: {DEFAULT_MIN_LOCALIZED_LIFT})"
        ),
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=DEFAULT_PATTERN_DIR, help="pattern report directory"
    )
    parser.add_argument(
        "--show-anomalies",
        type=int,
        default=5,
        help="how many anomalies to print in full, with their candidates (default: 5)",
    )
    parser.add_argument("--no-anomalies", action="store_true", help="skip the anomaly scan")
    parser.add_argument(
        "--anomaly-dir",
        type=Path,
        default=DEFAULT_ANOMALY_DIR,
        help=f"where to write the anomaly scan (default: {DEFAULT_ANOMALY_DIR.as_posix()})",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print the findings without writing a report"
    )
    return parser


def _selected(
    result: DiscoveryResult,
    *,
    kinds: list[str] | None,
    min_strength: float | None,
) -> tuple[Pattern, ...]:
    """Apply the display filters.

    Filtering is for reading convenience only -- it never changes what was
    tested, and the header still reports the full run, so a narrowed view cannot
    be mistaken for a smaller scan.
    """
    chosen = result.patterns
    if kinds:
        wanted = {PatternKind(name) for name in kinds}
        chosen = tuple(p for p in chosen if p.kind in wanted)
    if min_strength is not None:
        chosen = tuple(p for p in chosen if p.strength == p.strength and p.strength >= min_strength)
    return chosen


def _print_patterns(selected: tuple[Pattern, ...]) -> None:
    """Print the findings, tested ones first."""
    tested = [p for p in selected if p.kind in TESTED_KINDS]
    descriptive = [p for p in selected if p.kind not in TESTED_KINDS]

    print("\ntested findings")
    print("-" * 72)
    if not tested:
        print("  none")
    for pattern in tested:
        print(f"  {pattern.pattern_id}  {pattern.describe()}")
        print(f"    {pattern.description}")
        print(
            f"    test {pattern.evidence.test}, p={pattern.evidence.p_value:.3g}"
            + (
                f", adjusted p={pattern.evidence.adjusted_p_value:.3g}"
                if pattern.evidence.adjusted_p_value is not None
                else ""
            )
            + f", n={pattern.evidence.n_treatment} vs {pattern.evidence.n_control}"
        )
        for control in pattern.evidence.controls:
            print(f"    controlled: {control}")
        for note in pattern.evidence.notes:
            print(f"    note: {note}")
        print(f"    status: {pattern.status.value} -- {pattern.status_reason}")
        measured = [w for w in pattern.strength_by_window if w.measured]
        if measured:
            series = ", ".join(
                f"{w.window} {w.strength:.2f}" for w in measured if w.strength is not None
            )
            unmeasured = len(pattern.strength_by_window) - len(measured)
            print(
                f"    by window: {series}"
                + (f" (+{unmeasured} window(s) too sparse to measure)" if unmeasured else "")
            )

    if descriptive:
        print("\ndescriptive context (not part of the multiple-testing family)")
        print("-" * 72)
        for pattern in descriptive:
            print(f"  {pattern.pattern_id}  {pattern.describe()}")


def _print_interactions(result: DiscoveryResult) -> None:
    """Print the interaction verdicts (SPEC section 23)."""
    print("\ninteractions")
    print("-" * 72)
    if not result.interactions:
        print("  none testable: no two discovered patterns overlap in scope")
        return
    for item in result.interactions:
        print(f"  {item.describe()}")
    # Said explicitly, because "3 interactions tested" reads as "3 found".
    print(f"\n  supported by the evidence: {len(result.supported_interactions)}")


def main(argv: list[str] | None = None) -> int:
    """Discover, report and write. Returns a process exit code."""
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

    zones_path = source.parent / ZONES_FILENAME
    coordinates = (
        ZoneRegistry.from_csv(zones_path) if zones_path.is_file() else ZoneRegistry.empty()
    )

    # The source's local offset, read from the dataset config rather than
    # assumed: "Friday evening" is a local-clock fact, and a wrong offset would
    # smear the window across two hour buckets and hide the pattern.
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
    print(
        f"  {validated.stats.rows_in:,} rows -> {len(validated.valid):,} valid "
        f"-> {len(reconciled.unified):,} unified "
        f"({len(reconciled.duplicate_linked)} duplicate-linked, "
        f"{len(reconciled.conflicted_observations)} conflicted, both excluded)"
    )

    result = discover_patterns(
        reconciled,
        metric=args.metric,
        timezone_offset_hours=timezone_offset,
        fdr_level=args.fdr_level,
        min_localized_lift=args.min_lift,
    )

    # The header goes first, always. The findings mean nothing without it.
    print("\n" + "=" * 72)
    print(result.header())
    print("=" * 72)
    print(
        f"  readings analysed: {len(result.frame):,} "
        f"({result.frame.dropped_unknown:,} unknown dropped, never imputed)"
    )
    print(f"  local offset: UTC{timezone_offset:+.1f}")

    selected = _selected(result, kinds=args.kind, min_strength=args.min_strength)
    if len(selected) != len(result.patterns):
        print(
            f"  showing {len(selected)} of {len(result.patterns)} patterns "
            "(display filter only; the scan is unchanged)"
        )

    _print_patterns(selected)
    _print_interactions(result)

    # Held after the block so the scan can be written below. It stays None when
    # --no-anomalies was passed, which is a different thing from a scan that ran
    # and flagged nothing.
    scan: AnomalyResult | None = None
    if not args.no_anomalies:
        held = tuple(
            record.observation for record in validated.quarantined if record.observation is not None
        )
        population = pre_quarantine_population(validated.valid, held)
        scan = anomalies = detect_anomalies(
            population,
            patterns=result.patterns,
            source_health=validated.source_health,
            timezone_offset_hours=timezone_offset,
            metric=args.metric,
        )
        print("\nanomalies")
        print("-" * 72)
        print(
            f"  {len(anomalies.anomalies)} flagged of {anomalies.observations_scanned:,} "
            f"scanned ({anomalies.rate:.2%}) at {anomalies.threshold} MAD-sigma"
        )
        print(
            "  population: valid + quarantined observations, so the defects "
            "Phase 2 already removed are still visible here"
        )
        for kind, count in anomalies.by_leading_explanation.items():
            print(f"    leading candidate {kind}: {count}")
        print(f"    unexplained: {len(anomalies.unexplained)}")
        print("  none is classified as an error: is_error is None on every record")
        for anomaly in anomalies.anomalies[: max(0, args.show_anomalies)]:
            print()
            for line in anomaly.describe().splitlines():
                print(f"  {line}")

    if args.no_write:
        print("\n--no-write: nothing written")
        return 0

    store = PatternStore(args.reports_dir)
    target = store.write(result)
    if target is None:
        print("\nno pattern report: the run found nothing to write")
    else:
        print(f"\nwrote {target}")
        print(
            f"  {len(store.reports())} report(s) in {args.reports_dir}; "
            "runs append, never overwrite"
        )

    # Written separately from the patterns, and after them, because the two are
    # separate claims: a pattern is a regularity, an anomaly is a departure from
    # one. A run can legitimately produce either without the other, so neither
    # file's absence short-circuits the other's write.
    if scan is not None:
        anomaly_target = write_anomaly_report(scan, root=args.anomaly_dir)
        if anomaly_target is None:
            print("no anomaly report: the scan flagged nothing")
        else:
            print(f"wrote {anomaly_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
