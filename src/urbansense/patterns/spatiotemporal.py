"""The spatio-temporal scan (SPEC section 21).

    Zone B + Friday + 18:00-20:00 = Recurring congestion

Finding that without being told where to look is this module's whole job. It
tests every ``(zone, weekday, hour)`` cell and reports which ones genuinely depart
from their own context.

Three decisions make the difference between discovery and noise:

**The baseline is the same zone and the same hour on other weekdays.** Comparing a
cell against the zone's overall median would rediscover rush hour in every zone;
comparing against the city median would rediscover that arterials are busier than
side streets. Holding zone and hour fixed leaves only the question actually being
asked -- is *this day* different here?

**Multiple testing is corrected.** 4 zones x 7 weekdays x 24 hours is 672
hypotheses. At p<0.05 roughly 34 would clear by chance, so an uncorrected scan
would emit a page of confident nonsense. Benjamini-Hochberg controls the expected
share of false discoveries among those reported.

**A second pass removes the global weekday effect.** Friday is busier than
Tuesday everywhere, so every zone shows some Friday lift. Dividing each cell's
lift by the *median lift across zones* at that weekday and hour isolates what is
specific to the zone. Without it the scan reports four Friday patterns and cannot
say which zone actually has one.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from urbansense.patterns.frame import ObservationFrame, Reading
from urbansense.patterns.models import (
    Evidence,
    Pattern,
    PatternKind,
    PatternScope,
    PatternStatus,
)
from urbansense.patterns.statistics import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_MIN_GROUP,
    ComparisonResult,
    adjusted_p_values,
    benjamini_hochberg,
    compare_groups,
    confidence_from_p,
)

#: Localized lift a cell must show before it is reported, after the global
#: weekday effect is divided out. 1.25 keeps the scan away from the band where a
#: significant result is still too small to act on.
DEFAULT_MIN_LOCALIZED_LIFT = 1.25

#: Benjamini-Hochberg level for the scan.
DEFAULT_FDR_LEVEL = 0.05


@dataclass(frozen=True, slots=True)
class CellTest:
    """One cell's comparison against its own baseline.

    Attributes:
        location_id: Zone.
        weekday: Local day of week, Monday 0.
        hour: Local hour.
        comparison: Treatment-versus-baseline result.
        localized_lift: Raw lift divided by the median lift across zones at this
            weekday and hour -- the zone-specific part of the effect.
        peer_lift: The median lift across zones, kept so the adjustment is
            visible rather than buried.
    """

    location_id: str
    weekday: int
    hour: int
    comparison: ComparisonResult
    localized_lift: float
    peer_lift: float


def _scan_cells(
    frame: ObservationFrame,
    metric: str,
    *,
    min_group: int,
    seed: int,
) -> tuple[list[CellTest], int]:
    """Compare every cell against the same zone and hour on other weekdays.

    Returns:
        ``(tests, skipped)`` where ``skipped`` counts cells too sparse to test.
        Sparse cells are counted and reported rather than silently dropped.
    """
    cells = frame.cells(metric)
    by_zone_hour: dict[tuple[str, int], list[Reading]] = {}
    for (zone, _, hour), readings in cells.items():
        by_zone_hour.setdefault((zone, hour), []).extend(readings)

    raw: list[CellTest] = []
    skipped = 0
    for (zone, weekday, hour), readings in sorted(cells.items()):
        treatment = [item.value for item in readings]
        # Baseline: same zone, same hour, every *other* weekday. Holding hour
        # fixed is what stops the scan rediscovering rush hour.
        baseline = [
            item.value for item in by_zone_hour.get((zone, hour), []) if item.weekday != weekday
        ]
        # First pass without the bootstrap: the p-value and point ratio decide
        # which cells are worth reporting, and an interval is only needed for
        # those few. Bootstrapping every cell would compute 672 intervals and
        # discard all but a couple of them.
        comparison = compare_groups(
            treatment, baseline, min_group=min_group, seed=seed, bootstrap=False
        )
        if not comparison.tested:
            skipped += 1
            continue
        raw.append(
            CellTest(
                location_id=zone,
                weekday=weekday,
                hour=hour,
                comparison=comparison,
                localized_lift=float("nan"),
                peer_lift=float("nan"),
            )
        )

    # Second pass: divide out the lift every zone shows at this weekday and hour,
    # leaving what is specific to the zone.
    peers: dict[tuple[int, int], list[float]] = {}
    for test in raw:
        value = test.comparison.effect.value
        if value == value:  # not NaN
            peers.setdefault((test.weekday, test.hour), []).append(value)

    adjusted: list[CellTest] = []
    for test in raw:
        group = peers.get((test.weekday, test.hour), [])
        peer_lift = statistics.median(group) if group else float("nan")
        raw_lift = test.comparison.effect.value
        localized = (
            raw_lift / peer_lift
            if peer_lift == peer_lift and peer_lift != 0 and raw_lift == raw_lift
            else float("nan")
        )
        adjusted.append(
            CellTest(
                location_id=test.location_id,
                weekday=test.weekday,
                hour=test.hour,
                comparison=test.comparison,
                localized_lift=localized,
                peer_lift=peer_lift,
            )
        )
    return adjusted, skipped


def _merge_adjacent(tests: list[CellTest]) -> list[list[CellTest]]:
    """Group surviving cells that share a zone and weekday into hour runs.

    An 18:00-20:00 effect shows up as two adjacent single-hour cells. Reporting
    them separately would be technically true and useless: the pattern a person
    cares about is the window, so contiguous hours are merged into one pattern.
    """
    by_day: dict[tuple[str, int], list[CellTest]] = {}
    for test in tests:
        by_day.setdefault((test.location_id, test.weekday), []).append(test)

    runs: list[list[CellTest]] = []
    for group in by_day.values():
        group.sort(key=lambda item: item.hour)
        current: list[CellTest] = [group[0]]
        for test in group[1:]:
            if test.hour == current[-1].hour + 1:
                current.append(test)
            else:
                runs.append(current)
                current = [test]
        runs.append(current)
    return runs


def discover_spatiotemporal(
    frame: ObservationFrame,
    metric: str,
    *,
    fdr_level: float = DEFAULT_FDR_LEVEL,
    min_localized_lift: float = DEFAULT_MIN_LOCALIZED_LIFT,
    min_group: int = DEFAULT_MIN_GROUP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[list[Pattern], int, int]:
    """Scan for zone-and-weekday-and-hour patterns.

    Args:
        frame: The unified-layer view.
        metric: Metric to scan.
        fdr_level: Benjamini-Hochberg level.
        min_localized_lift: Zone-specific lift required, after the global weekday
            effect is divided out.
        min_group: Minimum observations per group before a cell is tested.
        seed: Bootstrap seed.

    Returns:
        ``(patterns, tests_run, cells_too_sparse)``. The counts are returned so a
        reader can judge the discoveries against the number of hypotheses --
        "three patterns found" means something very different out of 10 tests than
        out of 672.
    """
    tests, skipped = _scan_cells(frame, metric, min_group=min_group, seed=seed)
    if not tests:
        return [], 0, skipped

    raw_p = [test.comparison.p_value for test in tests]
    survives = benjamini_hochberg(raw_p, level=fdr_level)
    adjusted = adjusted_p_values(raw_p)

    surviving: list[CellTest] = []
    adjusted_by_cell: dict[tuple[str, int, int], float] = {}
    for test, passed, adjusted_value in zip(tests, survives, adjusted, strict=True):
        key = (test.location_id, test.weekday, test.hour)
        adjusted_by_cell[key] = adjusted_value
        lift = test.localized_lift
        # Both gates: statistically distinguishable from noise AND large enough
        # to matter once the effect every zone shares is removed.
        if passed and lift == lift and lift >= min_localized_lift:
            surviving.append(test)

    # Second pass: bootstrap an interval for the survivors only.
    by_zone_hour: dict[tuple[str, int], list[Reading]] = {}
    for (zone, _, hour), readings in frame.cells(metric).items():
        by_zone_hour.setdefault((zone, hour), []).extend(readings)

    detailed: list[CellTest] = []
    for test in surviving:
        cell = frame.cells(metric)[(test.location_id, test.weekday, test.hour)]
        baseline = [
            item.value
            for item in by_zone_hour[(test.location_id, test.hour)]
            if item.weekday != test.weekday
        ]
        full = compare_groups(
            [item.value for item in cell],
            baseline,
            min_group=min_group,
            seed=seed,
            bootstrap=True,
        )
        detailed.append(
            CellTest(
                location_id=test.location_id,
                weekday=test.weekday,
                hour=test.hour,
                comparison=full,
                localized_lift=test.localized_lift,
                peer_lift=test.peer_lift,
            )
        )

    patterns: list[Pattern] = []
    for run in _merge_adjacent(detailed):
        patterns.append(_pattern_from_run(run, frame, metric, adjusted_by_cell, fdr_level))

    patterns.sort(key=lambda item: item.pattern_id)
    return patterns, len(tests), skipped


def _pattern_from_run(
    run: list[CellTest],
    frame: ObservationFrame,
    metric: str,
    adjusted_by_cell: dict[tuple[str, int, int], float],
    fdr_level: float,
) -> Pattern:
    """Turn a contiguous run of surviving cells into one pattern."""
    zone = run[0].location_id
    weekday = run[0].weekday
    hours = tuple(test.hour for test in run)

    # The run's headline numbers come from the strongest cell, with the span
    # recorded in the scope. Averaging the cells would understate a sharp peak
    # that happens to sit next to a milder hour.
    strongest = max(
        run,
        key=lambda test: test.localized_lift if test.localized_lift == test.localized_lift else 0.0,
    )
    comparison = strongest.comparison
    key = (strongest.location_id, strongest.weekday, strongest.hour)

    readings = [
        item
        for item in frame.for_metric(metric)
        if item.location_id == zone and item.weekday == weekday and item.hour in hours
    ]
    first = min(item.local_time for item in readings)
    last = max(item.local_time for item in readings)

    scope = PatternScope(metric=metric, location_id=zone, weekday=weekday, hours=hours)
    window = f"{min(hours):02d}:00-{max(hours) + 1:02d}:00"

    return Pattern(
        pattern_id=f"ST-{zone}-{weekday}-{min(hours):02d}{max(hours) + 1:02d}",
        kind=PatternKind.SPATIOTEMPORAL,
        description=(
            f"{zone} on {_weekday_name(weekday)} {window} runs "
            f"x{strongest.localized_lift:.2f} above the same hours on other "
            f"weekdays, after removing the lift every zone shows then "
            f"(x{strongest.peer_lift:.2f})"
        ),
        scope=scope,
        first_detected=first,
        last_observed=last,
        frequency_per_week=len(hours) / 1.0,
        confidence=confidence_from_p(adjusted_by_cell.get(key, comparison.p_value)),
        evidence=Evidence(
            test="mann-whitney-u",
            statistic=comparison.statistic,
            p_value=comparison.p_value,
            adjusted_p_value=adjusted_by_cell.get(key),
            # The localized lift is the reported effect: the raw ratio still
            # carries the weekday effect common to every zone.
            effect=comparison.effect.model_copy(
                update={
                    "value": strongest.localized_lift,
                    "lower": comparison.effect.lower / strongest.peer_lift,
                    "upper": comparison.effect.upper / strongest.peer_lift,
                }
            ),
            n_treatment=comparison.n_treatment,
            n_control=comparison.n_control,
            controls=(
                "hour of day (same hour compared)",
                "zone level (same zone compared)",
                "global weekday effect (divided by the median lift across zones)",
            ),
            notes=(
                f"raw within-zone lift x{comparison.effect.value:.3f}; "
                f"peer lift x{strongest.peer_lift:.3f}",
                f"Benjamini-Hochberg at q={fdr_level}",
                "rainfall is not controlled; it is city-wide and should balance "
                "across weekdays, but a rainier Friday would inflate this",
            ),
        ),
        status=PatternStatus.INSUFFICIENT_HISTORY,
    )


def _weekday_name(weekday: int) -> str:
    """Weekday name for a description."""
    return (
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    )[weekday % 7]
