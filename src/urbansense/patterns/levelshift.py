"""Level-shift detection (SPEC sections 21, 22).

A city changes: a flyover opens and a corridor's baseline moves permanently. That
is not an anomaly and not a data error -- it is a pattern, and section 22's
``STRENGTHENING`` is the status for it.

The hard part is telling a level shift apart from seasonality. Traffic in June is
higher than in January everywhere, so a within-zone before-and-after comparison
reports a shift in every zone and cannot say which one actually changed. The fix
is **difference-in-differences**: divide each zone's before-and-after ratio by a
reference standing for what every zone shared. Seasonality cancels; a shift
specific to one corridor survives.

**The reference is the lower quartile of the shifts, and that choice carries an
assumption worth stating.** The obvious reference is the median across zones, and
it breaks whenever a large minority of zones has genuinely shifted: with two of
four zones moving, the median sits *between* the two groups, so the adjustment
splits the effect across both. On this dataset that mistake credits an 8% rise to
a zone that did not move, an 8% fall to another, and understates the real shift
from 18% to 8%.

The lower quartile instead assumes **at least some zones are unaffected**, which
is the identifying assumption of every difference-in-differences design, stated
here rather than left implicit. If every zone shifted equally this method reports
nothing, and that is the honest answer: a change affecting the whole city
uniformly cannot be told from seasonality without an external reference.

The change-point is found by scanning candidate months and taking the one that
maximizes the adjusted shift, rather than being told where to look.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from urbansense.patterns.frame import ObservationFrame
from urbansense.patterns.models import (
    EffectSize,
    Evidence,
    Pattern,
    PatternKind,
    PatternScope,
    PatternStatus,
)
from urbansense.patterns.statistics import (
    DEFAULT_BOOTSTRAP_SEED,
    compare_groups,
    confidence_from_p,
)

#: Adjusted shift a zone must show before a level shift is reported. Seasonality
#: and noise leave residual movement of a few percent even after the control
#: adjustment, so the floor sits clear of it.
DEFAULT_MIN_SHIFT = 1.08

#: Months of data required either side of a candidate change-point. A shift
#: detected from two weeks of data is a fluctuation.
DEFAULT_MIN_SIDE_MONTHS = 2

#: Quantile of the cross-zone shift distribution used as the shared-movement
#: reference. The median fails once a large minority of zones has genuinely
#: shifted, because it then sits between the shifted and unshifted groups.
DEFAULT_REFERENCE_QUANTILE = 0.25


@dataclass(frozen=True, slots=True)
class ShiftCandidate:
    """One zone's shift at one candidate change-point.

    Attributes:
        location_id: Zone.
        change_point: First month on the later side.
        raw_ratio: The zone's own after-to-before median ratio.
        peer_ratio: Reference ratio -- the lower quartile across zones, standing
            for the movement every zone shared.
        adjusted_ratio: ``raw / peer`` -- the zone-specific shift.
        n_before: Readings before the change-point.
        n_after: Readings after it.
    """

    location_id: str
    change_point: str
    raw_ratio: float
    peer_ratio: float
    adjusted_ratio: float
    n_before: int
    n_after: int


def _monthly_values(frame: ObservationFrame, metric: str) -> dict[str, dict[str, list[float]]]:
    """Values per zone per month."""
    zones = set(frame.zones(metric))
    grouped: dict[str, dict[str, list[float]]] = {}
    for reading in frame.for_metric(metric):
        if reading.location_id not in zones:
            continue
        grouped.setdefault(reading.location_id, {}).setdefault(reading.month_key, []).append(
            reading.value
        )
    return grouped


def scan_change_points(
    frame: ObservationFrame,
    metric: str,
    *,
    min_side_months: int = DEFAULT_MIN_SIDE_MONTHS,
    reference_quantile: float = DEFAULT_REFERENCE_QUANTILE,
) -> dict[str, ShiftCandidate]:
    """Find each zone's strongest candidate change-point.

    Returns:
        The best candidate per zone, keyed by zone. Zones with too few months to
        split are absent rather than reported with a guess.
    """
    monthly = _monthly_values(frame, metric)
    months = frame.windows()
    if len(months) < 2 * min_side_months:
        return {}

    candidates = months[min_side_months : len(months) - min_side_months + 1]
    best: dict[str, ShiftCandidate] = {}

    for change_point in candidates:
        index = months.index(change_point)
        before_months = months[:index]
        after_months = months[index:]

        ratios: dict[str, tuple[float, int, int]] = {}
        for zone, by_month in monthly.items():
            before = [v for m in before_months for v in by_month.get(m, [])]
            after = [v for m in after_months for v in by_month.get(m, [])]
            if not before or not after:
                continue
            before_median = statistics.median(before)
            if before_median == 0:
                continue
            ratios[zone] = (
                statistics.median(after) / before_median,
                len(before),
                len(after),
            )

        if len(ratios) < 2:
            continue
        # What every zone shared. The lower quartile rather than the median: the
        # median is contaminated once a large minority of zones has genuinely
        # shifted, since it then sits between the two groups and splits the effect
        # across both. This assumes at least some zones are unaffected -- the
        # identifying assumption, stated in the module docstring.
        peer = float(
            np.quantile(
                np.asarray([value for value, _, _ in ratios.values()], dtype=np.float64),
                reference_quantile,
            )
        )
        if peer == 0:
            continue

        for zone, (raw, n_before, n_after) in ratios.items():
            adjusted = raw / peer
            current = best.get(zone)
            if current is None or abs(adjusted - 1.0) > abs(current.adjusted_ratio - 1.0):
                best[zone] = ShiftCandidate(
                    location_id=zone,
                    change_point=change_point,
                    raw_ratio=raw,
                    peer_ratio=peer,
                    adjusted_ratio=adjusted,
                    n_before=n_before,
                    n_after=n_after,
                )
    return best


def discover_level_shifts(
    frame: ObservationFrame,
    metric: str,
    *,
    min_shift: float = DEFAULT_MIN_SHIFT,
    min_side_months: int = DEFAULT_MIN_SIDE_MONTHS,
    reference_quantile: float = DEFAULT_REFERENCE_QUANTILE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[Pattern]:
    """Discover sustained changes in a zone's baseline level.

    Returns:
        A pattern per zone whose adjusted shift clears ``min_shift`` and whose
        before-and-after distributions differ significantly. Zones whose movement
        is explained by seasonality produce nothing, which is the point.
    """
    candidates = scan_change_points(
        frame,
        metric,
        min_side_months=min_side_months,
        reference_quantile=reference_quantile,
    )
    monthly = _monthly_values(frame, metric)
    months = frame.windows()

    patterns: list[Pattern] = []
    for zone, candidate in sorted(candidates.items()):
        if abs(candidate.adjusted_ratio - 1.0) < (min_shift - 1.0):
            continue

        index = months.index(candidate.change_point)
        before = [v for m in months[:index] for v in monthly[zone].get(m, [])]
        after = [v for m in months[index:] for v in monthly[zone].get(m, [])]
        # Fewer resamples than the default: these groups hold thousands of
        # readings, and the median of a large sample is stable enough that 500
        # resamples give the same interval to three decimals for a fraction of
        # the cost.
        comparison = compare_groups(after, before, seed=seed, bootstrap=True, bootstrap_samples=500)
        if not comparison.tested or not comparison.effect.excludes_null:
            continue

        readings = [r for r in frame.for_metric(metric) if r.location_id == zone]
        first = min(r.local_time for r in readings)
        last = max(r.local_time for r in readings)
        direction = "rose" if candidate.adjusted_ratio > 1.0 else "fell"

        patterns.append(
            Pattern(
                pattern_id=f"LVL-{zone}-{candidate.change_point}",
                kind=PatternKind.LEVEL_SHIFT,
                description=(
                    f"{zone} baseline {direction} by "
                    f"x{candidate.adjusted_ratio:.3f} from {candidate.change_point}, "
                    f"after dividing out the x{candidate.peer_ratio:.3f} movement "
                    "every zone shared"
                ),
                scope=PatternScope(metric=metric, location_id=zone),
                first_detected=first,
                last_observed=last,
                # A level shift is not a recurring event; it happens once and
                # persists, so a per-week frequency would be misleading.
                frequency_per_week=0.0,
                confidence=confidence_from_p(comparison.p_value),
                evidence=Evidence(
                    test="mann-whitney-u + difference-in-differences",
                    statistic=comparison.statistic,
                    p_value=comparison.p_value,
                    effect=EffectSize(
                        value=candidate.adjusted_ratio,
                        lower=comparison.effect.lower / candidate.peer_ratio,
                        upper=comparison.effect.upper / candidate.peer_ratio,
                        kind="median_ratio",
                    ),
                    n_treatment=candidate.n_after,
                    n_control=candidate.n_before,
                    controls=(
                        "seasonality (divided by the lower-quartile shift across "
                        "zones, standing for the shared movement)",
                        "zone level (each zone compared against itself)",
                    ),
                    notes=(
                        f"raw within-zone shift x{candidate.raw_ratio:.3f}; "
                        f"peer shift x{candidate.peer_ratio:.3f}",
                        "change-point chosen by scanning candidate months, not supplied",
                        "assumes at least some zones are unaffected; a change "
                        "affecting every zone equally cannot be told from "
                        "seasonality without an external reference",
                        "a level shift is a change in the city, not a data error; "
                        "adapting a model to it is a separate decision",
                    ),
                ),
                status=PatternStatus.INSUFFICIENT_HISTORY,
            )
        )
    return patterns


def change_point_time(pattern: Pattern) -> datetime | None:
    """Parse a level-shift pattern's change-point month into a datetime."""
    if pattern.kind is not PatternKind.LEVEL_SHIFT:
        return None
    suffix = pattern.pattern_id.rsplit("-", 2)[-2:]
    try:
        year, month = int(suffix[0]), int(suffix[1])
    except (ValueError, IndexError):  # pragma: no cover - ids are built above
        return None
    return datetime(year, month, 1)
