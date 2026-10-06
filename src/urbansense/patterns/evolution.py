"""Pattern evolution (SPEC section 22).

    The system must continuously evaluate whether old patterns still hold.

Which only means something if strength is measured *per window*. A single number
cannot be strengthening. So every pattern's effect is recomputed month by month,
and the status comes from that series plus a trend test.

The status rules are deliberately conservative about the two interesting labels.
``STRENGTHENING`` and ``WEAKENING`` require a **significant monotonic trend**, not
just a recent window above the historical median -- adjacent windows differ by
noise constantly, and a status that flipped every month would carry no
information. ``CONFLICTING`` exists for the case the spec names: evidence that
disagrees with itself, where the effect reverses direction across windows rather
than merely varying in size.

Windows that could not be measured stay unmeasured. Interpolating them would
manufacture the very trend the test is looking for.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from urbansense.patterns.frame import ObservationFrame, Reading
from urbansense.patterns.models import (
    Pattern,
    PatternKind,
    PatternStatus,
    WindowStrength,
)
from urbansense.patterns.statistics import DEFAULT_MIN_GROUP, test_trend

#: Windows at the end of the series treated as "recent".
DEFAULT_RECENT_WINDOWS = 2

#: Minimum observations per side before a window's strength is computed.
DEFAULT_MIN_WINDOW_GROUP = 4

#: Relative difference between recent and historical strength below which the
#: pattern is called stable, absent a significant trend.
DEFAULT_STABLE_BAND = 0.15


@dataclass(frozen=True, slots=True)
class EvolutionSummary:
    """The outcome of tracking a pattern across windows.

    Attributes:
        windows: Per-window strength, including unmeasured ones.
        historical_strength: Median strength over the earlier windows.
        recent_strength: Median strength over the recent windows.
        status: How the pattern is behaving.
        reason: Why that status was assigned, in words.
    """

    windows: tuple[WindowStrength, ...]
    historical_strength: float | None
    recent_strength: float | None
    status: PatternStatus
    reason: str


def _window_strength_spatiotemporal(
    readings: list[Reading], pattern: Pattern, *, min_group: int
) -> tuple[float | None, int, int]:
    """Strength of a spatio-temporal pattern within one window.

    Same comparison as discovery -- the cell against the same hours on other
    weekdays -- so a window's strength is on the same scale as the headline
    figure and the two can be compared.
    """
    scope = pattern.scope
    treatment = [
        r.value
        for r in readings
        if r.location_id == scope.location_id
        and r.weekday == scope.weekday
        and r.hour in scope.hours
    ]
    control = [
        r.value
        for r in readings
        if r.location_id == scope.location_id
        and r.weekday != scope.weekday
        and r.hour in scope.hours
    ]
    if len(treatment) < min_group or len(control) < min_group:
        return None, len(treatment), len(control)
    control_median = statistics.median(control)
    if control_median == 0:
        return None, len(treatment), len(control)
    return statistics.median(treatment) / control_median, len(treatment), len(control)


def _window_strength_level(
    readings: list[Reading], pattern: Pattern, *, min_group: int
) -> tuple[float | None, int, int]:
    """Strength of a level-shift pattern within one window.

    The zone's median relative to the median across all zones in the same window.
    Expressing it as a share removes the seasonal movement the whole city shares,
    which is what makes the series comparable across months.
    """
    zone = pattern.scope.location_id
    mine = [r.value for r in readings if r.location_id == zone]
    others = [r.value for r in readings if r.location_id != zone]
    if len(mine) < min_group or len(others) < min_group:
        return None, len(mine), len(others)
    peer = statistics.median(others)
    if peer == 0:
        return None, len(mine), len(others)
    return statistics.median(mine) / peer, len(mine), len(others)


def _window_strength_conditional(
    readings: list[Reading], pattern: Pattern, *, min_group: int
) -> tuple[float | None, int, int]:
    """Strength of a conditional pattern within one window.

    Each wet reading is divided by the dry median of its own
    ``(zone, weekday, hour)`` cell, and the window's strength is the median of
    those ratios -- the same normalization the headline fit uses.

    Comparing raw wet and dry medians across a whole month instead would mix hours
    of day together, and wet hours are not spread evenly across the day: a month
    whose rain happened to fall overnight would look like the effect collapsing
    rather than like a quiet month. That is what produced a spurious CONFLICTING
    status before this was fixed.
    """
    baselines: dict[tuple[str, int, int], list[float]] = {}
    for reading in readings:
        if reading.rain_mm is not None and reading.rain_mm == 0.0:
            baselines.setdefault((reading.location_id, reading.weekday, reading.hour), []).append(
                reading.value
            )
    medians = {
        key: statistics.median(values) for key, values in baselines.items() if len(values) >= 2
    }

    ratios: list[float] = []
    for reading in readings:
        if not reading.is_wet:
            continue
        baseline = medians.get((reading.location_id, reading.weekday, reading.hour))
        if baseline is None or baseline == 0:
            continue
        ratios.append(reading.value / baseline)

    dry_count = sum(1 for r in readings if r.rain_mm is not None and r.rain_mm == 0.0)
    if len(ratios) < min_group or dry_count < min_group:
        return None, len(ratios), dry_count
    return statistics.median(ratios), len(ratios), dry_count


#: Kinds with a defined per-window strength measure. The descriptive profiles are
#: absent on purpose: a "peak-to-trough ratio for March" is computable but its
#: movement between months reflects which hours happened to be observed more than
#: anything about the city, so tracking it would manufacture statuses.
TRACKABLE_KINDS = frozenset(
    {
        PatternKind.SPATIOTEMPORAL,
        PatternKind.LEVEL_SHIFT,
        PatternKind.CONDITIONAL,
    }
)


def _strength_for(
    pattern: Pattern, readings: list[Reading], *, min_group: int
) -> tuple[float | None, int, int]:
    """Dispatch to the right per-window strength measure."""
    if pattern.kind is PatternKind.SPATIOTEMPORAL:
        return _window_strength_spatiotemporal(readings, pattern, min_group=min_group)
    if pattern.kind is PatternKind.LEVEL_SHIFT:
        return _window_strength_level(readings, pattern, min_group=min_group)
    if pattern.kind is PatternKind.CONDITIONAL:
        return _window_strength_conditional(readings, pattern, min_group=min_group)
    return None, 0, 0


def classify(
    windows: tuple[WindowStrength, ...],
    *,
    recent_windows: int = DEFAULT_RECENT_WINDOWS,
    stable_band: float = DEFAULT_STABLE_BAND,
) -> EvolutionSummary:
    """Assign a status from a strength series.

    The order of checks matters: insufficient history first, because every other
    label would be an overstatement; then disappearance and novelty, which are
    about presence rather than size; then trend, which needs the series.
    """
    measured = [w for w in windows if w.measured and w.strength is not None]
    if len(measured) < 3:
        return EvolutionSummary(
            windows=windows,
            historical_strength=(
                statistics.median([w.strength for w in measured if w.strength is not None])
                if measured
                else None
            ),
            recent_strength=None,
            status=PatternStatus.INSUFFICIENT_HISTORY,
            reason=(
                f"only {len(measured)} window(s) could be measured; evolution needs at least three"
            ),
        )

    recent_slice = windows[-recent_windows:]
    historical_slice = windows[:-recent_windows]

    recent_measured = [w.strength for w in recent_slice if w.measured and w.strength is not None]
    historical_measured = [
        w.strength for w in historical_slice if w.measured and w.strength is not None
    ]

    if historical_measured and not recent_measured:
        return EvolutionSummary(
            windows=windows,
            historical_strength=statistics.median(historical_measured),
            recent_strength=None,
            status=PatternStatus.DISAPPEARED,
            reason=("held in the earlier windows but could not be measured in the recent ones"),
        )
    if recent_measured and not historical_measured:
        return EvolutionSummary(
            windows=windows,
            historical_strength=None,
            recent_strength=statistics.median(recent_measured),
            status=PatternStatus.NEW,
            reason="first measurable in the most recent windows",
        )

    historical = statistics.median(historical_measured)
    recent = statistics.median(recent_measured)

    series = [w.strength for w in windows if w.measured and w.strength is not None]
    # Direction is taken relative to 1.0, the no-effect point for a ratio. A
    # series that crosses it has reversed sign, which is a different claim from
    # merely shrinking.
    above = [s for s in series if s > 1.0]
    below = [s for s in series if s < 1.0]
    if above and below and min(len(above), len(below)) >= 2:
        return EvolutionSummary(
            windows=windows,
            historical_strength=historical,
            recent_strength=recent,
            status=PatternStatus.CONFLICTING,
            reason=(
                f"the effect reverses direction across windows "
                f"({len(above)} above 1.0, {len(below)} below): the evidence "
                "disagrees with itself"
            ),
        )

    trend = test_trend(series)
    relative = (recent - historical) / abs(historical) if historical else 0.0

    if trend.direction == "up" and relative > 0:
        return EvolutionSummary(
            windows=windows,
            historical_strength=historical,
            recent_strength=recent,
            status=PatternStatus.STRENGTHENING,
            reason=(
                f"recent strength {recent:.3f} against historical {historical:.3f} "
                f"({relative:+.1%}), with a significant upward trend "
                f"(Spearman {trend.correlation:+.2f}, p={trend.p_value:.4f})"
            ),
        )
    if trend.direction == "down" and relative < 0:
        return EvolutionSummary(
            windows=windows,
            historical_strength=historical,
            recent_strength=recent,
            status=PatternStatus.WEAKENING,
            reason=(
                f"recent strength {recent:.3f} against historical {historical:.3f} "
                f"({relative:+.1%}), with a significant downward trend "
                f"(Spearman {trend.correlation:+.2f}, p={trend.p_value:.4f})"
            ),
        )

    if abs(relative) <= stable_band:
        reason = (
            f"recent strength {recent:.3f} within {stable_band:.0%} of historical "
            f"{historical:.3f} and no significant trend"
        )
    else:
        reason = (
            f"recent strength {recent:.3f} differs from historical "
            f"{historical:.3f} by {relative:+.1%}, but with no significant "
            f"monotonic trend (p={trend.p_value:.4f}) the movement is not "
            "distinguishable from fluctuation"
        )
    return EvolutionSummary(
        windows=windows,
        historical_strength=historical,
        recent_strength=recent,
        status=PatternStatus.STABLE,
        reason=reason,
    )


def track_evolution(
    pattern: Pattern,
    frame: ObservationFrame,
    *,
    recent_windows: int = DEFAULT_RECENT_WINDOWS,
    min_group: int = DEFAULT_MIN_WINDOW_GROUP,
) -> Pattern:
    """Recompute a pattern's strength per window and assign its status.

    Returns:
        A new pattern -- the model is frozen -- carrying the strength series,
        the historical and recent figures, and the status with its reason.
    """
    if pattern.kind not in TRACKABLE_KINDS:
        # Said plainly rather than reported as "insufficient history", which
        # would wrongly suggest more data would help.
        return pattern.model_copy(
            update={
                "status": PatternStatus.INSUFFICIENT_HISTORY,
                "status_reason": (
                    f"evolution is not tracked for {pattern.kind.value} patterns: "
                    "they are descriptive context rather than a tested effect, and "
                    "a per-window figure would move with which hours happened to "
                    "be observed"
                ),
            }
        )

    by_window: dict[str, list[Reading]] = {}
    for reading in frame.for_metric(pattern.scope.metric):
        by_window.setdefault(reading.month_key, []).append(reading)

    windows: list[WindowStrength] = []
    for label in frame.windows():
        readings = by_window.get(label, [])
        strength, n_treatment, n_control = _strength_for(pattern, readings, min_group=min_group)
        windows.append(
            WindowStrength(
                window=label,
                strength=strength,
                n_treatment=n_treatment,
                n_control=n_control,
                measured=strength is not None and not math.isnan(strength),
            )
        )

    summary = classify(tuple(windows), recent_windows=recent_windows)
    return pattern.model_copy(
        update={
            "strength_by_window": summary.windows,
            "historical_strength": summary.historical_strength,
            "recent_strength": summary.recent_strength,
            "status": summary.status,
            "status_reason": summary.reason,
        }
    )


def track_all(
    patterns: list[Pattern],
    frame: ObservationFrame,
    *,
    recent_windows: int = DEFAULT_RECENT_WINDOWS,
) -> list[Pattern]:
    """Track evolution for every pattern."""
    return [track_evolution(pattern, frame, recent_windows=recent_windows) for pattern in patterns]


__all__ = [
    "DEFAULT_MIN_GROUP",
    "DEFAULT_RECENT_WINDOWS",
    "TRACKABLE_KINDS",
    "EvolutionSummary",
    "classify",
    "track_all",
    "track_evolution",
]
