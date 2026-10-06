"""Temporal and spatial profiles (SPEC section 21).

    Temporal patterns: Hourly, Daily, Weekly, Monthly, Seasonal
    Spatial patterns: Hotspots, Repeated locations

These are the patterns the other analyses *condition on*, and they are emitted
rather than computed and discarded. A reader who sees "ZONE-B on Friday evening
runs 2.3x its other weekdays" needs to know that ZONE-B's evening peak is already
five times its overnight trough, or the first number is uninterpretable.

Each profile is tested for whether the grouping explains anything at all --
Kruskal-Wallis across the groups, which is the non-parametric analogue of a
one-way ANOVA and makes no assumption about the shape of the distribution. The
effect size is the peak-to-trough ratio, which is the thing a person would
actually quote.

A hotspot is a zone whose median materially exceeds the city median. "Repeated
locations" in the spec's sense is the same question asked over time, so the
hotspot test is run per window by the evolution tracker rather than being a
separate pattern kind.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

from scipy import stats

from urbansense.patterns.frame import ObservationFrame, Reading
from urbansense.patterns.models import (
    EffectSize,
    Evidence,
    Pattern,
    PatternKind,
    PatternScope,
    PatternStatus,
)
from urbansense.patterns.statistics import DEFAULT_MIN_GROUP, confidence_from_p

#: Peak-to-trough ratio a profile must show before it is reported as a pattern.
#: A profile that varies by 5% across the day is real and not worth a reader's
#: attention.
DEFAULT_MIN_PROFILE_RANGE = 1.5

#: Ratio against the city median before a zone counts as a hotspot.
DEFAULT_MIN_HOTSPOT_RATIO = 1.3


def _profile(
    readings: tuple[Reading, ...], key: Callable[[Reading], int]
) -> dict[int, list[float]]:
    """Group readings by a derived integer key."""
    grouped: dict[int, list[float]] = {}
    for reading in readings:
        grouped.setdefault(key(reading), []).append(reading.value)
    return grouped


def _profile_pattern(
    frame: ObservationFrame,
    metric: str,
    zone: str,
    kind: PatternKind,
    key: Callable[[Reading], int],
    label: str,
    *,
    min_group: int,
    min_range: float,
) -> Pattern | None:
    """Test whether a grouping explains variation, and describe it if so."""
    readings = tuple(item for item in frame.for_metric(metric) if item.location_id == zone)
    if not readings:
        return None

    groups = {
        bucket: values
        for bucket, values in _profile(readings, key).items()
        if len(values) >= min_group
    }
    if len(groups) < 2:
        return None

    medians = {bucket: statistics.median(values) for bucket, values in groups.items()}
    trough_bucket = min(medians, key=lambda b: medians[b])
    peak_bucket = max(medians, key=lambda b: medians[b])
    trough, peak = medians[trough_bucket], medians[peak_bucket]
    if trough <= 0:
        return None

    ratio = peak / trough
    if ratio < min_range:
        return None

    # Kruskal-Wallis: does the grouping explain anything? Non-parametric, so it
    # makes no claim about the shape of the distribution within a group.
    outcome = stats.kruskal(*groups.values())

    first = min(item.local_time for item in readings)
    last = max(item.local_time for item in readings)

    return Pattern(
        pattern_id=f"PRF-{kind.value}-{zone}",
        kind=kind,
        description=(
            f"{zone} {label} profile: peak at {label} {peak_bucket} "
            f"({peak:,.0f}) is x{ratio:.2f} the trough at {label} "
            f"{trough_bucket} ({trough:,.0f})"
        ),
        scope=PatternScope(metric=metric, location_id=zone),
        first_detected=first,
        last_observed=last,
        frequency_per_week=0.0,
        confidence=confidence_from_p(float(outcome.pvalue)),
        evidence=Evidence(
            test="kruskal-wallis",
            statistic=float(outcome.statistic),
            p_value=float(outcome.pvalue),
            effect=EffectSize(
                value=ratio,
                # A peak-to-trough ratio has no natural interval without
                # bootstrapping both ends; the Kruskal-Wallis p-value carries the
                # certainty and the bounds are left unclaimed rather than faked.
                lower=float("nan"),
                upper=float("nan"),
                kind="median_ratio",
            ),
            n_treatment=len(groups[peak_bucket]),
            n_control=len(groups[trough_bucket]),
            controls=(f"zone (single zone, {zone})",),
            notes=(
                f"{len(groups)} {label} groups compared",
                "descriptive: this is the context other patterns are measured "
                "against, not a claim about cause",
            ),
        ),
        status=PatternStatus.INSUFFICIENT_HISTORY,
    )


def discover_profiles(
    frame: ObservationFrame,
    metric: str,
    *,
    min_group: int = DEFAULT_MIN_GROUP,
    min_range: float = DEFAULT_MIN_PROFILE_RANGE,
) -> list[Pattern]:
    """Discover hourly, weekday and seasonal profiles per zone."""
    specifications = (
        (PatternKind.HOURLY_PROFILE, lambda r: r.hour, "hour"),
        (PatternKind.WEEKDAY_PROFILE, lambda r: r.weekday, "weekday"),
        (PatternKind.SEASONAL_PROFILE, lambda r: r.local_time.month, "month"),
    )
    patterns: list[Pattern] = []
    for zone in frame.zones(metric):
        for kind, key, label in specifications:
            found = _profile_pattern(
                frame,
                metric,
                zone,
                kind,
                key,
                label,
                min_group=min_group,
                min_range=min_range,
            )
            if found is not None:
                patterns.append(found)
    patterns.sort(key=lambda item: item.pattern_id)
    return patterns


def discover_hotspots(
    frame: ObservationFrame,
    metric: str,
    *,
    min_ratio: float = DEFAULT_MIN_HOTSPOT_RATIO,
    min_group: int = DEFAULT_MIN_GROUP,
) -> list[Pattern]:
    """Discover zones carrying materially more traffic than the city median.

    The comparison is against the median of *other* zones, not of all zones: a
    zone included in its own baseline drags the baseline toward itself and
    understates how far it stands out.
    """
    zones = frame.zones(metric)
    if len(zones) < 2:
        return []

    by_zone = {
        zone: [item.value for item in frame.for_metric(metric) if item.location_id == zone]
        for zone in zones
    }

    patterns: list[Pattern] = []
    for zone, values in sorted(by_zone.items()):
        if len(values) < min_group:
            continue
        others = [v for other, vs in by_zone.items() if other != zone for v in vs]
        if len(others) < min_group:
            continue

        peer_median = statistics.median(others)
        if peer_median <= 0:
            continue
        ratio = statistics.median(values) / peer_median
        if ratio < min_ratio:
            continue

        outcome = stats.mannwhitneyu(values, others, alternative="two-sided")
        readings = [item for item in frame.for_metric(metric) if item.location_id == zone]
        patterns.append(
            Pattern(
                pattern_id=f"HOT-{zone}",
                kind=PatternKind.HOTSPOT,
                description=(f"{zone} carries x{ratio:.2f} the median {metric} of the other zones"),
                scope=PatternScope(metric=metric, location_id=zone),
                first_detected=min(item.local_time for item in readings),
                last_observed=max(item.local_time for item in readings),
                frequency_per_week=0.0,
                confidence=confidence_from_p(float(outcome.pvalue)),
                evidence=Evidence(
                    test="mann-whitney-u",
                    statistic=float(outcome.statistic),
                    p_value=float(outcome.pvalue),
                    effect=EffectSize(
                        value=ratio,
                        lower=float("nan"),
                        upper=float("nan"),
                        kind="median_ratio",
                    ),
                    n_treatment=len(values),
                    n_control=len(others),
                    controls=("the zone is excluded from its own baseline",),
                    notes=(
                        "a hotspot is a statement about level, not about risk: a "
                        "busy arterial is not necessarily a problem",
                    ),
                ),
                status=PatternStatus.INSUFFICIENT_HISTORY,
            )
        )
    return patterns
