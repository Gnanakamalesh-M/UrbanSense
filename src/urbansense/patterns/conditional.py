"""Conditional patterns (SPEC section 21).

    Rain + Friday evening -> traffic increase

Rainfall arrives as its own observation series from its own source, so nothing in
the traffic rows says it was raining. The relationship has to be found by joining
the two series on time, which is the point: a conditional pattern is a discovery,
not a column.

Two decisions carry the weight:

**The baseline is built from dry hours only.** Normalizing each reading by its
``(zone, weekday, hour)`` median is what removes the diurnal and zone structure --
but if that median is computed over *all* hours it already contains the rain
effect, and the regression then measures rain against a baseline rain helped
create. The slope comes out biased toward zero. Using dry hours for the baseline
keeps the comparison honest.

**The slope is estimated on light rain.** Any real relationship saturates
eventually, and fitting a straight line through a saturating curve pulls the slope
down in proportion to how much heavy rain happens to be in the sample. Restricting
to light rain is a pre-committed choice that cannot be tuned after seeing the
answer, and the heavy-rain fit is reported alongside so the difference is visible.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from urbansense.patterns.frame import RAINFALL_METRIC, ObservationFrame
from urbansense.patterns.models import (
    Evidence,
    Pattern,
    PatternKind,
    PatternScope,
    PatternStatus,
)
from urbansense.patterns.statistics import (
    SlopeResult,
    confidence_from_p,
    fit_slope,
)

#: Rainfall in mm below which the relationship is assumed not yet to saturate.
#: Pre-committed, not fitted: choosing this after seeing the slope would be
#: tuning the answer.
DEFAULT_LIGHT_RAIN_MM = 10.0

#: Minimum dry readings in a cell before its median is used as a baseline. Below
#: this the "expected" value is noise and the ratio inherits that noise.
DEFAULT_MIN_BASELINE = 4

#: Slope per mm a conditional pattern must reach to be reported. A relationship
#: of 0.0001 per mm is real and irrelevant.
DEFAULT_MIN_SLOPE = 0.005


@dataclass(frozen=True, slots=True)
class ConditionalFit:
    """A fitted condition-to-response relationship.

    Attributes:
        condition: What was varied, e.g. ``rainfall``.
        light: Fit restricted to light values, the headline estimate.
        full: Fit over all values, reported so saturation is visible.
        wet_observations: Readings with the condition present.
        dry_observations: Readings used to build the baseline.
        baseline_cells: Cells with a usable dry-hour baseline.
    """

    condition: str
    light: SlopeResult
    full: SlopeResult
    wet_observations: int
    dry_observations: int
    baseline_cells: int


def _dry_baselines(
    frame: ObservationFrame, metric: str, *, min_baseline: int
) -> dict[tuple[str, int, int], float]:
    """Median value per ``(zone, weekday, hour)``, from dry hours only.

    Cells with too few dry hours are omitted rather than filled, so a reading in
    such a cell contributes nothing to the fit instead of contributing a
    fabricated ratio.
    """
    buckets: dict[tuple[str, int, int], list[float]] = {}
    for reading in frame.for_metric(metric):
        if reading.rain_mm is None or reading.rain_mm > 0.0:
            continue
        buckets.setdefault((reading.location_id, reading.weekday, reading.hour), []).append(
            reading.value
        )
    return {
        key: statistics.median(values)
        for key, values in buckets.items()
        if len(values) >= min_baseline
    }


def fit_rainfall_effect(
    frame: ObservationFrame,
    metric: str,
    *,
    light_rain_mm: float = DEFAULT_LIGHT_RAIN_MM,
    min_baseline: int = DEFAULT_MIN_BASELINE,
) -> ConditionalFit:
    """Estimate how much rainfall moves a metric, per millimetre.

    The response is ``value / dry_baseline - 1``, so the slope reads directly as a
    proportional change per mm and is comparable across zones that differ
    fourfold in volume.
    """
    baselines = _dry_baselines(frame, metric, min_baseline=min_baseline)

    light_x: list[float] = []
    light_y: list[float] = []
    full_x: list[float] = []
    full_y: list[float] = []
    wet = 0

    for reading in frame.for_metric(metric):
        rain = reading.rain_mm
        if rain is None or rain <= 0.0:
            continue
        baseline = baselines.get((reading.location_id, reading.weekday, reading.hour))
        if baseline is None or baseline == 0:
            continue
        wet += 1
        response = reading.value / baseline - 1.0
        full_x.append(rain)
        full_y.append(response)
        if rain <= light_rain_mm:
            light_x.append(rain)
            light_y.append(response)

    dry = sum(
        1
        for reading in frame.for_metric(metric)
        if reading.rain_mm is not None and reading.rain_mm == 0.0
    )

    return ConditionalFit(
        condition=RAINFALL_METRIC,
        light=fit_slope(light_x, light_y),
        full=fit_slope(full_x, full_y),
        wet_observations=wet,
        dry_observations=dry,
        baseline_cells=len(baselines),
    )


def discover_conditional(
    frame: ObservationFrame,
    metric: str,
    *,
    light_rain_mm: float = DEFAULT_LIGHT_RAIN_MM,
    min_slope: float = DEFAULT_MIN_SLOPE,
) -> list[Pattern]:
    """Discover conditional patterns for a metric.

    Returns:
        A pattern per condition whose slope is both distinguishable from zero and
        large enough to matter. An empty list when no condition qualifies -- which
        is a finding, not a failure.
    """
    fit = fit_rainfall_effect(frame, metric, light_rain_mm=light_rain_mm)
    if not fit.light.tested:
        return []

    effect = fit.light.effect
    if not effect.excludes_null or abs(effect.value) < min_slope:
        return []

    readings = [r for r in frame.for_metric(metric) if r.rain_mm is not None]
    if not readings:
        return []
    first = min(r.local_time for r in readings)
    last = max(r.local_time for r in readings)
    wet_hours = sum(1 for r in readings if r.is_wet)
    span_weeks = max(1.0, (last - first).days / 7.0)

    direction = "an increase" if effect.value > 0 else "a decrease"
    return [
        Pattern(
            pattern_id=f"CND-{metric}-{RAINFALL_METRIC}",
            kind=PatternKind.CONDITIONAL,
            description=(
                f"rainfall is associated with {direction} in {metric} of "
                f"{effect.value:+.4f} per mm "
                f"({effect.value * 100:+.2f}% per mm), estimated on rain up to "
                f"{light_rain_mm:g} mm"
            ),
            scope=PatternScope(metric=metric, condition=f"{RAINFALL_METRIC}_mm"),
            first_detected=first,
            last_observed=last,
            frequency_per_week=wet_hours / span_weeks,
            confidence=confidence_from_p(fit.light.p_value),
            evidence=Evidence(
                test="ols-slope",
                statistic=fit.light.effect.value,
                p_value=fit.light.p_value,
                effect=effect,
                n_treatment=fit.light.n,
                n_control=fit.dry_observations,
                controls=(
                    "hour of day, weekday and zone (each reading divided by its own cell median)",
                    "rain itself excluded from the baseline (dry hours only)",
                    f"saturation (fit restricted to rain <= {light_rain_mm:g} mm)",
                ),
                notes=(
                    f"slope over all rain: {fit.full.effect.describe()} "
                    f"(n={fit.full.n}) -- differs from the light-rain estimate by "
                    "the amount heavy rain saturates",
                    f"R^2 {fit.light.r_squared:.4f}: low is expected, since one "
                    "covariate should not explain most of the variation in traffic",
                    f"{fit.baseline_cells} cells had enough dry hours for a baseline",
                ),
            ),
            status=PatternStatus.INSUFFICIENT_HISTORY,
        )
    ]
