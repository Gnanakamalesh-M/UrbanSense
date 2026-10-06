"""Statistical tools for pattern discovery (SPEC sections 21, 38).

Four choices here do most of the work, and each is a deliberate guard against a
way discovery goes wrong.

**Non-parametric tests.** Traffic counts are skewed and heavy-tailed, so a t-test
on raw volumes answers a question about means that nobody asked. Mann-Whitney U
tests whether one group tends to exceed the other, which is the actual claim.

**Effect sizes, not just p-values.** With thousands of hours per zone, a 1.01x
difference is significant and worthless. Every finding carries a ratio of medians
with a bootstrap interval, so size and certainty stay separate numbers.

**Ratio of *medians*, not means.** A single festival evening moves a mean
noticeably; the median is unmoved. Since the point is to find the recurring
structure rather than the exceptions, the robust statistic is the right one.

**Multiple-testing correction.** Scanning 672 cells at p<0.05 yields about 34
false positives by construction. Benjamini-Hochberg controls the expected
proportion of false discoveries, which is the quantity that matters when the
output is a list of claims rather than one decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from urbansense.patterns.models import EffectSize

#: Bootstrap resamples for an effect-size interval. 2,000 is enough for a stable
#: percentile interval and cheap enough to run across hundreds of cells.
DEFAULT_BOOTSTRAP_SAMPLES = 2000

#: Seed for every bootstrap, so a discovery run is reproducible. Pattern
#: discovery that shifted between runs could not be used to track evolution.
DEFAULT_BOOTSTRAP_SEED = 20260101

#: Minimum observations in each group before a comparison is attempted. Below
#: this the median is noise and a "pattern" is an artifact of three readings.
DEFAULT_MIN_GROUP = 12


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The outcome of comparing two groups.

    Attributes:
        statistic: Mann-Whitney U statistic.
        p_value: Two-sided p-value.
        effect: Ratio of medians with a bootstrap interval.
        n_treatment: Observations in the treatment group.
        n_control: Observations in the control group.
        treatment_median: Median of the treatment group.
        control_median: Median of the control group.
        tested: Whether the comparison ran. ``False`` when a group was too small.
    """

    statistic: float
    p_value: float
    effect: EffectSize
    n_treatment: int
    n_control: int
    treatment_median: float
    control_median: float
    tested: bool


#: Either a plain sequence or an already-built float array. Both occur: the
#: scan hands over numpy slices it has already materialized, while callers
#: elsewhere pass lists.
Values = Sequence[float] | NDArray[np.float64]


def _as_array(values: Values) -> NDArray[np.float64]:
    """Drop unknown values rather than filling them.

    Missing stays missing (SPEC section 15): an absent reading contributes
    nothing to a median, and substituting anything would invent evidence.
    """
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return array[~np.isnan(array)]


def median_ratio(
    treatment: Values,
    control: Values,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> EffectSize:
    """Ratio of medians with a percentile bootstrap interval.

    The interval is bootstrapped rather than derived analytically because the
    ratio of two medians has no convenient closed form, and a delta-method
    approximation would understate the uncertainty on the small cells that matter
    most here.

    Returns:
        An :class:`EffectSize` of kind ``median_ratio``. A zero control median
        yields a NaN ratio rather than an infinity, since "infinitely more
        traffic than nothing" is not a usable effect size.
    """
    left = _as_array(treatment)
    right = _as_array(control)
    if left.size == 0 or right.size == 0:
        return EffectSize(
            value=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            kind="median_ratio",
            confidence_level=confidence_level,
        )

    control_median = float(np.median(right))
    point = float(np.median(left)) / control_median if control_median != 0 else float("nan")

    # Vectorized: all resamples drawn and reduced in two calls rather than a
    # Python loop. The scan runs this on hundreds of cells, and the loop version
    # took minutes where this takes under a second.
    generator = np.random.default_rng(seed)
    left_draws = left[generator.integers(0, left.size, size=(bootstrap_samples, left.size))]
    right_draws = right[generator.integers(0, right.size, size=(bootstrap_samples, right.size))]
    numerators = np.median(left_draws, axis=1)
    denominators = np.median(right_draws, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(denominators != 0, numerators / denominators, np.nan)

    finite = ratios[~np.isnan(ratios)]
    if finite.size == 0:
        lower = upper = float("nan")
    else:
        tail = (1.0 - confidence_level) / 2.0
        lower = float(np.quantile(finite, tail))
        upper = float(np.quantile(finite, 1.0 - tail))

    return EffectSize(
        value=point,
        lower=lower,
        upper=upper,
        kind="median_ratio",
        confidence_level=confidence_level,
    )


def compare_groups(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    min_group: int = DEFAULT_MIN_GROUP,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap: bool = True,
) -> ComparisonResult:
    """Test whether a treatment group differs from its control.

    Args:
        treatment: Values under the condition.
        control: Values from the comparison group.
        min_group: Minimum size for either group. Below it the comparison is not
            attempted and ``tested`` is ``False`` -- saying nothing is honest,
            whereas a median of four readings would look like a finding.
        bootstrap_samples: Resamples for the effect-size interval.
        seed: Bootstrap seed, for reproducibility.
        bootstrap: When ``False``, report the point ratio with no interval. A
            scan uses this for its first pass: the p-value and the point estimate
            decide which cells are worth reporting, and an interval is only
            needed for the few that survive. Bootstrapping all 672 cells costs
            twenty seconds to compute intervals that are then discarded.

    Returns:
        A :class:`ComparisonResult`.
    """
    left = _as_array(treatment)
    right = _as_array(control)

    if left.size < min_group or right.size < min_group:
        return ComparisonResult(
            statistic=float("nan"),
            p_value=float("nan"),
            effect=EffectSize(
                value=float("nan"),
                lower=float("nan"),
                upper=float("nan"),
                kind="median_ratio",
            ),
            n_treatment=int(left.size),
            n_control=int(right.size),
            treatment_median=float(np.median(left)) if left.size else float("nan"),
            control_median=float(np.median(right)) if right.size else float("nan"),
            tested=False,
        )

    outcome = stats.mannwhitneyu(left, right, alternative="two-sided")
    if bootstrap:
        effect = median_ratio(left, right, bootstrap_samples=bootstrap_samples, seed=seed)
    else:
        control_median = float(np.median(right))
        effect = EffectSize(
            value=(
                float(np.median(left)) / control_median if control_median != 0 else float("nan")
            ),
            lower=float("nan"),
            upper=float("nan"),
            kind="median_ratio",
        )

    return ComparisonResult(
        statistic=float(outcome.statistic),
        p_value=float(outcome.pvalue),
        effect=effect,
        n_treatment=int(left.size),
        n_control=int(right.size),
        treatment_median=float(np.median(left)),
        control_median=float(np.median(right)),
        tested=True,
    )


def benjamini_hochberg(p_values: Sequence[float], *, level: float = 0.05) -> list[bool]:
    """Which hypotheses survive Benjamini-Hochberg at the given level.

    Controls the expected *proportion* of false discoveries among those reported,
    which is the right quantity when the output is a list of claims. Bonferroni
    would control the chance of any false positive at all and would be far too
    conservative across 672 cells -- it would reject the real pattern along with
    the noise.

    Args:
        p_values: Raw p-values. ``NaN`` entries are treated as not significant
            rather than dropped, so the returned list stays aligned with the
            input.
        level: Target false-discovery rate.

    Returns:
        A list of booleans, aligned with ``p_values``.
    """
    count = len(p_values)
    if count == 0:
        return []

    indexed = [
        (index, value)
        for index, value in enumerate(p_values)
        if value is not None and not math.isnan(value)
    ]
    if not indexed:
        return [False] * count

    indexed.sort(key=lambda pair: pair[1])
    tested = len(indexed)

    # Walk from the largest p-value down and take everything at or below the
    # largest rank whose p-value clears its own threshold.
    threshold_rank = 0
    for rank, (_, value) in enumerate(indexed, start=1):
        if value <= level * rank / tested:
            threshold_rank = rank

    survivors = [False] * count
    for rank, (index, _) in enumerate(indexed, start=1):
        if rank <= threshold_rank:
            survivors[index] = True
    return survivors


def adjusted_p_values(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, aligned with the input.

    Reported alongside the raw value so a reader can see how much the correction
    cost, rather than only whether a finding survived it.
    """
    count = len(p_values)
    if count == 0:
        return []

    indexed = [
        (index, value)
        for index, value in enumerate(p_values)
        if value is not None and not math.isnan(value)
    ]
    adjusted = [float("nan")] * count
    if not indexed:
        return adjusted

    indexed.sort(key=lambda pair: pair[1], reverse=True)
    tested = len(indexed)
    running = 1.0
    for position, (index, value) in enumerate(indexed):
        rank = tested - position
        running = min(running, value * tested / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


@dataclass(frozen=True, slots=True)
class SlopeResult:
    """A fitted linear relationship between a condition and a response.

    Attributes:
        effect: Slope per unit, with its interval.
        intercept: Fitted intercept.
        r_squared: Share of variance explained. Low values are expected here and
            are not a failure: one covariate out of many should not explain most
            of the variation in traffic.
        n: Observations used.
        p_value: Two-sided p-value for the slope.
        tested: Whether the fit ran.
    """

    effect: EffectSize
    intercept: float
    r_squared: float
    n: int
    p_value: float
    tested: bool


def fit_slope(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    min_observations: int = 30,
    confidence_level: float = 0.95,
) -> SlopeResult:
    """Fit ``y = a + b x`` and report ``b`` with an interval.

    Used for a continuous condition such as rainfall, where the question is "how
    much per millimetre?" rather than "wet versus dry". Ordinary least squares is
    appropriate because the response here is a ratio already normalized by its
    cell median, so the heavy tail of raw counts has been divided out.

    Args:
        x_values: The condition.
        y_values: The response.
        min_observations: Minimum points before fitting.
        confidence_level: Coverage of the slope interval.

    Returns:
        A :class:`SlopeResult`; ``tested`` is ``False`` when there were too few
        points or no variation in ``x``.
    """
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]

    untested = SlopeResult(
        effect=EffectSize(
            value=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            kind="slope_per_unit",
            confidence_level=confidence_level,
        ),
        intercept=float("nan"),
        r_squared=float("nan"),
        n=int(x.size),
        p_value=float("nan"),
        tested=False,
    )
    if x.size < min_observations or float(np.std(x)) == 0.0:
        return untested

    fit = stats.linregress(x, y)
    # linregress reports the standard error of the slope; the interval uses the
    # t distribution rather than a normal approximation, which matters on the
    # smaller per-window fits.
    degrees = x.size - 2
    critical = float(stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, degrees))
    half_width = critical * float(fit.stderr)

    return SlopeResult(
        effect=EffectSize(
            value=float(fit.slope),
            lower=float(fit.slope) - half_width,
            upper=float(fit.slope) + half_width,
            kind="slope_per_unit",
            confidence_level=confidence_level,
        ),
        intercept=float(fit.intercept),
        r_squared=float(fit.rvalue) ** 2,
        n=int(x.size),
        p_value=float(fit.pvalue),
        tested=True,
    )


@dataclass(frozen=True, slots=True)
class TrendResult:
    """A monotonic trend test over an ordered series.

    Attributes:
        correlation: Spearman rank correlation with position.
        p_value: Two-sided p-value.
        direction: ``up``, ``down`` or ``flat``.
        n: Windows used.
        tested: Whether the test ran.
    """

    correlation: float
    p_value: float
    direction: str
    n: int
    tested: bool


def test_trend(
    values: Sequence[float | None], *, min_windows: int = 4, level: float = 0.05
) -> TrendResult:
    """Test whether a strength series trends up or down.

    Spearman rather than Pearson: the question is whether strength is rising, not
    whether it is rising *linearly*, and a rank test does not assume a shape.

    Unmeasured windows are skipped rather than interpolated -- a window with too
    little data says nothing about the trend, and filling it would manufacture
    one.
    """
    pairs = [
        (index, value)
        for index, value in enumerate(values)
        if value is not None and not math.isnan(value)
    ]
    if len(pairs) < min_windows:
        return TrendResult(
            correlation=float("nan"),
            p_value=float("nan"),
            direction="flat",
            n=len(pairs),
            tested=False,
        )

    positions = np.asarray([p for p, _ in pairs], dtype=np.float64)
    strengths = np.asarray([v for _, v in pairs], dtype=np.float64)
    if float(np.std(strengths)) == 0.0:
        return TrendResult(
            correlation=0.0, p_value=1.0, direction="flat", n=len(pairs), tested=True
        )

    outcome = stats.spearmanr(positions, strengths)
    correlation = float(outcome.statistic)
    p_value = float(outcome.pvalue)
    if p_value <= level and correlation > 0:
        direction = "up"
    elif p_value <= level and correlation < 0:
        direction = "down"
    else:
        direction = "flat"

    return TrendResult(
        correlation=correlation,
        p_value=p_value,
        direction=direction,
        n=len(pairs),
        tested=True,
    )


def confidence_from_p(p_value: float) -> float:
    """Turn a p-value into a reportable confidence, as ``1 - p``.

    A convenience for display only. It is emphatically *not* the probability the
    pattern is real -- that would need a prior -- and the field is named
    ``confidence`` rather than ``probability`` for that reason.
    """
    if p_value is None or math.isnan(p_value):
        return 0.0
    return max(0.0, min(1.0, 1.0 - p_value))
