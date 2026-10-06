"""Pattern interaction (SPEC section 23).

    Friday -> high traffic, Rain -> high traffic does NOT automatically mean
    Friday + Rain -> even higher traffic. The system must test the relationship
    against evidence.

So this module tests it, and the design is shaped by two things that make
interaction tests easy to get wrong.

**An interaction is only defined relative to a scale.** Two effects that compose
*multiplicatively* show no interaction on the log scale and a positive one on the
raw scale -- same data, opposite verdicts. The scale is therefore a named field on
the result, and the test runs on logs, where "no interaction" means "the two
effects compose independently". Reporting a verdict without the scale would be
reporting nothing.

**Interaction cells are small, so power is low.** The joint cell is the
intersection of two conditions, and here it holds roughly sixteen hours against
hundreds elsewhere. A point estimate from sixteen observations will be some
distance from zero by luck alone, so the verdict comes from the confidence
interval and ``NO_EVIDENCE`` is reported as *absence of evidence* rather than
evidence of absence. The cell counts travel with the result so a reader can see
how much the null is worth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from urbansense.patterns.frame import ObservationFrame, Reading
from urbansense.patterns.models import (
    EffectSize,
    InteractionTest,
    InteractionVerdict,
    PatternScope,
)

#: Minimum observations in every cell before the model is fitted. Below this the
#: interval is so wide that the test cannot distinguish anything from anything.
DEFAULT_MIN_CELL = 8

#: Scale the test runs on, recorded on every result.
LOG_SCALE = "log(value) - effects composing multiplicatively show no interaction"


@dataclass(frozen=True, slots=True)
class _Cells:
    """The four condition cells of a 2x2 interaction."""

    both: list[float]
    first_only: list[float]
    second_only: list[float]
    neither: list[float]

    @property
    def counts(self) -> dict[str, int]:
        """Observation counts per cell."""
        return {
            "both": len(self.both),
            "first_only": len(self.first_only),
            "second_only": len(self.second_only),
            "neither": len(self.neither),
        }

    @property
    def smallest(self) -> int:
        """Smallest cell, which bounds the power of the whole test."""
        return min(self.counts.values())


def _split_cells(
    readings: list[Reading],
    first: object,
    second: object,
) -> _Cells:
    """Partition readings by two boolean conditions."""
    both: list[float] = []
    first_only: list[float] = []
    second_only: list[float] = []
    neither: list[float] = []
    for reading in readings:
        in_first = bool(first(reading))  # type: ignore[operator]
        in_second = bool(second(reading))  # type: ignore[operator]
        if in_first and in_second:
            both.append(reading.value)
        elif in_first:
            first_only.append(reading.value)
        elif in_second:
            second_only.append(reading.value)
        else:
            neither.append(reading.value)
    return _Cells(both, first_only, second_only, neither)


def _ols_interaction(
    cells: _Cells, *, confidence_level: float = 0.95
) -> tuple[EffectSize, EffectSize, EffectSize]:
    """Fit ``log(y) ~ a + b1 x1 + b2 x2 + b12 x1 x2`` and return the three effects.

    Logs are taken because the generator of any multiplicative process is additive
    there: a pure "each condition scales the level" world has ``b12 == 0``, so a
    non-zero ``b12`` means something beyond independent composition.
    """
    rows: list[tuple[float, float, float]] = []
    values: list[float] = []
    for value in cells.neither:
        rows.append((0.0, 0.0, 0.0))
        values.append(value)
    for value in cells.first_only:
        rows.append((1.0, 0.0, 0.0))
        values.append(value)
    for value in cells.second_only:
        rows.append((0.0, 1.0, 0.0))
        values.append(value)
    for value in cells.both:
        rows.append((1.0, 1.0, 1.0))
        values.append(value)

    design = np.column_stack([np.ones(len(rows)), np.asarray(rows, dtype=np.float64)])
    # Values must be positive to take logs. A non-positive traffic count would
    # have been quarantined upstream, so this guards against a caller passing a
    # metric where zero is legitimate rather than against expected data.
    response = np.log(np.asarray(values, dtype=np.float64))

    coefficients, _residuals, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    degrees = design.shape[0] - rank
    residual = response - design @ coefficients
    variance = float(residual @ residual) / degrees
    covariance = variance * np.linalg.pinv(design.T @ design)
    errors = np.sqrt(np.diag(covariance))
    critical = float(stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, degrees))

    def effect(index: int) -> EffectSize:
        value = float(coefficients[index])
        half = critical * float(errors[index])
        return EffectSize(
            value=value,
            lower=value - half,
            upper=value + half,
            kind="log_difference",
            confidence_level=confidence_level,
        )

    return effect(1), effect(2), effect(3)


def test_interaction(
    frame: ObservationFrame,
    metric: str,
    *,
    location_id: str | None,
    first_name: str,
    first_condition: object,
    second_name: str,
    second_condition: object,
    hours: tuple[int, ...] = (),
    weekday: int | None = None,
    min_cell: int = DEFAULT_MIN_CELL,
) -> InteractionTest:
    """Test whether two conditions interact beyond composing independently.

    Args:
        frame: The unified-layer view.
        metric: Metric under test.
        location_id: Zone to restrict to, or ``None`` for all zones.
        first_name: Name of the first condition, for the report.
        first_condition: Predicate on a :class:`Reading`.
        second_name: Name of the second condition.
        second_condition: Predicate on a :class:`Reading`.
        hours: Local hours to restrict to. Empty means all.
        weekday: Local weekday to restrict to, or ``None`` for all.
        min_cell: Minimum observations in every cell.

    Returns:
        An :class:`InteractionTest`. ``UNTESTABLE`` when a cell was too small --
        which is a different statement from ``NO_EVIDENCE`` and is kept distinct.
    """
    readings = [
        reading
        for reading in frame.for_metric(metric)
        if (location_id is None or reading.location_id == location_id)
        and (not hours or reading.hour in hours)
        and (weekday is None or reading.weekday == weekday)
        and reading.value > 0
    ]
    cells = _split_cells(readings, first_condition, second_condition)
    scope = PatternScope(
        metric=metric,
        location_id=location_id,
        weekday=weekday,
        hours=hours,
        condition=f"{first_name} x {second_name}",
    )
    identifier = f"INT-{location_id or 'ALL'}-{first_name}-{second_name}".replace(" ", "_")

    if cells.smallest < min_cell:
        return InteractionTest(
            interaction_id=identifier,
            description=(
                f"cannot test {first_name} x {second_name}: the smallest cell has "
                f"{cells.smallest} observations, below the {min_cell} needed"
            ),
            scope=scope,
            first_condition=first_name,
            second_condition=second_name,
            verdict=InteractionVerdict.UNTESTABLE,
            scale=LOG_SCALE,
            cell_counts=cells.counts,
            notes=(
                "untestable is not the same as no interaction: the data simply cannot speak to it",
            ),
        )

    first_effect, second_effect, interaction = _ols_interaction(cells)

    if not interaction.excludes_null:
        verdict = InteractionVerdict.NO_EVIDENCE
        summary = (
            f"no evidence that {first_name} and {second_name} interact beyond "
            "composing independently"
        )
    elif interaction.value > 0:
        verdict = InteractionVerdict.SUPPORTED
        summary = (
            f"{first_name} and {second_name} together exceed what independent composition predicts"
        )
    else:
        verdict = InteractionVerdict.CONTRADICTED
        summary = (
            f"{first_name} and {second_name} together fall short of what "
            "independent composition predicts"
        )

    notes = [
        f"on the {LOG_SCALE}",
        f"smallest cell has {cells.smallest} observations, which bounds the power",
    ]
    if verdict is InteractionVerdict.NO_EVIDENCE:
        notes.append(
            "absence of evidence, not evidence of absence: with this cell size a "
            "real interaction of moderate size would also fail to reach "
            "significance"
        )

    return InteractionTest(
        interaction_id=identifier,
        description=summary,
        scope=scope,
        first_condition=first_name,
        second_condition=second_name,
        verdict=verdict,
        scale=LOG_SCALE,
        cell_counts=cells.counts,
        first_effect=first_effect,
        second_effect=second_effect,
        interaction_effect=interaction,
        notes=tuple(notes),
    )


def is_wet(reading: Reading) -> bool:
    """Condition: rain was falling."""
    return reading.is_wet


def on_weekday(weekday: int) -> object:
    """Condition: the reading falls on a given local weekday."""

    def condition(reading: Reading) -> bool:
        return reading.weekday == weekday

    return condition


def _weekday_label(weekday: int) -> str:
    """Weekday name, for a readable condition label."""
    return (
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    )[weekday % 7]


def discover_interactions(
    frame: ObservationFrame,
    metric: str,
    patterns: list[object],
) -> list[InteractionTest]:
    """Test each discovered spatio-temporal pattern against rainfall.

    The pairs tested come from what discovery actually found, not from a list
    someone wrote down. That keeps the interaction stage downstream of discovery
    rather than needing its own knowledge of where to look.
    """
    results: list[InteractionTest] = []
    for pattern in patterns:
        scope = getattr(pattern, "scope", None)
        if scope is None or scope.weekday is None or not scope.hours:
            continue
        # Hours are held fixed and the weekday becomes the condition, matching
        # how the spatio-temporal pattern was defined. Leaving hours free would
        # put 03:00 readings in the comparison cell, adding noise from a
        # completely different part of the day for no gain.
        results.append(
            test_interaction(
                frame,
                metric,
                location_id=scope.location_id,
                hours=scope.hours,
                first_name=_weekday_label(scope.weekday),
                first_condition=on_weekday(scope.weekday),
                second_name="rain",
                second_condition=is_wet,
            )
        )
    return results
