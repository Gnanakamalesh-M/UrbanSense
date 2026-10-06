"""Pattern records and the status taxonomy (SPEC sections 21, 22).

A discovered pattern is a *claim about a city*, and a claim needs its evidence
attached or it cannot be audited, revisited, or disbelieved. So every pattern
carries not just a strength but the test that produced it, the sample sizes
behind it, the confidence interval, and the per-window strength series that makes
its evolution measurable rather than asserted.

Two things are deliberately kept apart:

**Strength is an effect size, not a p-value.** A tiny effect measured over
thousands of hours is significant and uninteresting; the strength field says how
*large* the effect is and ``confidence`` says how sure we are it is not noise.
Collapsing the two would let a 1.01x "pattern" look as real as a 2.2x one.

**A pattern's status is derived from its history, never from one measurement.**
SPEC section 22 asks whether old patterns still hold, which only means something
if strength is tracked per window. ``STABLE`` and ``STRENGTHENING`` are claims
about a trend, so they are computed from the series and a trend test rather than
from comparing two numbers.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class PatternKind(StrEnum):
    """What sort of regularity a pattern describes (SPEC section 21)."""

    #: Volume by hour of day, within a zone.
    HOURLY_PROFILE = "hourly_profile"
    #: Volume by day of week, within a zone.
    WEEKDAY_PROFILE = "weekday_profile"
    #: Volume by month, within a zone. The seasonal shape.
    SEASONAL_PROFILE = "seasonal_profile"
    #: A zone carrying materially more traffic than the city median.
    HOTSPOT = "hotspot"
    #: A zone-and-weekday-and-hour window that departs from its own baseline.
    SPATIOTEMPORAL = "spatiotemporal"
    #: An external condition that moves the metric, such as rainfall.
    CONDITIONAL = "conditional"
    #: A sustained change in a zone's baseline level.
    LEVEL_SHIFT = "level_shift"


class PatternStatus(StrEnum):
    """How a pattern is behaving over time (SPEC section 22).

    Assigned from the per-window strength series plus a trend test, never from a
    single pair of numbers: two adjacent windows differ by noise all the time,
    and calling that ``STRENGTHENING`` would make the status meaningless.
    """

    #: First seen in the most recent windows only.
    NEW = "new"
    #: Present throughout, with no significant trend.
    STABLE = "stable"
    #: Present and growing, with a significant positive trend.
    STRENGTHENING = "strengthening"
    #: Present and shrinking, with a significant negative trend.
    WEAKENING = "weakening"
    #: Held historically, absent from the recent windows.
    DISAPPEARED = "disappeared"
    #: Direction reverses between windows: the evidence disagrees with itself.
    CONFLICTING = "conflicting"
    #: Too few windows to say anything about evolution. Stated, not guessed.
    INSUFFICIENT_HISTORY = "insufficient_history"


class InteractionVerdict(StrEnum):
    """Outcome of testing whether two conditions interact (SPEC section 23).

    ``NO_EVIDENCE`` is not a synonym for "no interaction". It means the data
    cannot distinguish an interaction from zero, which with a small cell count is
    the honest answer and a very different claim.
    """

    #: The interval excludes zero in the expected direction.
    SUPPORTED = "supported"
    #: The interval contains zero: absence of evidence, not evidence of absence.
    NO_EVIDENCE = "no_evidence"
    #: The interval excludes zero in the opposite direction.
    CONTRADICTED = "contradicted"
    #: Not enough data in one of the cells to fit the model at all.
    UNTESTABLE = "untestable"


class EffectSize(BaseModel):
    """How large an effect is, with its uncertainty.

    Attributes:
        value: The point estimate. A ratio for comparisons of level, a slope for
            a continuous condition.
        lower: Lower bound of the interval.
        upper: Upper bound of the interval.
        kind: What the number means -- ``median_ratio``, ``slope_per_unit``, or
            ``log_difference``. Named because a bare number invites the wrong
            interpretation.
        confidence_level: Coverage of the interval, e.g. 0.95.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float
    lower: float
    upper: float
    kind: str
    confidence_level: float = 0.95

    @property
    def excludes_null(self) -> bool:
        """Whether the interval excludes the no-effect value.

        The null is 1.0 for a ratio and 0.0 for a slope or difference, which is
        why ``kind`` has to be carried alongside the number.
        """
        null = 1.0 if self.kind == "median_ratio" else 0.0
        return not (self.lower <= null <= self.upper)

    def describe(self) -> str:
        """Human-readable rendering."""
        if self.kind == "median_ratio":
            return f"x{self.value:.3f} ({self.lower:.3f} to {self.upper:.3f})"
        return f"{self.value:+.4f} ({self.lower:+.4f} to {self.upper:+.4f})"


class Evidence(BaseModel):
    """The statistical basis for a pattern.

    Kept on the pattern rather than logged, so a claim can always be re-examined
    without re-running the discovery.

    Attributes:
        test: Name of the test used.
        statistic: Test statistic.
        p_value: Raw p-value, before any multiple-testing adjustment.
        adjusted_p_value: After Benjamini-Hochberg, when the pattern came from a
            scan. ``None`` for a single pre-specified test.
        effect: The effect size and its interval.
        n_treatment: Observations in the condition being tested.
        n_control: Observations in the comparison group.
        controls: Confounds held constant, named explicitly so a reader can see
            what was *not* controlled too.
        notes: Caveats worth carrying with the number.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    test: str
    statistic: float
    p_value: float
    effect: EffectSize
    n_treatment: int
    n_control: int
    adjusted_p_value: float | None = None
    controls: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def significance(self) -> float:
        """The p-value that should actually be judged against a threshold."""
        return self.p_value if self.adjusted_p_value is None else self.adjusted_p_value


class PatternScope(BaseModel):
    """Where and when a pattern applies.

    Attributes:
        location_id: Zone, or ``None`` for a city-wide pattern.
        metric: Metric the pattern is about.
        weekday: Day of week, Monday 0. ``None`` when the pattern spans days.
        hours: Local hours covered. Empty when the pattern spans the day.
        condition: External condition, for a conditional pattern.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    location_id: str | None = None
    weekday: int | None = None
    hours: tuple[int, ...] = ()
    condition: str | None = None

    def describe(self) -> str:
        """Human-readable rendering, in the order a person would say it."""
        parts: list[str] = []
        if self.location_id is not None:
            parts.append(self.location_id)
        if self.weekday is not None:
            parts.append(_WEEKDAY_NAMES[self.weekday % 7])
        if self.hours:
            if len(self.hours) == 1:
                parts.append(f"{self.hours[0]:02d}:00")
            else:
                parts.append(f"{min(self.hours):02d}:00-{max(self.hours) + 1:02d}:00")
        if self.condition is not None:
            parts.append(self.condition)
        parts.append(f"[{self.metric}]")
        return " + ".join(parts[:-1]) + " " + parts[-1] if len(parts) > 1 else parts[0]


_WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


class WindowStrength(BaseModel):
    """A pattern's strength over one time window.

    Attributes:
        window: Window label, e.g. ``2026-03``.
        strength: Effect size in that window.
        n_treatment: Observations in the condition.
        n_control: Observations in the comparison group.
        measured: Whether the window had enough data to measure. A window that
            could not be measured is recorded as such rather than filled in --
            missing stays missing (SPEC section 15).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    window: str
    strength: float | None
    n_treatment: int
    n_control: int
    measured: bool


class Pattern(BaseModel):
    """One discovered regularity, with its evidence and its history.

    Attributes:
        pattern_id: Stable identifier, derived from the scope so the same pattern
            keeps its id across runs and can be compared over time.
        kind: What sort of pattern it is.
        description: One line a person can read.
        scope: Where and when it applies.
        first_detected: Earliest observation supporting it.
        last_observed: Latest observation supporting it.
        frequency_per_week: How often the condition occurs.
        confidence: How sure we are the effect is not noise, as ``1 - p``.
        historical_strength: Strength over the earlier windows.
        recent_strength: Strength over the most recent windows.
        strength_by_window: The full series, which is what makes evolution
            measurable rather than asserted.
        status: How it is behaving over time.
        status_reason: Why that status was assigned.
        evidence: The statistical basis.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern_id: str
    kind: PatternKind
    description: str
    scope: PatternScope
    first_detected: datetime
    last_observed: datetime
    frequency_per_week: float
    confidence: float
    evidence: Evidence
    historical_strength: float | None = None
    recent_strength: float | None = None
    strength_by_window: tuple[WindowStrength, ...] = ()
    status: PatternStatus = PatternStatus.INSUFFICIENT_HISTORY
    status_reason: str = ""

    @property
    def strength(self) -> float:
        """The headline effect size."""
        return self.evidence.effect.value

    @property
    def is_significant(self) -> bool:
        """Whether the effect survived its significance threshold."""
        return self.evidence.effect.excludes_null

    def as_dict(self) -> dict[str, object]:
        """Serializable form, for the pattern store."""
        return {
            "pattern_id": self.pattern_id,
            "kind": self.kind.value,
            "description": self.description,
            "scope": {
                "metric": self.scope.metric,
                "location_id": self.scope.location_id,
                "weekday": self.scope.weekday,
                "hours": list(self.scope.hours),
                "condition": self.scope.condition,
            },
            "first_detected": self.first_detected.isoformat(),
            "last_observed": self.last_observed.isoformat(),
            "frequency_per_week": round(self.frequency_per_week, 4),
            "confidence": round(self.confidence, 6),
            "strength": round(self.strength, 6),
            "historical_strength": (
                None if self.historical_strength is None else round(self.historical_strength, 6)
            ),
            "recent_strength": (
                None if self.recent_strength is None else round(self.recent_strength, 6)
            ),
            "status": self.status.value,
            "status_reason": self.status_reason,
            "strength_by_window": [
                {
                    "window": item.window,
                    "strength": None if item.strength is None else round(item.strength, 6),
                    "n_treatment": item.n_treatment,
                    "n_control": item.n_control,
                    "measured": item.measured,
                }
                for item in self.strength_by_window
            ],
            "evidence": {
                "test": self.evidence.test,
                "statistic": round(self.evidence.statistic, 6),
                "p_value": self.evidence.p_value,
                "adjusted_p_value": self.evidence.adjusted_p_value,
                "effect": {
                    "kind": self.evidence.effect.kind,
                    "value": round(self.evidence.effect.value, 6),
                    "lower": round(self.evidence.effect.lower, 6),
                    "upper": round(self.evidence.effect.upper, 6),
                    "confidence_level": self.evidence.effect.confidence_level,
                    "excludes_null": self.evidence.effect.excludes_null,
                },
                "n_treatment": self.evidence.n_treatment,
                "n_control": self.evidence.n_control,
                "controls": list(self.evidence.controls),
                "notes": list(self.evidence.notes),
            },
        }

    def describe(self) -> str:
        """One-line human-readable summary."""
        return (
            f"{self.scope.describe()}: {self.evidence.effect.describe()} "
            f"[{self.kind.value}, {self.status.value}, "
            f"confidence {self.confidence:.3f}]"
        )


class InteractionTest(BaseModel):
    """The outcome of testing whether two conditions interact (SPEC section 23).

    Attributes:
        interaction_id: Stable identifier.
        description: One line a person can read.
        scope: Where the test was run.
        first_condition: Name of the first condition.
        second_condition: Name of the second condition.
        verdict: What the evidence supports.
        scale: Scale the test was run on. Named because "no interaction" is
            meaningless until the scale is stated: effects that compose
            multiplicatively show no interaction in logs and a positive one in
            levels.
        first_effect: Effect of the first condition alone.
        second_effect: Effect of the second condition alone.
        interaction_effect: The departure from independent composition.
        cell_counts: Observations per condition cell, which is what determines
            whether a null result means anything.
        notes: Caveats, including power.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    interaction_id: str
    description: str
    scope: PatternScope
    first_condition: str
    second_condition: str
    verdict: InteractionVerdict
    scale: str
    cell_counts: dict[str, int]
    first_effect: EffectSize | None = None
    second_effect: EffectSize | None = None
    interaction_effect: EffectSize | None = None
    notes: tuple[str, ...] = ()

    @property
    def min_cell_count(self) -> int:
        """Smallest cell, which bounds the power of the whole test."""
        return min(self.cell_counts.values()) if self.cell_counts else 0

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""

        def effect(item: EffectSize | None) -> dict[str, object] | None:
            if item is None:
                return None
            return {
                "kind": item.kind,
                "value": round(item.value, 6),
                "lower": round(item.lower, 6),
                "upper": round(item.upper, 6),
                "excludes_null": item.excludes_null,
            }

        return {
            "interaction_id": self.interaction_id,
            "description": self.description,
            "scope": {
                "metric": self.scope.metric,
                "location_id": self.scope.location_id,
                "weekday": self.scope.weekday,
                "hours": list(self.scope.hours),
            },
            "first_condition": self.first_condition,
            "second_condition": self.second_condition,
            "verdict": self.verdict.value,
            "scale": self.scale,
            "cell_counts": dict(sorted(self.cell_counts.items())),
            "min_cell_count": self.min_cell_count,
            "first_effect": effect(self.first_effect),
            "second_effect": effect(self.second_effect),
            "interaction_effect": effect(self.interaction_effect),
            "notes": list(self.notes),
        }

    def describe(self) -> str:
        """Human-readable summary that states the scale and the power."""
        lines = [
            f"{self.first_condition} x {self.second_condition} at "
            f"{self.scope.describe()}: {self.verdict.value.upper()}",
            f"  scale: {self.scale}",
        ]
        if self.interaction_effect is not None:
            lines.append(f"  interaction: {self.interaction_effect.describe()}")
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.cell_counts.items()))
        lines.append(f"  cells: {counts}")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


class DiscoveryStats(BaseModel):
    """Bookkeeping for one discovery run, so the numbers can be judged.

    Attributes:
        tests_run: Hypotheses tested in the scan. Without this, a count of
            discoveries is uninterpretable: 34 of 672 tests clearing p<0.05 is
            exactly what chance produces.
        fdr_level: Benjamini-Hochberg level applied.
        discoveries: Patterns surviving the FDR gate and the effect-size floor.
        cells_too_sparse: Cells skipped for want of data, reported rather than
            silently dropped.
        bootstrap_seed: Seed used, so a run is reproducible.
        source_rows: Observations discovery read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tests_run: int = 0
    fdr_level: float = 0.05
    discoveries: int = 0
    cells_too_sparse: int = 0
    bootstrap_seed: int = 0
    source_rows: int = 0
