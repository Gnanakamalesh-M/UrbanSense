"""Candidate explanations for an anomaly (SPEC section 24).

    Normal: 5000 vehicles/hour. Observed: 8700 vehicles/hour -> ANOMALY.
    Then examine possible explanations: weather, events, accidents, road
    closures, sensor failure, holidays.
    Do not automatically classify an anomaly as an error.

That last line shapes this entire module. Every explanation here is a
**candidate** with its supporting evidence attached, and there is no code path
that promotes one to a verdict. The reason is not caution for its own sake: the
spec's own list contains a sensor failure *and* a festival *and* an accident, and
a single number cannot tell them apart. A system that guessed would be wrong in
the most expensive direction -- discarding a real event as a glitch.

``UNEXPLAINED`` is a first-class outcome rather than an empty list. "We flagged
this and cannot account for it" is useful information; silently returning nothing
would read as "nothing to see".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from urbansense.patterns.frame import WET_THRESHOLD_MM
from urbansense.patterns.models import Pattern, PatternKind
from urbansense.schemas.enums import SourceStatus
from urbansense.schemas.observation import Observation


class ExplanationKind(StrEnum):
    """A candidate cause for an anomalous reading.

    Deliberately not exhaustive. Accidents, road closures and public holidays are
    in SPEC section 24's list and are absent here because this system has no
    feed that would evidence them -- naming a candidate it cannot support would
    be theatre.
    """

    #: Rain was falling, which is known to move traffic volume.
    RAINFALL = "rainfall"
    #: A discovered pattern covers this zone, weekday and hour.
    RECURRING_PATTERN = "recurring_pattern"
    #: The reading sits after a detected change in the zone's baseline.
    LEVEL_SHIFT = "level_shift"
    #: The reporting source is not healthy, or the reading repeats its neighbours.
    SENSOR_SUSPECT = "sensor_suspect"
    #: Nothing above fired. Stated rather than left as an empty list.
    UNEXPLAINED = "unexplained"


@dataclass(frozen=True, slots=True)
class Explanation:
    """One candidate cause, with its evidence.

    Attributes:
        kind: Which candidate.
        detail: What was observed that makes this plausible.
        support: How strongly the evidence points here, 0 to 1. A ranking aid
            only -- it is not a probability and does not sum to one across
            candidates, because the candidates are not mutually exclusive.
        pattern_id: The pattern that fired, when the candidate came from one.
    """

    kind: ExplanationKind
    detail: str
    support: float
    pattern_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "kind": self.kind.value,
            "detail": self.detail,
            "support": round(self.support, 3),
            "pattern_id": self.pattern_id,
        }


def _covers(pattern: Pattern, observation: Observation, local_offset: float) -> bool:
    """Whether a pattern's scope covers this observation."""
    scope = pattern.scope
    if scope.location_id is not None and scope.location_id != observation.location_id:
        return False
    if scope.metric != observation.metric:
        return False
    local = observation.event_time + timedelta(hours=local_offset)
    if scope.weekday is not None and local.weekday() != scope.weekday:
        return False
    if scope.hours and local.hour not in scope.hours:
        return False
    return True


def explain(
    observation: Observation,
    *,
    patterns: tuple[Pattern, ...] = (),
    rain_mm: float | None = None,
    source_health: SourceStatus | None = None,
    repeated_neighbours: int = 0,
    timezone_offset_hours: float = 0.0,
) -> tuple[Explanation, ...]:
    """Assemble candidate explanations for an anomalous reading.

    Args:
        observation: The flagged reading.
        patterns: Discovered patterns, used to see whether the reading sits
            inside a known regularity.
        rain_mm: Concurrent rainfall, when known.
        source_health: Health of the reporting source, when known.
        repeated_neighbours: How many adjacent readings share this exact value.
        timezone_offset_hours: Local offset, for matching a pattern's hours.

    Returns:
        Candidates ordered by support, strongest first. Always at least one
        element: ``UNEXPLAINED`` when nothing else fired.
    """
    found: list[Explanation] = []

    if rain_mm is not None and rain_mm > WET_THRESHOLD_MM:
        # Rain genuinely moves volume, so it is a candidate -- but a rainy hour
        # is common and a flagged reading is rare, so rain alone is weak evidence
        # for *this* reading being extreme.
        found.append(
            Explanation(
                kind=ExplanationKind.RAINFALL,
                detail=f"{rain_mm:.1f} mm of rain was falling in this hour",
                support=min(0.6, 0.2 + rain_mm / 50.0),
            )
        )

    for pattern in patterns:
        if not _covers(pattern, observation, timezone_offset_hours):
            continue
        if pattern.kind is PatternKind.SPATIOTEMPORAL:
            found.append(
                Explanation(
                    kind=ExplanationKind.RECURRING_PATTERN,
                    detail=(
                        f"a discovered pattern covers this zone, weekday and hour "
                        f"({pattern.description})"
                    ),
                    support=0.8,
                    pattern_id=pattern.pattern_id,
                )
            )
        elif pattern.kind is PatternKind.LEVEL_SHIFT:
            found.append(
                Explanation(
                    kind=ExplanationKind.LEVEL_SHIFT,
                    detail=(
                        f"the zone's baseline changed during this period ({pattern.description})"
                    ),
                    support=0.5,
                    pattern_id=pattern.pattern_id,
                )
            )

    if source_health is not None and source_health is not SourceStatus.OK:
        found.append(
            Explanation(
                kind=ExplanationKind.SENSOR_SUSPECT,
                detail=f"the reporting source is {source_health.value}",
                support=0.7 if source_health is SourceStatus.FAILED else 0.4,
            )
        )
    if repeated_neighbours >= 2:
        found.append(
            Explanation(
                kind=ExplanationKind.SENSOR_SUSPECT,
                detail=(
                    f"{repeated_neighbours} adjacent readings carry this exact "
                    "value, which a live sensor rarely does"
                ),
                support=0.75,
            )
        )

    if not found:
        found.append(
            Explanation(
                kind=ExplanationKind.UNEXPLAINED,
                detail=(
                    "no rainfall, no covering pattern, no level shift and no "
                    "source-health signal accounts for this reading"
                ),
                support=0.0,
            )
        )

    found.sort(key=lambda item: (-item.support, item.kind.value))
    return tuple(found)
