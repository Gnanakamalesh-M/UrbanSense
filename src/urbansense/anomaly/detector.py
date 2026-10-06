"""Anomaly detection with explanations (SPEC section 24).

Detection **reuses Phase 2's robust core** (``data_integrity.outliers``) rather
than reimplementing median/MAD. One definition of "unusual" in the codebase means
two stages cannot disagree about what counts, and the existing conditioning on
``(zone, metric, weekday, hour)`` is already what stops a recurring peak looking
anomalous -- within its own weekday-and-hour group, the Friday peak *is* the norm.

What this module adds is the second half of section 24: once something is
flagged, examine possible explanations. Each anomaly carries ranked candidates
with their evidence, and

    Do not automatically classify an anomaly as an error.

is enforced structurally: ``is_error`` is ``None`` on every record and there is no
setter. The spec's own candidate list holds a sensor failure, a festival and an
accident, and the number alone cannot separate them. Guessing would discard real
events as glitches, which is the expensive direction to be wrong in.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from urbansense.anomaly.explanations import Explanation, ExplanationKind, explain
from urbansense.data_integrity.outliers import (
    DEFAULT_MIN_GROUP_SIZE,
    DEFAULT_THRESHOLD,
    OutlierFlag,
    detect_outliers,
)
from urbansense.data_integrity.sensors import SourceHealth
from urbansense.patterns.frame import CITY_LOCATION, RAINFALL_METRIC
from urbansense.patterns.models import Pattern
from urbansense.schemas.enums import ValidationStatus
from urbansense.schemas.observation import Observation

#: How many neighbouring hours either side are checked for a repeated value, as
#: evidence of a stuck sensor.
NEIGHBOUR_WINDOW = 3


@dataclass(frozen=True, slots=True)
class Anomaly:
    """One flagged reading, with candidates but no verdict.

    Attributes:
        observation_id: The flagged observation.
        record_id: The source's own reading id.
        location_id: Zone.
        metric: Metric.
        event_time: When the measurement happened.
        value: The observed value.
        expected: Median of its comparison group.
        deviation: How many MAD-sigmas it sits from that median.
        group_size: Samples in the comparison group.
        explanations: Ranked candidate causes, never empty.
        is_error: Always ``None``. A flagged reading could be a sensor fault, a
            festival or an accident, and this system cannot tell which -- so it
            does not pretend to. The field exists so that a later phase with more
            evidence has somewhere to record a decision.
        status: Always ``SUSPECT``: retained and flagged, never discarded.
    """

    observation_id: str
    record_id: str
    location_id: str
    metric: str
    event_time: datetime
    value: float
    expected: float
    deviation: float
    group_size: int
    explanations: tuple[Explanation, ...]
    is_error: None = None
    status: ValidationStatus = ValidationStatus.SUSPECT

    @property
    def is_unexplained(self) -> bool:
        """Whether nothing accounted for the reading."""
        return any(item.kind is ExplanationKind.UNEXPLAINED for item in self.explanations)

    @property
    def leading_explanation(self) -> Explanation:
        """The best-supported candidate. Not a conclusion."""
        return self.explanations[0]

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "observation_id": self.observation_id,
            "record_id": self.record_id,
            "location_id": self.location_id,
            "metric": self.metric,
            "event_time": self.event_time.isoformat(),
            "value": self.value,
            "expected": round(self.expected, 3),
            "deviation_mad_sigma": round(self.deviation, 3),
            "group_size": self.group_size,
            "status": self.status.value,
            # Written out explicitly rather than omitted, so a reader sees that
            # the system declined to judge rather than forgot to.
            "is_error": None,
            "is_error_note": (
                "not classified: a flagged reading may be a sensor fault, an "
                "event or a genuine record, and this evidence cannot separate them"
            ),
            "explanations": [item.as_dict() for item in self.explanations],
        }

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{self.record_id} {self.location_id} {self.metric} {self.event_time.isoformat()}",
            f"  observed {self.value:,.0f} against an expected {self.expected:,.0f} "
            f"({self.deviation:.1f} MAD-sigma, n={self.group_size})",
            "  candidates (not a verdict):",
        ]
        lines.extend(
            f"    - {item.kind.value} ({item.support:.2f}): {item.detail}"
            for item in self.explanations
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class AnomalyResult:
    """Everything one anomaly run produced.

    Attributes:
        anomalies: Flagged readings with their candidates.
        observations_scanned: Readings examined.
        flags: The raw outlier flags, kept so the detection and the explanation
            layers can be inspected separately.
        threshold: MAD-sigma threshold used.
    """

    anomalies: tuple[Anomaly, ...]
    observations_scanned: int
    flags: dict[str, OutlierFlag]
    threshold: float

    @property
    def rate(self) -> float:
        """Share of scanned readings flagged."""
        if self.observations_scanned == 0:
            return 0.0
        return len(self.anomalies) / self.observations_scanned

    @property
    def unexplained(self) -> tuple[Anomaly, ...]:
        """Anomalies nothing accounted for."""
        return tuple(item for item in self.anomalies if item.is_unexplained)

    @property
    def by_leading_explanation(self) -> dict[str, int]:
        """Counts per best-supported candidate."""
        counts: dict[str, int] = {}
        for item in self.anomalies:
            key = item.leading_explanation.kind.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def for_zone(self, location_id: str) -> tuple[Anomaly, ...]:
        """Anomalies in one zone."""
        return tuple(item for item in self.anomalies if item.location_id == location_id)

    @property
    def record_ids(self) -> frozenset[str]:
        """Record ids of every flagged reading."""
        return frozenset(item.record_id for item in self.anomalies)


def _rain_by_time(observations: list[Observation]) -> dict[datetime, float]:
    """City rainfall keyed by event time."""
    return {
        observation.event_time: observation.value
        for observation in observations
        if observation.metric == RAINFALL_METRIC
        and observation.location_id == CITY_LOCATION
        and observation.value is not None
    }


def _repeated_neighbours(
    observations: list[Observation], window: int = NEIGHBOUR_WINDOW
) -> dict[str, int]:
    """How many adjacent readings share each reading's exact value.

    Evidence for a stuck sensor, which is one candidate explanation rather than a
    conclusion -- the frozen-value *detector* that quarantines such runs lives in
    Phase 2, and this is only the hint that one may apply.
    """
    series: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.value is None:
            continue
        series[(observation.source_id, observation.metric)].append(observation)

    counts: dict[str, int] = {}
    for group in series.values():
        group.sort(key=lambda item: item.event_time)
        for index, observation in enumerate(group):
            low = max(0, index - window)
            high = min(len(group), index + window + 1)
            same = sum(
                1
                for other in group[low:high]
                if other is not observation and other.value == observation.value
            )
            counts[observation.observation_id] = same
    return counts


def detect_anomalies(
    observations: list[Observation],
    *,
    patterns: tuple[Pattern, ...] = (),
    source_health: dict[str, SourceHealth] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    timezone_offset_hours: float = 0.0,
    metric: str | None = None,
) -> AnomalyResult:
    """Flag unexpected readings and attach candidate explanations.

    Args:
        observations: Readings to scan. The caller chooses the population: the
            unified layer for live monitoring, or the pre-quarantine population
            when checking what the statistical detector would have caught on its
            own.
        patterns: Discovered patterns, so a reading inside a known regularity can
            be recognized as such rather than reported as a surprise.
        source_health: Per-source health from Phase 2, when available.
        threshold: MAD-sigma deviation required to flag.
        min_group_size: Minimum comparison-group size, below which a group says
            nothing and is skipped.
        timezone_offset_hours: Local offset, for matching a pattern's hours.
        metric: Restrict to one metric, or ``None`` for all.

    Returns:
        An :class:`AnomalyResult`. Every anomaly carries at least one candidate
        and none carries a verdict.
    """
    scanned = [
        observation
        for observation in observations
        if metric is None or observation.metric == metric
    ]
    flags = detect_outliers(scanned, threshold=threshold, min_group_size=min_group_size)
    if not flags:
        return AnomalyResult(
            anomalies=(),
            observations_scanned=len(scanned),
            flags={},
            threshold=threshold,
        )

    rain = _rain_by_time(scanned)
    neighbours = _repeated_neighbours(scanned)
    by_id = {observation.observation_id: observation for observation in scanned}

    anomalies: list[Anomaly] = []
    for observation_id, flag in flags.items():
        observation = by_id.get(observation_id)
        if observation is None or observation.value is None:  # pragma: no cover
            continue
        health = source_health.get(observation.source_id) if source_health else None
        anomalies.append(
            Anomaly(
                observation_id=observation_id,
                record_id=str(observation.attributes.get("record_id", observation_id)),
                location_id=observation.location_id,
                metric=observation.metric,
                event_time=observation.event_time,
                value=observation.value,
                expected=flag.group_median,
                deviation=flag.deviation,
                group_size=flag.group_size,
                explanations=explain(
                    observation,
                    patterns=patterns,
                    rain_mm=rain.get(observation.event_time),
                    source_health=health.status if health is not None else None,
                    repeated_neighbours=neighbours.get(observation_id, 0),
                    timezone_offset_hours=timezone_offset_hours,
                ),
            )
        )

    anomalies.sort(key=lambda item: (item.event_time, item.location_id, item.metric))
    return AnomalyResult(
        anomalies=tuple(anomalies),
        observations_scanned=len(scanned),
        flags=flags,
        threshold=threshold,
    )


def pre_quarantine_population(
    valid: tuple[Observation, ...],
    quarantined_observations: tuple[Observation, ...],
) -> list[Observation]:
    """Rebuild the population Phase 2 saw, for scoring the detector.

    Zero planted point anomalies survive into the unified layer -- Phase 2
    quarantines them on range and frozen-run checks before discovery ever runs.
    Scoring the statistical detector therefore needs the population *before* that
    removal, which is this. It is a measurement tool, not a pipeline stage:
    discovery itself still reads the unified layer only.
    """
    merged = [*valid, *quarantined_observations]
    seen: set[str] = set()
    unique: list[Observation] = []
    for observation in merged:
        if observation.observation_id in seen:
            continue
        seen.add(observation.observation_id)
        unique.append(observation)
    return unique
