"""Sensor failure detection and source health (SPEC section 17).

Two failure modes matter, and they are opposites:

**A frozen sensor** keeps reporting, but reports the same number forever. This
is more dangerous than silence, because the value looks plausible: a pipeline
that only checks ranges accepts it, and a model trained on it learns a stuck
device. Detected as a run of identical consecutive readings.

**A dead or gappy stream** stops reporting. The crucial rule is that a gap is
never read as zero -- a sensor that reported nothing did not report no traffic.
Gaps are measured and reported as health, never filled here.

Health is per source, because the same road can be covered by a healthy sensor
and a broken one at different times, and the question a user asks is "can I
trust this feed right now?"
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise

from urbansense.schemas.enums import SourceStatus
from urbansense.schemas.observation import Observation

#: Identical consecutive readings needed before a run counts as frozen.
#: Six hours of a byte-identical vehicle count is not a quiet road; it is a
#: stuck device. Set from configuration in practice.
DEFAULT_MIN_FROZEN_RUN = 6

#: Gap beyond which a source is considered failed rather than merely late.
DEFAULT_FAILED_GAP = timedelta(hours=6)

#: Gap beyond which a source is flagged for attention.
DEFAULT_WARNING_GAP = timedelta(hours=2)

#: Missing-value rate above which a source is flagged, even with no long gap.
#: A feed dropping one reading in ten is degraded even if never silent.
DEFAULT_WARNING_MISSING_RATE = 0.10

#: Values whose repetition is physically normal and therefore not a freeze.
#:
#: Zero is the important case: a rain gauge reporting 0 mm through a dry April
#: is working perfectly, and a detector on a closed road legitimately counts no
#: vehicles for hours. Repetition only indicates a stuck device when the value
#: itself represents activity -- a sensor pinned at 316 vehicles/hour for two
#: and a half days is reporting something that never happens naturally.
#:
#: A long run of zeros is not ignored by the system as a whole: it still faces
#: the outlier detector, which will flag a daytime zero against its own hour's
#: median. That is the right division -- "no activity" is suspicious but
#: possible, whereas a frozen non-zero reading is not a measurement at all.
DEFAULT_IGNORED_FROZEN_VALUES: tuple[float, ...] = (0.0,)


@dataclass(frozen=True, slots=True)
class FrozenRun:
    """A detected run of identical consecutive readings.

    Attributes:
        source_id: Source that produced the run.
        location_id: Zone the readings are for.
        metric: Metric that froze.
        value: The repeated value.
        start: ``event_time`` of the first reading in the run.
        end: ``event_time`` of the last reading in the run.
        length: Number of readings in the run.
        observation_ids: Ids of every affected observation, so the quarantine
            records can name exactly what was rejected.
    """

    source_id: str
    location_id: str
    metric: str
    value: float
    start: datetime
    end: datetime
    length: int
    observation_ids: tuple[str, ...]

    @property
    def duration(self) -> timedelta:
        """Wall-clock span of the frozen window."""
        return self.end - self.start


@dataclass
class SourceHealth:
    """Health of one reporting source (SPEC section 17).

    Attributes:
        source_id: The source.
        status: ``OK`` / ``WARNING`` / ``FAILED``.
        observation_count: Readings seen from this source.
        missing_count: Readings whose value was absent. Counted, never zeroed.
        frozen_run_count: Frozen runs detected.
        frozen_observation_count: Readings inside frozen runs.
        last_observation_time: ``event_time`` of the most recent reading.
        max_gap: Largest gap between consecutive readings.
        gap_duration: Time from the last reading to the reference "now".
        detail: Human-readable summary of why the status is what it is.
        locations: Zones this source reported for.
    """

    source_id: str
    status: SourceStatus = SourceStatus.OK
    observation_count: int = 0
    missing_count: int = 0
    frozen_run_count: int = 0
    frozen_observation_count: int = 0
    last_observation_time: datetime | None = None
    max_gap: timedelta | None = None
    gap_duration: timedelta | None = None
    detail: str = ""
    locations: tuple[str, ...] = ()

    @property
    def missing_rate(self) -> float:
        """Fraction of readings with no value."""
        if self.observation_count == 0:
            return 0.0
        return self.missing_count / self.observation_count


def detect_frozen_runs(
    observations: Iterable[Observation],
    *,
    min_run_length: int = DEFAULT_MIN_FROZEN_RUN,
    ignored_values: Sequence[float] = DEFAULT_IGNORED_FROZEN_VALUES,
) -> tuple[FrozenRun, ...]:
    """Find runs of identical consecutive readings.

    Observations are grouped by ``(source_id, location_id, metric)`` and sorted
    by ``event_time``, since "consecutive" only means anything within one
    series. A missing value **breaks** a run rather than extending it: a gap is
    an absence of evidence, not a repeated reading.

    Args:
        observations: Observations to scan, in any order.
        min_run_length: Identical readings required to call a run frozen.
        ignored_values: Values whose repetition is physically normal -- zero by
            default. Without this, every dry spell in a rainfall series reads
            as a broken gauge, and a quiet overnight road as a broken detector.

    Returns:
        Every detected run, ordered by source, location, metric and start time.
    """
    if min_run_length < 2:
        raise ValueError(f"min_run_length must be at least 2, got {min_run_length}")

    ignored = set(ignored_values)

    series: dict[tuple[str, str, str], list[Observation]] = defaultdict(list)
    for observation in observations:
        key = (observation.source_id, observation.location_id, observation.metric)
        series[key].append(observation)

    runs: list[FrozenRun] = []
    for key, group in sorted(series.items()):
        ordered = sorted(group, key=lambda obs: obs.event_time)
        current: list[Observation] = []

        for observation in ordered:
            if observation.value is None:
                # A gap is an absence of evidence, not a repeated reading, so
                # it ends the run rather than extending it.
                runs.extend(_as_run(current, key, min_run_length, ignored))
                current = []
                continue
            if current and observation.value == current[-1].value:
                current.append(observation)
            else:
                runs.extend(_as_run(current, key, min_run_length, ignored))
                current = [observation]
        runs.extend(_as_run(current, key, min_run_length, ignored))

    return tuple(runs)


def _as_run(
    batch: Sequence[Observation],
    key: tuple[str, str, str],
    min_run_length: int,
    ignored: set[float],
) -> tuple[FrozenRun, ...]:
    """Turn a candidate batch into a frozen run, if it qualifies.

    A free function rather than a closure over the loop variables: the batch and
    its series key are passed explicitly, so the run can never be built against
    a key from a different iteration.
    """
    if len(batch) < min_run_length:
        return ()
    value = batch[0].value
    if value is None:  # pragma: no cover - callers filter missing values
        return ()
    if value in ignored:
        # Repetition of a no-activity value is not a stuck device.
        return ()

    source_id, location_id, metric = key
    return (
        FrozenRun(
            source_id=source_id,
            location_id=location_id,
            metric=metric,
            value=value,
            start=batch[0].event_time,
            end=batch[-1].event_time,
            length=len(batch),
            observation_ids=tuple(obs.observation_id for obs in batch),
        ),
    )


def assess_source_health(
    observations: Iterable[Observation],
    *,
    frozen_runs: Iterable[FrozenRun] = (),
    now: datetime | None = None,
    expected_interval: timedelta | None = None,
    warning_gap: timedelta = DEFAULT_WARNING_GAP,
    failed_gap: timedelta = DEFAULT_FAILED_GAP,
    warning_missing_rate: float = DEFAULT_WARNING_MISSING_RATE,
) -> dict[str, SourceHealth]:
    """Build a health record for every source seen.

    Args:
        observations: Every observation from the run, including ones that were
            quarantined -- a frozen sensor's readings are exactly the evidence
            that it is unhealthy, so excluding them would hide the failure.
        frozen_runs: Runs from :func:`detect_frozen_runs`.
        now: Reference time for the trailing gap. Defaults to the latest
            ``event_time`` seen, which makes the result reproducible for a
            historical file rather than depending on when it was processed.
        expected_interval: Expected spacing between readings. When given, only
            gaps larger than one interval count as gaps.
        warning_gap: Gap beyond which a source is flagged.
        failed_gap: Gap beyond which a source is considered failed.
        warning_missing_rate: Missing rate beyond which a source is flagged.

    Returns:
        Health keyed by ``source_id``.
    """
    by_source: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_source[observation.source_id].append(observation)

    frozen_by_source: dict[str, list[FrozenRun]] = defaultdict(list)
    for run in frozen_runs:
        frozen_by_source[run.source_id].append(run)

    latest_overall: datetime | None = None
    for group in by_source.values():
        for observation in group:
            if latest_overall is None or observation.event_time > latest_overall:
                latest_overall = observation.event_time
    reference = now if now is not None else latest_overall

    health: dict[str, SourceHealth] = {}
    for source_id, group in sorted(by_source.items()):
        ordered = sorted(group, key=lambda obs: obs.event_time)
        times = [obs.event_time for obs in ordered]

        # Gaps are measured between distinct timestamps: several metrics share
        # one timestamp, and counting those as zero-length gaps would say
        # nothing about the feed's continuity.
        distinct = sorted(set(times))
        gaps = [later - earlier for earlier, later in pairwise(distinct)]
        if expected_interval is not None:
            gaps = [gap for gap in gaps if gap > expected_interval]
        max_gap = max(gaps) if gaps else None

        last_seen = distinct[-1] if distinct else None
        trailing = (
            reference - last_seen if reference is not None and last_seen is not None else None
        )

        runs = frozen_by_source.get(source_id, [])
        frozen_observations = sum(run.length for run in runs)
        missing = sum(1 for obs in ordered if obs.value is None)

        record = SourceHealth(
            source_id=source_id,
            observation_count=len(ordered),
            missing_count=missing,
            frozen_run_count=len(runs),
            frozen_observation_count=frozen_observations,
            last_observation_time=last_seen,
            max_gap=max_gap,
            gap_duration=trailing,
            locations=tuple(sorted({obs.location_id for obs in ordered})),
        )

        reasons: list[str] = []
        status = SourceStatus.OK

        if runs:
            status = SourceStatus.FAILED
            longest = max(runs, key=lambda run: run.length)
            reasons.append(
                f"{len(runs)} frozen run(s); longest {longest.length} identical "
                f"readings of {longest.value:g} over {longest.duration}"
            )

        if trailing is not None and trailing > failed_gap:
            status = SourceStatus.FAILED
            reasons.append(f"no observation for {trailing} (failed threshold {failed_gap})")
        elif max_gap is not None and max_gap > failed_gap:
            if status is not SourceStatus.FAILED:
                status = SourceStatus.WARNING
            reasons.append(f"largest gap {max_gap} exceeds the failed threshold {failed_gap}")
        elif trailing is not None and trailing > warning_gap:
            if status is SourceStatus.OK:
                status = SourceStatus.WARNING
            reasons.append(f"no observation for {trailing}")
        elif max_gap is not None and max_gap > warning_gap:
            if status is SourceStatus.OK:
                status = SourceStatus.WARNING
            reasons.append(f"largest gap {max_gap}")

        if record.missing_rate > warning_missing_rate and status is SourceStatus.OK:
            status = SourceStatus.WARNING
            reasons.append(f"{record.missing_rate:.1%} of readings are missing (missing, not zero)")

        record.status = status
        record.detail = "; ".join(reasons) if reasons else "no issues detected"
        health[source_id] = record

    return health
