"""Real-time stream simulation (SPEC sections 4B, 37).

Replays the held-out period one observation at a time, so the adaptive
machinery in later phases has something live-shaped to react to without needing
a real city feed.

The held-out period is the point: it sits *after* the historical range, so no
model can have trained on it. Replaying history as if it were live would make
every later evaluation meaningless.

What makes this a real-time source rather than a slow file read is
``received_time``. Each observation is stamped as it is emitted, so
``received_time`` advances while ``event_time`` stays fixed at when the
measurement happened -- exactly the distinction that lets late-arriving data be
recognized as late (SPEC section 4B).

Time is injected rather than taken from the global clock: ``sleep`` and ``clock``
are parameters, so tests run instantly and deterministically, and ``speed``
gives a fast mode for demos.
"""

from __future__ import annotations

import csv
import time as time_module
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from urbansense.config.models import Settings
from urbansense.ingestion.builders import (
    PIPELINE_VERSION,
    ZoneRegistry,
    to_observation,
    to_quarantine_record,
)
from urbansense.ingestion.result import IngestionResult, IngestionStats
from urbansense.ingestion.rows import RowParseError, parse_row
from urbansense.schemas.enums import ProblemType, SourceType
from urbansense.schemas.observation import Observation
from urbansense.schemas.quarantine import QuarantineRecord

__all__ = ["PIPELINE_VERSION", "StreamSimulator"]

#: Default wall-clock gap between emitted observations.
DEFAULT_INTERVAL_SECONDS = 1.0


class StreamSimulator:
    """Emits observations from a held-out CSV as though they were arriving live.

    Attributes:
        path: Held-out CSV to replay.
        interval_seconds: Wall-clock gap between emissions at ``speed=1``.
        speed: Speed-up factor. ``1.0`` is real time; ``60`` replays a minute
            per second; ``0`` is fast mode and never waits at all, which is what
            tests and demos use.
    """

    def __init__(
        self,
        path: Path,
        *,
        settings: Settings | None = None,
        problem_type: ProblemType = ProblemType.TRAFFIC,
        measurement_interval: timedelta = timedelta(hours=1),
        registry: ZoneRegistry | None = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        speed: float = 0.0,
        sleep: Callable[[float], None] = time_module.sleep,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the simulator.

        Args:
            path: Held-out CSV to replay. Opened read-only.
            settings: Problem configuration for alias and unit resolution.
            problem_type: Problem domain being replayed.
            measurement_interval: Length of each measurement interval.
            registry: Zone coordinates.
            interval_seconds: Wall-clock gap between emissions at ``speed=1``.
            speed: Speed-up factor; ``0`` disables waiting entirely.
            sleep: Injected sleep, so tests never actually wait.
            clock: Injected clock returning an aware UTC datetime, so
                ``received_time`` is deterministic under test.

        Raises:
            ValueError: if ``interval_seconds`` or ``speed`` is negative.
        """
        if interval_seconds < 0:
            raise ValueError(f"interval_seconds must not be negative, got {interval_seconds}")
        if speed < 0:
            raise ValueError(f"speed must not be negative, got {speed}")

        self.path = path
        self.settings = settings
        self.problem_type = problem_type
        self.measurement_interval = measurement_interval
        self.registry = registry if registry is not None else ZoneRegistry.empty()
        self.interval_seconds = interval_seconds
        self.speed = speed
        self._sleep = sleep
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

        self.rows_read = 0
        self.rejected: list[QuarantineRecord] = []

    @property
    def delay_seconds(self) -> float:
        """Wall-clock delay between emissions. Zero in fast mode."""
        if self.speed == 0.0:
            return 0.0
        return self.interval_seconds / self.speed

    def observations(self, *, limit: int | None = None) -> Iterator[Observation]:
        """Yield observations one at a time, waiting between them.

        Rows that cannot be parsed are collected into :attr:`rejected` rather
        than raising: a live feed that dies on one malformed message is not a
        live feed. Nothing is dropped silently -- every row ends up in the
        yielded stream or in ``rejected``.

        Args:
            limit: Stop after this many observations. ``None`` replays the file.

        Yields:
            Canonical observations in file order, each stamped with the
            ``received_time`` at which it was emitted.
        """
        delay = self.delay_seconds
        emitted = 0

        self.rows_read = 0
        self.rejected = []

        with self.path.open("r", encoding="utf-8", newline="") as handle:
            for index, payload in enumerate(csv.DictReader(handle), start=1):
                if limit is not None and emitted >= limit:
                    break

                self.rows_read += 1
                cleaned = {key: ("" if value is None else value) for key, value in payload.items()}

                metric_name = (cleaned.get("metric") or "").strip()
                configured = (
                    self.settings.resolve_metric(self.problem_type, metric_name)
                    if self.settings is not None and metric_name
                    else None
                )
                units = (
                    frozenset(configured.accepted_units)
                    if configured is not None and configured.accepted_units
                    else None
                )

                parsed = parse_row(
                    cleaned, interval=self.measurement_interval, accepted_units=units
                )

                # Stamp receipt now, not when the file was opened: this is what
                # makes received_time advance across the stream.
                received = self._clock()

                if isinstance(parsed, RowParseError):
                    self.rejected.append(
                        to_quarantine_record(
                            parsed,
                            source_path=self.path,
                            received_time=received,
                            index=index,
                        )
                    )
                    continue

                if delay > 0:
                    self._sleep(delay)
                    received = self._clock()

                yield to_observation(
                    parsed,
                    received_time=received,
                    source_type=SourceType.REAL_TIME,
                    problem_type=self.problem_type,
                    registry=self.registry,
                    canonical_metric=configured.name if configured is not None else None,
                    source_row=cleaned,
                )
                emitted += 1

    def collect(self, *, limit: int | None = None) -> IngestionResult:
        """Drain the stream into an :class:`IngestionResult`.

        Convenience for tests and demos that want the whole held-out period at
        once; the streaming interface is :meth:`observations`.
        """
        observations = tuple(self.observations(limit=limit))
        return IngestionResult(
            observations=observations,
            rejected=tuple(self.rejected),
            stats=IngestionStats(
                rows_read=self.rows_read,
                observations=len(observations),
                rejected=len(self.rejected),
                missing_values=sum(1 for obs in observations if obs.value is None),
                source_path=self.path,
            ),
        )
