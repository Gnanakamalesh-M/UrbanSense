"""The result of a validation run (SPEC sections 8, 16, 18).

A deliberate **two-bucket partition**: every input row ends up either usable or
quarantined, and :attr:`ValidatedResult.accounts_for_every_row` asserts it.
Anything else would mean records vanishing between stages, which the project
forbids outright.

Suspicious observations are *not* a third bucket. An outlier is real data until
proven otherwise (SPEC section 16), so it stays in ``valid`` carrying
``validation_status=SUSPECT``, and :attr:`suspicious` is a view over ``valid``
rather than a separate collection. That keeps the partition exact while still
letting a consumer ask for the flagged rows -- or exclude them -- by choice.
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from urbansense.data_integrity.outliers import OutlierFlag
from urbansense.data_integrity.sensors import FrozenRun, SourceHealth
from urbansense.schemas.enums import QuarantineReason, Recency, SourceStatus, ValidationStatus
from urbansense.schemas.observation import Observation
from urbansense.schemas.quarantine import QuarantineRecord


class ValidationStats(BaseModel):
    """Counters for one validation run.

    Attributes:
        rows_in: Input rows, taken from the ingestion result so the invariant
            is checked against the file rather than against our own bookkeeping.
        valid: Observations that came through usable.
        suspicious: Usable observations flagged for investigation.
        quarantined: Records held back.
        missing_values: Usable observations with no value. Reported separately
            because "58,000 valid rows" means something different when some
            carry no measurement -- and they are still missing, never zero.
        carried_forward: Quarantine records inherited from ingestion rather than
            produced here, so the two stages' contributions stay separable.
        normalized_units: Observations whose unit was converted.
        source_path: File the data came from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows_in: int = 0
    valid: int = 0
    suspicious: int = 0
    quarantined: int = 0
    missing_values: int = 0
    carried_forward: int = 0
    normalized_units: int = 0
    source_path: Path | None = None


class ValidatedResult(BaseModel):
    """Everything one validation run produced.

    Attributes:
        valid: Usable observations, normalized, each ``VALID`` or ``SUSPECT``.
        quarantined: Records held back, each with its original payload and a
            reason.
        source_health: Health per source id.
        frozen_runs: Frozen-sensor runs detected.
        outlier_flags: Outlier detail keyed by observation id, for the rows
            flagged ``SUSPECT``.
        stats: Counters for the run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    valid: tuple[Observation, ...] = ()
    quarantined: tuple[QuarantineRecord, ...] = ()
    source_health: dict[str, SourceHealth] = Field(default_factory=dict)
    frozen_runs: tuple[FrozenRun, ...] = ()
    outlier_flags: dict[str, OutlierFlag] = Field(default_factory=dict)
    stats: ValidationStats = Field(default_factory=ValidationStats)

    @property
    def accounts_for_every_row(self) -> bool:
        """Whether every input row became a usable or a quarantined record.

        If this is ever false, records have been lost silently between
        ingestion and validation, which the project forbids.
        """
        return self.stats.rows_in == len(self.valid) + len(self.quarantined)

    @property
    def suspicious(self) -> tuple[Observation, ...]:
        """Usable observations flagged for investigation.

        A view over ``valid``, not a separate bucket: an outlier is retained
        data, so removing it from ``valid`` would both break the partition and
        quietly discard a possibly-genuine event.
        """
        return tuple(obs for obs in self.valid if obs.validation_status is ValidationStatus.SUSPECT)

    @property
    def clean(self) -> tuple[Observation, ...]:
        """Usable observations with nothing flagged against them."""
        return tuple(obs for obs in self.valid if obs.validation_status is ValidationStatus.VALID)

    @property
    def by_reason(self) -> dict[QuarantineReason, int]:
        """Quarantine counts per reason, highest first."""
        counts = Counter(record.reason for record in self.quarantined)
        return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0].value)))

    @property
    def by_recency(self) -> dict[Recency, int]:
        """Usable observation counts per recency band."""
        counts: Counter[Recency] = Counter()
        for observation in self.valid:
            raw = observation.attributes.get("recency")
            if isinstance(raw, str):
                counts[Recency(raw)] += 1
        return dict(sorted(counts.items(), key=lambda pair: pair[0].value))

    @property
    def unhealthy_sources(self) -> dict[str, SourceHealth]:
        """Sources that are not ``OK``."""
        return {
            source_id: health
            for source_id, health in sorted(self.source_health.items())
            if health.status is not SourceStatus.OK
        }

    @property
    def total_frozen_observations(self) -> int:
        """Readings held back because their sensor was frozen."""
        return sum(run.length for run in self.frozen_runs)

    @property
    def frozen_span(self) -> timedelta | None:
        """Span from the first to the last frozen reading, if any."""
        if not self.frozen_runs:
            return None
        return max(run.end for run in self.frozen_runs) - min(run.start for run in self.frozen_runs)

    def __len__(self) -> int:
        """Number of usable observations."""
        return len(self.valid)
