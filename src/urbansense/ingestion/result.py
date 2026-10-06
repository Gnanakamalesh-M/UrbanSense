"""The result of an ingestion run (SPEC sections 4, 18).

Ingestion returns both what it produced *and* what it could not, because the
alternative -- logging a warning and moving on -- loses records. The invariant
:meth:`IngestionResult.accounts_for_every_row` makes that explicit and is
asserted in the tests: observations plus rejections must equal the input.

Rejections are **returned, not written**. Ingestion does no IO beyond reading
its input, so the decision of where quarantine lives stays with the caller and
nothing in this package can touch the raw files.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from urbansense.schemas.observation import Observation
from urbansense.schemas.quarantine import QuarantineRecord


class IngestionStats(BaseModel):
    """Counters for one ingestion run.

    Attributes:
        rows_read: Data rows read from the source, excluding the header.
        observations: Rows turned into canonical observations.
        rejected: Rows that could not be represented and were quarantined.
        missing_values: Observations whose value is absent. Counted separately
            because "we ingested 50,000 rows" means something different when a
            tenth of them carry no measurement.
        source_path: File the rows came from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows_read: int = 0
    observations: int = 0
    rejected: int = 0
    missing_values: int = 0
    source_path: Path | None = None


class IngestionResult(BaseModel):
    """Everything one ingestion run produced.

    Attributes:
        observations: Canonical observations, in input order.
        rejected: Rows that could not become observations, each carrying its
            original payload for review.
        stats: Counters for the run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    observations: tuple[Observation, ...] = ()
    rejected: tuple[QuarantineRecord, ...] = ()
    stats: IngestionStats = Field(default_factory=IngestionStats)

    @property
    def accounts_for_every_row(self) -> bool:
        """Whether every input row became an observation or a rejection.

        If this is ever false, ingestion has lost data silently, which the
        project forbids outright.
        """
        return self.stats.rows_read == len(self.observations) + len(self.rejected)

    def __len__(self) -> int:
        """Number of observations produced."""
        return len(self.observations)
