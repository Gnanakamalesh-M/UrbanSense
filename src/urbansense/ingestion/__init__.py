"""Source-specific ingestion (SPEC sections 4, 37).

Historical, real-time and document streams are ingested through *independent*
pipelines and never mixed before validation, so a wrong number stays traceable
to the stream that produced it.

Three rules hold across every loader here:

**Raw data is immutable.** Nothing in this package opens a file for writing.
Inputs are read-only, and rejected rows are *returned* rather than written, so
the decision of where quarantine lives stays with the caller.

**Nothing is dropped silently.** Every input row becomes either an observation
or a :class:`~urbansense.schemas.quarantine.QuarantineRecord` carrying its
original payload. :attr:`~urbansense.ingestion.result.IngestionResult.accounts_for_every_row`
states that invariant and the tests assert it.

**Ingestion transcribes; it does not vouch.** Observations arrive as
``PENDING``. A negative vehicle count parses fine and passes straight through --
judging values is the integrity engine's job (SPEC section 8), and doing it here
would pre-empt it.

Implemented in Phase 1: historical CSV loading and the real-time stream
simulator, both for traffic. Document ingestion arrives with Phase 2.
"""

from urbansense.ingestion.builders import (
    PIPELINE_VERSION,
    ZoneRegistry,
    aggregation_for,
    to_observation,
    to_quarantine_record,
)
from urbansense.ingestion.historical import load_historical
from urbansense.ingestion.realtime import DEFAULT_INTERVAL_SECONDS, StreamSimulator
from urbansense.ingestion.result import IngestionResult, IngestionStats
from urbansense.ingestion.rows import (
    REQUIRED_COLUMNS,
    ParsedRow,
    RowParseError,
    parse_row,
)

__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "PIPELINE_VERSION",
    "REQUIRED_COLUMNS",
    "IngestionResult",
    "IngestionStats",
    "ParsedRow",
    "RowParseError",
    "StreamSimulator",
    "ZoneRegistry",
    "aggregation_for",
    "load_historical",
    "parse_row",
    "to_observation",
    "to_quarantine_record",
]
