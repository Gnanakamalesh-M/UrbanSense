"""Shared temporal validation (SPEC sections 12, 33).

Two rules apply to every timestamp in the system:

1. It must be timezone-aware, and it is stored normalized to UTC. A naive
   datetime is ambiguous, and an ambiguous instant cannot be compared across
   sources -- so it is rejected at the boundary rather than guessed at. Local
   wall-clock time belongs in provenance, never in ``event_time``.
2. Event time and receipt time are distinct. "When it happened" is never
   conflated with "when we were told", which is what makes late-arriving data
   and newly uploaded old reports behave correctly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from pydantic import AfterValidator

#: How far past "now" an ``event_time`` may sit before it is treated as a clock
#: error or a leakage attempt (SPEC section 33). Small, non-zero tolerance to
#: absorb clock skew between a sensor and this process.
FUTURE_TOLERANCE = timedelta(minutes=5)


def require_utc(value: datetime) -> datetime:
    """Reject naive datetimes; normalize aware ones to UTC.

    Raises:
        ValueError: if ``value`` has no timezone information.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            "datetime must be timezone-aware; got a naive value. "
            "Attach a timezone (e.g. datetime.now(UTC)) -- an instant without "
            "an offset cannot be compared across sources."
        )
    return value.astimezone(UTC)


#: A timezone-aware datetime, stored in UTC.
UTCDateTime = Annotated[datetime, AfterValidator(require_utc)]


def is_future(
    value: datetime, *, now: datetime | None = None, tolerance: timedelta = FUTURE_TOLERANCE
) -> bool:
    """Return whether ``value`` lies beyond ``now`` by more than ``tolerance``."""
    reference = now if now is not None else datetime.now(UTC)
    return value > reference + tolerance
