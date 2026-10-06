"""Recency classification (SPEC section 12).

A newly uploaded report about 2022 is not news about today. The spec is blunt
about it: data whose period long predates its receipt "must remain historical".

Recency is derived from the lag between ``event_time`` and ``received_time``,
which Phase 1 keeps as two separate fields precisely so this question can be
asked. Nothing here modifies either timestamp.

**``source_type`` is deliberately left alone.** It records *which independent
stream* a value arrived on, and SPEC section 4 requires that to stay traceable.
Recency is a second, independent axis: a reading can arrive on the live feed and
still be months old. Collapsing the two would make "what did the real-time feed
tell us?" unanswerable, and overwriting the stream label to mean "old" would
lose the provenance the project is built around.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from urbansense.schemas.enums import Recency

#: Lag at or under which an observation still describes current conditions.
DEFAULT_CURRENT_WITHIN = timedelta(hours=2)

#: Lag at or under which an observation is late but still informs the present.
DEFAULT_RECENT_WITHIN = timedelta(hours=48)


def arrival_lag(event_time: datetime, received_time: datetime) -> timedelta:
    """How long after the event the system learned about it.

    Negative when ``received_time`` precedes ``event_time``, which is a clock
    error rather than prescience; the schema only permits that combination on a
    quarantined record, and the caller decides what to do with it.
    """
    return received_time - event_time


def classify_recency(
    event_time: datetime,
    received_time: datetime,
    *,
    current_within: timedelta = DEFAULT_CURRENT_WITHIN,
    recent_within: timedelta = DEFAULT_RECENT_WITHIN,
) -> Recency:
    """Classify how old an observation was when it arrived.

    Args:
        event_time: When the measured phenomenon occurred.
        received_time: When this system received the information.
        current_within: Lag at or under which the data counts as current.
        recent_within: Lag at or under which the data counts as recent.

    Returns:
        The :class:`Recency` band. A negative lag classifies as ``CURRENT``,
        since a clock error is not evidence of age -- the timestamp validators
        handle that case on its own terms rather than having it silently
        reappear as a freshness claim.
    """
    lag = arrival_lag(event_time, received_time)
    if lag <= current_within:
        return Recency.CURRENT
    if lag <= recent_within:
        return Recency.RECENT
    return Recency.HISTORICAL


def is_historical(
    event_time: datetime,
    received_time: datetime,
    *,
    recent_within: timedelta = DEFAULT_RECENT_WITHIN,
) -> bool:
    """Whether this observation must not be treated as current."""
    return arrival_lag(event_time, received_time) > recent_within
