"""The quarantine record (SPEC sections 14, 18).

Questionable information is never destroyed. When a payload cannot even be
parsed into an :class:`~urbansense.schemas.observation.Observation`, it still has
to be kept -- a record that fails schema validation is exactly the record a
reviewer most needs to see.

:class:`QuarantineRecord` therefore stores the *raw payload* as given, plus the
reason and the error, and makes the parsed observation optional. This is the one
type in the schema layer that deliberately accepts malformed input.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from urbansense.schemas.enums import QuarantineReason
from urbansense.schemas.observation import Observation
from urbansense.schemas.temporal import UTCDateTime


class QuarantineRecord(BaseModel):
    """A rejected or suspicious record, preserved with its original payload.

    Attributes:
        quarantine_id: Identifier for this quarantine entry.
        reason: Why the record was quarantined.
        detail: Human-readable explanation, e.g. the validation error text.
        quarantined_at: When it was quarantined (UTC).
        source_id: Source the payload came from, when known.
        raw_payload: The payload exactly as received. Never normalized, so the
            original evidence survives even if our parsing logic was wrong.
        observation: The parsed observation, when parsing succeeded and the
            record was quarantined on a content judgement rather than a schema
            failure.
        detector: Which check raised it, for auditing the detectors themselves.
        resolved: Whether a reviewer has dispositioned this entry.
        resolution_notes: What the reviewer decided and why.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    quarantine_id: str = Field(min_length=1)
    reason: QuarantineReason
    detail: str = Field(min_length=1)
    quarantined_at: UTCDateTime
    source_id: str | None = None
    raw_payload: dict[str, Any]
    observation: Observation | None = None
    detector: str = Field(min_length=1)
    resolved: bool = False
    resolution_notes: str | None = None
