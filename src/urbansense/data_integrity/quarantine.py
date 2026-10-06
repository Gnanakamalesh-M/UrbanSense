"""Writing quarantined records to disk (SPEC section 18).

    Never destroy questionable information without traceability.

So the quarantine directory is append-only in spirit: each run writes a new
timestamped file and nothing overwrites a previous one. A record that failed
validation is exactly what a reviewer needs, and it is written with its **raw
payload unaltered** -- normalizing it on the way out would destroy the evidence
of what the source actually sent.

Suspicious observations are written separately, under ``review/``. They are
**notes, not removals**: those rows remain in the usable stream (SPEC section
16), and the note exists so a human can look at a flagged value without having
to re-run the pipeline. Keeping them in a different directory is what stops a
reader mistaking "worth a look" for "rejected".
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from urbansense.data_integrity.conflicts import ConflictGroup
from urbansense.data_integrity.outliers import OutlierFlag
from urbansense.schemas.observation import Observation
from urbansense.schemas.quarantine import QuarantineRecord

#: Subdirectory for review notes about retained-but-flagged observations.
REVIEW_SUBDIR = "review"

#: Subdirectory for the conflict review queue (SPEC section 10). Separate from
#: ``review/`` because these are not observations held for a second look -- they
#: are disagreements awaiting a decision, and conflating the two would let
#: "we are unsure about this value" read as "we could not decide which value".
CONFLICTS_SUBDIR = "conflicts"


def _stamp(now: datetime | None = None) -> str:
    """Filename-safe UTC timestamp."""
    moment = now if now is not None else datetime.now(UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ")


class QuarantineWriter:
    """Writes quarantine records and review notes into a directory.

    Attributes:
        root: Quarantine directory. Created on demand.
    """

    def __init__(self, root: Path) -> None:
        """Initialize the writer.

        Args:
            root: Directory to write into, typically ``quarantine/``.
        """
        self.root = root

    def write_records(
        self,
        records: Sequence[QuarantineRecord],
        *,
        label: str = "validation",
        now: datetime | None = None,
    ) -> Path | None:
        """Write quarantine records as one timestamped JSON file.

        Args:
            records: The records to persist, payloads intact.
            label: Prefix for the filename, naming the stage that produced it.
            now: Timestamp for the filename. Defaults to now.

        Returns:
            The file written, or ``None`` when there was nothing to write. An
            empty run deliberately leaves no file, so the presence of a file
            means something actually needed review.
        """
        if not records:
            return None

        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{label}-{_stamp(now)}.json"
        payload = [record.model_dump(mode="json") for record in records]
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return target

    def write_review_notes(
        self,
        observations: Iterable[Observation],
        flags: dict[str, OutlierFlag],
        *,
        label: str = "suspicious",
        now: datetime | None = None,
    ) -> Path | None:
        """Write notes about retained-but-flagged observations.

        These rows are **not** quarantined -- they stay in the usable stream.
        The note records why they were flagged so a reviewer can judge whether
        the value was an incident, an event, or a sensor problem
        (SPEC section 16).

        Args:
            observations: The flagged observations.
            flags: Outlier detail keyed by observation id.
            label: Prefix for the filename.
            now: Timestamp for the filename.

        Returns:
            The file written, or ``None`` when nothing was flagged.
        """
        notes: list[dict[str, Any]] = []
        for observation in observations:
            flag = flags.get(observation.observation_id)
            notes.append(
                {
                    "observation_id": observation.observation_id,
                    "record_id": observation.attributes.get("record_id"),
                    "source_id": observation.source_id,
                    "location_id": observation.location_id,
                    "metric": observation.metric,
                    "event_time": observation.event_time.isoformat(),
                    "value": observation.value,
                    "unit": observation.unit,
                    "validation_status": observation.validation_status.value,
                    "retained": True,
                    "note": (
                        flag.detail
                        if flag is not None
                        else "flagged as suspicious; retained for investigation"
                    ),
                    "deviation_mad_sigma": None if flag is None else round(flag.deviation, 2),
                    "group_median": None if flag is None else flag.group_median,
                    "group_size": None if flag is None else flag.group_size,
                }
            )

        if not notes:
            return None

        directory = self.root / REVIEW_SUBDIR
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{label}-{_stamp(now)}.json"
        target.write_text(
            json.dumps(notes, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return target

    def write_conflicts(
        self,
        groups: Sequence[ConflictGroup],
        *,
        label: str = "conflicts",
        now: datetime | None = None,
    ) -> Path | None:
        """Write detected conflicts to the review queue (SPEC section 10).

        Every candidate value is written with its source and **no winner is
        marked**. A reviewer decides; the file records what was observed and
        that nothing has been decided yet.

        Args:
            groups: The conflicts to persist.
            label: Prefix for the filename.
            now: Timestamp for the filename. Defaults to now.

        Returns:
            The file written, or ``None`` when there were no conflicts. An
            empty run leaves no file, so the presence of one means a decision
            is genuinely outstanding.
        """
        if not groups:
            return None

        directory = self.root / CONFLICTS_SUBDIR
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{label}-{_stamp(now)}.json"
        payload = [group.as_dict() for group in groups]
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return target
