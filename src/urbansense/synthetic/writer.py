"""CSV output for generated data (SPEC sections 4A, 39).

Two properties matter here beyond "write a file":

**The CSV contains only what a real source would send.** No label columns, no
``is_friday_evening``, no flag marking the planted duplicates. A label in the
data is a leakage channel, and a later phase that could read the answer off its
input would be credited with a discovery it never made. Every label lives in
``ground_truth.json`` instead.

**Nothing is overwritten by accident.** Raw data is immutable by project rule,
so the writer refuses to replace an existing file unless the caller explicitly
forces it. Writes go to a temporary file and are then moved into place, so an
interrupted run cannot leave a half-written dataset that looks complete.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

#: The exact CSV columns, in order. This is the contract between the generator
#: and ingestion, and it is deliberately the minimum a real traffic feed would
#: provide. Adding a column that encodes planted truth would break the dataset's
#: purpose, so the tests assert this tuple exactly.
CSV_COLUMNS: tuple[str, ...] = (
    "record_id",
    "source_id",
    "interval_end_local",
    "location_id",
    "metric",
    "value",
    "unit",
)

#: Columns of the location registry written beside the data. Coordinates live
#: here rather than being repeated on every reading, because that is how a real
#: feed works -- a sensor sends a station id, not its own latitude 70,000 times
#: -- and because SPEC section 13 wants a canonical location registry.
ZONE_COLUMNS: tuple[str, ...] = (
    "location_id",
    "description",
    "latitude",
    "longitude",
    "sensor_id",
)

#: Column-name shapes that would constitute label leakage. Checked by tests
#: against the real header, so a future change cannot quietly reintroduce one.
FORBIDDEN_COLUMN_PATTERNS: tuple[str, ...] = (
    "is_",
    "_label",
    "label_",
    "injected",
    "expected",
    "ground_truth",
    "truth_",
    "anomal",
    "duplicate",
    "conflict",
)


class RawRow(BaseModel):
    """One row as a source would emit it.

    Deliberately stringly-typed for ``value``: an empty string means *missing*,
    which is distinct from ``"0"``, and the planted invalid rows need to carry
    values a typed field would reject (an unparseable timestamp, for instance).
    Turning these into typed fields is ingestion's job, and failing to is how
    ingestion discovers what it must reject.

    Attributes:
        record_id: Stable reading id. Ground truth references these.
        source_id: Which sensor or feed produced the row.
        interval_end_local: Timestamp as the source stamped it: the **end** of
            the measurement interval, in the source's own local wall-clock time
            with offset, e.g. ``2026-01-02 19:00:00+05:30`` for the 18:00-19:00
            hour. The column is named for the convention so ingestion never has
            to guess it. The sensor's own clock skew is baked in, as it would be
            in reality -- a pipeline cannot subtract an offset it does not know.
        location_id: Zone the reading is for. Coordinates are looked up in the
            location registry, not carried on every row.
        metric: Metric name as the source calls it.
        value: The measurement, or ``""`` when missing. Never ``"0"`` for
            missing (SPEC section 15).
        unit: Unit as the source declares it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    source_id: str
    interval_end_local: str
    location_id: str
    metric: str
    value: str
    unit: str

    def as_csv_row(self) -> list[str]:
        """Render in :data:`CSV_COLUMNS` order."""
        return [getattr(self, column) for column in CSV_COLUMNS]


class ZoneRow(BaseModel):
    """One entry in the location registry written beside the data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    location_id: str
    description: str
    latitude: str
    longitude: str
    sensor_id: str

    def as_csv_row(self) -> list[str]:
        """Render in :data:`ZONE_COLUMNS` order."""
        return [getattr(self, column) for column in ZONE_COLUMNS]


class WriteRefused(RuntimeError):
    """Raised when writing would overwrite existing data without ``force``."""


def write_rows(
    path: Path,
    rows: Iterable[RawRow] | Iterable[ZoneRow],
    *,
    force: bool = False,
    columns: Sequence[str] = CSV_COLUMNS,
) -> int:
    """Write ``rows`` to ``path`` as CSV and return the number written.

    Args:
        path: Destination file. Parent directories are created as needed.
        rows: Rows to write, in order.
        force: Permit replacing an existing file. Without this, an existing
            destination raises rather than being overwritten -- raw data is
            immutable by project rule, and a generator rerun is the most likely
            way to destroy a dataset someone meant to keep.
        columns: Header to write. Defaults to :data:`CSV_COLUMNS`.

    Returns:
        Number of data rows written.

    Raises:
        WriteRefused: if ``path`` exists and ``force`` is false.
    """
    if path.exists() and not force:
        raise WriteRefused(
            f"{path} already exists. Pass force=True (or --force) to replace it; "
            "generated data is not overwritten by default."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    written = 0

    # newline="" per the csv module's contract; lineterminator is pinned so the
    # file is byte-identical on Windows and POSIX and regeneration stays
    # reproducible across machines.
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row.as_csv_row())
            written += 1

    os.replace(temporary, path)
    return written
