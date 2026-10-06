"""Materializing the pipeline's output for the API to read.

The API serves stored artifacts. Every endpoint but one already had a file to
read -- patterns, outcomes, drift events, candidates, the model registry, the
quarantine queue -- because earlier phases wrote them as they worked. The
unified data layer did not: it is rebuilt from CSV every time, which takes about
half a minute.

So this module writes it down once. The alternative was to rebuild it inside the
API process and cache it, and that would have made the API no longer a read-only
thing: it would run the ingestion, validation and reconciliation pipeline on a
request, hold the result in memory, and lose it on restart. Exporting instead
keeps the serving path to "open a file", and the export is itself an inspectable
artifact a person can grep.

Three files, into ``data/processed/`` which the repository already gitignores:

``unified.jsonl``
    One row per unified observation, **with its provenance**, so "where did this
    value come from?" is answerable from an API response alone rather than
    requiring a second lookup.

``conflicts.json``
    Recorded disagreements, every candidate value kept. Nothing is resolved by
    overwriting, so a conflict is a hole in the unified layer plus this record
    of what the sources each claimed.

``data_quality.json``
    Counts and per-source health. Written here rather than derived in the API
    because the counts come from the validation run, which the API never sees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from urbansense.config import Settings, load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.data_integrity.result import ValidatedResult
from urbansense.data_integrity.unified import ReconciledResult
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.preprocessing import load_location_registry
from urbansense.schemas.observation import Observation
from urbansense.synthetic.generator import HISTORY_FILENAME, ZONES_FILENAME

#: Default directory the export is written to. Gitignored: it is derived data,
#: regenerable from the committed CSVs by re-running the export.
DEFAULT_EXPORT_DIR = Path("data") / "processed"

#: Default source dataset.
DEFAULT_DATA_DIR = Path("data") / "sample"

UNIFIED_FILENAME = "unified.jsonl"
CONFLICTS_FILENAME = "conflicts.json"
QUALITY_FILENAME = "data_quality.json"


def observation_row(observation: Observation) -> dict[str, object]:
    """One unified observation as a JSON row, provenance included.

    The three timestamps are all kept. They answer different questions -- when
    the measurement happened, when the source stamped it, when we received it --
    and collapsing them would make a late arrival indistinguishable from a
    prompt one.
    """
    provenance = observation.provenance
    return {
        "observation_id": observation.observation_id,
        "record_id": observation.attributes.get("record_id"),
        "source_id": observation.source_id,
        "source_type": observation.source_type.value,
        "problem_type": observation.problem_type.value,
        "location_id": observation.location_id,
        "metric": observation.metric,
        "value": observation.value,
        "unit": observation.unit,
        "event_time": observation.event_time.isoformat(),
        "source_time": (observation.source_time.isoformat() if observation.source_time else None),
        "received_time": observation.received_time.isoformat(),
        "aggregation_level": observation.aggregation_level.value,
        "validation_status": observation.validation_status.value,
        "is_imputed": observation.is_imputed,
        "duplicate_of": observation.duplicate_of,
        "conflicts_with": list(observation.conflicts_with),
        "provenance": provenance.model_dump(mode="json"),
    }


@dataclass(frozen=True)
class ExportResult:
    """What one export wrote.

    Attributes:
        unified: Path to the unified rows.
        conflicts: Path to the conflict groups.
        quality: Path to the data-quality summary.
        rows: Unified rows written.
    """

    unified: Path
    conflicts: Path
    quality: Path
    rows: int

    def describe(self) -> str:
        """Human-readable summary."""
        return (
            f"exported {self.rows:,} unified row(s) to {self.unified}, plus "
            f"{self.conflicts.name} and {self.quality.name}"
        )


def quality_summary(validated: ValidatedResult, reconciled: ReconciledResult) -> dict[str, object]:
    """Counts and source health, from the run that produced them.

    Reported rather than derived later because the interesting numbers -- rows
    in, what was quarantined and why, which sources are unhealthy -- exist only
    during validation. By the time the unified layer is on disk, the records
    that were held back are no longer in it.
    """
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "rows_in": validated.stats.rows_in,
        "valid": len(validated.valid),
        "suspicious": len(validated.suspicious),
        "quarantined": len(validated.quarantined),
        "missing_values": validated.stats.missing_values,
        "normalized_units": validated.stats.normalized_units,
        "accounts_for_every_row": validated.accounts_for_every_row,
        "quarantine_by_reason": {
            reason.value: count for reason, count in validated.by_reason.items()
        },
        "source_health": {
            source_id: {
                "status": health.status.value,
                "observations": health.observation_count,
                "missing": health.missing_count,
                "frozen_runs": health.frozen_run_count,
                "frozen_observations": health.frozen_observation_count,
                "last_observation_time": (
                    health.last_observation_time.isoformat()
                    if health.last_observation_time
                    else None
                ),
                "max_gap_hours": (
                    round(health.max_gap.total_seconds() / 3600.0, 2)
                    if health.max_gap is not None
                    else None
                ),
                "detail": health.detail,
                "locations": sorted(health.locations),
            }
            for source_id, health in sorted(validated.source_health.items())
        },
        "unified_rows": len(reconciled.unified),
        "duplicate_linked": len(reconciled.duplicate_linked),
        "conflicted_observations": len(reconciled.conflicted_observations),
        "conflict_groups": len(reconciled.conflicts),
        "accounts_for_every_observation": reconciled.accounts_for_every_observation,
        "notes": [
            "Missing values are counted, never filled. A row with no measurement is still a row.",
            "Quarantined, duplicate-linked and conflicted records are excluded "
            "from the unified layer but retained in their own buckets.",
        ],
    }


def export_api_data(
    *,
    source: Path | None = None,
    destination: Path | None = None,
    settings: Settings | None = None,
) -> ExportResult:
    """Run the pipeline once and write what the API needs.

    Args:
        source: Dataset directory holding the history CSV. Defaults to
            ``data/sample``.
        destination: Where to write. Defaults to ``data/processed``.
        settings: Problem configuration.

    Returns:
        An :class:`ExportResult`.

    Raises:
        FileNotFoundError: if the history CSV is absent, with the command that
            generates it.
    """
    data_dir = source if source is not None else DEFAULT_DATA_DIR
    out = destination if destination is not None else DEFAULT_EXPORT_DIR
    history = data_dir / HISTORY_FILENAME
    if not history.is_file():
        raise FileNotFoundError(
            f"{history} not found. Generate the dataset first:\n"
            f"  python scripts/generate_demo_data.py --out {data_dir.as_posix()}"
        )

    resolved = settings if settings is not None else load_settings()
    zones = data_dir / ZONES_FILENAME
    registry = ZoneRegistry.from_csv(zones) if zones.is_file() else ZoneRegistry.empty()
    validated = validate_and_normalize(
        load_historical(history, registry=registry),
        settings=resolved,
        registry=load_location_registry(),
    )
    reconciled = reconcile(validated, settings=resolved)

    out.mkdir(parents=True, exist_ok=True)
    unified_path = out / UNIFIED_FILENAME
    with unified_path.open("w", encoding="utf-8", newline="\n") as handle:
        for observation in reconciled.unified:
            handle.write(json.dumps(observation_row(observation), sort_keys=True) + "\n")

    conflicts_path = out / CONFLICTS_FILENAME
    conflicts_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "groups": [group.as_dict() for group in reconciled.conflicts],
                "note": (
                    "Conflicts are recorded, never resolved by overwriting. Each "
                    "group keeps every candidate value and the sources that "
                    "claimed it; the disputed slot is a hole in the unified layer "
                    "rather than a guess."
                ),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    quality_path = out / QUALITY_FILENAME
    quality_path.write_text(
        json.dumps(quality_summary(validated, reconciled), indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return ExportResult(
        unified=unified_path,
        conflicts=conflicts_path,
        quality=quality_path,
        rows=len(reconciled.unified),
    )


def export_is_present(destination: Path | None = None) -> bool:
    """Whether a usable export already exists."""
    out = destination if destination is not None else DEFAULT_EXPORT_DIR
    return all(
        (out / name).is_file() for name in (UNIFIED_FILENAME, CONFLICTS_FILENAME, QUALITY_FILENAME)
    )
