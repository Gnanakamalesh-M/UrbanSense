"""The pattern store (SPEC sections 22, 38).

Patterns are written as JSON under ``reports/patterns/``, one file per run. Each
run writes a new timestamped file rather than overwriting, for the same reason
the quarantine queue does: the point of tracking evolution is comparing what was
believed then against what is believed now, and overwriting destroys the earlier
belief.

The run header is written alongside the patterns and is not optional. A list of
findings without the number of hypotheses behind it cannot be judged -- one
pattern out of 672 tests is a different claim from one out of ten -- so the file
carries the FDR level, the test count, the sparse-cell count and the seed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from urbansense.patterns.discovery import DiscoveryResult

#: Default directory for pattern reports.
DEFAULT_PATTERN_DIR = Path("reports") / "patterns"


def _stamp(now: datetime | None = None) -> str:
    """Filename-safe UTC timestamp."""
    moment = now if now is not None else datetime.now(UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ")


class PatternStore:
    """Writes and reads pattern reports.

    Attributes:
        root: Directory the reports live in. Created on demand.
    """

    def __init__(self, root: Path = DEFAULT_PATTERN_DIR) -> None:
        """Initialize.

        Args:
            root: Directory to write into, typically ``reports/patterns/``.
        """
        self.root = root

    def payload(self, result: DiscoveryResult) -> dict[str, object]:
        """Build the serializable report.

        The header comes first so that anything reading the file -- a person or a
        later phase -- sees the multiple-testing context before the findings.
        """
        return {
            "generated_at": result.generated_at.isoformat(),
            "run": {
                "tests_run": result.stats.tests_run,
                "fdr_level": result.stats.fdr_level,
                "discoveries": result.stats.discoveries,
                "cells_too_sparse": result.stats.cells_too_sparse,
                "bootstrap_seed": result.stats.bootstrap_seed,
                "source_rows": result.stats.source_rows,
                "timezone_offset_hours": result.frame.timezone_offset_hours,
                "readings_used": len(result.frame),
                "readings_dropped_unknown": result.frame.dropped_unknown,
                "summary": result.header(),
            },
            "patterns": [pattern.as_dict() for pattern in result.patterns],
            "interactions": [item.as_dict() for item in result.interactions],
            "notes": [
                "Discovery reads the unified layer only; quarantined, "
                "duplicate-linked and conflicted records are excluded upstream.",
                "Strengths are effect sizes, not p-values. Confidence is 1 - p "
                "and is not the probability the pattern is real.",
                "Descriptive profiles are reported but were not part of the "
                "multiple-testing family, so they are excluded from the "
                "discovery count.",
                "Unknown readings are dropped, never imputed; cells with too "
                "little data are reported as untested.",
            ],
        }

    def write(
        self,
        result: DiscoveryResult,
        *,
        label: str = "patterns",
        now: datetime | None = None,
    ) -> Path | None:
        """Write a discovery result as one timestamped JSON file.

        Returns:
            The file written, or ``None`` when the run found nothing at all. An
            empty run leaves no file, so the presence of one means there is
            something to read.
        """
        if not result.patterns and not result.interactions:
            return None

        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{label}-{_stamp(now or result.generated_at)}.json"
        target.write_text(
            json.dumps(self.payload(result), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return target

    def reports(self) -> tuple[Path, ...]:
        """Every report written, oldest first."""
        if not self.root.is_dir():
            return ()
        return tuple(sorted(self.root.glob("*.json")))

    def load(self, path: Path) -> dict[str, object]:
        """Read one report back."""
        loaded: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def latest(self) -> dict[str, object] | None:
        """Read the most recent report, or ``None`` when there are none."""
        found = self.reports()
        return self.load(found[-1]) if found else None
