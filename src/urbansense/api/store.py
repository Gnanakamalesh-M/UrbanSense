"""Reading the stored artifacts the API serves.

Every method here opens a file. Nothing runs the pipeline, trains a model or
recomputes a statistic -- the API's job is to make what the pipeline already
produced inspectable, and a request that quietly triggered half an hour of
feature engineering would be a different kind of thing.

**A missing artifact is a normal state, not an error.** A fresh clone has no
pattern report because discovery has not been run, and no experiment report
because the experiment takes twelve minutes. Those endpoints return an empty
page with a note naming the file that would have been read, so a reader can tell
"nothing found" from "nothing run yet". :attr:`ArtifactStore.availability` is
what ``/health`` reports.

Artifacts are cached after the first read. They are written by batch jobs and do
not change under a running server, so re-parsing a 35MB JSONL on every page of
results would be waste rather than freshness.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from urbansense.api.export import (
    CONFLICTS_FILENAME,
    DEFAULT_EXPORT_DIR,
    QUALITY_FILENAME,
    UNIFIED_FILENAME,
)

#: Return type of a cached artifact read.
T = TypeVar("T")

#: Where each artifact family lives, relative to the repository root.
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_MODELS_DIR = Path("models") / "registry"
DEFAULT_QUARANTINE_DIR = Path("quarantine")


def _read_json(path: Path) -> object | None:
    """Read one JSON file, or ``None`` when it is absent.

    Returns ``object`` rather than a concrete shape on purpose: these files are
    written by several different phases and a caller has to narrow with
    ``isinstance`` anyway. Claiming a type here would be claiming the file's
    schema, which this module is in no position to guarantee.
    """
    if not path.is_file():
        return None
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file, or an empty list when it is absent."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _latest(directory: Path, pattern: str) -> Path | None:
    """The most recent matching file, by name.

    Report filenames carry a UTC timestamp, so sorting by name is sorting by
    time. Several phases append a new file per run rather than overwriting, and
    the API shows the newest.
    """
    if not directory.is_dir():
        return None
    found = sorted(directory.glob(pattern))
    return found[-1] if found else None


@dataclass
class ArtifactStore:
    """Opens and caches the artifacts the endpoints read.

    Attributes:
        root: Repository root. Every other path is derived from it, so a test
            can point the whole store at a temporary directory.
    """

    root: Path = Path(".")
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def export_dir(self) -> Path:
        """Where the materialized unified layer lives."""
        return self.root / DEFAULT_EXPORT_DIR

    @property
    def reports_dir(self) -> Path:
        """Where per-run reports live."""
        return self.root / DEFAULT_REPORTS_DIR

    @property
    def models_dir(self) -> Path:
        """Where model metadata lives."""
        return self.root / DEFAULT_MODELS_DIR

    @property
    def quarantine_dir(self) -> Path:
        """Where held-back records live."""
        return self.root / DEFAULT_QUARANTINE_DIR

    def _cached(self, key: str, load: Callable[[], T]) -> T:
        """Memoize one artifact read."""
        if key not in self._cache:
            self._cache[key] = load()
        cached: T = self._cache[key]
        return cached

    def clear(self) -> None:
        """Drop the cache, so a re-exported artifact is picked up."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # The artifacts
    # ------------------------------------------------------------------

    def unified(self) -> list[dict[str, Any]]:
        """Unified observations, provenance included."""
        return list(
            self._cached("unified", lambda: _read_jsonl(self.export_dir / UNIFIED_FILENAME))
        )

    def conflicts(self) -> list[dict[str, Any]]:
        """Recorded conflict groups.

        Reads the export first and falls back to the quarantine queue, because
        Phase 3's ``reconcile_demo`` writes conflicts there and a reader may have
        run that without running the API export.
        """

        def load() -> list[dict[str, Any]]:
            payload = _read_json(self.export_dir / CONFLICTS_FILENAME)
            if isinstance(payload, dict) and isinstance(payload.get("groups"), list):
                return list(payload["groups"])
            fallback = _latest(self.quarantine_dir / "conflicts", "conflicts-*.json")
            if fallback is None:
                return []
            found = _read_json(fallback)
            if isinstance(found, dict):
                groups = found.get("groups") or found.get("conflicts")
                return list(groups) if isinstance(groups, list) else []
            return list(found) if isinstance(found, list) else []

        return list(self._cached("conflicts", load))

    def data_quality(self) -> dict[str, Any] | None:
        """Counts and per-source health."""
        payload = self._cached("quality", lambda: _read_json(self.export_dir / QUALITY_FILENAME))
        return payload if isinstance(payload, dict) else None

    def quarantine(self) -> list[dict[str, Any]]:
        """Held-back records, newest report first."""

        def load() -> list[dict[str, Any]]:
            path = _latest(self.quarantine_dir, "validation-*.json")
            if path is None:
                return []
            payload = _read_json(path)
            if isinstance(payload, list):
                return list(payload)
            if isinstance(payload, dict) and isinstance(payload.get("records"), list):
                return list(payload["records"])
            return []

        return list(self._cached("quarantine", load))

    def patterns(self) -> list[dict[str, Any]]:
        """Discovered patterns from the most recent discovery run."""

        def load() -> list[dict[str, Any]]:
            path = _latest(self.reports_dir / "patterns", "patterns-*.json")
            payload = _read_json(path) if path is not None else None
            if isinstance(payload, dict) and isinstance(payload.get("patterns"), list):
                return list(payload["patterns"])
            return []

        return list(self._cached("patterns", load))

    def pattern_run(self) -> dict[str, Any] | None:
        """The discovery run header.

        Served alongside the patterns because a list of findings without the
        number of hypotheses behind it cannot be judged -- one pattern out of
        672 tests is a different claim from one out of ten.
        """

        def load() -> dict[str, Any] | None:
            path = _latest(self.reports_dir / "patterns", "patterns-*.json")
            payload = _read_json(path) if path is not None else None
            if isinstance(payload, dict) and isinstance(payload.get("run"), dict):
                return dict(payload["run"])
            return None

        found = self._cached("pattern_run", load)
        return found if isinstance(found, dict) else None

    def anomalies(self) -> list[dict[str, Any]]:
        """Flagged readings with their candidate explanations."""

        def load() -> list[dict[str, Any]]:
            path = _latest(self.reports_dir / "anomalies", "anomalies-*.json")
            payload = _read_json(path) if path is not None else None
            if isinstance(payload, dict) and isinstance(payload.get("anomalies"), list):
                return list(payload["anomalies"])
            return []

        return list(self._cached("anomalies", load))

    def predictions(self) -> list[dict[str, Any]]:
        """Recorded prediction outcomes.

        The outcome log is what makes ``/predictions`` answerable from storage:
        it holds the prediction, the actual, the calibrated probability, the
        severity and the conditions, per zone and target hour.
        """
        return list(
            self._cached(
                "predictions", lambda: _read_jsonl(self.reports_dir / "outcomes" / "outcomes.jsonl")
            )
        )

    def drift_events(self) -> list[dict[str, Any]]:
        """Drift evaluations in time order."""
        return list(
            self._cached(
                "drift_events", lambda: _read_jsonl(self.reports_dir / "drift" / "events.jsonl")
            )
        )

    def drift_episodes(self) -> list[dict[str, Any]]:
        """Drift episodes."""
        return list(
            self._cached(
                "drift_episodes",
                lambda: _read_jsonl(self.reports_dir / "drift" / "episodes.jsonl"),
            )
        )

    def candidates(self) -> list[dict[str, Any]]:
        """Candidate updates and their decisions."""
        return list(
            self._cached(
                "candidates",
                lambda: _read_jsonl(self.reports_dir / "adaptation" / "candidates.jsonl"),
            )
        )

    def models(self) -> list[dict[str, Any]]:
        """Model versions, oldest first."""

        def load() -> list[dict[str, Any]]:
            if not self.models_dir.is_dir():
                return []
            found: list[dict[str, Any]] = []
            for path in sorted(self.models_dir.glob("*.json")):
                payload = _read_json(path)
                if isinstance(payload, dict):
                    found.append(payload)
            found.sort(key=lambda item: str(item.get("created_at", "")))
            return found

        return list(self._cached("models", load))

    def experiments(self) -> list[dict[str, Any]]:
        """Experiment reports, newest first."""

        def load() -> list[dict[str, Any]]:
            directory = self.reports_dir / "experiments"
            if not directory.is_dir():
                return []
            found: list[dict[str, Any]] = []
            for path in sorted(directory.glob("experiment-*.json"), reverse=True):
                payload = _read_json(path)
                if isinstance(payload, dict):
                    payload = {**payload, "report_file": path.name}
                    found.append(payload)
            return found

        return list(self._cached("experiments", load))

    # ------------------------------------------------------------------
    # What is and is not there
    # ------------------------------------------------------------------

    @property
    def availability(self) -> dict[str, bool]:
        """Which artifact families have something to serve.

        ``/health`` reports this so a reader can tell an empty endpoint from a
        broken one without reading the server's logs.
        """
        return {
            "unified": bool(self.unified()),
            "conflicts": bool(self.conflicts()),
            "data_quality": self.data_quality() is not None,
            "quarantine": bool(self.quarantine()),
            "patterns": bool(self.patterns()),
            "anomalies": bool(self.anomalies()),
            "predictions": bool(self.predictions()),
            "drift": bool(self.drift_events()),
            "models": bool(self.models()),
            "experiments": bool(self.experiments()),
        }

    def missing_note(self, family: str) -> str | None:
        """Why a family is empty, and what to run to fill it.

        Returned as a ``note`` on the response rather than raised, because "no
        experiment has been run" is a true and useful answer to "list the
        experiments" -- not a server fault.
        """
        hints = {
            "unified": (
                f"no export at {(self.export_dir / UNIFIED_FILENAME).as_posix()}; "
                "run: python scripts/serve_api.py --rebuild"
            ),
            "conflicts": ("no conflict record; run: python scripts/reconcile_demo.py"),
            "data_quality": (
                f"no summary at {(self.export_dir / QUALITY_FILENAME).as_posix()}; "
                "run: python scripts/serve_api.py --rebuild"
            ),
            "quarantine": "no validation report; run: python scripts/validate_demo.py",
            "patterns": "no discovery report; run: python scripts/discover_patterns.py",
            "anomalies": "no anomaly report; run: python scripts/discover_patterns.py",
            "predictions": "no outcome log; run: python scripts/error_report.py",
            "drift": "no drift log; run: python scripts/replay_adaptive.py",
            "models": "no registry entry; run: python scripts/train_baseline.py",
            "experiments": "no experiment report; run: python scripts/run_experiment.py",
        }
        if self.availability.get(family, False):
            return None
        return hints.get(family)


def parse_time(value: str | None) -> datetime | None:
    """Parse an ISO-8601 instant from a query parameter, or ``None``."""
    if value is None:
        return None
    return datetime.fromisoformat(value)
