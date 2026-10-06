"""Where the dashboard gets its data (SPEC section 43).

Two backends behind one protocol, because the dashboard has two jobs that pull
in opposite directions.

:class:`ArtifactLoader`
    Opens the stored files directly, through the same
    :class:`~urbansense.api.store.ArtifactStore` the API uses. This is the
    default so ``python scripts/serve_dashboard.py`` works on a fresh clone with
    nothing else running -- no server to start first, no port to pick.

:class:`ApiLoader`
    Fetches the same things over HTTP from the read-only API. This is what
    ``docker-compose`` uses, so the two containers genuinely exercise the API
    layer rather than sitting side by side ignoring each other.

Both satisfy :class:`DashboardData`, so :mod:`urbansense.dashboard.views` never
learns which one it was given.

**Neither imports streamlit, and neither computes anything.** Every method is a
file read or an HTTP GET. A dashboard that retrained a model to draw a chart
would make the picture disagree with the reports, and would do it slowly.

``ApiLoader`` uses :mod:`urllib.request` from the standard library rather than
``httpx``, which is a dev-only dependency here. The HTTP path therefore adds no
runtime dependency.

A missing artifact is a normal state, not an error: a fresh clone has no
discovery report because nobody has run discovery. Every method returns empty
rather than raising, and :meth:`DashboardData.missing_note` carries the command
that would produce the file -- the same string the API serves.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from urbansense.api.store import ArtifactStore

#: How long to wait on the API before giving up, in seconds. Short: the
#: dashboard is a local tool, and a page that hangs for thirty seconds is worse
#: than one that says the API is unreachable.
HTTP_TIMEOUT = 10.0

#: Rows pulled per paged request. The unified layer runs to tens of thousands
#: of rows and no page displays more than a few hundred, so the dashboard asks
#: for what it shows rather than streaming the lot.
PAGE_LIMIT = 500


@runtime_checkable
class DashboardData(Protocol):
    """What a page needs, however it is fetched.

    Deliberately shaped as "return the artifact rows" rather than "return a
    chart": the shaping lives in :mod:`urbansense.dashboard.views`, so both
    backends share one set of tested transformations instead of each growing
    its own.
    """

    def availability(self) -> dict[str, bool]:
        """Which artifact families have something to show."""
        ...

    def missing_note(self, family: str) -> str | None:
        """Why a family is empty, and what to run. ``None`` when it is present."""
        ...

    def unified(self, *, zone: str | None = None, limit: int = PAGE_LIMIT) -> list[dict[str, Any]]:
        """Unified observations, provenance included."""
        ...

    def data_quality(self) -> dict[str, Any] | None:
        """Counts and per-source health."""
        ...

    def conflicts(self) -> list[dict[str, Any]]:
        """Recorded disagreements between sources."""
        ...

    def quarantine(self) -> list[dict[str, Any]]:
        """Records held back, payload as received."""
        ...

    def patterns(self) -> list[dict[str, Any]]:
        """Discovered patterns with their evidence."""
        ...

    def pattern_run(self) -> dict[str, Any] | None:
        """The discovery run header, for the hypothesis count."""
        ...

    def anomalies(self) -> list[dict[str, Any]]:
        """Flagged readings with ranked candidate explanations."""
        ...

    def predictions(self, *, zone: str | None = None) -> list[dict[str, Any]]:
        """Recorded predictions and, where known, their outcomes."""
        ...

    def drift(self) -> dict[str, Any]:
        """Drift events, episodes and candidate decisions together."""
        ...

    def models(self) -> list[dict[str, Any]]:
        """Model versions, oldest first."""
        ...

    def experiments(self) -> list[dict[str, Any]]:
        """Experiment reports, newest first."""
        ...

    def zones(self) -> list[dict[str, Any]]:
        """Zone coordinates, for the map."""
        ...


@dataclass
class ArtifactLoader:
    """Reads the stored artifacts straight off disk.

    A thin adapter over :class:`~urbansense.api.store.ArtifactStore`, which
    already does the reading, the caching and the "what is missing" reporting.
    Re-implementing any of that here would give the dashboard a second opinion
    about the same files.

    Attributes:
        root: Repository root. Every path is derived from it, so a test can
            point the whole loader at a temporary directory.
        store: The underlying artifact store.
    """

    root: Path = Path(".")
    store: ArtifactStore = field(init=False)

    def __post_init__(self) -> None:
        """Build the store under ``root``."""
        self.store = ArtifactStore(root=self.root)

    def availability(self) -> dict[str, bool]:
        """Which artifact families have something to show."""
        return self.store.availability

    def missing_note(self, family: str) -> str | None:
        """Why a family is empty, and what to run."""
        return self.store.missing_note(family)

    def unified(self, *, zone: str | None = None, limit: int = PAGE_LIMIT) -> list[dict[str, Any]]:
        """Unified observations, optionally for one zone."""
        rows = self.store.unified()
        if zone is not None:
            rows = [row for row in rows if row.get("location_id") == zone]
        return rows[:limit]

    def data_quality(self) -> dict[str, Any] | None:
        """Counts and per-source health."""
        return self.store.data_quality()

    def conflicts(self) -> list[dict[str, Any]]:
        """Recorded disagreements between sources."""
        return self.store.conflicts()

    def quarantine(self) -> list[dict[str, Any]]:
        """Records held back, payload as received."""
        return self.store.quarantine()

    def patterns(self) -> list[dict[str, Any]]:
        """Discovered patterns with their evidence."""
        return self.store.patterns()

    def pattern_run(self) -> dict[str, Any] | None:
        """The discovery run header."""
        return self.store.pattern_run()

    def anomalies(self) -> list[dict[str, Any]]:
        """Flagged readings with their candidate explanations."""
        return self.store.anomalies()

    def predictions(self, *, zone: str | None = None) -> list[dict[str, Any]]:
        """Recorded predictions, optionally for one zone."""
        rows = self.store.predictions()
        if zone is not None:
            rows = [row for row in rows if row.get("location_id") == zone]
        return rows

    def drift(self) -> dict[str, Any]:
        """Drift events, episodes and candidates, shaped like the API's payload."""
        return {
            "events": self.store.drift_events(),
            "episodes": self.store.drift_episodes(),
            "candidates": self.store.candidates(),
        }

    def models(self) -> list[dict[str, Any]]:
        """Model versions, oldest first."""
        return self.store.models()

    def experiments(self) -> list[dict[str, Any]]:
        """Experiment reports, newest first."""
        return self.store.experiments()

    def zones(self) -> list[dict[str, Any]]:
        """Zone coordinates from the committed dataset.

        Read from ``data/sample/zones.csv`` rather than from an observation,
        because an observation carries a ``location_id`` and no geometry. Parsed
        with the csv module rather than pandas: four rows do not need a
        dataframe, and keeping pandas out of the loaders is what lets the tests
        run without the dashboard extra installed.
        """
        import csv

        path = self.root / "data" / "sample" / "zones.csv"
        if not path.is_file():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return [_zone_row(row) for row in csv.DictReader(handle)]

    def clear(self) -> None:
        """Drop the cache, so a re-exported artifact is picked up."""
        self.store.clear()


def _zone_row(row: dict[str, str]) -> dict[str, Any]:
    """One zone with its coordinates as floats, or ``None`` when unparseable.

    A zone whose coordinates cannot be read keeps its row and loses its point.
    Dropping the zone entirely would make it vanish from the overview, which is
    a worse failure than a missing marker.
    """
    return {
        "location_id": row.get("location_id", ""),
        "description": row.get("description", ""),
        "latitude": _as_float(row.get("latitude")),
        "longitude": _as_float(row.get("longitude")),
        "sensor_id": row.get("sensor_id", ""),
    }


def _note_of(payload: object) -> str | None:
    """The ``note`` field of an API payload, when it carries a string one."""
    if not isinstance(payload, dict):
        return None
    note = payload.get("note")
    return note if isinstance(note, str) else None


def _as_float(value: str | None) -> float | None:
    """Parse a float, or ``None`` when absent or malformed."""
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass
class ApiLoader:
    """Fetches the same artifacts over HTTP from the read-only API.

    Used by ``docker-compose``, where the dashboard container talks to the API
    container. The point is not that HTTP is better here -- it is slower than
    opening the file -- but that the compose demo should prove the API works
    rather than merely ship it.

    Every request failure degrades to empty data plus a note naming the API, so
    an unreachable server produces a dashboard that says so instead of a page
    of tracebacks.

    Attributes:
        base_url: Where the API lives, e.g. ``http://127.0.0.1:8000``.
        timeout: Seconds to wait per request.
    """

    base_url: str
    timeout: float = HTTP_TIMEOUT
    _reachable: bool | None = field(default=None, repr=False)

    def _get(self, path: str, **params: str | int | None) -> object | None:
        """GET one endpoint and decode its JSON, or ``None`` on any failure.

        Returns ``None`` rather than raising: the dashboard's job is to show
        what it can, and one unavailable endpoint should not take down the page
        that happens to render it first.

        Typed as ``object`` rather than a concrete shape because these payloads
        come from eleven different endpoints and every caller narrows with
        ``isinstance`` anyway. Claiming a type here would be claiming the API's
        schema from the wrong side of the wire.
        """
        query = {key: str(value) for key, value in params.items() if value is not None}
        url = f"{self.base_url.rstrip('/')}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                payload: object = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            self._reachable = False
            return None
        self._reachable = True
        return payload

    def _items(self, path: str, **params: str | int | None) -> list[dict[str, Any]]:
        """The ``items`` of a paged endpoint, or an empty list."""
        payload = self._get(path, **params)
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            return [item for item in payload["items"] if isinstance(item, dict)]
        return []

    def _note(self, path: str, **params: str | int | None) -> str | None:
        """The ``note`` a paged endpoint attached, when it attached one."""
        payload = self._get(path, **params)
        return _note_of(payload)

    @property
    def reachable(self) -> bool | None:
        """Whether the last request succeeded. ``None`` before the first one."""
        return self._reachable

    def availability(self) -> dict[str, bool]:
        """What ``/health`` says is present."""
        payload = self._get("/health")
        if isinstance(payload, dict) and isinstance(payload.get("artifacts"), dict):
            return {str(key): bool(value) for key, value in payload["artifacts"].items()}
        return {}

    def missing_note(self, family: str) -> str | None:
        """The note the matching endpoint attaches when it is empty."""
        routes = {
            "unified": "/data/unified",
            "conflicts": "/conflicts",
            "quarantine": "/quarantine",
            "patterns": "/patterns",
            "anomalies": "/anomalies",
            "predictions": "/predictions",
            "models": "/models",
            "experiments": "/experiments",
        }
        if family == "data_quality":
            return _note_of(self._get("/data-quality"))
        if family == "drift":
            return _note_of(self._get("/drift"))
        route = routes.get(family)
        return self._note(route, limit=1) if route is not None else None

    def unified(self, *, zone: str | None = None, limit: int = PAGE_LIMIT) -> list[dict[str, Any]]:
        """Unified observations, optionally for one zone."""
        return self._items("/data/unified", zone=zone, limit=limit)

    def data_quality(self) -> dict[str, Any] | None:
        """Counts and per-source health."""
        payload = self._get("/data-quality")
        return payload if isinstance(payload, dict) else None

    def conflicts(self) -> list[dict[str, Any]]:
        """Recorded disagreements between sources."""
        return self._items("/conflicts", limit=PAGE_LIMIT)

    def quarantine(self) -> list[dict[str, Any]]:
        """Records held back, payload as received."""
        return self._items("/quarantine", limit=PAGE_LIMIT)

    def patterns(self) -> list[dict[str, Any]]:
        """Discovered patterns with their evidence."""
        return self._items("/patterns", limit=PAGE_LIMIT)

    def pattern_run(self) -> dict[str, Any] | None:
        """The discovery run header, taken from any one pattern's detail route."""
        found = self.patterns()
        if not found:
            return None
        identifier = found[0].get("pattern_id")
        if not isinstance(identifier, str):
            return None
        payload = self._get(f"/patterns/{urllib.parse.quote(identifier)}")
        if not isinstance(payload, dict):
            return None
        run = payload.get("run")
        return dict(run) if isinstance(run, dict) else None

    def anomalies(self) -> list[dict[str, Any]]:
        """Flagged readings with their candidate explanations."""
        return self._items("/anomalies", limit=PAGE_LIMIT)

    def predictions(self, *, zone: str | None = None) -> list[dict[str, Any]]:
        """Recorded predictions, optionally for one zone."""
        return self._items("/predictions", zone=zone, limit=PAGE_LIMIT)

    def drift(self) -> dict[str, Any]:
        """Drift events, episodes and candidate decisions."""
        payload = self._get("/drift")
        if isinstance(payload, dict):
            return payload
        return {"events": [], "episodes": [], "candidates": []}

    def models(self) -> list[dict[str, Any]]:
        """Model versions, oldest first."""
        return self._items("/models", limit=PAGE_LIMIT)

    def experiments(self) -> list[dict[str, Any]]:
        """Experiment reports, newest first."""
        return self._items("/experiments", limit=PAGE_LIMIT)

    def zones(self) -> list[dict[str, Any]]:
        """Zone coordinates, derived from the unified layer's zone ids.

        The API serves no geometry -- it was never asked to -- so over HTTP the
        map degrades to the zones that appear in the data, without points. The
        overview still lists them; it just cannot place them. Stated here rather
        than hidden, because a silently empty map looks like a bug.
        """
        seen: dict[str, dict[str, Any]] = {}
        for row in self.unified(limit=PAGE_LIMIT):
            zone = row.get("location_id")
            if isinstance(zone, str) and zone not in seen:
                seen[zone] = {
                    "location_id": zone,
                    "description": "",
                    "latitude": None,
                    "longitude": None,
                    "sensor_id": str(row.get("source_id", "")),
                }
        return [seen[key] for key in sorted(seen)]


def build_loader(root: Path | None = None, api_url: str | None = None) -> DashboardData:
    """Pick a backend.

    Args:
        root: Repository root for the direct backend.
        api_url: When given, read over HTTP from this API instead of from disk.

    Returns:
        Whichever loader was asked for. Direct is the default because it works
        with nothing else running.
    """
    if api_url:
        return ApiLoader(base_url=api_url)
    return ArtifactLoader(root=root if root is not None else Path("."))
