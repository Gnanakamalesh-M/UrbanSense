"""The FastAPI application (SPEC section 44).

    A read-only API over the data layers, patterns, predictions, anomalies,
    drift events, models and data quality.

Every route is a ``GET`` and every route opens a file. :class:`ArtifactStore`
does the reading and the caching; this module filters, pages and types the
result. The division matters: an endpoint that recomputed features or refitted a
model on request would turn "inspect what the pipeline produced" into "run the
pipeline again, differently", and the numbers served would stop matching the
numbers reported.

Three postures are deliberate and testable:

**Read-only.** Only ``@app.get`` is registered. No write, update or delete route
exists anywhere, so a ``POST`` returns 405 from the router rather than from a
guard that could be forgotten. A test walks ``/openapi.json`` and asserts no
path declares a write method.

**Unauthenticated, and loud about it.** There is no auth layer. Rather than
leave that implicit, ``/health`` returns ``authentication: "none"`` and the
bind hint, and the bundled script binds ``127.0.0.1``. Anyone exposing this
beyond localhost is then doing so knowingly.

**A missing artifact is not a failure.** A fresh clone has no experiment report
and no discovery run. Those endpoints return an empty page carrying a ``note``
with the command that would produce the file, because "nothing has been
computed yet" is a true answer to "list the experiments" and a 500 is not.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, TypeVar

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from urbansense import __version__
from urbansense.api.models import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    AnomalyRecord,
    DataQuality,
    DriftEventRecord,
    DriftSummary,
    ExperimentSummary,
    Health,
    ModelSummary,
    Page,
    PatternDetail,
    PatternSummary,
    PredictionOutcome,
    UnifiedObservation,
)
from urbansense.api.store import ArtifactStore, parse_time

#: Where :mod:`scripts.serve_api` binds. Reported by ``/health``.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

ModelT = TypeVar("ModelT", bound=BaseModel)

LimitQuery = Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)]
OffsetQuery = Annotated[int, Query(ge=0)]


def _page(
    rows: list[dict[str, Any]],
    model: type[ModelT],
    *,
    limit: int,
    offset: int,
    note: str | None = None,
) -> Page[ModelT]:
    """Page and validate a slice of rows.

    ``total`` counts the rows that matched **before** paging, so a client can
    distinguish "that is everything" from "there are more behind this page".
    """
    window = rows[offset : offset + limit]
    return Page[model](  # type: ignore[valid-type]
        items=[model.model_validate(row) for row in window],
        total=len(rows),
        limit=limit,
        offset=offset,
        note=note if not rows else None,
    )


def _in_window(
    row: dict[str, Any],
    field: str,
    start: datetime | None,
    end: datetime | None,
) -> bool:
    """Whether a row's timestamp falls in ``[start, end)``.

    A row whose timestamp is absent or unparseable is **excluded** when a time
    filter is given, rather than passed through. A caller asking for a window
    has asked a question about time, and a row that cannot answer it should not
    be silently counted as matching.
    """
    if start is None and end is None:
        return True
    raw = row.get(field)
    if not isinstance(raw, str):
        return False
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if start is not None and moment < start:
        return False
    return not (end is not None and moment >= end)


def _not_found(what: str, identifier: str, searched: str) -> HTTPException:
    """A 404 that names the id and what was searched.

    "unknown pattern 'PAT-9'; 41 pattern(s) in the latest discovery report" and
    "no discovery report has been written yet" are different problems with
    different fixes, so the detail distinguishes them instead of saying only
    "not found".
    """
    return HTTPException(status_code=404, detail=f"unknown {what} '{identifier}'; {searched}")


def create_app(store: ArtifactStore | None = None) -> FastAPI:
    """Build the application.

    Args:
        store: Where to read artifacts from. Defaults to the repository root;
            a test passes a store rooted at a temporary directory.

    Returns:
        The configured app. Routes are registered inside the factory so each
        app instance closes over its own store -- otherwise tests would share
        one cache and one root.
    """
    artifacts = store if store is not None else ArtifactStore()

    app = FastAPI(
        title="UrbanSense API",
        version=__version__,
        summary="Read-only access to the stored data layers, patterns, predictions and models.",
        description=(
            "Every endpoint reads an artifact the pipeline already wrote; nothing is "
            "recomputed per request. All routes are GET: there are no write endpoints. "
            "There is no authentication, so the bundled script binds 127.0.0.1 only. "
            "Endpoints whose artifact has not been produced yet return an empty page "
            "with a note naming the command that would produce it, not an error."
        ),
    )

    # Closed over rather than injected with ``Depends``. A dependency alias
    # defined in here is invisible to FastAPI: under ``from __future__ import
    # annotations`` it resolves route annotations as strings against module
    # globals, so a local ``Store`` alias is silently read as a required query
    # parameter and every endpoint answers 422. Closing over the store keeps
    # one store per app, which is what the factory exists for.
    store = artifacts

    # ------------------------------------------------------------------
    # Service
    # ------------------------------------------------------------------

    @app.get("/health", response_model=Health, tags=["service"])
    def health() -> Health:
        """Liveness, posture, and which artifacts are actually present."""
        return Health(
            status="ok",
            version=__version__,
            read_only=True,
            authentication="none",
            bind_hint=f"{DEFAULT_HOST}:{DEFAULT_PORT} (localhost only; no authentication)",
            artifacts=store.availability,
        )

    # ------------------------------------------------------------------
    # Data layers
    # ------------------------------------------------------------------

    @app.get("/data/unified", response_model=Page[UnifiedObservation], tags=["data"])
    def unified(
        zone: str | None = None,
        metric: str | None = None,
        from_: Annotated[str | None, Query(alias="from")] = None,
        to: str | None = None,
        limit: LimitQuery = DEFAULT_PAGE_LIMIT,
        offset: OffsetQuery = 0,
    ) -> Page[UnifiedObservation]:
        """Unified observations, each with the provenance of its value.

        Filter by zone, metric and event-time window. Every item carries
        ``source_id``, ``source_type``, the three timestamps and the
        ``provenance`` block, so "where did this value come from?" is
        answerable from one response.
        """
        try:
            start, end = parse_time(from_), parse_time(to)
        except ValueError as error:
            raise HTTPException(
                status_code=422, detail=f"from/to must be ISO-8601 instants: {error}"
            ) from error
        rows = [
            row
            for row in store.unified()
            if (zone is None or row.get("location_id") == zone)
            and (metric is None or row.get("metric") == metric)
            and _in_window(row, "event_time", start, end)
        ]
        return _page(
            rows,
            UnifiedObservation,
            limit=limit,
            offset=offset,
            note=store.missing_note("unified"),
        )

    @app.get("/conflicts", response_model=Page[dict[str, Any]], tags=["data"])
    def conflicts(
        limit: LimitQuery = DEFAULT_PAGE_LIMIT,
        offset: OffsetQuery = 0,
    ) -> Page[dict[str, Any]]:
        """Recorded disagreements between sources.

        Conflicts are never resolved by overwriting: each group keeps every
        candidate value and who claimed it, and the disputed slot stays absent
        from the unified layer rather than being guessed.
        """
        rows = store.conflicts()
        window = rows[offset : offset + limit]
        return Page[dict[str, Any]](
            items=window,
            total=len(rows),
            limit=limit,
            offset=offset,
            note=store.missing_note("conflicts") if not rows else None,
        )

    @app.get("/quarantine", response_model=Page[dict[str, Any]], tags=["data"])
    def quarantine(
        reason: str | None = None,
        source_id: str | None = None,
        limit: LimitQuery = DEFAULT_PAGE_LIMIT,
        offset: OffsetQuery = 0,
    ) -> Page[dict[str, Any]]:
        """Records held back from the dataset, payload as received.

        Nothing is dropped: a quarantined record keeps its raw payload, the
        detector that caught it and the reason, so a rejection can be audited
        or reversed.
        """
        rows = [
            row
            for row in store.quarantine()
            if (reason is None or row.get("reason") == reason)
            and (source_id is None or row.get("source_id") == source_id)
        ]
        window = rows[offset : offset + limit]
        return Page[dict[str, Any]](
            items=window,
            total=len(rows),
            limit=limit,
            offset=offset,
            note=store.missing_note("quarantine") if not rows else None,
        )

    @app.get("/data-quality", response_model=DataQuality, tags=["data"])
    def data_quality() -> DataQuality:
        """Counts and per-source health.

        ``missing_values`` is reported separately from ``valid`` because a
        usable observation may carry no measurement -- and it stays missing
        rather than becoming a zero.
        """
        payload = store.data_quality()
        if payload is None:
            return DataQuality(note=store.missing_note("data_quality"))
        return DataQuality.model_validate(payload)

    # ------------------------------------------------------------------
    # Patterns and anomalies
    # ------------------------------------------------------------------

    @app.get("/patterns", response_model=Page[PatternSummary], tags=["discovery"])
    def patterns(
        kind: str | None = None,
        status: str | None = None,
        limit: LimitQuery = DEFAULT_PAGE_LIMIT,
        offset: OffsetQuery = 0,
    ) -> Page[PatternSummary]:
        """Discovered patterns, each with the evidence behind the claim."""
        rows = [
            row
            for row in store.patterns()
            if (kind is None or row.get("kind") == kind)
            and (status is None or row.get("status") == status)
        ]
        return _page(
            rows, PatternSummary, limit=limit, offset=offset, note=store.missing_note("patterns")
        )

    @app.get("/patterns/{pattern_id}", response_model=PatternDetail, tags=["discovery"])
    def pattern(pattern_id: str) -> PatternDetail:
        """One pattern, with the discovery run that found it.

        The run header travels with the pattern because a finding cannot be
        weighed without the number of hypotheses tested alongside it.
        """
        found = store.patterns()
        for row in found:
            if row.get("pattern_id") == pattern_id:
                return PatternDetail(
                    pattern=PatternSummary.model_validate(row), run=store.pattern_run()
                )
        searched = (
            f"{len(found)} pattern(s) in the latest discovery report"
            if found
            else str(store.missing_note("patterns"))
        )
        raise _not_found("pattern", pattern_id, searched)

    @app.get("/anomalies", response_model=Page[AnomalyRecord], tags=["discovery"])
    def anomalies(
        zone: str | None = None,
        limit: LimitQuery = DEFAULT_PAGE_LIMIT,
        offset: OffsetQuery = 0,
    ) -> Page[AnomalyRecord]:
        """Readings that deviate from their own cell's history.

        An anomaly is flagged, not judged. ``is_error`` is always null: the
        system cannot tell a sensor fault from a festival, and says so rather
        than deciding.
        """
        rows = [row for row in store.anomalies() if zone is None or row.get("location_id") == zone]
        return _page(
            rows, AnomalyRecord, limit=limit, offset=offset, note=store.missing_note("anomalies")
        )

    # ------------------------------------------------------------------
    # Predictions, drift, models, experiments
    # ------------------------------------------------------------------

    @app.get("/predictions", response_model=Page[PredictionOutcome], tags=["prediction"])
    def predictions(
        zone: str | None = None,
        date: str | None = None,
        model_version: str | None = None,
        limit: LimitQuery = DEFAULT_PAGE_LIMIT,
        offset: OffsetQuery = 0,
    ) -> Page[PredictionOutcome]:
        """Recorded predictions and, where known, what actually happened.

        These are logged outcomes, not fresh inference: the API serves what was
        predicted at the time, which is the only version of a prediction that
        can be scored honestly.

        Filter by ``zone``, by ``model_version``, and by ``date`` -- one target
        date as ``YYYY-MM-DD``.
        """
        if date is not None:
            try:
                datetime.fromisoformat(date)
            except ValueError as error:
                raise HTTPException(
                    status_code=422, detail=f"date must be YYYY-MM-DD: {error}"
                ) from error
        rows = [
            row
            for row in store.predictions()
            if (zone is None or row.get("location_id") == zone)
            and (model_version is None or row.get("model_version") == model_version)
            and (date is None or str(row.get("target_time", "")).startswith(date))
        ]
        return _page(
            rows,
            PredictionOutcome,
            limit=limit,
            offset=offset,
            note=store.missing_note("predictions"),
        )

    @app.get("/drift", response_model=DriftSummary, tags=["adaptation"])
    def drift(firing_only: bool = False) -> DriftSummary:
        """Drift evaluations, episodes and the candidate decisions they raised.

        Quiet evaluations are kept by default. A detector that did not fire is
        evidence too: the false-alarm rate cannot be read from firings alone.
        """
        events = store.drift_events()
        shown = [row for row in events if row.get("is_drift")] if firing_only else events
        return DriftSummary(
            events=[DriftEventRecord.model_validate(row) for row in shown],
            episodes=store.drift_episodes(),
            candidates=store.candidates(),
            note=store.missing_note("drift") if not events else None,
        )

    @app.get("/models", response_model=Page[ModelSummary], tags=["models"])
    def models(
        status: str | None = None,
        limit: LimitQuery = DEFAULT_PAGE_LIMIT,
        offset: OffsetQuery = 0,
    ) -> Page[ModelSummary]:
        """Model versions, oldest first. Exactly one carries champion status."""
        rows = [row for row in store.models() if status is None or row.get("status") == status]
        return _page(
            rows, ModelSummary, limit=limit, offset=offset, note=store.missing_note("models")
        )

    @app.get("/models/{version}", response_model=dict[str, Any], tags=["models"])
    def model(version: str) -> dict[str, Any]:
        """One model version in full.

        The whole registry entry is returned rather than a summary: the
        training and test periods, the gap, the leakage checks, the excluded-row
        accounting and the metrics are what make a reported score checkable.
        """
        found = store.models()
        for row in found:
            if row.get("version") == version:
                return row
        searched = (
            f"{len(found)} version(s) in the registry: "
            + ", ".join(str(row.get("version")) for row in found)
            if found
            else str(store.missing_note("models"))
        )
        raise _not_found("model version", version, searched)

    @app.get("/experiments", response_model=Page[ExperimentSummary], tags=["experiments"])
    def experiments(
        limit: LimitQuery = DEFAULT_PAGE_LIMIT,
        offset: OffsetQuery = 0,
    ) -> Page[ExperimentSummary]:
        """Experiment reports, newest first.

        Each carries its pre-registered configuration beside its verdict, since
        a result without the parameters fixed in advance is not evidence of
        much.
        """
        return _page(
            store.experiments(),
            ExperimentSummary,
            limit=limit,
            offset=offset,
            note=store.missing_note("experiments"),
        )

    return app
