"""The read-only API (SPEC section 44).

Every test here drives the app through ``TestClient``, against a store rooted in
a temporary directory holding hand-written artifacts. Nothing runs the pipeline:
these tests are about what the HTTP layer does with files, and a fixture that
took thirty seconds to build would discourage adding cases.

Four properties carry the weight:

**Read-only, provably.** The OpenAPI schema is walked and asserted to declare no
``post``, ``put``, ``patch`` or ``delete`` on any path, and a ``POST`` is checked
to return 405. A guard that could be forgotten on one route would not survive
either assertion.

**Provenance survives the serialization.** ``/data/unified`` items must carry
``source_id``, ``source_type``, the three timestamps and the ``provenance``
block. The project exists to answer "where did this value come from?", and a
response that dropped it on the way out would defeat that at the last step.

**A missing artifact is an empty page with a note, not a 500.** The fixtures
include an *empty* store for exactly this, because a fresh clone is the common
case and a stack trace is the wrong answer to "no experiment has been run yet".

**Pagination is bounded.** An over-large ``limit`` must be rejected rather than
quietly serving tens of thousands of rows.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from urbansense.api.app import create_app
from urbansense.api.models import MAX_PAGE_LIMIT
from urbansense.api.store import ArtifactStore

#: Every documented endpoint, as a GET with no required parameters. Listed once
#: so a new route cannot be added without appearing in the schema and
#: empty-store tests below.
ENDPOINTS = (
    "/health",
    "/data/unified",
    "/conflicts",
    "/quarantine",
    "/data-quality",
    "/patterns",
    "/predictions",
    "/anomalies",
    "/drift",
    "/models",
    "/experiments",
)

#: Endpoints returning a paged envelope, as opposed to a single object.
PAGED = (
    "/data/unified",
    "/conflicts",
    "/quarantine",
    "/patterns",
    "/predictions",
    "/anomalies",
    "/models",
    "/experiments",
)


def _observation(
    record_id: str,
    *,
    zone: str = "ZONE-A",
    event_time: str = "2026-01-15T08:30:00+00:00",
    value: float | None = 4_200.0,
    metric: str = "traffic_volume",
) -> dict[str, Any]:
    """One unified row, shaped exactly as the exporter writes it."""
    return {
        "observation_id": f"OBS-{record_id}",
        "record_id": record_id,
        "source_id": "sensor-A12",
        "source_type": "historical",
        "problem_type": "traffic",
        "location_id": zone,
        "metric": metric,
        "value": value,
        "unit": "vehicles/hour",
        "event_time": event_time,
        "source_time": event_time,
        "received_time": "2026-10-04T10:00:00+00:00",
        "aggregation_level": "hourly",
        "validation_status": "valid",
        "is_imputed": False,
        "duplicate_of": None,
        "conflicts_with": [],
        "provenance": {
            "kind": "direct",
            "origin": "sensor-A12",
            "ingested_at": "2026-10-04T10:00:00Z",
            "pipeline_version": "ingestion-1.0.0",
            "upstream_observation_ids": [],
        },
    }


@pytest.fixture
def populated(tmp_path: Path) -> ArtifactStore:
    """A store with one artifact of every family written by hand.

    Hand-written rather than generated so a test can assert on specific ids and
    so the fixture stays readable: these tests check the HTTP layer, and the
    pipeline that produces the real files has its own suite.
    """
    export = tmp_path / "data" / "processed"
    export.mkdir(parents=True)
    rows = [
        _observation("REC-001"),
        _observation("REC-002", event_time="2026-01-16T09:30:00+00:00"),
        _observation("REC-003", zone="ZONE-B", event_time="2026-02-01T18:30:00+00:00"),
        # A usable observation carrying no measurement. Missing stays missing:
        # the API must serve null, never 0.
        _observation(
            "REC-004",
            zone="ZONE-B",
            event_time="2026-02-02T08:30:00+00:00",
            value=None,
            metric="average_speed",
        ),
    ]
    (export / "unified.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (export / "conflicts.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "location_id": "ZONE-C",
                        "metric": "traffic_volume",
                        "event_time": "2026-01-20T12:30:00+00:00",
                        "candidates": [
                            {"source_id": "sensor-C1", "value": 3_000.0},
                            {"source_id": "sensor-C2", "value": 5_400.0},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (export / "data_quality.json").write_text(
        json.dumps(
            {
                "rows_in": 100,
                "valid": 96,
                "quarantined": 4,
                "missing_values": 7,
                "unified_rows": 94,
                "quarantine_by_reason": {"out_of_range": 4},
                "source_health": {"sensor-A12": {"status": "healthy", "observations": 96}},
            }
        ),
        encoding="utf-8",
    )

    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    (quarantine / "validation-20261004T100000Z.json").write_text(
        json.dumps(
            [
                {
                    "quarantine_id": "Q-1",
                    "source_id": "sensor-A12",
                    "reason": "out_of_range",
                    "detector": "range",
                    "raw_payload": {"traffic_volume": "-1648"},
                    "resolved": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    patterns = tmp_path / "reports" / "patterns"
    patterns.mkdir(parents=True)
    (patterns / "patterns-20261004T100000Z.json").write_text(
        json.dumps(
            {
                "run": {"tests_run": 672, "fdr_level": 0.1, "discoveries": 2},
                "patterns": [
                    {
                        "pattern_id": "PAT-001",
                        "kind": "recurring_peak",
                        "description": "ZONE-A peaks on Friday evenings",
                        "status": "stable",
                        "confidence": 0.99,
                        "scope": {"location_id": "ZONE-A"},
                        "evidence": {"n_treatment": 120},
                    },
                    {
                        "pattern_id": "PAT-002",
                        "kind": "correlation",
                        "description": "rainfall is associated with higher volume",
                        "status": "weakening",
                        "confidence": 0.95,
                        "scope": {},
                        "evidence": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    anomalies = tmp_path / "reports" / "anomalies"
    anomalies.mkdir(parents=True)
    (anomalies / "anomalies-20261004T100000Z.json").write_text(
        json.dumps(
            {
                "run": {"observations_scanned": 1_000, "flagged": 1},
                "anomalies": [
                    {
                        "record_id": "REC-900",
                        "location_id": "ZONE-B",
                        "metric": "traffic_volume",
                        "event_time": "2026-01-22T14:30:00+00:00",
                        "value": -1648.0,
                        "expected": 3_425.5,
                        "deviation_mad_sigma": 8.73,
                        "status": "suspect",
                        "is_error": None,
                        "explanations": [{"kind": "sensor_suspect", "support": 0.75}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    outcomes = tmp_path / "reports" / "outcomes"
    outcomes.mkdir(parents=True)
    (outcomes / "outcomes.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "prediction_id": "Traffic-v1.1:ZONE-A:2026-05-14T02:30:00+00:00",
                    "model_version": "Traffic-v1.1",
                    "location_id": "ZONE-A",
                    "target_time": "2026-05-14T02:30:00+00:00",
                    "predicted": 8_316.7,
                    "actual": 8_264.0,
                    "error": 52.7,
                    "probability": 0.99,
                    "severity": "high",
                    "congested_predicted": True,
                    "congested_actual": True,
                    "conditions": {"rain_band": "dry", "local_hour": 8},
                },
                {
                    "prediction_id": "Traffic-v1.1:ZONE-B:2026-05-15T10:30:00+00:00",
                    "model_version": "Traffic-v1.1",
                    "location_id": "ZONE-B",
                    "target_time": "2026-05-15T10:30:00+00:00",
                    "predicted": 2_100.0,
                    "actual": None,
                    "error": None,
                    "probability": 0.12,
                    "severity": "none",
                    "congested_predicted": False,
                    "congested_actual": None,
                    "conditions": {},
                },
            )
        ),
        encoding="utf-8",
    )

    drift = tmp_path / "reports" / "drift"
    drift.mkdir(parents=True)
    (drift / "events.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "detected_at": "2026-02-12T17:30:00+00:00",
                    "is_drift": False,
                    "kinds": [],
                    "episode_id": None,
                    "magnitude": 0.4,
                    "signals": [{"family": "feature", "fired": False}],
                },
                {
                    "detected_at": "2026-03-05T17:30:00+00:00",
                    "is_drift": True,
                    "kinds": ["performance"],
                    "episode_id": "EP-1",
                    "magnitude": 2.1,
                    "signals": [{"family": "performance", "fired": True}],
                },
            )
        ),
        encoding="utf-8",
    )
    (drift / "episodes.jsonl").write_text(
        json.dumps({"episode_id": "EP-1", "events": 1}) + "\n", encoding="utf-8"
    )

    adaptation = tmp_path / "reports" / "adaptation"
    adaptation.mkdir(parents=True)
    (adaptation / "candidates.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "CAND-1",
                "status": "promoted",
                "promoted_version": "Traffic-v1.2",
                "episode_id": "EP-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    registry = tmp_path / "models" / "registry"
    registry.mkdir(parents=True)
    (registry / "Traffic-v1.0.json").write_text(
        json.dumps(
            {
                "version": "Traffic-v1.0",
                "status": "archived",
                "problem_type": "traffic",
                "target": "traffic_volume",
                "horizon_hours": 1,
                "algorithm": "HistGradientBoostingRegressor",
                "created_at": "2026-10-02T12:00:00+00:00",
                "leakage_checks": {"probe": "clean"},
            }
        ),
        encoding="utf-8",
    )
    (registry / "Traffic-v1.1.json").write_text(
        json.dumps(
            {
                "version": "Traffic-v1.1",
                "status": "champion",
                "problem_type": "traffic",
                "target": "traffic_volume",
                "horizon_hours": 1,
                "algorithm": "HistGradientBoostingRegressor",
                "created_at": "2026-10-03T12:00:00+00:00",
                "training_period": {"start": "2025-10-01", "end": "2025-12-14"},
                "gap_hours": 168,
            }
        ),
        encoding="utf-8",
    )

    experiments = tmp_path / "reports" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "experiment-20261004T100000Z.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-10-04T10:00:00+00:00",
                "pre_registered_config": {"seeds": [1, 2], "retrain_every_days": 28},
                "verdict": {"best": "scheduled", "ties": [["scheduled", "adaptive"]]},
                "arms": {"scheduled": {"mae": {"mean": 230.7}}},
                "limitations": ["Synthetic data."],
            }
        ),
        encoding="utf-8",
    )

    return ArtifactStore(root=tmp_path)


@pytest.fixture
def client(populated: ArtifactStore) -> Iterator[TestClient]:
    """A client over the populated store."""
    with TestClient(create_app(populated)) as test_client:
        yield test_client


@pytest.fixture
def bare(tmp_path: Path) -> Iterator[TestClient]:
    """A client over an empty directory -- the fresh-clone case."""
    with TestClient(create_app(ArtifactStore(root=tmp_path / "empty"))) as test_client:
        yield test_client


class TestEveryEndpointAnswers:
    """Each documented endpoint returns 200 and its declared shape."""

    @pytest.mark.parametrize("path", ENDPOINTS)
    def test_the_endpoint_returns_200(self, client: TestClient, path: str) -> None:
        """Nothing documented is missing or broken."""
        assert client.get(path).status_code == 200

    @pytest.mark.parametrize("path", PAGED)
    def test_a_paged_endpoint_returns_the_envelope(self, client: TestClient, path: str) -> None:
        """items, total, limit and offset, every time.

        ``total`` is the count before paging, which is what lets a client tell
        "that is all of them" from "there are more".
        """
        payload = client.get(path).json()
        assert set(payload) >= {"items", "total", "limit", "offset"}
        assert isinstance(payload["items"], list)
        assert payload["total"] >= len(payload["items"])

    def test_health_reports_its_posture_and_what_it_can_serve(self, client: TestClient) -> None:
        """Read-only and unauthenticated are stated in the payload, not just a README."""
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["read_only"] is True
        assert payload["authentication"] == "none"
        assert "127.0.0.1" in payload["bind_hint"]
        assert payload["artifacts"]["unified"] is True
        assert payload["artifacts"]["patterns"] is True

    def test_data_quality_counts_missing_values_separately(self, client: TestClient) -> None:
        """96 valid rows means something different when 7 carry no measurement."""
        payload = client.get("/data-quality").json()
        assert payload["valid"] == 96
        assert payload["missing_values"] == 7
        assert payload["quarantined"] == 4
        assert payload["source_health"]["sensor-A12"]["status"] == "healthy"

    def test_drift_keeps_the_quiet_evaluations(self, client: TestClient) -> None:
        """A detector that did not fire is evidence too.

        The false-alarm rate cannot be computed from firings alone, so the
        default response carries both and ``firing_only`` is opt-in.
        """
        payload = client.get("/drift").json()
        assert len(payload["events"]) == 2
        assert [item["is_drift"] for item in payload["events"]] == [False, True]
        assert payload["episodes"] and payload["candidates"]

        firing = client.get("/drift?firing_only=true").json()
        assert len(firing["events"]) == 1
        assert firing["events"][0]["kinds"] == ["performance"]

    def test_an_anomaly_is_flagged_not_judged(self, client: TestClient) -> None:
        """is_error stays null and the explanations stay plural."""
        item = client.get("/anomalies").json()["items"][0]
        assert item["is_error"] is None
        assert item["explanations"]
        assert item["deviation_mad_sigma"] == pytest.approx(8.73)

    def test_a_missing_actual_is_served_as_null(self, client: TestClient) -> None:
        """An unresolved prediction has no error, and that is not a zero."""
        rows = client.get("/predictions").json()["items"]
        pending = next(row for row in rows if row["location_id"] == "ZONE-B")
        assert pending["actual"] is None
        assert pending["error"] is None
        assert pending["congested_actual"] is None

    def test_an_experiment_carries_its_pre_registration(self, client: TestClient) -> None:
        """A result without the parameters fixed in advance is not evidence."""
        item = client.get("/experiments").json()["items"][0]
        assert item["pre_registered_config"]["retrain_every_days"] == 28
        assert item["verdict"]["best"] == "scheduled"
        assert item["report_file"].startswith("experiment-")

    def test_a_conflict_keeps_every_candidate_value(self, client: TestClient) -> None:
        """Conflicts are recorded, never resolved by overwriting."""
        group = client.get("/conflicts").json()["items"][0]
        assert len(group["candidates"]) == 2
        assert {item["source_id"] for item in group["candidates"]} == {"sensor-C1", "sensor-C2"}

    def test_a_quarantined_record_keeps_its_raw_payload(self, client: TestClient) -> None:
        """A rejection has to be auditable, so the payload is kept as received."""
        item = client.get("/quarantine").json()["items"][0]
        assert item["raw_payload"] == {"traffic_volume": "-1648"}
        assert item["reason"] == "out_of_range"

    def test_the_openapi_document_and_docs_page_are_served(self, client: TestClient) -> None:
        """Auto docs, so the schema is readable without issuing requests."""
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200


class TestProvenance:
    """Asking where a value came from must be answerable from one response."""

    def test_a_unified_item_carries_its_source_references(self, client: TestClient) -> None:
        """Source, stream, the three timestamps and the provenance block."""
        item = client.get("/data/unified?limit=1").json()["items"][0]
        assert item["source_id"] == "sensor-A12"
        assert item["source_type"] == "historical"
        assert item["event_time"] and item["source_time"] and item["received_time"]
        assert item["provenance"]["origin"] == "sensor-A12"
        assert item["provenance"]["pipeline_version"] == "ingestion-1.0.0"
        assert item["provenance"]["kind"] == "direct", "extra provenance fields must survive"

    def test_every_unified_item_has_provenance(self, client: TestClient) -> None:
        """Not a sample of them. A row with no origin should never have been served."""
        payload = client.get("/data/unified").json()
        assert len(payload["items"]) == payload["total"] > 0
        for item in payload["items"]:
            assert item["provenance"], f"{item['record_id']} was served without provenance"

    def test_a_missing_value_is_null_and_not_imputed(self, client: TestClient) -> None:
        """Missing stays missing: null, with is_imputed false."""
        rows = client.get("/data/unified?metric=average_speed").json()["items"]
        assert rows and all(row["value"] is None for row in rows)
        assert all(row["is_imputed"] is False for row in rows)


class TestFiltering:
    """The filters narrow what is served without changing what it means."""

    def test_filtering_by_zone(self, client: TestClient) -> None:
        """Zone filter, and the total reflects the filter rather than the file."""
        payload = client.get("/data/unified?zone=ZONE-B").json()
        assert payload["total"] == 2
        assert {item["location_id"] for item in payload["items"]} == {"ZONE-B"}

    def test_filtering_by_event_time_window(self, client: TestClient) -> None:
        """Half-open window: the start is included, the end is not."""
        payload = client.get(
            "/data/unified?from=2026-01-15T00:00:00%2B00:00&to=2026-01-16T00:00:00%2B00:00"
        ).json()
        assert payload["total"] == 1
        assert payload["items"][0]["record_id"] == "REC-001"

    def test_a_malformed_timestamp_is_rejected_not_ignored(self, client: TestClient) -> None:
        """Silently ignoring an unparseable filter would answer a different question."""
        response = client.get("/data/unified?from=last-tuesday")
        assert response.status_code == 422
        assert "ISO-8601" in response.json()["detail"]

    def test_filtering_predictions_by_zone_and_date(self, client: TestClient) -> None:
        """Zone and target date, the two things a reader asks for."""
        assert client.get("/predictions?zone=ZONE-A").json()["total"] == 1
        assert client.get("/predictions?date=2026-05-15").json()["total"] == 1
        assert (
            client.get("/predictions?date=2026-05-14").json()["items"][0]["location_id"] == "ZONE-A"
        )

    def test_a_malformed_date_is_rejected(self, client: TestClient) -> None:
        """A typo should not quietly return everything."""
        assert client.get("/predictions?date=14-05-2026").status_code == 422

    def test_filtering_models_by_status(self, client: TestClient) -> None:
        """Exactly one champion, which is the registry's own invariant."""
        champions = client.get("/models?status=champion").json()
        assert champions["total"] == 1
        assert champions["items"][0]["version"] == "Traffic-v1.1"

    def test_filtering_patterns_by_kind(self, client: TestClient) -> None:
        """Kind filter."""
        payload = client.get("/patterns?kind=correlation").json()
        assert payload["total"] == 1
        assert payload["items"][0]["pattern_id"] == "PAT-002"

    def test_a_filter_matching_nothing_is_an_empty_page_not_a_404(self, client: TestClient) -> None:
        """No rows matching is a successful answer to a question about rows."""
        payload = client.get("/data/unified?zone=ZONE-Z").json()
        assert payload["total"] == 0
        assert payload["items"] == []


class TestDetailRoutes:
    """Single-item routes, and the 404s that name what was looked for."""

    def test_a_pattern_is_served_with_its_discovery_run(self, client: TestClient) -> None:
        """One finding out of 672 tests is a different claim from one out of ten."""
        payload = client.get("/patterns/PAT-001").json()
        assert payload["pattern"]["pattern_id"] == "PAT-001"
        assert payload["run"]["tests_run"] == 672
        assert payload["run"]["fdr_level"] == pytest.approx(0.1)

    def test_a_model_version_is_served_in_full(self, client: TestClient) -> None:
        """The whole registry entry: periods, gap and checks are what make a score checkable."""
        payload = client.get("/models/Traffic-v1.1").json()
        assert payload["status"] == "champion"
        assert payload["gap_hours"] == 168
        assert payload["training_period"]["end"] == "2025-12-14"

    def test_an_unknown_pattern_is_a_404_naming_the_id(self, client: TestClient) -> None:
        """The detail distinguishes "no such pattern" from "no report yet"."""
        response = client.get("/patterns/PAT-999")
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert "PAT-999" in detail
        assert "2 pattern(s)" in detail

    def test_an_unknown_model_is_a_404_listing_what_exists(self, client: TestClient) -> None:
        """Naming the available versions turns a 404 into a usable answer."""
        response = client.get("/models/Traffic-v9.9")
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert "Traffic-v9.9" in detail
        assert "Traffic-v1.1" in detail

    def test_a_404_on_an_empty_store_says_nothing_has_been_written(self, bare: TestClient) -> None:
        """A different problem with a different fix, so a different message."""
        detail = bare.get("/patterns/PAT-001").json()["detail"]
        assert "discover_patterns" in detail


class TestPagination:
    """Bounded, and honest about what it did not return."""

    @pytest.mark.parametrize("path", PAGED)
    def test_a_limit_above_the_maximum_is_rejected(self, client: TestClient, path: str) -> None:
        """422, not a silent full scan.

        The real unified export is tens of thousands of rows; serving all of
        them because a client asked for a million would be the server's fault,
        not the client's.
        """
        assert client.get(f"{path}?limit={MAX_PAGE_LIMIT + 1}").status_code == 422

    @pytest.mark.parametrize("path", PAGED)
    def test_a_nonpositive_limit_is_rejected(self, client: TestClient, path: str) -> None:
        """A page of zero rows is a mistake, not a request."""
        assert client.get(f"{path}?limit=0").status_code == 422

    def test_a_negative_offset_is_rejected(self, client: TestClient) -> None:
        """Paging backwards past the start is a mistake too."""
        assert client.get("/data/unified?offset=-1").status_code == 422

    def test_the_limit_bounds_the_page_and_the_total_does_not_move(
        self, client: TestClient
    ) -> None:
        """Total is the matching count, not the page size."""
        payload = client.get("/data/unified?limit=2").json()
        assert len(payload["items"]) == 2
        assert payload["total"] == 4
        assert payload["limit"] == 2

    def test_paging_through_covers_every_row_exactly_once(self, client: TestClient) -> None:
        """Offsets partition the result rather than overlapping it."""
        first = client.get("/data/unified?limit=2&offset=0").json()["items"]
        second = client.get("/data/unified?limit=2&offset=2").json()["items"]
        ids = [item["record_id"] for item in first + second]
        assert ids == ["REC-001", "REC-002", "REC-003", "REC-004"]

    def test_an_offset_past_the_end_is_an_empty_page_with_the_real_total(
        self, client: TestClient
    ) -> None:
        """Running off the end is not an error, and the total still tells the truth."""
        payload = client.get("/data/unified?offset=1000").json()
        assert payload["items"] == []
        assert payload["total"] == 4
        assert payload["offset"] == 1000


class TestMissingArtifactsAreNotFailures:
    """A fresh clone has computed nothing, and that is a normal state."""

    @pytest.mark.parametrize("path", ENDPOINTS)
    def test_every_endpoint_still_answers_200(self, bare: TestClient, path: str) -> None:
        """No artifact, no stack trace."""
        assert bare.get(path).status_code == 200

    @pytest.mark.parametrize("path", PAGED)
    def test_an_empty_page_carries_the_command_that_would_fill_it(
        self, bare: TestClient, path: str
    ) -> None:
        """Nothing run yet stays distinguishable from nothing found."""
        payload = bare.get(path).json()
        assert payload["items"] == []
        assert payload["total"] == 0
        assert payload["note"], f"{path} gave no hint about why it is empty"
        assert "run:" in payload["note"]

    def test_health_reports_nothing_available(self, bare: TestClient) -> None:
        """So an empty endpoint can be told from a broken one."""
        artifacts = bare.get("/health").json()["artifacts"]
        assert artifacts and not any(artifacts.values())

    def test_data_quality_says_why_it_has_no_counts(self, bare: TestClient) -> None:
        """Nulls plus a note, rather than zeros that would read as measurements."""
        payload = bare.get("/data-quality").json()
        assert payload["valid"] is None
        assert payload["note"]

    def test_drift_says_why_it_has_no_events(self, bare: TestClient) -> None:
        """An empty drift log is not the same as a quiet one."""
        payload = bare.get("/drift").json()
        assert payload["events"] == []
        assert payload["note"]

    def test_a_populated_endpoint_carries_no_note(self, client: TestClient) -> None:
        """The note is for absence only; it must not become decoration."""
        assert client.get("/patterns").json()["note"] is None


class TestReadOnly:
    """No write endpoint exists, and the schema proves it."""

    def test_the_schema_declares_no_write_methods(self, client: TestClient) -> None:
        """Walked rather than spot-checked, so a new route cannot slip one in.

        This is the assertion that makes "read-only" a property of the
        application instead of a claim in its README.
        """
        schema = client.get("/openapi.json").json()
        assert schema["paths"], "the schema declared no paths at all"
        for path, operations in schema["paths"].items():
            declared = set(operations)
            assert declared <= {"get"}, f"{path} declares {declared - {'get'}}"

    @pytest.mark.parametrize("path", ENDPOINTS)
    def test_a_post_is_rejected(self, client: TestClient, path: str) -> None:
        """405 from the router, not from a guard that could be forgotten."""
        assert client.post(path, json={}).status_code == 405

    @pytest.mark.parametrize("method", ("put", "patch", "delete"))
    def test_the_other_write_methods_are_rejected(self, client: TestClient, method: str) -> None:
        """Nothing mutates anything, by any verb."""
        assert getattr(client, method)("/data/unified").status_code == 405

    def test_a_request_does_not_create_or_modify_any_file(
        self, populated: ArtifactStore, client: TestClient
    ) -> None:
        """The API reads; it does not write, and it does not run the pipeline.

        Checked by fingerprinting the whole tree before and after hitting every
        endpoint -- an endpoint that rebuilt a cache on disk would show up here.
        """

        def fingerprint() -> dict[str, int]:
            return {
                str(path.relative_to(populated.root)): path.stat().st_size
                for path in sorted(populated.root.rglob("*"))
                if path.is_file()
            }

        before = fingerprint()
        for path in ENDPOINTS:
            client.get(path)
        assert fingerprint() == before


class TestStoreIsolation:
    """Each app reads its own store, so tests cannot leak into each other."""

    def test_two_apps_over_different_roots_do_not_share_artifacts(
        self, populated: ArtifactStore, tmp_path: Path
    ) -> None:
        """The factory exists so the store is per-app rather than global."""
        with (
            TestClient(create_app(populated)) as full,
            TestClient(create_app(ArtifactStore(root=tmp_path / "other"))) as empty,
        ):
            assert full.get("/patterns").json()["total"] == 2
            assert empty.get("/patterns").json()["total"] == 0

    def test_clearing_the_cache_picks_up_a_rewritten_artifact(
        self, populated: ArtifactStore
    ) -> None:
        """Artifacts are cached because batch jobs write them, not requests.

        A re-export during a running server is picked up by clearing the cache,
        which is what the ``--rebuild`` path does before serving.
        """
        with TestClient(create_app(populated)) as test_client:
            assert test_client.get("/data/unified").json()["total"] == 4
            target = populated.export_dir / "unified.jsonl"
            target.write_text(json.dumps(_observation("REC-010")) + "\n", encoding="utf-8")
            assert test_client.get("/data/unified").json()["total"] == 4, "cache should hold"
            populated.clear()
            assert test_client.get("/data/unified").json()["total"] == 1
