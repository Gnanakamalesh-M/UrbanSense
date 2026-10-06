"""The dashboard's data layer (SPEC section 43).

**Nothing here launches Streamlit, and nothing here imports it.** That is the
point of the package split: :mod:`urbansense.dashboard.loaders` and
:mod:`urbansense.dashboard.views` are plain Python, so the figures the pages
display can be asserted on directly. A test that had to spin up a browser would
be slow enough that nobody would add cases to it, and the numbers are the part
worth protecting.

Four properties carry the weight:

**Both backends agree.** The same artifacts read off disk and read over HTTP
must produce the same view data. If they diverged, the dashboard would show one
thing locally and another under ``docker-compose``, and only one of them could
be right.

**Missing stays missing.** A reading with no value arrives ``None`` and leaves
``None``; an average over nothing is ``None`` rather than ``0.0``. A chart that
drew those as zero would invent a measurement, which is the failure this whole
project is built to avoid.

**The dashboard cannot contradict the report.** The experiment view reads the
stored verdict rather than ranking the arms itself, so it reports that scheduled
retraining came first when the report says so. This is the test that would catch
a page quietly written around a more flattering conclusion.

**An empty store is a normal state.** A fresh clone has no discovery report, and
every view must return empty data plus the command that would produce it, rather
than raising.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from urbansense.api.app import create_app
from urbansense.api.store import ArtifactStore
from urbansense.dashboard import views
from urbansense.dashboard.loaders import (
    ApiLoader,
    ArtifactLoader,
    DashboardData,
    build_loader,
)

# Reused from the API suite's shape: hand-written artifacts, so a test can
# assert on specific ids and the fixture stays readable.
ZONES_CSV = (
    "location_id,description,latitude,longitude,sensor_id\n"
    "ZONE-A,six-lane arterial corridor,13.082700,80.270700,sensor-A12\n"
    "ZONE-B,market district,13.060000,80.250000,sensor-B7\n"
)


def _observation(
    record_id: str,
    *,
    zone: str = "ZONE-A",
    event_time: str = "2026-01-15T08:30:00+00:00",
    value: float | None = 4_200.0,
    metric: str = "traffic_volume",
) -> dict[str, Any]:
    """One unified row, shaped as the exporter writes it."""
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
        },
    }


@pytest.fixture
def populated(tmp_path: Path) -> Path:
    """A repository root holding one artifact of every family."""
    sample = tmp_path / "data" / "sample"
    sample.mkdir(parents=True)
    (sample / "zones.csv").write_text(ZONES_CSV, encoding="utf-8")

    export = tmp_path / "data" / "processed"
    export.mkdir(parents=True)
    rows = [
        _observation("REC-001"),
        _observation("REC-002", zone="ZONE-B", event_time="2026-01-16T09:30:00+00:00"),
        # A usable observation with no measurement. Missing stays missing.
        _observation("REC-003", zone="ZONE-B", value=None, metric="average_speed"),
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
                "suspicious": 2,
                "quarantined": 4,
                "missing_values": 7,
                "unified_rows": 94,
                "duplicate_linked": 3,
                "conflicted_observations": 2,
                "quarantine_by_reason": {"out_of_range": 3, "sensor_failure": 1},
                "source_health": {
                    "sensor-A12": {
                        "status": "healthy",
                        "observations": 60,
                        "missing": 2,
                        "frozen_runs": 0,
                        "max_gap_hours": 1.0,
                    },
                    "sensor-B7": {
                        "status": "failed",
                        "observations": 36,
                        "missing": 5,
                        "frozen_runs": 1,
                        "max_gap_hours": 60.0,
                        "detail": "frozen for 2.5 days",
                    },
                },
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
                    "raw_payload": {"traffic_volume": "-1648"},
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
                "run": {
                    "tests_run": 672,
                    "fdr_level": 0.1,
                    "discoveries": 2,
                    "cells_too_sparse": 5,
                },
                "patterns": [
                    {
                        "pattern_id": "PAT-001",
                        "kind": "recurring_peak",
                        "status": "stable",
                        "description": "ZONE-B peaks on Friday evenings",
                        "strength": 2.287,
                        "confidence": 0.99,
                        "evidence": {"effect": {"value": 2.287, "lower": 1.939, "upper": 2.404}},
                        "first_detected": "2026-01-02",
                        "last_observed": "2026-03-27",
                    },
                    {
                        "pattern_id": "PAT-002",
                        "kind": "correlation",
                        "status": "weakening",
                        "description": "rainfall raises volume",
                        "strength": 0.0309,
                        "confidence": 0.95,
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
                "run": {"observations_scanned": 1_000, "flagged": 2},
                "anomalies": [
                    {
                        "record_id": "REC-900",
                        "location_id": "ZONE-B",
                        "metric": "traffic_volume",
                        "event_time": "2026-01-22T14:30:00+00:00",
                        "value": -1648.0,
                        "expected": 3_425.5,
                        "deviation_mad_sigma": 8.73,
                        "is_error": None,
                        "explanations": [{"kind": "sensor_suspect", "support": 0.75}],
                    },
                    {
                        "record_id": "REC-901",
                        "location_id": "ZONE-A",
                        "metric": "traffic_volume",
                        "event_time": "2026-02-02T18:30:00+00:00",
                        "value": 12_000.0,
                        "expected": 6_000.0,
                        "deviation_mad_sigma": 5.1,
                        "is_error": None,
                        "explanations": [{"kind": "unexplained", "support": 0.0}],
                    },
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
                    "probability": 0.9,
                    "severity": "high",
                    "congested_predicted": True,
                    "congested_actual": True,
                    "conditions": {"rain_band": "dry"},
                },
                {
                    "prediction_id": "Traffic-v1.1:ZONE-A:2026-05-14T03:30:00+00:00",
                    "model_version": "Traffic-v1.1",
                    "location_id": "ZONE-A",
                    "target_time": "2026-05-14T03:30:00+00:00",
                    "predicted": 7_000.0,
                    "actual": 7_100.0,
                    "error": -100.0,
                    "probability": 0.7,
                    "severity": "moderate",
                    "congested_predicted": True,
                    "congested_actual": True,
                    "conditions": {},
                },
                {
                    # Unresolved: no actual, no error. Must stay None.
                    "prediction_id": "Traffic-v1.1:ZONE-B:2026-05-15T10:30:00+00:00",
                    "model_version": "Traffic-v1.1",
                    "location_id": "ZONE-B",
                    "target_time": "2026-05-15T10:30:00+00:00",
                    "predicted": 2_100.0,
                    "actual": None,
                    "error": None,
                    "probability": 0.1,
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
                {"detected_at": "2026-02-12T17:30:00+00:00", "is_drift": False, "kinds": []},
                {
                    "detected_at": "2026-03-05T17:30:00+00:00",
                    "is_drift": True,
                    "kinds": ["performance"],
                    "episode_id": "EP-1",
                    "magnitude": 2.1,
                    "strongest_subject": "traffic_volume",
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
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "candidate_id": "CAND-1",
                    "raised_at": "2026-03-05T17:30:00+00:00",
                    "decision_time": "2026-03-12T00:00:00+00:00",
                    "status": "promoted",
                    "champion_version": "Traffic-v1.1",
                    "promoted_version": "Traffic-v1.2",
                    "evaluations": [{"version": "Traffic-v1.2"}],
                    "notes": ["margin met"],
                },
                {
                    "candidate_id": "CAND-2",
                    "raised_at": "2026-04-02T17:30:00+00:00",
                    "decision_time": "2026-04-09T00:00:00+00:00",
                    "status": "rejected",
                    "champion_version": "Traffic-v1.2",
                    "promoted_version": None,
                    "evaluations": [{"version": "Traffic-v1.3"}],
                    "notes": ["margin not met"],
                },
            )
        ),
        encoding="utf-8",
    )

    registry = tmp_path / "models" / "registry"
    registry.mkdir(parents=True)
    (registry / "Traffic-v1.0.json").write_text(
        json.dumps(
            {
                "version": "Traffic-v1.0",
                "status": "archived",
                "created_at": "2026-10-02T12:00:00+00:00",
                "algorithm": "HistGradientBoostingRegressor",
                "horizon_hours": 1,
                "gap_hours": 168,
                "metrics": {"test": {"mae": 310.2, "rmse": 540.0}},
            }
        ),
        encoding="utf-8",
    )
    (registry / "Traffic-v1.1.json").write_text(
        json.dumps(
            {
                "version": "Traffic-v1.1",
                "status": "champion",
                "created_at": "2026-10-03T12:00:00+00:00",
                "algorithm": "HistGradientBoostingRegressor",
                "horizon_hours": 1,
                "gap_hours": 168,
                "metrics": {"test": {"mae": 284.9, "rmse": 508.9}},
            }
        ),
        encoding="utf-8",
    )

    experiments = tmp_path / "reports" / "experiments"
    experiments.mkdir(parents=True)
    # Shaped exactly like a real report, and carrying the real Phase 8 finding:
    # scheduled first, tied with adaptive by the pre-registered rule.
    (experiments / "experiment-20261004T114237Z.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-10-04T11:42:37+00:00",
                "pre_registered_config": {
                    "pre_registered": "2026-10-04",
                    "seeds": [20260101, 20260202],
                    "retrain_every_days": 28,
                },
                "verdict": {
                    "ranking_by_mae": ["scheduled", "adaptive", "static"],
                    "best": "scheduled",
                    "ties": [["scheduled", "adaptive"]],
                    "conclusion": (
                        "scheduled and adaptive are not distinguished by this "
                        "experiment: a gap smaller than the seed-to-seed spread"
                    ),
                },
                "arms": {
                    "static": {
                        "mae": {"mean": 292.6, "sd": 6.3, "seeds": 2},
                        "rmse": {"mean": 532.5, "sd": 23.8},
                        "congestion": {"f1": {"mean": 0.69, "sd": 0.026}},
                        "brier": {"mean": 0.096, "sd": 0.0038},
                        "cost": {
                            "retrains": {"mean": 0.0},
                            "fits": {"mean": 0.0},
                            "training_rows": {"mean": 0.0},
                        },
                        "slice_mae": {"zone ZONE-A": {"mean": 567.4}},
                    },
                    "scheduled": {
                        "mae": {"mean": 229.5, "sd": 4.9, "seeds": 2},
                        "rmse": {"mean": 388.0, "sd": 9.5},
                        "congestion": {"f1": {"mean": 0.806, "sd": 0.004}},
                        "brier": {"mean": 0.0651, "sd": 0.0009},
                        "cost": {
                            "retrains": {"mean": 4.0},
                            "fits": {"mean": 4.0},
                            "training_rows": {"mean": 77629.0},
                        },
                        "slice_mae": {"zone ZONE-A": {"mean": 343.3}},
                    },
                    "adaptive": {
                        "mae": {"mean": 237.5, "sd": 9.4, "seeds": 2},
                        "rmse": {"mean": 400.2, "sd": 14.8},
                        "congestion": {"f1": {"mean": 0.794, "sd": 0.007}},
                        "brier": {"mean": 0.0682, "sd": 0.002},
                        "cost": {
                            "retrains": {"mean": 3.2},
                            "fits": {"mean": 7.6},
                            "training_rows": {"mean": 97788.0},
                        },
                        "slice_mae": {"zone ZONE-A": {"mean": 367.6}},
                    },
                },
                "regressions": {
                    "static": ["zone ZONE-A"],
                    "scheduled": [],
                    "adaptive": ["zone ZONE-A"],
                },
                "limitations": ["Synthetic data.", "One drift type."],
            }
        ),
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def direct(populated: Path) -> ArtifactLoader:
    """The on-disk backend."""
    return ArtifactLoader(root=populated)


@pytest.fixture
def over_http(populated: Path) -> Iterator[ApiLoader]:
    """The HTTP backend, driven through the real API in-process.

    ``_get`` is pointed at ``TestClient`` rather than at a socket, so the loader
    is exercised against the payloads the API genuinely returns without this
    suite having to bind a port. The parsing under test is the loader's, and the
    payloads under test are the API's.
    """
    from fastapi.testclient import TestClient

    client = TestClient(create_app(ArtifactStore(root=populated)))
    loader = ApiLoader(base_url="http://testserver")

    def fetch(path: str, **params: str | int | None) -> object | None:
        query = {key: value for key, value in params.items() if value is not None}
        response = client.get(path, params=query)
        if response.status_code != 200:
            return None
        return response.json()

    # Replacing the one method that touches the network. Everything above it --
    # paging, note extraction, shaping -- is the real implementation.
    loader._get = fetch  # type: ignore[method-assign]
    with client:
        yield loader


@pytest.fixture
def empty(tmp_path: Path) -> ArtifactLoader:
    """A loader over an empty directory -- the fresh-clone case."""
    return ArtifactLoader(root=tmp_path / "nothing")


class TestLoadersSatisfyTheProtocol:
    """Both backends are interchangeable by construction."""

    def test_the_direct_loader_is_dashboard_data(self, direct: ArtifactLoader) -> None:
        """Structural typing, checked at runtime."""
        assert isinstance(direct, DashboardData)

    def test_the_api_loader_is_dashboard_data(self) -> None:
        """Same protocol, different transport."""
        assert isinstance(ApiLoader(base_url="http://example.invalid"), DashboardData)

    def test_build_loader_defaults_to_reading_files(self, populated: Path) -> None:
        """No API URL means no server needed, which is the point of the default."""
        assert isinstance(build_loader(root=populated), ArtifactLoader)

    def test_build_loader_switches_on_an_api_url(self) -> None:
        """An API URL selects HTTP."""
        loader = build_loader(api_url="http://127.0.0.1:8000")
        assert isinstance(loader, ApiLoader)
        assert loader.base_url == "http://127.0.0.1:8000"


class TestNoStreamlitRequired:
    """The data layer must not need the dashboard extra installed."""

    def test_neither_module_imports_streamlit(self) -> None:
        """Only ``app.py`` may import it.

        Asserted on the source rather than by mocking the import system,
        because this is the property CI relies on: it does not install
        streamlit, so a stray import in ``views`` or ``loaders`` would break
        the build in a way that looks unrelated.
        """
        package = Path(views.__file__).parent
        for name in ("loaders.py", "views.py", "__init__.py"):
            source = (package / name).read_text(encoding="utf-8")
            code = [
                line
                for line in source.splitlines()
                if line.startswith(("import ", "from ")) and "streamlit" in line
            ]
            assert not code, f"{name} imports streamlit: {code}"

    def test_the_views_module_does_not_need_pandas_either(self) -> None:
        """Pandas arrives with streamlit, so it is confined to ``app.py`` too."""
        source = Path(views.__file__).read_text(encoding="utf-8")
        assert "import pandas" not in source


class TestCityOverview:
    """Risk by zone, and the headline figures."""

    def test_zone_risk_averages_the_recorded_probabilities(self, direct: ArtifactLoader) -> None:
        """Risk is the mean calibrated probability over stored outcomes."""
        panel = views.zone_risk(direct)
        by_zone = {row["location_id"]: row for row in panel.rows}

        assert by_zone["ZONE-A"]["risk"] == pytest.approx((0.9 + 0.7) / 2)
        assert by_zone["ZONE-A"]["predictions"] == 2
        assert by_zone["ZONE-A"]["mae"] == pytest.approx((52.7 + 100.0) / 2)
        assert by_zone["ZONE-B"]["risk"] == pytest.approx(0.1)

    def test_a_zone_with_no_resolved_prediction_has_no_error(self, direct: ArtifactLoader) -> None:
        """None, not zero: ZONE-B's only prediction is unresolved."""
        by_zone = {row["location_id"]: row for row in views.zone_risk(direct).rows}
        assert by_zone["ZONE-B"]["mae"] is None

    def test_zone_coordinates_are_attached_for_the_map(self, direct: ArtifactLoader) -> None:
        """Read from the committed zones.csv, parsed as floats."""
        by_zone = {row["location_id"]: row for row in views.zone_risk(direct).rows}
        assert by_zone["ZONE-A"]["latitude"] == pytest.approx(13.0827)
        assert by_zone["ZONE-A"]["longitude"] == pytest.approx(80.2707)
        assert by_zone["ZONE-A"]["description"] == "six-lane arterial corridor"

    def test_map_points_use_the_column_names_streamlit_expects(
        self, direct: ArtifactLoader
    ) -> None:
        """``lat``/``lon``, and a size that grows with risk."""
        points = views.map_points(direct)
        assert len(points) == 2
        assert {"lat", "lon", "size"} == set(points[0])
        sizes = [point["size"] for point in points]
        assert all(size > 0 for size in sizes)

    def test_a_zone_without_coordinates_is_listed_but_not_mapped(
        self, direct: ArtifactLoader, populated: Path
    ) -> None:
        """Losing a marker beats losing the zone.

        A zone that appears in the predictions but not in zones.csv still gets
        a row in the table; it simply cannot be placed on the map.
        """
        (populated / "data" / "sample" / "zones.csv").write_text(
            "location_id,description,latitude,longitude,sensor_id\nZONE-A,arterial,,,sensor-A12\n",
            encoding="utf-8",
        )
        direct.clear()
        zones = {row["location_id"] for row in views.zone_risk(direct).rows}
        assert "ZONE-A" in zones and "ZONE-B" in zones
        assert views.map_points(direct) == []

    def test_the_headline_reads_every_figure_from_an_artifact(self, direct: ArtifactLoader) -> None:
        """Champion, anomalies, drift, and the data-quality counts."""
        headline = views.overview_headline(direct)
        assert headline["champion"] == "Traffic-v1.1"
        assert headline["model_versions"] == 2
        assert headline["anomalies"] == 2
        assert headline["unexplained_anomalies"] == 1
        assert headline["drift_evaluations"] == 2
        assert headline["drift_firings"] == 1
        assert headline["quarantined"] == 4
        assert headline["missing_values"] == 7
        assert headline["conflicts"] == 1


class TestPredictions:
    """Logged predictions, with their outcomes where known."""

    def test_filtering_by_zone_and_date(self, direct: ArtifactLoader) -> None:
        """Both pickers narrow the same underlying log."""
        assert len(views.prediction_rows(direct, zone="ZONE-A").rows) == 2
        assert len(views.prediction_rows(direct, zone="ZONE-B").rows) == 1
        assert len(views.prediction_rows(direct, date="2026-05-14").rows) == 2
        assert len(views.prediction_rows(direct, zone="ZONE-B", date="2026-05-14").rows) == 0

    def test_an_unresolved_prediction_keeps_null_actual_and_error(
        self, direct: ArtifactLoader
    ) -> None:
        """Pending is not zero error.

        The single most load-bearing assertion in this file: a prediction whose
        outcome is unknown must not be rendered as a perfect one.
        """
        panel = views.prediction_rows(direct, zone="ZONE-B")
        row = panel.rows[0]
        assert row["actual"] is None
        assert row["error"] is None
        assert row["congested_actual"] is None
        assert panel.extra["pending"] == 1
        assert panel.extra["resolved"] == 0
        assert panel.extra["mae"] is None

    def test_the_mae_counts_only_resolved_predictions(self, direct: ArtifactLoader) -> None:
        """An unresolved row must not dilute the error."""
        panel = views.prediction_rows(direct, zone="ZONE-A")
        assert panel.extra["resolved"] == 2
        assert panel.extra["mae"] == pytest.approx((52.7 + 100.0) / 2)

    def test_rows_are_ordered_by_target_time(self, direct: ArtifactLoader) -> None:
        """A time series has to be drawn in time order."""
        times = [row["target_time"] for row in views.prediction_rows(direct).rows]
        assert times == sorted(times)

    def test_the_pickers_offer_what_exists(self, direct: ArtifactLoader) -> None:
        """Zones and dates come from the log, not from a hard-coded list."""
        assert views.prediction_zones(direct) == ["ZONE-A", "ZONE-B"]
        assert views.prediction_dates(direct, zone="ZONE-A") == ["2026-05-14"]


class TestPatternsAndAnomalies:
    """Claims with their evidence, and flags without verdicts."""

    def test_the_hypothesis_count_travels_with_the_findings(self, direct: ArtifactLoader) -> None:
        """One pattern out of 672 tests is a different claim from one out of ten."""
        panel = views.pattern_rows(direct)
        assert len(panel) == 2
        assert panel.extra["tests_run"] == 672
        assert panel.extra["fdr_level"] == pytest.approx(0.1)
        assert panel.extra["cells_too_sparse"] == 5

    def test_the_effect_interval_is_carried_through(self, direct: ArtifactLoader) -> None:
        """A strength without its interval cannot be judged."""
        rows = {row["pattern_id"]: row for row in views.pattern_rows(direct).rows}
        assert rows["PAT-001"]["effect"] == pytest.approx(2.287)
        assert rows["PAT-001"]["lower"] == pytest.approx(1.939)
        assert rows["PAT-001"]["upper"] == pytest.approx(2.404)

    def test_patterns_are_counted_by_status(self, direct: ArtifactLoader) -> None:
        """New, recurring, strengthening, weakening (SPEC section 43)."""
        assert views.pattern_rows(direct).extra["by_status"] == {"stable": 1, "weakening": 1}

    def test_an_anomaly_is_never_presented_as_an_error(self, direct: ArtifactLoader) -> None:
        """is_error stays null on every row, and the explanations stay plural."""
        panel = views.anomaly_rows(direct)
        assert len(panel) == 2
        assert all(row["is_error"] is None for row in panel.rows)
        assert "is_error is null" in panel.extra["verdict"]

    def test_the_leading_explanation_is_a_candidate_not_a_cause(
        self, direct: ArtifactLoader
    ) -> None:
        """Ranked candidates, carried through in order."""
        rows = {row["record_id"]: row for row in views.anomaly_rows(direct).rows}
        assert rows["REC-900"]["leading_explanation"] == "sensor_suspect"
        assert rows["REC-901"]["leading_explanation"] == "unexplained"


class TestDataHealth:
    """Per-source first, then the unified counts."""

    def test_source_health_comes_per_source(self, direct: ArtifactLoader) -> None:
        """SPEC section 20: a healthy total hides an unhealthy feed."""
        panel = views.source_health(direct)
        rows = {row["source_id"]: row for row in panel.rows}
        assert rows["sensor-A12"]["status"] == "healthy"
        assert rows["sensor-B7"]["status"] == "failed"
        assert panel.extra["unhealthy"] == ["sensor-B7"]

    def test_missing_values_are_reported_beside_valid_rows(self, direct: ArtifactLoader) -> None:
        """96 valid rows means something different when 7 carry no measurement."""
        summary = views.data_health(direct)
        assert summary["valid"] == 96
        assert summary["missing_values"] == 7
        assert "never filled" in summary["missing_policy"]

    def test_quarantine_is_counted_by_reason(self, direct: ArtifactLoader) -> None:
        """From the records themselves where they exist."""
        summary = views.data_health(direct)
        assert summary["quarantine_by_reason"]["out_of_range"] >= 1

    def test_a_conflict_keeps_every_candidate_value(self, direct: ArtifactLoader) -> None:
        """Recorded, never resolved by overwriting."""
        panel = views.conflict_rows(direct)
        assert len(panel) == 1
        assert panel.rows[0]["candidates"] == 2
        assert "sensor-C1" in panel.rows[0]["sources"]

    def test_the_unified_sample_carries_provenance(self, direct: ArtifactLoader) -> None:
        """Asking where a value came from is answerable from the row on screen."""
        panel = views.unified_sample(direct)
        assert len(panel) == 3
        assert panel.rows[0]["origin"] == "sensor-A12"
        assert panel.rows[0]["pipeline_version"] == "ingestion-1.0.0"
        assert panel.rows[0]["source_type"] == "historical"

    def test_a_missing_measurement_stays_null_through_the_view(
        self, direct: ArtifactLoader
    ) -> None:
        """Never zero, and never marked imputed."""
        panel = views.unified_sample(direct)
        blanks = [row for row in panel.rows if row["value"] is None]
        assert len(blanks) == 1
        assert blanks[0]["is_imputed"] is False
        assert panel.extra["missing_values"] == 1

    def test_filtering_the_sample_by_zone(self, direct: ArtifactLoader) -> None:
        """The zone filter reaches the loader."""
        assert len(views.unified_sample(direct, zone="ZONE-B").rows) == 2


class TestModelEvolution:
    """Versions, champion, drift and candidate decisions."""

    def test_exactly_one_champion_is_surfaced(self, direct: ArtifactLoader) -> None:
        """The registry's invariant, shown so a breach would be visible."""
        panel = views.model_timeline(direct)
        assert panel.extra["champion"] == "Traffic-v1.1"
        assert panel.extra["champion_count"] == 1
        assert panel.extra["versions"] == 2

    def test_archived_versions_keep_their_rows(self, direct: ArtifactLoader) -> None:
        """Knowing which updates failed is the experimental record."""
        statuses = {row["version"]: row["status"] for row in views.model_timeline(direct).rows}
        assert statuses == {"Traffic-v1.0": "archived", "Traffic-v1.1": "champion"}

    def test_registry_metrics_are_read_from_the_test_segment(self, direct: ArtifactLoader) -> None:
        """Test-period error, not validation, is what a reader wants."""
        rows = {row["version"]: row for row in views.model_timeline(direct).rows}
        assert rows["Traffic-v1.1"]["test_mae"] == pytest.approx(284.9)

    def test_quiet_drift_evaluations_are_kept(self, direct: ArtifactLoader) -> None:
        """A false-alarm rate cannot be read from firings alone."""
        panel = views.drift_rows(direct)
        assert panel.extra["evaluations"] == 2
        assert panel.extra["firings"] == 1
        assert [row["is_drift"] for row in panel.rows] == [False, True]

    def test_both_promoted_and_rejected_candidates_are_shown(self, direct: ArtifactLoader) -> None:
        """A rejected candidate is the system working, not a gap in the log."""
        panel = views.candidate_rows(direct)
        assert panel.extra["candidates"] == 2
        assert panel.extra["promoted"] == 1
        assert panel.extra["rejected"] == 1


class TestExperimentViewCannotContradictTheReport:
    """The page reports what the experiment recorded, flattering or not."""

    def test_the_verdict_is_read_from_the_report(self, direct: ArtifactLoader) -> None:
        """Not recomputed.

        The stored report ranks scheduled retraining first and calls it a tie
        with the adaptive system. The view must say exactly that: a page that
        derived its own ordering could quietly tell a better story than the
        experiment found.
        """
        panel = views.experiment_summary(direct)
        assert panel.extra["best"] == "scheduled"
        assert panel.extra["ranking"] == ["scheduled", "adaptive", "static"]
        assert ["scheduled", "adaptive"] in panel.extra["ties"]
        assert "not distinguished" in panel.extra["conclusion"]

    def test_the_arms_are_ordered_by_the_reports_ranking(self, direct: ArtifactLoader) -> None:
        """Scheduled ahead of adaptive, because the report says so."""
        arms = [row["arm"] for row in views.experiment_summary(direct).rows]
        assert arms == ["scheduled", "adaptive", "static"]

    def test_cost_is_reported_beside_accuracy(self, direct: ArtifactLoader) -> None:
        """An accuracy table alone lets the most expensive arm look best free."""
        rows = {row["arm"]: row for row in views.experiment_summary(direct).rows}
        assert rows["scheduled"]["mae"] == pytest.approx(229.5)
        assert rows["scheduled"]["training_rows"] == pytest.approx(77629.0)
        assert rows["adaptive"]["mae"] == pytest.approx(237.5)
        assert rows["adaptive"]["training_rows"] == pytest.approx(97788.0)
        assert rows["adaptive"]["fits"] == pytest.approx(7.6)
        # The finding, as data rather than as prose: adaptive cost more.
        assert rows["adaptive"]["training_rows"] > rows["scheduled"]["training_rows"]

    def test_where_each_arm_lost_is_carried_through(self, direct: ArtifactLoader) -> None:
        """Including for the winning arm."""
        extra = views.experiment_summary(direct).extra
        assert extra["regressions"]["adaptive"] == ["zone ZONE-A"]
        assert extra["regressions"]["scheduled"] == []

    def test_the_limitations_travel_with_the_result(self, direct: ArtifactLoader) -> None:
        """A number without its caveats is not a finding."""
        assert views.experiment_summary(direct).extra["limitations"]

    def test_a_report_that_said_adaptive_won_would_be_shown_that_way(
        self, direct: ArtifactLoader, populated: Path
    ) -> None:
        """The view has no favourite.

        The counterpart to the test above: rewrite the stored verdict so
        adaptive wins, and the page must follow. A view hard-coded either way
        would fail one of these two.
        """
        path = next((populated / "reports" / "experiments").glob("experiment-*.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["verdict"] = {
            "ranking_by_mae": ["adaptive", "scheduled", "static"],
            "best": "adaptive",
            "ties": [],
            "conclusion": "adaptive has the lowest error",
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        direct.clear()

        panel = views.experiment_summary(direct)
        assert panel.extra["best"] == "adaptive"
        assert [row["arm"] for row in panel.rows] == ["adaptive", "scheduled", "static"]

    def test_per_slice_errors_are_available_for_every_arm(self, direct: ArtifactLoader) -> None:
        """The counterweight to the headline table."""
        panel = views.experiment_slices(direct)
        row = next(item for item in panel.rows if item["slice"] == "zone ZONE-A")
        assert row["scheduled"] == pytest.approx(343.3)
        assert row["adaptive"] == pytest.approx(367.6)


class TestBothBackendsAgree:
    """Reading files and reading the API must produce the same view data."""

    def test_the_same_artifacts_give_the_same_headline(
        self, direct: ArtifactLoader, over_http: ApiLoader
    ) -> None:
        """Otherwise the dashboard says one thing locally and another in Docker."""
        assert views.overview_headline(direct) == views.overview_headline(over_http)

    def test_the_same_artifacts_give_the_same_predictions(
        self, direct: ArtifactLoader, over_http: ApiLoader
    ) -> None:
        """Including the unresolved row's nulls."""
        assert views.prediction_rows(direct).rows == views.prediction_rows(over_http).rows

    def test_the_same_artifacts_give_the_same_patterns(
        self, direct: ArtifactLoader, over_http: ApiLoader
    ) -> None:
        """Run header included, which the API serves on the detail route."""
        left, right = views.pattern_rows(direct), views.pattern_rows(over_http)
        assert left.rows == right.rows
        assert left.extra["tests_run"] == right.extra["tests_run"] == 672

    def test_the_same_artifacts_give_the_same_experiment_verdict(
        self, direct: ArtifactLoader, over_http: ApiLoader
    ) -> None:
        """The finding must not depend on the transport."""
        assert (
            views.experiment_summary(direct).extra["best"]
            == views.experiment_summary(over_http).extra["best"]
            == "scheduled"
        )

    def test_the_same_artifacts_give_the_same_data_health(
        self, direct: ArtifactLoader, over_http: ApiLoader
    ) -> None:
        """Counts and per-source rows alike."""
        assert views.data_health(direct) == views.data_health(over_http)
        assert views.source_health(direct).rows == views.source_health(over_http).rows

    def test_the_api_backend_reports_availability(self, over_http: ApiLoader) -> None:
        """Read from /health rather than guessed."""
        availability = over_http.availability()
        assert availability["patterns"] is True
        assert availability["experiments"] is True

    def test_the_api_backend_has_no_geometry_and_says_so(self, over_http: ApiLoader) -> None:
        """The API serves no coordinates, so the map degrades rather than lies.

        Documented behaviour, not an oversight: zones still appear, without
        points, and the page says the map is empty by design.
        """
        zones = over_http.zones()
        assert {zone["location_id"] for zone in zones} == {"ZONE-A", "ZONE-B"}
        assert all(zone["latitude"] is None for zone in zones)
        assert views.map_points(over_http) == []


class TestAnUnreachableApiDegrades:
    """A dead server produces a dashboard that says so, not a traceback."""

    def test_every_method_returns_empty_rather_than_raising(self) -> None:
        """Nothing here may propagate a connection error."""
        loader = ApiLoader(base_url="http://127.0.0.1:1", timeout=0.3)
        assert loader.availability() == {}
        assert loader.unified() == []
        assert loader.patterns() == []
        assert loader.predictions() == []
        assert loader.models() == []
        assert loader.experiments() == []
        assert loader.data_quality() is None
        assert loader.drift() == {"events": [], "episodes": [], "candidates": []}
        assert loader.reachable is False

    def test_the_views_survive_an_unreachable_api(self) -> None:
        """Every page renders empty rather than failing."""
        loader = ApiLoader(base_url="http://127.0.0.1:1", timeout=0.3)
        assert views.zone_risk(loader).is_empty
        assert views.pattern_rows(loader).is_empty
        assert views.experiment_summary(loader).is_empty
        assert views.overview_headline(loader)["champion"] is None


class TestAnEmptyStoreIsNormal:
    """A fresh clone has computed nothing, and that is not an error."""

    def test_every_view_returns_empty_without_raising(self, empty: ArtifactLoader) -> None:
        """No artifact, no exception."""
        assert views.zone_risk(empty).is_empty
        assert views.prediction_rows(empty).is_empty
        assert views.pattern_rows(empty).is_empty
        assert views.anomaly_rows(empty).is_empty
        assert views.source_health(empty).is_empty
        assert views.conflict_rows(empty).is_empty
        assert views.unified_sample(empty).is_empty
        assert views.model_timeline(empty).is_empty
        assert views.drift_rows(empty).is_empty
        assert views.candidate_rows(empty).is_empty
        assert views.experiment_summary(empty).is_empty
        assert views.experiment_slices(empty).is_empty

    @pytest.mark.parametrize(
        "panel_name",
        ["zone_risk", "pattern_rows", "anomaly_rows", "model_timeline", "experiment_summary"],
    )
    def test_an_empty_panel_names_the_command_that_would_fill_it(
        self, empty: ArtifactLoader, panel_name: str
    ) -> None:
        """Nothing run yet has to stay distinguishable from nothing found."""
        panel = getattr(views, panel_name)(empty)
        assert panel.note, f"{panel_name} gave no hint about why it is empty"
        assert "run:" in panel.note

    def test_the_headline_shows_blanks_not_zeros(self, empty: ArtifactLoader) -> None:
        """A zero is a measurement; a blank is the absence of one."""
        headline = views.overview_headline(empty)
        assert headline["champion"] is None
        assert headline["quarantined"] is None
        assert headline["missing_values"] is None
        assert headline["unified_rows"] is None

    def test_data_health_reports_nulls_with_a_note(self, empty: ArtifactLoader) -> None:
        """Rather than a tidy row of zeros that would read as counts."""
        summary = views.data_health(empty)
        assert summary["valid"] is None
        assert summary["note"]

    def test_readiness_lists_every_family_as_missing(self, empty: ArtifactLoader) -> None:
        """Drives the sidebar, so an empty page is never mistaken for a broken one."""
        rows = views.readiness(empty)
        assert rows
        assert not any(row["ready"] for row in rows)
        assert all(row["note"] for row in rows)

    def test_the_synthetic_banner_exists_and_says_synthetic(self) -> None:
        """Shown on every page; the most important caveat in the project."""
        assert "synthetic" in views.SYNTHETIC_BANNER.lower()
        assert "no real city" in views.SYNTHETIC_BANNER.lower()
